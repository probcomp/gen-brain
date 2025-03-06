import numpy as np
import jax.numpy as jnp
import jax
import math

np.seterr(all="ignore")

def normalize(x):
    return np.array(x) / sum(x)

class Resampler():
    def __init__(self, particles):
        self.particles = particles
        self.particles_before_resampling = copy.deepcopy(particles)
        self.normalizer_spikes = {str(p_id): [] for p_id in range(len(self.particles))}
        self.resampler_spikes = {str(p_id): [] for p_id in range(len(self.particles))}
        self.component_dict = {
            "norm_neurons": self.normalizer_spikes,
            "resampler_wta": self.resampler_spikes,
        }
        self.t = 0
        self.normalizer_λ = 0.1
        self.winners = []
        self.log_weights = 0
        self.log_total_weight = 0
        self.ml_setpoint = 1
        self.ess_threshold = jnp.inf
        self.resampled_on_step = True
        self.normalizer_λ = 0.5
        self.resampler_probs = []
        self.resampler_start_time = 0
        self.resampler_end_time = 0
        self.ess_threshold = len(particles) / 40

    def populate_normalizer(self, length_sim, norm_probs):
        sim_lambdas = self.normalizer_λ * norm_probs
        for i, λ in enumerate(sim_lambdas):
            self.normalizer_spikes[str(i)] = poisson_process(λ, length_sim, 0)

    def make_new_particles(self):
        new_particles = [
            Particle(
                self.particles[0].neurons_per_assembly,
                self.particles[0].latent_variables,
                self.particles[0].observed_variables,
            )
            for i in range(len(self.particles))
        ]
        return new_particles

    def clip_normalizer_spikes(self):
        last_resampler_spike = np.max(
            np.concatenate([spikes for i, spikes in self.resampler_spikes.items()])
        )
        for k in self.normalizer_spikes.keys():
            self.normalizer_spikes[k] = self.normalizer_spikes[k][
                self.normalizer_spikes[k] <= last_resampler_spike
            ]
        self.resampler_end_time = self.resampler_start_time + last_resampler_spike

    def find_last_particle_spike(self):
        last_spike = 0
        for p in self.particles:
            for ss in p.samplescores.values():
                candidate_last_spike = ss.score_complete_time
                if candidate_last_spike > last_spike:
                    last_spike = candidate_last_spike
        self.resampler_start_time = last_spike

    # this is accomplishing the goals of the particle mux.
    # for now, the circuit isn't operating on the state buffer
    def detect_winner(self):
        spike_at_t = [ns[-1] for ns in self.normalizer_spikes.values()]
        if sum(spike_at_t) > 0:
            spiking_norm_neurons = np.nonzero(spike_at_t)[0]
            if len(spiking_norm_neurons) == 1:
                winner = spiking_norm_neurons[0]
            else:
                winner = np.random.randint(
                    spiking_norm_neurons[0], spiking_norm_neurons[-1] + 1
                )
        else:
            winner = float("NaN")

        if math.isnan(winner):
            self.pad_resampler()
        else:
            for i, rs_key in enumerate(self.resampler_spikes.keys()):
                if i == winner:
                    self.resampler_spikes[rs_key].append(1)
                    self.winners.append(i)
                else:
                    self.resampler_spikes[rs_key].append(0)
        return winner

    def norm_and_resample(self):
        self.log_weights = np.array(
            list(map(lambda p: p.score + p.prevstep_score, self.particles))
        )
        self.log_total_weight = jax.nn.logsumexp(self.log_weights)
        starting_unnormed_probs = np.exp(self.log_weights)
        if np.sum(starting_unnormed_probs) == 0:
            starting_unnormed_probs = np.ones(len(self.particles))
        # np.exp takes all of the finite probabilities and makes them 0.0.
        # then you have fully unnormed weights.
        normalizer_probs = normalize(starting_unnormed_probs)
        print("resampler score")
        print(normalizer_probs)
        self.resampler_probs = normalizer_probs
        self.populate_normalizer(1000, normalizer_probs)
        # just have to find which normalizer each of the first n_particles spikes come from.
        # then call switch_states().
        normalizer_stimes = [
            self.normalizer_spikes[str(p_id)] for p_id, p in enumerate(self.particles)
        ]
        maxlen_stimes = np.max([len(v) for v in normalizer_stimes])
        norm_spike_arrays_padded = np.array(
            [
                np.pad(
                    v, (0, maxlen_stimes - len(v)), "constant", constant_values=np.nan
                )
                for v in normalizer_stimes
            ]
        )
        winning_indices = np.argsort(norm_spike_arrays_padded.flatten())[
            : len(self.particles)
        ]
        winning_assemblies_and_indices = np.unravel_index(
            winning_indices, norm_spike_arrays_padded.shape
        )
        self.winners = winning_assemblies_and_indices[0]
        # index 0 is the neuron, index 1 is the spike index within the neuron. just need to put the
        # spiketimes into resampler_spikes at the correct index. use np.concatenate like:
        for neuron, spiketime in zip(
            winning_assemblies_and_indices[0], winning_assemblies_and_indices[1]
        ):
            self.resampler_spikes[str(neuron)].append(
                normalizer_stimes[neuron][spiketime]
            )
        self.find_last_particle_spike()
        self.clip_normalizer_spikes()
        self.switch_states(self.ess_threshold)

    # this is actually a very complex biophysical problem!
    # i think here you just make a new particle and assign its choices.
    # but then you'll lose the spikes.
    def switch_states(self, ess_threshold):
        new_particles = self.make_new_particles()
        log_ess = -jax.nn.logsumexp(2 * (self.log_weights - self.log_total_weight))
        ess = jnp.exp(log_ess)
        if ess >= ess_threshold:
            self.winners = jnp.arange(len(self.winners))
            self.resampled_on_step = False
            for i, new_p in enumerate(new_particles):
                new_p.prevstep_score = self.particles[i].score
        for w, p in zip(self.winners, new_particles):
            p.resampled_choicemap = self.particles[w].choicemap
        self.particles = new_particles

#        self.populate_state_buffer()


class Particle():
    def __init__(self, neurons_per_assembly, latent_variables, observed_variables):
        self.latent_variables = latent_variables
        self.observed_variables = observed_variables
        self.samplescores = {
            v["variable"]: lambda pq_probs: SampleScore(
                neurons_per_assembly, pq_probs
            )
            for v in latent_variables
        }
        self.likelihood_circuits = {}
        for v in observed_variables:
            if len(v["subtraced"]) == 0:
                self.likelihood_circuits[v["variable"]] = lambda probs: P_Unit(
                    neurons_per_assembly, probs
                )
            else:
                self.likelihood_circuits[v["variable"]] = (
                    lambda probs: ProbabilityMap(probs)
                )
        self.variables = latent_variables + observed_variables
        self.p_units = {
            v["variable"]: lambda probs: P_Unit_Analog(neurons_per_assembly, probs)
            for v in self.variables
        }
        self.score = 0
        self.variable_ids = [v["variable"] for v in self.variables]
        self.completed_variables = []
        self.resampled_choicemap = {}
        self.choicemap = {}
        self.timestamp = 0
        self.likelihood_score = 0.0
        self.prevstep_score = 0.0
        self.neurons_per_assembly = neurons_per_assembly

    def populate_state_buffer(self):
        choices = self.resampled_choicemap
        for k, v in self.samplescores.items():
            state = choices[k]
            v.state_buffer[str(state)].append(0.0)

    def start_sampler(self, v, prbs, race_start_time):
        self.samplescores[v] = self.samplescores[v](prbs)
        self.samplescores[v].race_start_time = race_start_time
        self.samplescores[v].sample_proposal()
        self.choicemap[v] = self.samplescores[v].state
        return self.samplescores[v].sample_time

    def start_pq_scoring(self, v, prbs):
        self.samplescores[v].initialize_p_assemblies(prbs)
        self.samplescores[v].run_scoring_circuitry()
        self.completed_variables.append(v)

    def score_likelihood(self, varbs_and_probvecs, obs_state):
        for v, prbs in varbs_and_probvecs.items():
            self.likelihood_circuits[v] = self.likelihood_circuits[v](
                prbs
            ).run_scoring_circuitry(obs_state)
            self.choicemap[v] = obs_state
            self.completed_variables.append(v)
        self.likelihood_score = np.sum(
            [lc.p for lc in self.likelihood_circuits.values()]
        )

    def score_particle(self):
        if set(self.completed_variables) == set(self.variable_ids):
            self.score = (
                np.sum([ss.p + ss.one_over_q for ss in self.samplescores.values()])
                + self.likelihood_score
            )
            return self
        else:
            raise Exception("not all scores accumulated")

    def timestamp_particle(self):
        self.timestamp = np.max(ss.scoring_time for ss in self.samplescores.values())


class SampleScore():
    def __init__(self, neurons_per_assembly, catprobs):
        self.neurons_per_assembly = neurons_per_assembly
        self.num_states = len(catprobs[0])
        # wtf for some reason this is always vy.
        self.kq = 20
        self.kp = 60
        # want to eventually have this set by a biological neuron. population lambda should be a norm setpoint.
        self.population_λ = {"q": 0.1, "p": 0.1}
        self.p_initialized = False
        q_assembly_indices = ["q" + str(i) for i in range(self.num_states)]
        catprobs_q = catprobs[0]
        self.catprobs = [catprobs_q, []]
        q_lambdas = catprobs_q * self.population_λ["q"]
        self.assemblies = {
            i: [[] for i in range(self.neurons_per_assembly)]
            for i in q_assembly_indices
        }
        self.sim_lambdas = dict(zip(q_assembly_indices, q_lambdas))
        self.tik = {"q": [], "p": []}
        self.wta = {str(i): [] for i in range(self.num_states)}
        self.p = 0
        self.one_over_q = 0
        self.p_tik = 0
        self.q_tik = 0
        self.race_start_time = 0
        self.sample_time = 0
        self.score_start_time = 0
        self.score_complete_time = 0
        self.state = float("NaN")
        self.component_dict = {
            "mux": self.mux,
            "tik": self.tik,
            "accum": self.accum,
            "wta": self.wta,
            "assemblies": self.assemblies,
        }
        if len(catprobs) == 2:
            catprobs_p = catprobs[1]
            self.initialize_p_assemblies(catprobs_p)
        self.num_pp_generated = {"p": 0, "q": 0}
        self.max_recursion_passes = 3
        self.pp_length = 20
        staterange = np.hstack((np.arange(self.num_states), np.arange(self.num_states)))
        ps_qs = ["p"] * self.num_states + ["q"] * self.num_states
        self.mux = {pq + str(s): [] for pq, s in zip(ps_qs, staterange)}
        self.accum = {str(s): [] for s in np.arange(self.neurons_per_assembly)}
        self.state_buffer = {str(s): [] for s in np.arange(self.num_states)}
        self.component_dict["state_buffer"] = self.state_buffer
        self.component_dict["accum"] = self.accum
        self.component_dict["mux"] = self.mux

        # update this at resample time (i.e. make state_buffer the resampled state for each particle
        # in switch states.
        #        self.kp = int(self.pp_length / 10)
        self.kp = 40
        self.kq = 10
        self.population_λ = {"q": 0.3, "p": 0.3}

    def update_sim_lambdas(self, p_or_q):
        lambdas = self.catprobs[int(p_or_q == "p")] * self.population_λ[p_or_q]
        assembly_indices = [p_or_q + str(i) for i in range(self.num_states)]
        sim_lambdas = zip(assembly_indices, lambdas)
        self.sim_lambdas.update(dict(sim_lambdas))
        return assembly_indices

    def initialize_p_assemblies(self, catprobs_p):
        self.catprobs[1] = catprobs_p
        p_assembly_indices = self.update_sim_lambdas("p")
        self.assemblies.update(
            {
                i: [[] for n in range(self.neurons_per_assembly)]
                for i in p_assembly_indices
            }
        )
        self.populate_assemblies("p", self.pp_length, self.score_start_time)

    def populate_assemblies(self, q_or_p, length_sim, starttime):
        for k, λ in self.sim_lambdas.items():
            if k[0] == q_or_p:
                for neuron in range(self.neurons_per_assembly):
                    self.assemblies[k][neuron] = np.concatenate(
                        (
                            self.assemblies[k][neuron],
                            poisson_process(λ, length_sim, starttime),
                        )
                    )

    def clip_assemblies_to_scoretime(self):
        for k in self.assemblies.keys():
            if k[0] == "p":
                sct = self.tik["p"][0]
            elif k[0] == "q":
                sct = self.tik["q"][0]
            for i in range(self.neurons_per_assembly):
                self.assemblies[k][i] = self.assemblies[k][i][
                    self.assemblies[k][i] <= sct
                ]

    def sample_proposal(self):
        self.populate_assemblies("q", self.pp_length, self.race_start_time)
        q_assembly_spiketimes = [
            np.sort(np.concatenate(self.assemblies["q" + str(i)]))
            for i in range(self.num_states)
        ]
        first_spiketimes = [
            st[0] if not len(st) == 0 else np.nan for st in q_assembly_spiketimes
        ]
        if np.isnan(first_spiketimes).all():
            # this is the L4 autonorm for Q sampling.
            print("Q autonorm")
            self.population_λ["q"] *= 10
            self.update_sim_lambdas("q")
            print(self.sim_lambdas)
            return self.sample_proposal()
        winner = np.nanargmin(first_spiketimes)
        self.sample_time = first_spiketimes[winner]
        self.wta[str(winner)] = [self.sample_time]
        self.state = winner
        # clip all assemblies after this, or repopulate all of them in another function.

    def run_scoring_circuitry(self):
        # we don't care which q it is. we just want the kth element of it.
        def populate_accumulator_spikes(k, cliptime):
            if k == self.neurons_per_assembly:
                return
            else:
                spikes = np.sort(
                    np.concatenate(
                        [sts[k] for ak, sts in self.assemblies.items() if ak[0] == "q"]
                    )
                )
                self.accum[str(k)] = spikes[spikes <= cliptime]
                populate_accumulator_spikes(k + 1, cliptime)

        # assembly spikes will all be regenerated either here or before here.
        q_spikes_by_assembly = [
            np.sort(np.concatenate(sts))
            for k, sts in self.assemblies.items()
            if k[0] == "q"
        ]
        p_spikes_by_assembly = [
            np.sort(np.concatenate(sts))
            for k, sts in self.assemblies.items()
            if k[0] == "p"
        ]
        winning_q_assembly_spikes = q_spikes_by_assembly[self.state]
        winning_p_assembly_spikes = p_spikes_by_assembly[self.state]

        np.sort(np.concatenate(q_spikes_by_assembly))
        all_p_spikes = np.sort(np.concatenate(p_spikes_by_assembly))

        q_complete = len(winning_q_assembly_spikes) >= self.kq
        p_complete = len(all_p_spikes) >= self.kp

        # definitely the right idea here is to add on to the previous assemblies, which is already
        # done by populate assemblies. the first arg is how many more spikes to generate,
        # and the second arg is the start time of that generation.

        if not q_complete:
            self.num_pp_generated["q"] += 1
            self.populate_assemblies(
                "q", self.pp_length, self.pp_length * self.num_pp_generated["q"]
            )
            if self.num_pp_generated["q"] < self.max_recursion_passes:
                return self.run_scoring_circuitry()
        #            else:
        #                raise Exception("MAX RECURSION Q REACHED")

        elif not p_complete:
            self.num_pp_generated["p"] += 1
            self.populate_assemblies(
                "p", self.pp_length, self.pp_length * self.num_pp_generated["p"]
            )
            if self.num_pp_generated["p"] < self.max_recursion_passes:
                return self.run_scoring_circuitry()
        #            else:
        #                raise Exception("MAX RECURSION P REACHED")

        # both mux pass spikes from the winning assemblies.
        # the accum in q spikes for every assembly. whats passing
        # spikes out is the accum in Q and the mux in P.

        # there is a case where you get NO spikes in this variable. what that means is that
        # no spikes were generated in the distribution at all before max recursion.

        spikes_entering_p_tik = all_p_spikes[0 : self.kp + 1]
        if len(spikes_entering_p_tik) == 0:
            # this is the L4 autonorm for P scoring
            print("P Autonorm")
            self.population_λ["p"] *= 10
            self.update_sim_lambdas("p")
            print("Catprobs P")
            print(np.sum(self.catprobs[1]))
            print(len(self.catprobs[1]))
            return self.run_scoring_circuitry()
        self.mux["p" + str(self.state)] = winning_p_assembly_spikes[
            winning_p_assembly_spikes <= spikes_entering_p_tik[-1]
        ]

        if p_complete:
            self.tik["p"] = [spikes_entering_p_tik[-1]]
        else:
            #            print('p not complete')
            self.tik["p"] = [self.max_recursion_passes * self.pp_length]

        p_ratio = len(self.mux["p" + str(self.state)]) / self.kp
        if p_ratio == 0:
            self.p = -np.inf
        else:
            self.p = np.log(p_ratio)
        self.mux["q" + str(self.state)] = winning_q_assembly_spikes[0 : self.kq + 1]
        if q_complete:
            self.tik["q"] = [self.mux["q" + str(self.state)][-1]]
        else:
            #            print("q not complete")
            self.tik["q"] = [self.max_recursion_passes * self.pp_length]
        populate_accumulator_spikes(0, self.tik["q"][0])
        #        self.accum["q"] = all_q_spikes[all_q_spikes < self.tik["q"][0]]
        num_accumulator_spikes = np.sum(len(sp) for sp in self.accum.values())
        self.one_over_q = np.log(num_accumulator_spikes / self.kq)
        self.score_complete_time = np.max([self.tik["q"][0], self.tik["p"][0]])
        self.clip_assemblies_to_scoretime()

class ProbabilityMap(): 
    def __init__(self, neurons_per_assembly, probability_array, support, distribution):
        self.distribution = distribution
        self.probability_array = probability_array
        self.support = support




def poisson_process(λ, length_sim, starttime):
    rate = λ * length_sim
    spikenum = np.random.poisson(rate)
    spiketimes = starttime + np.sort(length_sim * np.random.uniform(0, 1, spikenum))
    return spiketimes
