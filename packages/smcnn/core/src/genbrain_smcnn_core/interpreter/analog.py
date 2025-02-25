import numpy as np
import jax
from genbrain_smcnn_core.interpreter.digital import (
    Resampler,
    Particle,
    SampleScore_Base,
    PixelBasedLikelihood,
)

np.seterr(all="ignore")

# 5/8/2024: code in an ess threshold for resampling. it should be easy.
# code in particle outputs in HOT, which can be silenced by state.

# with the recursive adding of poisson spikes to
# assemblies, the timing has gone off.


def normalize(x):
    return np.array(x) / sum(x)


class Resampler_Analog(Resampler):
    def __init__(self, particles):
        super().__init__(particles)
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
            Particle_Analog(
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


#        self.populate_state_buffer()


class Particle_Analog(Particle):
    def __init__(self, neurons_per_assembly, latent_variables, observed_variables):
        super().__init__(neurons_per_assembly, latent_variables, observed_variables)
        self.samplescores = {
            v["variable"]: lambda pq_probs: SampleScore_Analog(
                neurons_per_assembly, pq_probs
            )
            for v in latent_variables
        }
        self.likelihood_circuits = {}
        for v in observed_variables:
            if len(v["subtraced"]) == 0:
                self.likelihood_circuits[v["variable"]] = lambda probs: P_Unit_Analog(
                    neurons_per_assembly, probs
                )
            else:
                self.likelihood_circuits[v["variable"]] = (
                    lambda probs: PixelBasedLikelihood_Analog(probs)
                )
        self.variables = latent_variables + observed_variables
        self.p_units = {
            v["variable"]: lambda probs: P_Unit_Analog(neurons_per_assembly, probs)
            for v in self.variables
        }

    def populate_state_buffer(self):
        choices = self.resampled_choicemap
        for k, v in self.samplescores.items():
            state = choices[k]
            v.state_buffer[str(state)].append(0.0)


class SampleScore_Analog(SampleScore_Base):
    def __init__(self, neurons_per_assembly, catprobs):
        self.num_pp_generated = {"p": 0, "q": 0}
        self.max_recursion_passes = 3
        self.pp_length = 20
        super().__init__(neurons_per_assembly, catprobs)
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


class SampleScore_VariableK(SampleScore_Analog):
    def __init__(self, neurons_per_assembly, catprobs_q, catprobs_p, kq, kp):
        super().__init__(neurons_per_assembly, (catprobs_q,))
        self.populate_assemblies("q", self.pp_length, 0)
        self.initialize_p_assemblies(catprobs_p)
        self.kq = kq
        self.kp = kp
        self.component_dict = {}
        self.component_dict["mux"] = self.mux
        self.component_dict["accum"] = self.accum
        self.component_dict["tik"] = self.tik

    def sample_and_score_time(self):
        self.sample_proposal()
        self.run_scoring_circuitry()
        return self.sample_time, self.score_complete_time


class ClampedSampleScore(SampleScore_Analog):
    def __init__(
        self, neurons_per_assembly, clamped_state, catprobs_q, catprobs_p, kq, kp
    ):
        super().__init__(neurons_per_assembly, (catprobs_q,))
        self.populate_assemblies("q", self.pp_length, 0)
        self.initialize_p_assemblies(catprobs_p)
        self.state = clamped_state
        self.kq = kq
        self.kp = kp
        self.component_dict = {}
        self.component_dict["mux"] = self.mux
        self.component_dict["accum"] = self.accum
        self.component_dict["tik"] = self.tik

    def run_scoring(self):
        self.run_scoring_circuitry()


class P_Unit_Analog(SampleScore_Analog):
    def __init__(self, neurons_per_assembly, catprobs):
        super().__init__(
            neurons_per_assembly, (np.ones(len(catprobs)) / len(catprobs), catprobs)
        )

    def clip_assemblies_to_scoretime(self):
        for k in self.assemblies.keys():
            if k[0] == "p":
                for i in range(self.neurons_per_assembly):
                    self.assemblies[k][i] = self.assemblies[k][i][
                        self.assemblies[k][i] <= self.score_complete_time
                    ]

    def run_scoring_circuitry(self, state):
        p_spikes_by_assembly = [
            np.sort(np.concatenate(sts))
            for k, sts in self.assemblies.items()
            if k[0] == "p"
        ]
        # currently using 0 indexed pix values but have to make sure
        # you keep track of the pixel support
        state = int(state)
        winning_p_assembly_spikes = p_spikes_by_assembly[state]
        all_p_spikes = np.sort(np.concatenate(p_spikes_by_assembly))
        p_complete = len(all_p_spikes) >= self.kp
        if not p_complete:
            self.num_pp_generated["p"] += 1
            self.populate_assemblies(
                "p", self.pp_length, self.pp_length * self.num_pp_generated["p"]
            )
            if self.num_pp_generated["p"] < self.max_recursion_passes:
                return self.run_scoring_circuitry(state)

        if p_complete:
            spikes_entering_p_tik = all_p_spikes[0 : self.kp + 1]
            self.tik["p"] = [spikes_entering_p_tik[-1]]
            self.score_complete_time = self.tik["p"][0]
        else:
            spikes_entering_p_tik = all_p_spikes
            self.tik["p"] = []
            self.score_complete_time = all_p_spikes[-1]

        self.mux["p"] = winning_p_assembly_spikes[
            winning_p_assembly_spikes <= spikes_entering_p_tik[-1]
        ]
        self.p = np.log(len(self.mux["p"]) / self.kp)
        self.clip_assemblies_to_scoretime()
        return self

    def sample_from_prior(self):
        self.populate_assemblies("p", self.pp_length, 0)
        p_assembly_spiketimes = [
            np.sort(np.concatenate(self.assemblies["p" + str(i)]))
            for i in range(self.num_states)
        ]
        first_spiketimes = [
            st[0] if not len(st) == 0 else np.nan for st in p_assembly_spiketimes
        ]
        winner = np.nanargmin(first_spiketimes)
        self.sample_time = first_spiketimes[winner]
        self.wta[str(winner)] = [self.sample_time]
        self.state = winner
        return self.state


class PixelBasedLikelihood_Analog(PixelBasedLikelihood):
    def __init__(self, probvecs):
        super().__init__(probvecs)
        self.p_units = {
            "pix" + str(i): P_Unit_Analog(self.assembly_size, probs)
            for i, probs in enumerate(probvecs)
        }

    # traditional log addition of all scores. however, if even one score is inf,
    # will put the whole render to 0 probability.


def poisson_process(λ, length_sim, starttime):
    rate = λ * length_sim
    spikenum = np.random.poisson(rate)
    spiketimes = starttime + np.sort(length_sim * np.random.uniform(0, 1, spikenum))
    return spiketimes


# ss = SampleScore_Analog(3, (jnp.array([.2, .2, .6]),))
# ss_d = SampleScore(3, (jnp.array([.2, .2, .6]),))

# ss.sample_proposal()
# ss_d.sample_proposal()

# ss.initialize_p_assemblies(np.array([.1, .1, .8]))
# ss_d.initialize_p_assemblies(np.array([.1, .1, .8]))


# ss.run_scoring_circuitry()
# p_unit = P_Unit_Analog(3, jnp.array([.2, .2, .6]))


# for timing, store the race start time as the sample time of the
# parent variables. add it to the poisson process.

# stimes = [v for v in ss.assemblies.values()]
# maxlen_stimes = np.max([len(v) for v in stimes])
# spike_arrays_padded = np.array([np.pad(v, (0, maxlen_stimes-len(v)), 'constant', constant_values=np.nan) for v in stimes])
# # Convert the padded list of lists to a NumPy array
# # Flatten the array to 1D and get the indices of the 10 smallest values
# indices = np.argsort(spike_arrays_padded.flatten())[:10]

# # Convert the flattened indices back to 2D indices
# indices_2d = np.unravel_index(indices, spike_arrays_padded.shape)


# definitely the right idea here is to add on to the previous assemblies, which is already
# done by populate assemblies. the first arg is how many more spikes to generate,
# and the second arg is the start time of that generation.
