import math
import numpy as np
import jax.numpy as jnp
import copy
import jaxlib
import jax

# add a single mux per assembly. add an accum per multiple assemblies with a free param to dictate which is which.
# make a poisson process for assemblies instead of stepping per div. it doesn't look natural that there are so many
# aligned spikes per div.


def normalize(x):
    return x / jnp.sum(x)


array_type = jaxlib.xla_extension.ArrayImpl


class Resampler:
    def __init__(self, particles):
        self.particles = particles
        self.particles_before_resampling = copy.deepcopy(particles)
        self.normalizer_spikes = {str(p_id): [] for p_id in range(len(self.particles))}
        self.resampler_spikes = {str(p_id): [] for p_id in range(len(self.particles))}
        self.neural_float_e = {"0": []}
        self.neural_float_i = {"0": []}
        self.component_dict = {
            "norm_neurons": self.normalizer_spikes,
            "resampler_wta": self.resampler_spikes,
        }
        # "nfp_excitatory" : self.neural_float_e,
        # "nfp_inhibitory" : self.neural_float_i }
        self.t = 0
        self.normalizer_λ = 0.1
        self.winners = []
        self.log_weights = 0
        self.log_total_weight = 0
        self.ml_setpoint = 1
        self.nfp_delta = 0.1
        self.ess_threshold = jnp.inf
        self.resampled_on_step = True
        # fast enough to sample in a reasonable time with a very low joint probability,
        # slow enough to prevent all cells from spiking per step; currently
        # resolving ties by uniform draw, but you don't want severely different
        # normalized probabilities to be arriving simultaneously.
        # good to do PL proof of this!
        # for now, you want to write a loop where you inspect the marginal likelihood and bump it by
        # .1 distributed over

    def poisson_spike(self, probs):
        # sometimes this is a nan list – figure out tomorrow
        rate = self.normalizer_λ * probs
        for r, k in enumerate(self.normalizer_spikes.keys()):
            self.normalizer_spikes[k].append(np.random.poisson(rate[r], 1)[0])
        self.t += 1

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

    def pad_resampler(self):
        for i, p in enumerate(self.particles):
            self.resampler_spikes[str(i)].append(0)

    def nfp_norm_and_resample(self):
        starting_probs = list(map(lambda p: p.score, self.particles))
        self.marginal_likelihood = jnp.logsumexp(starting_probs)
        # here the inhibitory neuron is going to fire an amount of spikes proportional to the
        # marginal likelihood to stop the autonorm excitatory neuron from delivering "1".
        # we want the total activity to be close to 1. the joint scores are going to be
        # extremely low without

        # all particle scores converge to an inhibitory neuron.
        # if particle scores are high, turns off bias from excitatory neuron
        # if particle scores are low, the excitatory neuron brings the
        # particle probabilities up until the total prob = 1.
        # the inhibitory neuron is the marginal likelihood.

    def step_norm_and_resample_circuit(self, normalizer_probs):
        try:
            self.poisson_spike(normalizer_probs)
        except ValueError:
            print("All NaN Particle Scores")
            normalizer_probs = normalize(np.ones(len(self.particles)))
            self.poisson_spike(normalizer_probs)
        winner = self.detect_winner()
        if math.isnan(winner):
            return 0
        else:
            return 1

    def norm_and_resample(self):
        # right now this just does the entire job of neural floating point and ignores marginal likelihood
        # update this so that normalization is done by
        self.log_weights = list(
            map(lambda p: p.score + p.prevstep_score, self.particles)
        )
        self.log_total_weight = jax.nn.logsumexp(self.log_weights)
        starting_unnormed_probs = np.exp(self.log_weights)
        normalizer_probs = normalize(starting_unnormed_probs)
        total_spikes = 0
        while True:
            total_spikes += self.step_norm_and_resample_circuit(normalizer_probs)
            if total_spikes == len(self.particles):
                break
        # print("resampler score")
        # print(normalizer_probs)
        self.switch_states(jnp.inf)

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


# you can initialize a Particle by currying
# SampleScores using lambdas.
# i.e. { varb : lambda p, q: SampleScore(p, q, npa, varb) } – nice this worked well.
# a pattern might be, if you call varb on this dictionary, it
# automatically creates the SampleScore and runs smcnn.
# once all keys have been called, you score the entire particle.
# you can say "if i score all of the variables in the sample scores dict,
# I yield a joint score and mark myself as complete.


class Particle:
    def __init__(self, neurons_per_assembly, latent_variables, observed_variables):
        self.latent_variables = latent_variables
        self.observed_variables = observed_variables
        self.samplescores = {
            v["variable"]: lambda pq_probs: SampleScore(neurons_per_assembly, pq_probs)
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
                    lambda probs: PixelBasedLikelihood(probs)
                )
        self.variables = latent_variables + observed_variables
        self.p_units = {
            v["variable"]: lambda probs: P_Unit(neurons_per_assembly, probs)
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
            #            np.sum([lc.p for lc in self.likelihood_circuits.values()])
            return self
        else:
            raise Exception("not all scores accumulated")

    def timestamp_particle(self):
        self.timestamp = np.max(ss.scoring_time for ss in self.samplescores.values())


# need to set P AFTER you get the p_probs. you won't have p_probs until you've sampled its parents.
# but you still need to be able to sample Q as you get to its variable. so

# Catprobs is a tuple. You can either pass it with a one tuple or a two tuple. if its two, its q and p. if one, its q.


class SampleScore_Base:
    def __init__(self, neurons_per_assembly, catprobs):
        self.neurons_per_assembly = neurons_per_assembly
        self.num_states = len(catprobs[0])
        # wtf for some reason this is always vy.
        self.kq = 200
        self.kp = 600
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
        self.mux = {"q": [], "p": []}
        self.tik = {"q": [], "p": []}
        self.accum = {"q": []}
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

    def initialize_p_assemblies(self, catprobs_p):
        p_lambdas = catprobs_p * self.population_λ["p"]
        p_assembly_indices = ["p" + str(i) for i in range(self.num_states)]
        # assembly ps should have leading 0s of len assemblies q.
        self.assemblies.update(
            {
                i: [
                    np.zeros(len(self.assemblies["q0"][0])).tolist()
                    for i in range(self.neurons_per_assembly)
                ]
                for i in p_assembly_indices
            }
        )
        self.component_dict["assemblies"] = self.assemblies
        p_sim_lambdas = zip(p_assembly_indices, p_lambdas)
        self.sim_lambdas.update(dict(p_sim_lambdas))


# Likelihood is only a P_unit. Also encapsulates a Sampler from P.


class SampleScore(SampleScore_Base):
    def __init__(self, neurons_per_assembly, catprobs):
        self.t = 0
        SampleScore_Base.__init__(self, neurons_per_assembly, catprobs)

    # As if any assembly neurons spiked during a timestep.

    def poisson_spike(self):
        for assembly in self.assemblies.keys():
            rate = self.sim_lambdas[assembly]
            # if math.isnan(rate):
            #     print("NaN Poisson Rate")
            #     print(rate)
            #     rate = 0
            for i in range(self.neurons_per_assembly):
                self.assemblies[assembly][i].append(np.random.poisson(rate, 1)[0])
        self.t += 1

    # this can go much faster if you index it with an assembly but use numpy for the addition.

    # Count the amount of spikes that happened in each assembly at the specified time.

    def find_spikes_in_assemblies(self, assembly):
        spikes = sum(
            [self.assemblies[assembly][i][-1] for i in range(self.neurons_per_assembly)]
        )
        return spikes

    # This is the WTA. It asks whether any assemblies spiked at the given timepoint.
    # If not, reports there was no winner. If there's a tie, randomly sample the winner from the
    # set of assemblies that tied.
    def detect_winner(self):
        if math.isnan(self.state):
            spikes_per_assembly = list(
                map(
                    lambda x: self.find_spikes_in_assemblies("q" + str(x)),
                    range(self.num_states),
                )
            )
            if sum(spikes_per_assembly) > 0:
                assembly_spikes = np.nonzero(spikes_per_assembly)[0]
                if len(assembly_spikes) == 1:
                    winner = assembly_spikes[0]
                else:
                    winner = np.random.randint(
                        assembly_spikes[0], assembly_spikes[-1] + 1
                    )
            else:
                winner = float("NaN")
            for i in range(self.num_states):
                if i == winner:
                    self.wta[str(i)].append(1)
                else:
                    self.wta[str(i)].append(0)
            return winner
        else:
            self.pad_wtas()
            return np.nan

    # Wait to score until state has been established by the detect_winner (WTA) function.
    # Once it has, count how many spikes come out of the assemblies for P and Q.
    # Once you hit kp and kq, stop the sample score and return "True" that you've completed scoring.

    def run_scoring_circuitry(self):
        while True:
            self.poisson_spike()
            self.pad_wtas()
            state = self.state
            qmux_spikes = self.find_spikes_in_assemblies("q" + str(state))
            pmux_spikes = self.find_spikes_in_assemblies("p" + str(state))
            total_p_assembly_spikes = sum(
                map(
                    lambda x: self.find_spikes_in_assemblies("p" + str(x)),
                    range(self.num_states),
                )
            )
            total_q_assembly_spikes = sum(
                map(
                    lambda x: self.find_spikes_in_assemblies("q" + str(x)),
                    range(self.num_states),
                )
            )
            if self.p_tik < self.kp:
                self.p_tik += total_p_assembly_spikes
                self.mux["p"].append(pmux_spikes)
                if self.p_tik >= self.kp:
                    self.tik["p"].append(1)
                else:
                    self.tik["p"].append(0)
            else:
                self.mux["p"].append(0)
                self.tik["p"].append(0)

            if self.q_tik < self.kq:
                self.q_tik += qmux_spikes
                self.mux["q"].append(qmux_spikes)
                self.accum["q"].append(total_q_assembly_spikes)
                if self.q_tik >= self.kq:
                    self.tik["q"].append(1)
                else:
                    self.tik["q"].append(0)
            else:
                self.mux["q"].append(0)
                self.accum["q"].append(0)
                self.tik["q"].append(0)

            if self.p_tik >= self.kp and self.q_tik >= self.kq:
                self.p = np.log(sum(self.spikes_during_step("mux", "p")) / self.kp)
                self.one_over_q = np.log(
                    sum(self.spikes_during_step("accum", "q")) / self.kq
                )
                return True

    def pad_wtas(self):
        for i in range(self.num_states):
            self.wta[str(i)].append(0)

    def pad_scoring(self):
        self.mux["q"].append(0)
        self.mux["p"].append(0)
        self.tik["p"].append(0)
        self.tik["q"].append(0)
        self.accum["q"].append(0)

    # Run the loop of stepping each assembly neuron forward in time, checking to see which assemblies spiked in the WTA,
    # and scoring the detected state. once scoring is complete, calculate the p and q scores from spiking in the scoring circuitry.

    def sample_proposal(self):
        while True:
            # Step the assemblies forward
            self.poisson_spike()
            self.pad_scoring()
            # Detect whether a spike occurred in the WTA.
            self.state = self.detect_winner()
            if not math.isnan(self.state):
                self.sample_time = np.copy(self.t) + self.race_start_time
                break

    def spikes_during_step(self, component, p_or_q, *assembly_neuron_id):
        spiketrain = self.component_dict[component][p_or_q]
        if component == "assemblies" and assembly_neuron_id != ():
            return spiketrain[assembly_neuron_id[0]]
        else:
            return spiketrain


class P_Unit(SampleScore):
    def __init__(self, neurons_per_assembly, catprobs):
        super().__init__(
            neurons_per_assembly, (np.ones(len(catprobs)) / len(catprobs), catprobs)
        )

    def run_scoring_circuitry(self, state):
        while True:
            self.poisson_spike()
            pmux_spikes = self.find_spikes_in_assemblies("p" + str(int(state)))
            total_p_assembly_spikes = sum(
                map(
                    lambda x: self.find_spikes_in_assemblies("p" + str(x)),
                    range(self.num_states),
                )
            )
            if self.p_tik < self.kp:
                self.p_tik += total_p_assembly_spikes
                self.mux["p"].append(pmux_spikes)
                if self.p_tik >= self.kp:
                    self.tik["p"].append(1)
                else:
                    self.tik["p"].append(0)

            if self.p_tik >= self.kp:
                self.p = np.log(sum(self.spikes_during_step("mux", "p")) / self.kp)
                return self

    def detect_winner(self):
        if math.isnan(self.state):
            spike_at_t = list(
                map(
                    lambda x: self.find_spikes_in_assemblies("p" + str(x)),
                    range(self.num_states),
                )
            )
            if sum(spike_at_t) > 0:
                spiking_assemblies = np.nonzero(spike_at_t)[0]
                if len(spiking_assemblies) == 1:
                    winner = spiking_assemblies[0]
                else:
                    winner = np.random.randint(
                        spiking_assemblies[0], spiking_assemblies[-1] + 1
                    )
            else:
                winner = float("NaN")
            for i in range(self.num_states):
                if i == winner:
                    self.wta[str(i)].append(1)
                else:
                    self.wta[str(i)].append(0)
            return winner
        else:
            for i in range(self.num_states):
                self.wta[str(i)].append(0)
            return self.state

    def sample_from_prior(self):
        self.state = np.nan
        while True:
            self.poisson_spike()
            state = self.detect_winner()
            if not math.isnan(state):
                self.run_scoring_circuitry(state)
                self.state = state
                return state


# non-bug issue that a single inf score (i.e. 0 spikes out of K) makes the whole particle inf.
class PixelBasedLikelihood:
    def __init__(self, probvecs):
        self.assembly_size = 3
        self.p_units = {
            "pix" + str(i): P_Unit(self.assembly_size, probs)
            for i, probs in enumerate(probvecs)
        }
        #        for k, p_unit in self.p_units.items():
        #           p_unit.kp = 40
        self.pixel_probs = []
        self.p = jnp.nan

    # traditional log addition of all scores. however, if even one score is inf,
    # will put the whole render to 0 probability.
    def run_scoring_circuitry(self, observation):
        # switch to vmap
        pixel_probs = []
        for k, obs in zip(self.p_units.keys(), observation):
            self.p_units[k].run_scoring_circuitry(obs)
            pixel_probs.append(self.p_units[k].p)
        self.pixel_probs = jnp.array(pixel_probs)
        joint_score = jnp.sum(self.pixel_probs)
        self.p = joint_score
        if math.isnan(self.p):
            print("NAN LIKELIHOOD")
        return self

    # next try neural floating point for each pixel.
    # you need to speed up the assemblies. would have its input multiplexed in from its selected assembly. implementing neural floating point would be pretty easy here.
    # there's a time component in the normalizer. verify that you still get unbiased scores out of it. have speedups after a reasonable time. put up assembly sizes until it works.
