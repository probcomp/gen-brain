import jax
import jax.numpy as jnp
import numpy as np
from genjax import ChoiceMapBuilder as CMB
from . import master as master

np.seterr(divide="ignore")
key = jax.random.PRNGKey(10000)
key, subkey = jax.random.split(key, 2)

# 3/6/25: replace these. 

def get_categorical_probs_obs(key, genfunc_sim, v, args):
    trace = genfunc_sim(key, args)
    if isinstance(v, tuple):
        probs = trace.get_subtrace((v[0],)).inner.get_subtrace((v[1],)).args
    else:
        probs = trace.get_subtrace((v,)).args
    return probs[0]

def get_categorical_probs(key, genfunc_imp, v, args, constraints):
    trace, w = genfunc_imp(key, constraints, args)
    if isinstance(v, tuple):
        probs = (
            trace.get_subtrace((v[0],))
            .subtraces[trace.get_subtrace((v[0],)).args[0]]
            .get_subtrace((v[1],))
            .args
        )
    else:
        probs = trace.get_subtrace((v,)).args
    return probs[0]


def get_variable_id(variable_dict, prop_model_obs):
    var = variable_dict["variable"]
    for pmo in variable_dict["subtraced"]:
        if pmo[0] == prop_model_obs:
            var = (pmo[1], variable_dict["variable"])
    return var

def filter_variable(v, variables):
    return list(filter(lambda x: x["variable"] == v, variables))[0]

def smcnn_particle_filter_step_variables(
    key,
    proposal,
    proposal_args,
    proposal_variables,
    model,
    model_args,
    model_variables,
    particles,
    init_or_step,
):
    proposal_label = init_or_step + "_proposal"
    model_label = init_or_step + "_model"

    def sample_full_proposal(key, particle, sampled_list, prop_args):
        if set([v["variable"] for v in proposal_variables]) == set(sampled_list):
            return particle
        if sampled_list == []:
            empty_cm = CMB.d({})
            for prop_v in proposal_variables:
                if prop_v["parents"] == []:
                    q_probs = get_categorical_probs(
                        key,
                        proposal,
                        get_variable_id(prop_v, proposal_label),
                        prop_args,
                        empty_cm,
                    )
                    if not np.isfinite(q_probs).all():
                        print("Nan prb in proposal layer 1")
                        print(prop_args)
                        print(prop_v["variable"])
                    race_start_time = 0
                    particle.start_sampler(
                        prop_v["variable"], (q_probs,), race_start_time
                    )
                    sampled_list.append(prop_v["variable"])
        else:
            for prop_v in proposal_variables:
                if (set(prop_v["parents"]) <= set(sampled_list)) and (
                    prop_v["variable"] not in particle.choicemap.keys()
                ):
                    parent_states = CMB.d(
                        {
                            get_variable_id(
                                filter_variable(v, proposal_variables), proposal_label
                            ): filter_variable(v, proposal_variables)["support"][
                                particle.choicemap[v]
                            ]
                            for v in prop_v["parents"]
                        }
                    )
                    parent_sample_times = [
                        particle.samplescores[v].sample_time for v in prop_v["parents"]
                    ]
                    q_probs = get_categorical_probs(
                        key,
                        proposal,
                        get_variable_id(prop_v, proposal_label),
                        prop_args,
                        parent_states,
                    )
                    race_start_time = np.max(parent_sample_times)
                    particle.start_sampler(
                        prop_v["variable"], (q_probs,), race_start_time
                    )
                    sampled_list.append(prop_v["variable"])
        return sample_full_proposal(key, particle, sampled_list, prop_args)

    subkeys = jax.random.split(key, len(particles))
    # can easily make this a vmap.
    _ = list(
        map(
            lambda sb_key, particle, prop_args: sample_full_proposal(
                sb_key, particle, [], prop_args
            ),
            subkeys,
            particles,
            proposal_args,
        )
    )

    def assess_under_model(key, particle, mod_args):
        for model_variable in model_variables:
            parent_states = CMB.d(
                {
                    get_variable_id(
                        filter_variable(v, model_variables), model_label
                    ): filter_variable(v, model_variables)["support"][
                        particle.choicemap[v]
                    ]
                    for v in model_variable["parents"]
                }
            )
            parents_sample_times = [
                particle.samplescores[v].sample_time for v in model_variable["parents"]
            ]
            self_sample_time = particle.samplescores[
                model_variable["variable"]
            ].sample_time
            score_start_time = np.max(parents_sample_times + [self_sample_time])
            p_probs = get_categorical_probs(
                key,
                model,
                get_variable_id(model_variable, model_label),
                mod_args,
                parent_states,
            )

            if not np.isfinite(p_probs).all():
                print("non finite p probs")
                print(key)
                print(model_variable["variable"])
                print(p_probs)
                print(mod_args)
                print(model_variable["parents"])
                print([parent_states[v] for v in model_variable["parents"]])

            particle.samplescores[
                model_variable["variable"]
            ].score_start_time = score_start_time
            particle.set_score_start_time(model_variable["variable"], score_start_time)
            particle.start_pq_scoring(model_variable["variable"], p_probs)

    subkeys = jax.random.split(subkeys[0], len(particles))
    _ = list(
        map(
            lambda sb_key, particle, mod_args: assess_under_model(
                sb_key, particle, mod_args
            ),
            subkeys,
            particles,
            model_args,
        )
    )
    if init_or_step == "step":
        _ = list(map(lambda p: p.populate_state_buffer(), particles))
    return particles


# will need to rewrite if there are dependencies in the obs model.
def smcnn_particle_filter_score_obs(
    key, observation, obs_model, obs_args, obs_variables, model_variables, particles
):
    obs_indexer = 0
    for obs_variable in obs_variables:
        subkey = jax.random.split(key, len(particles))
        for obs_arg, particle in zip(obs_args, particles):
            key, subkey = jax.random.split(key, 2)
            probs = get_categorical_probs_obs(
                subkey, obs_model, get_variable_id(obs_variable, "obs"), obs_arg
            )
            if obs_variable["support"].shape == ():
                state = jnp.where(obs_variable["support"] == observation[obs_indexer])[
                    0
                ][0]
            else:
                state = jnp.where(
                    (obs_variable["support"] == observation[obs_indexer]).all(axis=1)
                )[0][0]
            particle.score_likelihood({obs_variable["variable"]: probs}, state)
        obs_indexer += 1
    return particles

def initialize_smcnn_particle_filter(
    key,
    variables,
    initial_model,
    initial_proposal,
    obs_model,
    neurons_per_assembly,
    num_particles,
    first_observation,
    dig_or_analog,
):
    
    latent_variables, obs_variables = variables
    key, subkey = jax.random.split(key, 2)
    
    particles = [
            master.Particle(neurons_per_assembly, latent_variables, obs_variables)
            for i in range(num_particles)
        ]
    # this is correct. format so initial model always takes no arguments, proposal only takes
    # the first observation.
    model_args = [() for i in range(num_particles)]
    proposal_args = [first_observation for i in range(num_particles)]
    particles = smcnn_particle_filter_step_variables(
        key,
        initial_proposal,
        proposal_args,
        initial_model,
        model_args,
        latent_variables,
        particles,
        "init",
    )
    # make sure obs args are arranged in order in metadata
    obs_args = [
        tuple([v["support"][p.choicemap[v["variable"]]] for v in latent_variables])
        for p in particles
    ]
    
    particles = smcnn_particle_filter_score_obs(
        subkey,
        first_observation,
        obs_model,
        obs_args,
        obs_variables,
        latent_variables,
        particles,
    )
    particles = list(map(lambda p: p.score_particle(), particles))
    return particles


def run_smcnn_particle_filter(
    variables,
    initial_model,
    step_model,
    initial_proposal,
    step_proposal,
    obs_model,
    assembly_size,
    num_particles,
    observations,
    dig_or_analog,
):
    print("Jitting Generative Functions")
    initial_model = jax.jit(initial_model.importance)
    step_model = jax.jit(step_model.importance)
    initial_proposal = jax.jit(initial_proposal.importance)
    step_proposal = jax.jit(step_proposal.importance)
    obs_model = jax.jit(obs_model.simulate)

    key = jax.random.PRNGKey(5000)
    print("Initializing SMCNN Particle Filter")
    particles = initialize_smcnn_particle_filter(
        key,
        variables,
        initial_model,
        initial_proposal,
        obs_model,
        assembly_size,
        num_particles,
        observations[0],
        dig_or_analog,
    )
    print("Initialized SMCNN Particle Filter")
    model_variables, proposal_variables, obs_variables = variables
    particles_per_step = [particles]
    resampler_per_step = []
    resampler = master.Resampler(particles)
    resampler.norm_and_resample()
    resampler_per_step.append(resampler)

    for step, observation in enumerate(observations[1:]):
        print("step", step)
        particles = resampler.particles
        key, subkey = jax.random.split(key, 2)
        model_args = [
            tuple(
                [
                    v["support"][p.resampled_choicemap[v["variable"]]]
                    for v in model_variables
                ]
            )
            for p in particles
        ]
        proposal_args = [m_args + tuple([*observation]) for m_args in model_args]
        particles = smcnn_particle_filter_step_variables(
            key,
            step_proposal,
            proposal_args,
            proposal_variables,
            step_model,
            model_args,
            model_variables,
            particles,
            "step",
        )
        obs_args = [
            tuple([v["support"][p.choicemap[v["variable"]]] for v in model_variables])
            for p in particles
        ]
        particles = smcnn_particle_filter_score_obs(
            subkey,
            observation,
            obs_model,
            obs_args,
            obs_variables,
            model_variables,
            particles,
        )
        particles = list(map(lambda p: p.score_particle(), particles))
        particles_per_step.append(particles)
        resampler = master.Resampler(particles)
        resampler.norm_and_resample()
        resampler_per_step.append(resampler)

    #   return particles_per_step, resampler_per_step
    # this return will let you step forward one more time.
    return (
        key,
        particles_per_step,
        resampler_per_step,
        step_model,
        step_proposal,
        obs_model,
        variables,
    )


def particle_validation(pf_results, particle_id):
    variables = pf_results[1][0][1].samplescores.keys()
    num_steps = len(pf_results[1])
    particle_over_time = [pf_results[1][step][particle_id] for step in range(num_steps)]
    p_scores = {v: [p.samplescores[v].p for p in particle_over_time] for v in variables}
    q_scores = {
        v: [p.samplescores[v].one_over_q for p in particle_over_time] for v in variables
    }
    likelihood_scores = [p.likelihood_score for p in particle_over_time]
    total_scores = [p.score for p in particle_over_time]
    particle_state = [p.choicemap for p in particle_over_time]
    resampled_on_step = [
        pf_results[2][step].resampled_on_step for step in range(num_steps)
    ]
    resampled_ind = [
        pf_results[2][step].winners[particle_id] for step in range(num_steps)
    ]
    full_validation = {
        "p": p_scores,
        "q": q_scores,
        "l": likelihood_scores,
        "total": total_scores,
        "state": particle_state,
        "resampled_index": resampled_ind,
        "resample": resampled_on_step,
    }
    return full_validation


# note if you want to set q to run scoring directly after sampling, you simply take the start of scoring time
# and subtract sample time from it, removing all of the leading 0s corresponding to this value.
