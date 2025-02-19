import genjax
import jax
import jax.numpy as jnp
from jax import tree_util, jit
import numpy as np
from genjax import switch
from genjax import ChoiceMapBuilder as CMB
import tensorflow_probability as tfp
tfd = tfp.distributions
from interpreter_genjax04.snmc_distributions04 import (  # type: ignore
    discrete_norm,
    labeled_categorical,
    normalize,
    unicat,
    upweight_zone,
)

def index_pytree(pytree, idx):
    return tree_util.tree_map(lambda x: x[idx], pytree)

def maybe_resample(key, log_weights_raw, ess_threshold):
    key, subkey = jax.random.split(key, 2)
    # nans times zero is nan! + a number is nan. 
   # log_weights = log_weights_raw * jnp.isfinite(log_weights_raw).all() + jnp.zeros(len(log_weights_raw)) * (1-jnp.isfinite(log_weights_raw).all())
    log_weights = log_weights_raw
    resample_inds = genjax.categorical.repeat(n=len(log_weights))
    # this automatically samples all zeros if fed a full nan vector
    resampled_inds = resample_inds(log_weights).simulate(subkey, ()).get_retval()
    log_total_weight = jax.nn.logsumexp(log_weights)
    log_normalized_weights = log_weights - log_total_weight
    log_ess = - jax.nn.logsumexp(2 * log_normalized_weights)
    ess = jnp.exp(log_ess)
    particle_inds = ((ess < ess_threshold) * resampled_inds) + ((ess >= ess_threshold) * jnp.arange(len(log_weights)))
   # particle_inds = jnp.arange(len(log_weights))
    return particle_inds, log_total_weight, log_normalized_weights

def run_single_step_smc_jax(key, obs_trace, proposal, 
                            model, obs_model, cm_translator,
                            prevstate_and_score, n_particles):
    last_step_traces, resampled_traces, log_total_weight, log_norm_weights, pqobs = prevstate_and_score
    subkeys = jax.random.split(key, n_particles)

    """ THESE ARE YOUR PROPOSED SAMPLES """
    proposal_map = jax.vmap(
        lambda k, rst: proposal.propose(
            k, rst.get_retval() + (obs_trace.get_retval(),)))(subkeys, resampled_traces)
    
    """ EACH SAMPLE IS SCORED UNDER THE SAME ARGUMENTS IN THE GENERATIVE MODEL """
    model_map = jax.vmap(
        lambda k, propmap, rst: model.importance(
            k,
            cm_translator(propmap),
            rst.get_retval()))(subkeys, proposal_map[0], resampled_traces)
    obs_map = jax.vmap(lambda k, mm: obs_model.importance(
        k,
        obs_trace.get_choices(), mm.get_retval()))(subkeys, model_map[0])
    p_q_obs = (model_map[1], proposal_map[1], obs_map[1])
    jax.debug.print("Values: P = {p_val}, Q = {q_val}, O = {o_val}", p_val=p_q_obs[0], q_val=p_q_obs[1], o_val=p_q_obs[2]) 
    scores = model_map[1] + obs_map[1] - proposal_map[1]
    key, subkey = jax.random.split(key, 2)
    resampled_particles, log_total_weight, log_norm_weights = maybe_resample(subkey, scores, jnp.inf) 
    next_parent_state = jax.vmap(lambda i: index_pytree(model_map[0], i))(resampled_particles)
    return (model_map[0], next_parent_state, log_total_weight, log_norm_weights, p_q_obs)

# the state at each step here in the resampled state. this is not necessarily what you want.
# you probably just want the resampled state to be the parents. but this is already
# done in snmc. but this is exactly why particle dist is so uniform. just try to implement this as
# "next parent state". should be fine! 

@genjax.gen
def initializing_func():
    return ()

def run_particle_filter(obs_traces, n_particles, num_steps, 
                        genfns, 
                        pfkey, prop_to_model_cm_translator):
    init_proposal, init_latent_model, step_proposal, step_latent_model, render = genfns
    subkeys = jax.random.split(pfkey, n_particles)
    _, init_key = jax.random.split(pfkey, 2)
    init_traces = jax.vmap(lambda k: initializing_func.simulate(k, ()))(subkeys)
    init_states_and_scores = run_single_step_smc_jax(init_key, 
                                                     index_pytree(obs_traces, 0),
                                                     init_proposal,
                                                     init_latent_model, 
                                                     render, 
                                                     prop_to_model_cm_translator,
                                                     (init_traces, init_traces, 0, 0, ()), n_particles)
    # seems like you have to run the first step outside the loop b/c its pytree is different than the init prop.
    _, pf_key_firststep = jax.random.split(init_key, 2)
    #return init_states_and_scores
    first_step_states_and_scores = run_single_step_smc_jax(pf_key_firststep, 
                                                           index_pytree(obs_traces, 1),
                                                           step_proposal,
                                                           step_latent_model, 
                                                           render,
                                                           prop_to_model_cm_translator,
                                                           init_states_and_scores, n_particles)
    # this is very easy just incorporate scores into the end of states (i.e. make it a tuple).
    # make maybe resample return scores.
    def particle_step(states, obs_and_key):
        obs, pfkey = obs_and_key
        newstates = run_single_step_smc_jax(pfkey, obs, step_proposal, step_latent_model,
                                            render, 
                                            prop_to_model_cm_translator, states, n_particles)
        return newstates, newstates

    unrolled_keys = jax.random.split(pf_key_firststep, num_steps - 2)
    unrolled_pf = jax.lax.scan(particle_step, first_step_states_and_scores, (jax.tree_map(lambda x: x[2:num_steps], obs_traces), unrolled_keys))
    return init_states_and_scores, first_step_states_and_scores, unrolled_pf
