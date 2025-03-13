import genbrain_model_3dot as model
from genbrain_smcnn_core.interpreter import initialize_smcnn_particle_filter, get_categorical_probs
import genbrain_smcnn_core.interpreter.master as smcnn
import numpy as np
import jax.numpy as jnp
import jax
from genjax import ChoiceMapBuilder as CMB
from PIL import Image
import re
import os


def collect_frames(directory="../data/", file_pattern="frame-*.png"):
    frame_regex = re.compile(r"frame-(\d+)\.png")
    frames = []
    for filename in os.listdir(directory):
        match = frame_regex.match(filename)
        if match:
            frame_number = int(match.group(1))
            frames.append((frame_number, os.path.join(directory, filename)))
    frames.sort(key=lambda x: x[0])
    frame_paths = [path for _, path in frames]
    numpy_frames = []
    for path in frame_paths:
        with Image.open(path) as img:
            # this must be going downwards? 
            im = img.convert("L").point(lambda p: 1 if p > 0 else 0)
            numpy_frames.append((np.transpose(np.flipud(np.array(im))) > 0).astype(int))
    return numpy_frames

key = jax.random.PRNGKey(100)
init_mod_imp = model.initial_model.importance
init_prop_imp = model.initial_proposal.importance
obs_mod_imp = model.obs_model.importance
no_constraint = CMB.d({})
args = ()
xy_obs_frames = collect_frames()
vis_angle_observations = jax.vmap(lambda obs: model.find_occupied_2d_angles(obs))(jnp.array(xy_obs_frames))

tr_mod, w = init_mod_imp(key, no_constraint, args)
tr_prop, w = init_prop_imp(key, no_constraint, (vis_angle_observations[1],))

# get_categorical_probs now finds its way through vmap and switches to the bottom level variables. 
mod_probs = get_categorical_probs(key, init_mod_imp, ("ego_pos", "ego_matter"), (), CMB.d({}))
prop_probs = get_categorical_probs(key, init_prop_imp, ("dot", "ego_pos", "ego_matter"), (vis_angle_observations[1],), CMB.d({}))

#step 2. make sure constraints work. start by creating a probability of "xyz" conditioned only on vis_angle_observations[1] as the input arg. 
prop_probs_xyz = get_categorical_probs(
    key, init_prop_imp, ("dot", "xyz"), (vis_angle_observations[1],), CMB.d({}))

# next make a constraint based on the same argument (index 1). this is an array of 0s and 1s that represent voxel occupancy in spherical space. for the argument vis_angle_observations[1], the ego constraint is occupancy at indices (Array([2192, 2205, 2621], dtype=int32),). This is the exact same as the value for the retval of tr_prop. 

ego_constraint = tr_prop.get_choices()["dot", "ego_pos", "ego_matter"]

# now add a totally different argument (index 20). make sure retval is the same as tr_prop. it is. 
tr_constr, w = init_prop_imp(key,  
                             CMB.d({ ("dot", "ego_pos", "ego_matter") : ego_constraint}), 
                             (vis_angle_observations[20],)) 

# constraining the choicemaps works. retvals and choicemaps are the same. assert these as tests. 
assert((tr_prop.get_retval()[2] == tr_constr.get_retval()[2]).all())
assert((ego_constraint.value == tr_constr.get_choices()["dot", "ego_pos", "ego_matter"].value).all())

# now test that the probabilities of xyz with a different input argument but the exact same ego_pos constraint correspond to the correct xyz interpretation. if constraint isn't working, probabilities will be very similar. 
prop_probs_xyz_ego_constrained = get_categorical_probs(
    key, init_prop_imp, ("dot", "xyz"), (vis_angle_observations[20],), 
    CMB.d({ ("dot", "ego_pos", "ego_matter") : ego_constraint}))

prop_probs_20_arg_no_constraint = get_categorical_probs(
    key, init_prop_imp, ("dot", "xyz"), (vis_angle_observations[20],), 
    CMB.d({}))

# All patterns work. If you constrain ego_pos, you get the same probabilities at xyz regardless of argument. If you let the different arguments control the sampling with no constraint, probs at xyz end up different. assert that these logical statements are true to make a test. 
assert((prop_probs_xyz == prop_probs_xyz_ego_constrained).all())
assert(not (prop_probs_20_arg_no_constraint == prop_probs_xyz_ego_constrained).all())

# Test constraint of observation model 

obs_constraint = CMB.d({("obs", "pix") : vis_angle_observations[1]})
obs_tr_constrained, w = obs_mod_imp(key, obs_constraint, tr_prop.get_retval())

assert((obs_tr_constrained.get_choices()["obs", "pix"] == obs_constraint[("obs", "pix")]).all())


# Test classes from master interpreter 

q_probs = jnp.array([.1, .2, .7])
p_probs = jnp.array([.1, .2, .7])
probmap_q = [jnp.array([.1, .9]), 
jnp.array([.001, .999]), jnp.array([.2, .8])]
probmap_p = [jnp.array([.9, .1]), 
jnp.array([.2, .8]), jnp.array([.5, .5])]
neurons_per_assembly = 10

print("testing samplescores")
ss = smcnn.SampleScore(neurons_per_assembly, (q_probs,), False)
print("starting proposal")
ss.sample_proposal(0.0)
print("initializing p assemblies")
ss.initialize_p_assemblies(p_probs)
print("scoring")
ss.run_scoring_circuitry()

print("testing probability maps")
pm = smcnn.ProbabilityMap(neurons_per_assembly, (probmap_q,))
pm.sample_proposal(0.0)
pm.initialize_p_assemblies(probmap_p)
pm.run_scoring_circuitry()
print(pm.total_score)

# Next up test Particle. We'll need metadata to make one, and we'll test the new compressed version of the metadata. 
# test metadata formatting. "variable" is the variable name in the generative model. only requirement of latent_variables is that ALL latent variable values are the return from the proposal and model, and they are in order in the metadata accordint to their order in the return value. you want to add in the support of the last variable in any tuple description for support. Probmap will handle categorical indexing. 
latent_variables = [
    {"variable": "v3d", "q_id": ("dot", "v3d"), "q_parents": [], "p_parents": [], "support": model.xyz_vels, "type": "distribution"}, 
    {"variable": "xyz", "q_id": ("dot", "xyz"), "q_parents": [("ego_pos", "ego_matter")], "p_parents": [],  "support": model.xyz_point_cloud, "type": "distribution"},
    {"variable": ("ego_pos", "ego_matter"), "q_id": ("dot", "ego_pos", "ego_matter"), "q_parents": [], "p_parents": ["xyz"], "support": model.bool_support, "type": "probmap"},
    {"variable": "lights", "q_id": ("dot", "lights"), 
     "p_parents": [], "q_parents": [], "support": model.bool_support, "type": "distribution"}, 
    {"variable": "diam", "q_id": ("dot", "diam"),  
     "p_parents": [], "q_parents": [], "support": model.diams, "type": "distribution"}
    ]
    
obs_variables = [
    {
        "variable": ("obs", "pix"),
        "parents": [],
        "support": model.bool_support,
        "type" : "probmap"
    }
]

# could probably change up samplescores a bit to not be a function
def apply(func, arg):
    return func(arg)
print("initializing particle")
particle = smcnn.Particle(neurons_per_assembly, latent_variables, obs_variables)

#start a samplescore sampler
print("starting samplers")
particle.start_sampler("lights", (jnp.array([.1, .9]),), 0.0)
#start a probabilitymap sampler
particle.start_sampler(("ego_pos", "ego_matter"), (mod_probs,), 0.0)

#test score time method
print("setting score times")
particle.set_score_time("lights", 3.0)
particle.set_score_time(("ego_pos", "ego_matter"), 3.0)
# pq_scoring. 
print("starting pq scoring")
particle.start_pq_scoring("lights", jnp.array([.1, .9]))
particle.start_pq_scoring(("ego_pos", "ego_matter"), prop_probs)

#likelihood
print("starting likelihood scoring")
args_to_obs = tr_prop.get_retval()
obs_probs = get_categorical_probs(key, obs_mod_imp, ("obs", "pix"), args_to_obs, CMB.d({}))
particle.score_likelihood((obs_probs,), vis_angle_observations[1])

print("initializing particle filter")

sfp_proposal = jax.jit(init_prop_imp)
sfp_model = jax.jit(init_mod_imp)
sfp_obs = jax.jit(obs_mod_imp)
# this shows you that the error stems from get_categorical_probs operating on  ("dot", "lights") with the correct argument structure. 
particles = initialize_smcnn_particle_filter(key, 
                                             (latent_variables, obs_variables), sfp_model, 
                                             sfp_proposal, sfp_obs, 2, 1, (vis_angle_observations[1],))

# replicated lambda error outside of the loop! 
sfp_proposal = init_prop_imp

def sample_full_proposal(key, particle, sampled_list, prop_args):
    empty_cm = CMB.d({})
    for lv in latent_variables:
        if lv["q_parents"] == []:
            print(lv)
            q_probs = get_categorical_probs(
                key,
                sfp_proposal,
                lv["q_id"],
                prop_args,
                empty_cm
            )
            if not np.isfinite(q_probs).all():
                print("Nan prb in proposal layer 1")
                print(prop_args)
                print(lv["variable"])
            race_start_time = 0
            particle.start_sampler(
                lv["variable"], (q_probs,), race_start_time
            )
            sampled_list.append(lv["q_id"])

input_particle = smcnn.Particle(
                         neurons_per_assembly, 
                         latent_variables, obs_variables) 
                         
sample_full_proposal(key, input_particle, 
                     [], (vis_angle_observations[1],))
get_categorical_probs(key, 
                      sfp_proposal, ('dot','lights'), 
                      (vis_angle_observations[1],), 
                      CMB.d({}))


