import genbrain_model_3dot as model
#import genbrain_smcnn_core.interpreter as smcnn
import numpy as np
import jax.numpy as jnp
import jax
from genjax import ChoiceMapBuilder as CMB
from PIL import Image
import re
import os

def get_catprobs(trace, v):
    if isinstance(v, tuple):
        probs, support = trace.get_subtrace(*v).get_args()
    else:
        probs, support = trace.get_subtrace((v,)).args
    return probs

def get_categorical_probs_obs_test(key, genfunc_sim, v, args):
    trace = genfunc_sim(key, args)
    return get_catprobs(trace, v)
    
def get_categorical_probs_test(key, genfunc_imp, v, args, constraints):
    trace, w = genfunc_imp(key, constraints, args)
    return get_catprobs(trace, v)

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
no_constraint = CMB.d({})
args = ()
xy_obs_frames = collect_frames()
vis_angle_observations = jax.vmap(lambda obs: model.find_occupied_2d_angles(obs))(jnp.array(xy_obs_frames))

tr_mod, w = init_mod_imp(key, no_constraint, args)
tr_prop, w = init_prop_imp(key, no_constraint, (vis_angle_observations[1],))

# get_categorical_probs now finds its way through vmap and switches to the bottom level variables. 
mod_probs = get_categorical_probs_test(key, init_mod_imp, ("ego_pos", "ego_matter"), (), CMB.d({}))
prop_probs = get_categorical_probs_test(key, init_prop_imp, ("dot", "ego_pos", "ego_matter"), (vis_angle_observations[1],), CMB.d({}))

#step 2. make sure constraints work. start by creating a probability of "xyz" conditioned only on vis_angle_observations[1] as the input arg. 
prop_probs_xyz = get_categorical_probs_test(
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
prop_probs_xyz_ego_constrained = get_categorical_probs_test(
    key, init_prop_imp, ("dot", "xyz"), (vis_angle_observations[20],), 
    CMB.d({ ("dot", "ego_pos", "ego_matter") : ego_constraint}))

prop_probs_20_arg_no_constraint = get_categorical_probs_test(
    key, init_prop_imp, ("dot", "xyz"), (vis_angle_observations[20],), 
    CMB.d({}))

# All patterns work. If you constrain ego_pos, you get the same probabilities at xyz regardless of argument. If you let the different arguments control the sampling with no constraint, probs at xyz end up different. assert that these logical statements are true to make a test. 
assert((prop_probs_xyz == prop_probs_xyz_ego_constrained).all())
assert(not (prop_probs_20_arg_no_constraint == prop_probs_xyz_ego_constrained).all())


# test metadata formatting. "variable" is the variable name in the generative model. 

latent_variables = [
    {"variable": "lights", "q_id": ("dot", "lights"), 
     "p_parents": [], "q_parents": [], "support": model.bool_support, "type": "distribution"}, 
    {"variable": "diam", "q_id": ("dot", "diam"),  
     "p_parents": [], "q_parents": [], "support": model.diams, "type": "distribution"},
    {"variable": "v3d", "q_id": ("dot", "v3d"), "q_parents": [], "p_parents": [], "support": model.xyz_vels, "type": "distribution"}, 
    {"variable": "xyz", "q_parents": ["ego_pos"], "p_parents": [],  "support": model.xyz_point_cloud, "type": "distribution"},
    {"variable": ("ego_pos", "ego_matter"), "q_id": ("dot", "ego_pos", "ego_matter"), "q_parents": [], "p_parents": ["xyz"], "support": model.egocentric_3d_map, "type": "probmap"}
    ]
    
obs_variables = [
    {
        "variable": "obs",
        "parents": [],
        "support": model.bool_support
    }
]