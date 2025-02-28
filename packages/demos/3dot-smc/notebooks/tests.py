import genstudio.plot as Plot
import genbrain_model_3dot as model
import genbrain_smcnn_core.interpreter as smcnn
import genbrain_utils_genjax as gjutils
import numpy as np
import jax.numpy as jnp
import jax
import genjax
from genjax import ChoiceMapBuilder as CMB
from PIL import Image
import re
import os
console = genjax.pretty()
# test smcnn get_categorical_probs

switch_type = genjax._src.generative_functions.combinators.switch.SwitchTrace

def get_catprobs_recur(key, genfunc_imp, v, args, constraints):
    trace, w = genfunc_imp(key, constraints, args)
    def recur_subtrace(strace, variables):
        if len(variables) == 0:
            probs, support = strace.args
            return probs
        elif type(strace) == switch_type:
            return recur_subtrace(strace.subtraces[strace.get_args()[0]], variables)
        else:
            return recur_subtrace(strace.get_subtrace((variables[0],)), variables[1:])
    probs = recur_subtrace(trace, v)
    return probs

def get_categorical_probs_test(key, genfunc_imp, v, args, constraints):
    trace, w = genfunc_imp(key, constraints, args)
    if isinstance(v, tuple):
        probs, support = trace.get_subtrace(*v).get_args()
    else:
        probs, support = trace.get_subtrace((v,)).args
    return probs



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


mod_probs = get_categorical_probs_test(key, init_mod_imp, ("ego_pos", "ego_matter"), (), CMB.d({}))
mod_probs = get_catprobs_recur(key, init_mod_imp, ("ego_pos", "ego_matter"), (), CMB.d({}))

prop_probs = get_catprobs_recur(key, init_prop_imp, ("dot", "ego_pos", "ego_matter"), (vis_angle_observations[1],), CMB.d({}))

prop_probs = get_categorical_probs_test(key, init_prop_imp, ("dot", "ego_pos", "ego_matter"), (vis_angle_observations[1],), CMB.d({}))
#step 2 
prop_probs_xyz = get_categorical_probs_test(key, init_prop_imp, ("dot", "xyz"), (vis_angle_observations[1],), CMB.d({}))
ego_constraint = tr_prop.get_retval()[2]
# constraint doesn't affect xyz_probs here. see if init_prop_imp is properly constrained. 
prop_probs_xyz_ego_constrained = get_categorical_probs_test(
    key, init_prop_imp, ("dot", "xyz"), (vis_angle_observations[1],), CMB.d({("dot", "ego_pos") : ego_constraint}))
# this is not propoerly constrained. the retval which is egopos is not the same as the constraint. 
tr_constr, w = init_prop_imp(key,  CMB.d({("dot", "ego_pos", "ego_matter") : ego_constraint}), 
                             (vis_angle_observations[20],))
(tr_constr.get_retval()[2] == ego_constraint).all()

cm = tr_constr.get_choices()

cm["dot", "ego_pos", "ego_matter"].value == ego_constraint



# if sampled in a switch, all cm vals will have a mask. have to check for switch. 



# one issue here is that the interpreter cares about the order of the proposal. the order is different in the proposal switch combinator -- if it goes down the vis path, you go obs->latents. if it goes the other way, you go markov. i think this might be a problem. better to keep the model the way it is and fix the ordering issue in the interpreter (i.e. provide dep maps for both switch outcomes, which is not a huge deal). 