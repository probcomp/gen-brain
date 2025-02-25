import itertools
import jax

from genbrain_smcnn_core.distributions import (
    discrete_norm,
    disc_gauss_unnorm,
    labeled_categorical,
    unicat,
)

import genjax
import jax.numpy as jnp
from jax import tree_util, jit
import numpy as np
from genjax import ChoiceMapBuilder as CMB
import tensorflow_probability as tfp

# from mpl_toolkits.mplot3d.art3d import Line3DCollection
tfd = tfp.distributions

# from smc_genjax import run_particle_filter
# from interpreter_genjax04.snmc_distributions04 import (  # type: ignore
#     discrete_norm,
#     disc_gauss_unnorm,

#     discrete_truncnorm,
#     labeled_categorical,
#     normalize,
#     unicat,
#     upweight_zone,
# )

# from interpreter_genjax04.snmc_utils_genjax04 import (  # type: ignore
#     run_snmc_livedemo,
#     run_snmc_particle_filter,
#     constrained_step_demo,
# )
# from interpreter_genjax04.plot_snmc_run import (
#     plot_particle_weight_and_state,
#     snmc_spikes_wrapper,
#     selectivity_index,
#     my_tab20,
#     merge_particle_dicts,
#     merge_component_dicts,
#     get_components,
#     invert_spiketime_labels,
#     lfp_and_spikes,
#     lfp_and_spikes_animated,
#     eeg,
#     direction_selectivity,
#     animate_snmc_spikes,
#     organize_labels_into_layers
#     )

np.seterr(divide="ignore")
# console = genjax.pretty()

# object blinks or not by having a latent
# say is it visible or not
# blinking frequency like fireflies.
# (you can really show interesting posteriors)
# amount of 3D starts at 0 then goes up.
# minimal model for stars. prob of transitioning from visible to invisible and back, rather than
# a pre-specified frequency. can do two numbers (.1 LtoD, .3 DtoL).

# reduce the resolution and check that results are reasonable.

# TODO 10/30:
# 1. plot the results from SMCNNs just like your current animations.

# Unusual that you'd have such high depth uncertainty. You can see its size, but you can use a few occluders to show that you have uncertainty. Give the occluder positions and sizes as arguments to the model. You know the size of the ball, and every frame you observe a size and an x and y coordinate such that everything COULD be null. Observation model is you don't really get a null unless its behind an occluder. As you do particle filtering, you don't know whats up when its behind the occluder. When it emerges N particles extend to 3N and then return to N.

# Play a bit with the size, play a bit with the perspective, hard code some 3D trajectories, add size add occluders, add line perspectives. Do the size changes correctly w.r.t. the cube. Don't hack the size changes.

# note observer is at y = 0. the image is 1000,1000, so each div is 100 pixels. to calculate the angle falling on the observer, the 2D pixel grid can be thought of as starting at 60 pixels away,

ystart = 1.0
observer = ystart * -100.0
xyz_step = 1
xs = jnp.arange(-5.0, 6.0, xyz_step)
ys = jnp.arange(ystart, ystart + 10.0, xyz_step)
zs = jnp.arange(-5.0, 6.0, xyz_step)
xyz_point_cloud = jnp.array(list(itertools.product(xs, ys, zs)))
vel_step = 1
# in reality you want your velocity vector to be capable of ANY transition.
# currently it is not. there is an interesting tradeoff of "being able to go anywhere" and
# the size of the arena.
xyz_vel_mags = jnp.arange(-3.0, 3.5, vel_step)
xyz_vels = jnp.array(list(itertools.product(xyz_vel_mags, xyz_vel_mags, xyz_vel_mags)))
angle_div = jnp.pi / 32
visual_angles = jnp.arange(-jnp.pi / 2, jnp.pi / 2, angle_div)
egocentric_2d_map = jnp.array(list(itertools.product(visual_angles, visual_angles)))
objects = jnp.array([0])
bool_support = jnp.array([0, 1])
diams = jnp.arange(1.0, 3.0, 0.1)

σ_ang = 0.05 # type: ignore
σ_diam = 0.1 
σ_pos = 0.1
σ_vel = 0.1
σ_pos_model = 1
σ_vel_model = 1
σ_ang_obs = 0.1
pixflip_noise = 0.03

def probvecs_to_R3(arr1, arr2, arr3):
    probmat2d = jnp.outer(arr1, arr2)
    result = probmat2d[:, :, None] * arr3
    return jnp.ravel(result)

def probvecs_to_R2(arr1, arr2):
    probmat2d = jnp.outer(arr1, arr2)
    return jnp.ravel(probmat2d)

def index_pytree(pytree, idx):
    return tree_util.tree_map(lambda x: x[idx], pytree)

# there are incompatible combinations of z and r, b/c r is the hyp and
# can't be smaller than z.

# round the outputs of these transforms to the supports of each


def round_to_support(v, supp):
    idx = jax.vmap(lambda sv: jnp.linalg.norm(sv - v))(supp).argmin()
    return supp[idx]


def xyz_to_spherical(x, y, z):
    r = jnp.linalg.norm(jnp.array([x, y, z]))
    θ = jnp.atan(x / y)
    ϕ = jnp.asin(z / r)
    return θ, ϕ, r


def spherical_to_xyz(θ, ϕ, r):
    x = r * jnp.cos(ϕ) * jnp.sin(θ)
    y = r * jnp.cos(ϕ) * jnp.cos(θ)
    z = r * jnp.sin(ϕ)
    return x, y, z


depth_div = 1.0
depths = jnp.arange(
    1, jnp.linalg.norm(jnp.array([xs[-1], ys[-1], zs[-1]])) + depth_div, depth_div
)
x_init, y_init, z_init = (xs[-1], ys[int(len(ys) / 2)], zs[0])
egocentric_3d_map = jnp.array(
    list(itertools.product(visual_angles, visual_angles, depths))
)

# this tells you at what index in the xyz_point_cloud each spherical coordinate is.
spherical_map_indices = jax.vmap(
    lambda egopos: jnp.where(
        jnp.all(
            round_to_support(jnp.array(spherical_to_xyz(*egopos)), xyz_point_cloud)
            == xyz_point_cloud,
            axis=1,
        ),
        size=1,
    )[0][0]
)(egocentric_3d_map)

xyz_map_indices = jax.vmap(
    lambda xyz: jnp.where(
        jnp.all(
            round_to_support(jnp.array(xyz_to_spherical(*xyz)), egocentric_3d_map)
            == egocentric_3d_map,
            axis=1,
        ),
        size=1,
    )[0][0]
)(xyz_point_cloud)

# this is an array of 3D gaussians for each occupied spherical coordinate

spherical_probabilities = jax.vmap(
    lambda tpr: probvecs_to_R3(
        disc_gauss_unnorm(tpr[0], σ_ang, visual_angles),
        disc_gauss_unnorm(tpr[1], σ_ang, visual_angles),
        disc_gauss_unnorm(tpr[2], σ_pos, depths),
    )
)(egocentric_3d_map)


def xyz_to_spherical_index(x, y, z):
    r = jnp.linalg.norm(jnp.array([x, y, z]))
    θ = jnp.atan(x / y)
    ϕ = jnp.asin(z / r)
    sph = jnp.array(
        [
            round_to_support(θ, visual_angles),
            round_to_support(ϕ, visual_angles),
            round_to_support(r, depths),
        ]
    )
    spherical_index = jnp.where(jnp.all(egocentric_3d_map == sph, axis=1), size=1)[0][0]
    return spherical_index


xyz_to_spher_index_map = jax.vmap(lambda xyz: xyz_to_spherical_index(*xyz))(
    xyz_point_cloud
)

xyz_probability_map = jax.vmap(
    lambda p: probvecs_to_R3(
        disc_gauss_unnorm(xyz_point_cloud[p][0], σ_pos, xs),
        disc_gauss_unnorm(xyz_point_cloud[p][1], σ_pos, ys),
        disc_gauss_unnorm(xyz_point_cloud[p][2], σ_pos, zs),
    )
)(xyz_to_spher_index_map)

# we are making the assumption that movement in 3D space is uncorrelated. i.e. you can change severely in one direction without affecting the others. this is completely true with paramecia.

# MT or MST has an allocentric velocity percept.
# V1 has an egocentric 3D
# grid_pair(v1, v2, v1_catprob, v2_catprob, noise) = [p1*p2 for p1 in truncated_discretized_gaussian(
#                                                         v1, noise, v1_catprob) for p2 in truncated_discretized_gaussian(
#                                                             v2, noise, v2_catprob)]


def out_of_boundary(θ, ϕ, r):
    x_prop = (r * jnp.cos(ϕ) * jnp.cos(θ)) > xs[-1]
    y_prop = (r * jnp.cos(ϕ) * jnp.sin(θ)) > ys[-1]
    z_prop = (r * jnp.sin(ϕ)) > zs[-1]
    return (x_prop.astype(int) + y_prop.astype(int) + z_prop.astype(int)) > 0


def is_inside_grid(angs):
    θ, ϕ = angs
    r_vec = jax.vmap(
        lambda theta, phi, r: out_of_boundary(theta, phi, r), in_axes=(None, None, 0)
    )(θ, ϕ, depths)
    return (r_vec == 0).astype(int)


def is_greater_than_z(z, pvec):
    depths_greater_than_z = depths >= z
    return pvec * depths_greater_than_z


# The function you want is going to be computing a vector for each grid member and then constructing a matrix by
# multiplying them together. then unravel that matrix and sample.

""" GENERATIVE MODEL """
# Egocentric voxel grid model.

# this is going to take an xyz position and size. it will generate an array of bools for xyz locations
# where there is matter. eventually want to add self pose here.

# this is putting 0 probability at a lot of places. you should add a floor probability to this.

obj_to_ego_rendering_noise = 1e-5


# spherical_probability_map contains a 3D gaussian, represented as an array, for every angle/depth pair. this function takes the xyz generated by the model and asks what other xyz coordinates fall within its volume. in xyz_to_spher_index_map, there is a mapping to ego3d for every xyz combination in xyz_point_cloud. this automatically constrains the location of object matter to the cube b/c all xyz coords examined are in xyz_point_cloud
def obj_to_ego_matter(xyz, radius):
    bool_xyz = jax.vmap(lambda pos: jnp.linalg.norm(pos - xyz) <= radius)(
        xyz_point_cloud
    ).astype(int)
    gaussian_sums = jax.vmap(
        lambda b_xyz, sp_ind: b_xyz * spherical_probabilities[sp_ind], in_axes=(0, 0)
    )(bool_xyz, xyz_to_spher_index_map)
    probability_map = jnp.sum(gaussian_sums, axis=0) + obj_to_ego_rendering_noise
    return (probability_map * 1 / jnp.max(probability_map)).astype(float)


# the angles are correct here, but the arg of the highest ego prob corresponds to depth 1, and it is 5. i think this is because the close gaussians are cut off and renormed, so nearby gaussians will add more weight (i.e. the gaussian cloud will be cut off and renormed). use an unnormed truncated gaussian for addition.

@genjax.vmap(in_axes=0)
@genjax.gen
def egocentric_matter_map(prob_occupied):
    matter = (
        labeled_categorical(jnp.array([1 - prob_occupied, prob_occupied]), bool_support)
        @ "ego_matter"
    )
    return matter

@genjax.gen
def initial_model():
    # Allocentric velocity model (xyz)
    lights = labeled_categorical(jnp.array([0.0, 1.0]), bool_support) @ "lights"
    diam = labeled_categorical(unicat(diams), diams) @ "diam"
    vₜ = labeled_categorical(unicat(xyz_vels), xyz_vels) @ "v3d"
    # Egocentric xyz position vectors
    xyzₜ = labeled_categorical(unicat(xyz_point_cloud), xyz_point_cloud) @ "xyz"
    # Start transformation to egocentric spherical space.
    egocentric_probability_map = obj_to_ego_matter(xyzₜ, diam / 2)
    vc_θϕr = egocentric_matter_map(egocentric_probability_map) @ "ego_pos"
    return (vₜ, xyzₜ, vc_θϕr, lights, diam)

@genjax.gen
def step_model(vₚ, xyzₚ, vc_θϕrₚ, lightsₚ, diamₚ):
    # markov model for lights, diam a gaussian.
    lights_on = lightsₚ * 0.8 + 0.1
    lights_off = (1 - lightsₚ) * 0.8 + 0.1
    lights = (
        labeled_categorical(jnp.array([lights_off, lights_on]), bool_support) @ "lights"
    )
    diam = labeled_categorical(discrete_norm(diamₚ, σ_diam, diams), diams) @ "diam"
    v_probvec = probvecs_to_R3(
        discrete_norm(vₚ[0], σ_vel_model, xyz_vel_mags),
        discrete_norm(vₚ[1], σ_vel_model, xyz_vel_mags),
        discrete_norm(vₚ[2], σ_vel_model, xyz_vel_mags),
    )
    vₜ = labeled_categorical(v_probvec, xyz_vels) @ "v3d"
    dx, dy, dz = vₜ
    xyz_probvec = probvecs_to_R3(
        discrete_norm(xyzₚ[0] + dx, σ_pos_model, xs),
        discrete_norm(xyzₚ[1] + dy, σ_pos_model, ys),
        discrete_norm(xyzₚ[2] + dz, σ_pos_model, zs),
    )
    xyzₜ = labeled_categorical(xyz_probvec, xyz_point_cloud) @ "xyz"
    egocentric_probability_map = obj_to_ego_matter(xyzₜ, diam / 2)
    vc_θϕr = egocentric_matter_map(egocentric_probability_map) @ "ego_pos"
    return (vₜ, xyzₜ, vc_θϕr, lights, diam)


""" OBSERVATION MODEL """


def compress_3D_to_2D(vc_θϕr):
    partitioned_matter_map = jnp.array(jnp.split(vc_θϕr, len(egocentric_2d_map)))
    compressed_map = jax.vmap(lambda x: jnp.minimum(jnp.sum(x), 1.0))(
        partitioned_matter_map
    )
    return compressed_map


@genjax.vmap(in_axes=0)
@genjax.gen
def render_to_2D(pixprob):
    obs_prob = pixprob * jnp.array([pixflip_noise, 1 - pixflip_noise]) + (
        1 - pixprob
    ) * jnp.array([1 - pixflip_noise, pixflip_noise])
    pix = labeled_categorical(obs_prob, bool_support) @ "pix"
    return pix


@genjax.gen
def obs_model(vₜ, xyzₜ, vc_θϕr, lights, diam):
    r_marginalized_map = compress_3D_to_2D(vc_θϕr)
    obs = render_to_2D(r_marginalized_map) @ "obs"
    return obs


""" GETTING OBSERVATIONS """


@jit
def dig_2d_array(white_indices):
    def ind_to_dig(i):
        return jnp.any(white_indices == i)

    return jax.vmap(lambda wi: ind_to_dig(wi))(
        jnp.arange(len(egocentric_2d_map))
    ).astype(int)

def generate_pix_to_ego_2d_map(frm_indices_array, frame_shape):
    def map_to_ego_2d(pix):
        x_pix, y_pix = frame_shape
        θ = round_to_support(
            jnp.arctan((pix[0] - (x_pix / 2)) / jnp.abs(observer)), visual_angles
        )
        ϕ = round_to_support(
            jnp.arctan((pix[1] - (y_pix / 2)) / jnp.abs(observer)), visual_angles
        )
        ego_ind = jnp.where(
            jnp.all(egocentric_2d_map == jnp.array([θ, ϕ]), axis=1), size=1
        )[0][0]
        return ego_ind
    return jax.vmap(lambda pix: map_to_ego_2d(pix))(frm_indices_array)

frame_shape = (600, 600)
frame_indices = jnp.indices(frame_shape)
frame_indices_array = jnp.stack(
    [frame_indices[0].ravel(), frame_indices[1].ravel()], axis=-1
)
pix_to_ego_2d_map = generate_pix_to_ego_2d_map(
    frame_indices_array, frame_shape).reshape(frame_shape)


def find_occupied_2d_angles(frame):
    mask = jnp.where(frame, 1.0, jnp.nan)
    occupied_inds = mask * pix_to_ego_2d_map
    return dig_2d_array(occupied_inds)


# observations = jax.vmap(lambda obs: find_occupied_2d_angles(obs))(jnp.array(obs_frames))

# np.save("observations.npy", np.array(observations[0:5]))
# loaded_obs = np.load("observations.npy")
# observations = jnp.array(loaded_obs)


# def show_obs_as_image(obs):
#     im = obs.reshape(len(visual_angles), len(visual_angles))
#     fig, ax = plt.subplots()
#     ax.imshow(im, cmap="viridis")
#     ax.set_xticks(jnp.arange(len(visual_angles)))
#     ax.set_yticks(jnp.arange(len(visual_angles)))
#     ax.set_xticklabels(visual_angles, fontsize=4, rotation=90)
#     ax.set_yticklabels((-1 * visual_angles), fontsize=4)
#     ax.set_xlabel("Θ")
#     ax.set_ylabel("ϕ")
#     plt.imshow(im)


def generate_obs_traces(observations):
    key = jax.random.PRNGKey(1000)
    random_args = initial_model.simulate(key, ()).get_retval()

    def imp(ob):
        obs_tr, w = obs_model.importance(key, CMB.d({("obs", "pix"): ob}), random_args)
        return obs_tr

    obs_traces = jax.vmap(lambda o: imp(o))(observations)
    return obs_traces


""" INFERENCE PROPOSALS """


@jit
def dig_xyz(arr):
    return jax.vmap(
        lambda i: jax.lax.cond(jnp.any(arr == i), lambda x: 1.0, lambda x: jnp.nan, 1)
    )(jnp.arange(len(xyz_point_cloud)))


# need to find all the indices in egocentric space where rs are possible and thetas and phis match the dot. add a gaussian at each index. just like before. you are going to make a digital array of theta phi r occupancy.


def map_2d_to_Θϕr(ego_2d_occupancy, r_μ):
    def zero_array(Θϕ):
        return jnp.zeros(len(egocentric_3d_map))

    def map_2d_occupancy_to_3d_gaussian(Θϕ):
        return probvecs_to_R3(
            discrete_norm(Θϕ[0], σ_ang, visual_angles),
            discrete_norm(Θϕ[1], σ_ang, visual_angles),
            discrete_norm(r_μ, σ_pos, depths),
        )

    gaussians = jax.vmap(
        lambda occ, ego2d: jax.lax.cond(
            occ, map_2d_occupancy_to_3d_gaussian, zero_array, ego2d
        )
    )(ego_2d_occupancy, egocentric_2d_map)
    probability_map = jnp.minimum(jnp.sum(gaussians, axis=0), 1.0)
    return probability_map


def sphere_diameter(num_voxels):
    sphere_radius = (3 * num_voxels / (4 * jnp.pi)) ** (1 / 3)
    return 2 * sphere_radius


# obs is not a pixel grid. its a mapped find_occupied_2d_angles of pixel grids.

init_depth = round(jnp.linalg.norm(jnp.array([x_init, y_init, z_init]))).astype(float)


def occupied_values(dig, support):
    occupied_vals = support[np.where(dig)]
    return occupied_vals


@genjax.gen
def initial_proposal_vis(obs):
    # go from 2d occupancy to 3D probs. each will be a gaussian centered on a angle-depth tuple. you make the gaussians by cycling through positive ego occupancies and making a probvecs to R3 call with discrete_norm on theta, phi, and R. Then add them up and normalize maximum to 1, and sample from the vmap over bernoullis.
    ego_probability = map_2d_to_Θϕr(obs, init_depth)
    vc_θϕr = egocentric_matter_map(ego_probability) @ "ego_pos"
    xyz_occupancy = jnp.where(vc_θϕr, 1, jnp.nan) * spherical_map_indices
    xyz_mean = round_to_support(
        jnp.nanmean(dig_xyz(xyz_occupancy)[:, None] * xyz_point_cloud, axis=0),
        xyz_point_cloud,
    )
    xyz_probvec = probvecs_to_R3(
        discrete_norm(xyz_mean[0], σ_pos, xs),
        discrete_norm(xyz_mean[1], σ_pos, ys),
        discrete_norm(xyz_mean[2], σ_pos, zs),
    )
    xyzₜ = labeled_categorical(xyz_probvec, xyz_point_cloud) @ "xyz"
    # could constrain this initial estimate to values that fit a delta angular velocity.
    vₜ = labeled_categorical(unicat(xyz_vels), xyz_vels) @ "v3d"
    diam = (
        labeled_categorical(
            discrete_norm(
                round_to_support(sphere_diameter(jnp.sum(vc_θϕr)), diams), σ_diam, diams
            ),
            diams,
        )
        @ "diam"
    )
    lights = labeled_categorical(jnp.array([0.0, 1.0]), bool_support) @ "lights"
    return (vₜ, xyzₜ, vc_θϕr, lights, diam)


@genjax.gen
def step_proposal_vis(vₚ, xyzₚ, vc_θϕrₚ, lightsₚ, diamₚ, obs):
    step_xyz = vₚ + xyzₚ
    r_calc = round(jnp.linalg.norm(step_xyz)).astype(float)
    ego_probability = map_2d_to_Θϕr(obs, r_calc)
    vc_θϕr = egocentric_matter_map(ego_probability) @ "ego_pos"

    xyz_occupancy = jnp.where(vc_θϕr, 1, jnp.nan) * spherical_map_indices
    # dig_xyz yields a boolean occupancy allocentric map. if you multiply by 0 it adds all the 000s. have to put into nanspace.
    xyz_mean = round_to_support(
        jnp.nanmean(dig_xyz(xyz_occupancy)[:, None] * xyz_point_cloud, axis=0),
        xyz_point_cloud,
    )
    xyz_probvec = probvecs_to_R3(
        discrete_norm(xyz_mean[0], σ_pos, xs),
        discrete_norm(xyz_mean[1], σ_pos, ys),
        discrete_norm(xyz_mean[2], σ_pos, zs),
    )
    xyzₜ = labeled_categorical(xyz_probvec, xyz_point_cloud) @ "xyz"
    # could constrain this initial estimate to values that fit a delta angular velocity.
    dx, dy, dz = (xyzₜ - xyzₚ).astype(float)
    v_probvec = probvecs_to_R3(
        discrete_norm(dx, σ_vel, xyz_vel_mags),
        discrete_norm(dy, σ_vel, xyz_vel_mags),
        discrete_norm(dz, σ_vel, xyz_vel_mags),
    )
    vₜ = labeled_categorical(v_probvec, xyz_vels) @ "v3d"
    diam = (
        labeled_categorical(
            discrete_norm(
                round_to_support(sphere_diameter(jnp.sum(vc_θϕr)), diams), σ_diam, diams
            ),
            diams,
        )
        @ "diam"
    )
    lights = labeled_categorical(jnp.array([0.01, 0.99]), bool_support) @ "lights"
    return (vₜ, xyzₜ, vc_θϕr, lights, diam)


@genjax.gen
def initial_proposal(obs):
    islit = jnp.max(obs)
    rv = genjax.switch(initial_model, initial_proposal_vis)(islit, (), (obs,)) @ "dot"
    return rv


@genjax.gen
def step_proposal(vₚ, xyzₚ, vc_θϕrₚ, lightsₚ, diamₚ, obs):
    islit = jnp.max(obs)
    rv = (
        genjax.switch(step_model, step_proposal_vis)(
            islit,
            (vₚ, xyzₚ, vc_θϕrₚ, lightsₚ, diamₚ),
            (vₚ, xyzₚ, vc_θϕrₚ, lightsₚ, diamₚ, obs),
        )
        @ "dot"
    )
    return rv


""" GenJAX Particle Filter """


def translate_proposal_cm_to_model(cm):
    model_cm = CMB.d(
        {
            "xyz": cm["dot", "xyz"],
            ("ego_pos", "ego_matter"): cm["dot", "ego_pos", "ego_matter"],
            "v3d": cm["dot", "v3d"],
            "diam": cm["dot", "diam"],
            "lights": cm["dot", "lights"],
        }
    )
    return model_cm


def xyz_and_particle_scores(init_pf, fs_pf, unrolled_pf):
    xyz_init = init_pf[0].get_retval()[1]
    xyz_step1 = fs_pf[0].get_retval()[1]
    xyz_rest = unrolled_pf[1][1].get_retval()[1]
    all_xyz = jnp.vstack([xyz_init[None, :], xyz_step1[None, :], xyz_rest])

    init_scores = jax.tree.reduce(lambda x, y: x + y, init_pf[4])
    fs_scores = jax.tree.reduce(lambda x, y: x + y, fs_pf[4])
    unrolled_scores = jax.vmap(
        lambda scores: jax.tree.reduce(lambda x, y: x + y, scores)
    )(unrolled_pf[1][4])
    all_scores = [init_scores, fs_scores, *unrolled_scores]
    return all_xyz, all_scores


# gen_funs = [initial_proposal, initial_model, step_proposal, step_model, obs_model]
# num_particles = 50
# key = jax.random.PRNGKey(200)
# len_sim = 50
# obs_traces = generate_obs_traces(observations[0:len_sim])
# init_states_and_scores, first_step_states_and_scores, unrolled_pf = run_particle_filter(
#     obs_traces, num_particles, len_sim, gen_funs, key, translate_proposal_cm_to_model
# )
# xyz, pscores = xyz_and_particle_scores(
#     init_states_and_scores, first_step_states_and_scores, unrolled_pf
# )

""" Tests """


def test_simulate():
    ky = jax.random.PRNGKey(1000)
    init_tr = initial_model.simulate(ky, ())
    step_tr = step_model.simulate(ky, init_tr.get_retval())
    rv = init_tr.get_retval()
    obs = obs_model.simulate(ky, rv)
    # obs = index_pytree(obs_traces, 1)
    init_prop_tr = initial_proposal.simulate(ky, (obs.get_retval(),))
    step_prop_tr = step_proposal.simulate(ky, rv + (obs.get_retval(),))
    return init_tr, step_tr, init_prop_tr, step_prop_tr


def test_assess_and_propose(obs_traces):
    ky = jax.random.PRNGKey(100)
    obs = tree_util.tree_map(lambda x: x[0], obs_traces).get_retval()
    init_prop_tr = initial_proposal.simulate(ky, (obs,))
    # rv = init_prop_tr.get_retval()
    # init_prop_test = initial_proposal.propose(ky, (obs,))
    print("init prop score")
    print(init_prop_tr.get_score())
    init_assess = initial_model.assess(
        translate_proposal_cm_to_model(init_prop_tr.get_sample()), ()
    )
    print("init assess score")
    print(init_assess)
    init_mod_importance, _ = initial_model.importance(ky, init_prop_tr.get_sample(), ())
    init_mod_importance_w_translation, _ = initial_model.importance(
        ky, translate_proposal_cm_to_model(init_prop_tr.get_sample()), ()
    )
    # step_prop_test = step_proposal.propose(ky, rv + (obs,))
    # step_prop_tr = step_proposal.simulate(ky, rv + (obs,))
    # mod_cm = translate_proposal_cm_to_model(step_prop_tr.get_sample())
    # step_assess = step_model.assess(mod_cm, rv)
    return (
        init_prop_tr.get_sample(),
        init_mod_importance,
        init_mod_importance_w_translation,
    )


def test_obj_to_ego():
    probmap = obj_to_ego_matter(jnp.array([0.0, 5.0, 0.0]), 1.0)
    # highprob_args = jnp.argwhere(probmap > 0.003)
    # plot_spherical_probs(probmap)
    return probmap


def test_egocentric_matter_score(obs_traces):
    ky = jax.random.PRNGKey(100)
    obs = tree_util.tree_map(lambda x: x[1], obs_traces).get_retval()
    init_prop_tr = initial_proposal.simulate(ky, (obs,))
    xyz_retval = init_prop_tr.get_retval()[1]
    print("inferred xyz")
    print(xyz_retval)
    diam = init_prop_tr.get_retval()[-1]
    egomap = obj_to_ego_matter(xyz_retval, diam / 2.0)
    print("observation angles")
    print(egocentric_2d_map[np.where(obs)])
    print("egomap highprobs")
    # these are totally wrong. only very negative angles. it also doesn't fit with the inferred xyz.
    print(egocentric_3d_map[egomap > 0.03][:, 0:2])
    vcr = egocentric_matter_map.simulate(ky, (egomap,))
    return egomap, vcr


def test_obs_constraint(observation):
    key = jax.random.PRNGKey(200)
    obs_trace = generate_obs_traces(jnp.array([observation]))
    print("Obs is return value")
    print((obs_trace.get_retval() == observation).all())
    ky = jax.random.PRNGKey(100)
    tr = initial_model.simulate(key, ())
    tr, w = obs_model.importance(ky, obs_trace.get_choices(), tr.get_retval())
    print((tr.get_retval() == observation).all())
    return tr, w


""" Plotting """


# def animate_current_particle_locs(
#     observations, points_3D_seq, scores, plot_history, interval=50
# ):
#     observation_grids = [
#         np.transpose(np.fliplr(o.reshape(len(visual_angles), len(visual_angles))))
#         for o in observations
#     ]
#     fig = plt.figure(figsize=(12, 6))
#     cmap = my_tab20(len(scores[0]))
#     ax2D = fig.add_subplot(121)
#     ax3D = fig.add_subplot(122, projection="3d")
#     win = 0.3
#     ax2D.set_xlabel("θ")
#     ax2D.set_ylabel("ϕ")
#     ax3D.set_xlabel("X")
#     ax3D.set_ylabel("Y")
#     ax3D.set_zlabel("Z")
#     ax3D.set_title("3D Hypotheses (Y = Depth)")
#     ax2D.set_title("2D Observation")
#     ax3D.set_xlim([xs[0], xs[-1]])
#     ax3D.set_ylim([ys[0], ys[-1]])
#     ax3D.set_zlim([zs[0], zs[-1]])
#     num_particles = len(points_3D_seq[0])
#     im = ax2D.imshow(
#         observation_grids[0], cmap="gray", interpolation="none", vmin=0, vmax=1
#     )
#     particles_3D = [ax3D.plot([], [], [], "o")[0] for _ in range(num_particles)]
#     tails_3D = [
#         ax3D.plot([], [], [], "-", color=cmap[i], alpha=0.3, linewidth=0.5)[0]
#         for i in range(num_particles)
#     ]

#     def update(frame):
#         im.set_array(observation_grids[frame])
#         for i, (particle, (x, y, z)) in enumerate(
#             zip(particles_3D, points_3D_seq[frame])
#         ):
#             particle.set_data([x], [y])
#             particle.set_alpha(float(jnp.exp(float(scores[frame][i])) ** 0.1))
#             particle.set_3d_properties([z])
#             particle.set_color(cmap[i])
#             if plot_history:
#                 history = np.array(points_3D_seq[: frame + 1])[:, i, :]
#                 tails_3D[i].set_data(history[:, 0], history[:, 1])
#                 tails_3D[i].set_3d_properties(history[:, 2])
#         return [im] + particles_3D + tails_3D

#     anim = FuncAnimation(
#         fig, update, frames=len(observations), interval=interval, blit=True
#     )
#     return anim


# pscores = [jnp.zeros(num_particles) for i in range(len_sim + 1)]
# animate_current_particle_locs(observations[0:len_sim], xyz, pscores, True)


# def animate_trajectories_only_3D(xyz_inferences, scores, interval=50):
#     fig = plt.figure(figsize=(12, 6))
#     cmap = colormaps["plasma"]
#     ax3D = fig.add_subplot(111, projection="3d")
#     win = 0.3
#     ax3D.set_xlabel("X")
#     ax3D.set_ylabel("Y")
#     ax3D.set_zlabel("Z")
#     ax3D.set_xlim([xs[0], xs[-1]])
#     ax3D.set_ylim([ys[0], ys[-1]])
#     ax3D.set_zlim([zs[0], zs[-1]])
#     ax3D.set_title("Inferred 3D Trajectories")
#     num_steps = len(scores)
#     num_particles = len(scores[0])
#     norm = plt.Normalize(0, num_steps)
#     particle_segments = []
#     num_steps = len(xyz_inferences)
#     num_particles = len(scores[0])
#     cmap_by_step = np.linspace(0, num_steps, num_steps)
#     segments_by_step = []
#     seg_colors_by_step = []
#     for curr_step in range(num_steps):
#         step_segs = []
#         step_colors = []
#         for p_ind in range(num_particles):
#             xyz = xyz_inferences[:, p_ind]
#             segs = [[list(xyz[i]), list(xyz[i + 1])] for i in range(curr_step)]
#             colors = [cmap_by_step[i] for i in range(curr_step)]
#             step_segs = step_segs + segs
#             step_colors = step_colors + colors
#         segments_by_step.append(step_segs)
#         seg_colors_by_step.append(step_colors)
#     segments_by_step = segments_by_step[1:]
#     seg_colors_by_step = seg_colors_by_step[1:]

#     def update(frame):
#         ax3D.cla()
#         ax3D.set_xlabel("X")
#         ax3D.set_ylabel("Y")
#         ax3D.set_zlabel("Z")
#         ax3D.set_xlim([xs[0], xs[-1]])
#         ax3D.set_ylim([ys[0], ys[-1]])
#         ax3D.set_zlim([zs[0], zs[-1]])
#         ax3D.set_title("3D Hypotheses (Y = Depth)")
#         lc = Line3DCollection(
#             segments_by_step[frame], cmap=cmap, norm=norm, linewidths=1
#         )
#         lc.set_array(seg_colors_by_step[frame])
#         ax3D.add_collection3d(lc)
#         return (lc,)

#     anim = FuncAnimation(
#         fig, update, frames=num_steps - 1, interval=interval, blit=False
#     )
#     return anim


# def make_probability_heatmap(matrix):
#     if len(matrix.shape) == 3:
#         x, y, z = np.indices(matrix.shape)
#         fig = plt.figure(figsize=(10, 7))
#         ax = fig.add_subplot(111, projection="3d")
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Z")
#         scatter = ax.scatter(x, y, z, c=matrix.flatten(), cmap="viridis")
#         fig.colorbar(scatter, ax=ax, label="Value")
#     plt.show()


# """ SMCNNs """


# def extract_xyz_from_snmc(pf_results, obs, particles_to_animate):
#     xyz_inferences = []
#     p_scores = []
#     particles_per_step = pf_results[1]
#     resampler_per_step = pf_results[2]
#     for step in range(len(obs)):
#         particles = particles_per_step[step]
#         particle_choicemaps = [
#             p.choicemap for i, p in enumerate(particles) if i in particles_to_animate
#         ]
#         particle_scores = resampler_per_step[step].log_weights
#         # note you would normally index the supports b/c
#         # choicemap is an assembly index. but
#         xyz = np.array([xyz_point_cloud[cm["xyz"]] for cm in particle_choicemaps])
#         xyz_inferences.append(xyz)
#         p_scores.append(particle_scores)
#     return np.array(xyz_inferences), p_scores


# model_variables = [
#     {"variable": "v3d", "parents": [], "support": xyz_vels, "subtraced": []},
#     {
#         "variable": "xyz",
#         "parents": ["v3d"],
#         "support": xyz_point_cloud,
#         "subtraced": [],
#     },
#     {
#         "variable": "ego_pos",
#         "parents": ["xyz"],
#         "support": egocentric_3d_map,
#         "subtraced": [],
#     },
# ]

# proposal_variables = [
#     {
#         "variable": "ego_pos",
#         "parents": [],
#         "support": egocentric_3d_map,
#         "subtraced": [],
#     },
#     {
#         "variable": "xyz",
#         "parents": ["ego_pos"],
#         "support": xyz_point_cloud,
#         "subtraced": [],
#     },
#     {"variable": "v3d", "parents": ["xyz"], "support": xyz_vels, "subtraced": []},
# ]

# obs_variables = [
#     {
#         "variable": "obs_angles",
#         "parents": [],
#         "support": egocentric_2d_map,
#         "subtraced": [],
#     }
# ]

# variables = [model_variables, proposal_variables, obs_variables]
# assembly_size = 10


# def make_lfp_and_spikes(pf_results, obs):
#     spikes = merge_particle_dicts(
#         snmc_spikes_wrapper(
#             pf_results,
#             "r",
#             range(len(obs)),
#             range(num_particles),
#             get_components(depths, range(num_particles), ["ctx"]),
#             False,
#         )[0]
#     )

#     phase_precession_neurons = ["assemblies_p" + str(i) for i, x in enumerate(depths)]
#     particle = 1
#     r_spikes_particle = snmc_spikes_wrapper(
#         pf_results, "r", range(len(obs)), [particle], phase_precession_neurons, False
#     )[0]
#     # these are a merging of assembly neurons in a single particle. i.e. assembly neuron 1-1 will be merged with 1-2.
#     merged_assemblies = merge_component_dicts(
#         r_spikes_particle[0], phase_precession_neurons
#     )

#     merge_components = merge_component_dicts(
#         spikes, get_components(depths, range(num_particles), "ctx")
#     )
#     layered = organize_labels_into_layers(merge_components)
#     # return merged_assemblies
#     #    lfp = lfp_and_spikes(merged_assemblies, eeg(spikes))
#     lfp = lfp_and_spikes(merge_components, eeg(spikes), False)
#     #    anim = lfp_and_spikes_animated(merged_assemblies, lfp)
#     anim = lfp_and_spikes_animated(invert_spiketime_labels(layered), lfp)
#     #    return anim
#     return anim
