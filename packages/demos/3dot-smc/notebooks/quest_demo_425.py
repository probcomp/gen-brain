import genbrain_model_3dot as model
from genbrain_3dot_smc import collect_frames
from genbrain_smcnn_core.interpreter import run_smcnn_particle_filter
from genbrain_3dot_smc.viz import *
import genbrain_utils_genjax as gjutils
import jax.numpy as jnp
import jax
import genstudio.plot as Plot
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
# %matplotlib tk

# First we will collect the generative functions from the 3dot model. 
genfns = [
    model.initial_proposal,
    model.initial_model,
    model.step_proposal,
    model.step_model,
    model.obs_model,
]

# Here we convert digital x,y numpy frames into a spherical occupancy map (i.e. a visual angle occupancy grid)
xy_obs_frames = collect_frames()[1:]
vis_angle_observations = jax.vmap(lambda obs: model.find_occupied_2d_angles(obs))(
    jnp.array(xy_obs_frames)
)
len_sim = 58
obs_traces = model.generate_obs_traces(vis_angle_observations[0:len_sim])

# choose a random seed so you can repeat the experiment
# looks really cool w key = 2000
key = jax.random.PRNGKey(2000)
num_particles = 20

def animate_current_particle_locs(observations, points_3D_seq, scores, plot_history, visual_angles, xyz_bounds, interval=50):
    observation_grids = [np.transpose(np.fliplr(o.reshape(len(visual_angles), len(visual_angles)))) for o in observations]
    xs, ys, zs = xyz_bounds
    fig = plt.figure(figsize=(12, 6))
    cmap = my_tab20(len(scores[0]))
    ax2D = fig.add_subplot(121)
    ax3D = fig.add_subplot(122, projection='3d')
    win = .3
    ax2D.set_xlabel("θ")
    ax2D.set_ylabel("ϕ")
    ax3D.set_xlabel('X')
    ax3D.set_ylabel('Y')
    ax3D.set_zlabel('Z')
    ax3D.set_title("3D Hypotheses (Y = Depth)")
    ax2D.set_title("2D Observation")
    ax3D.set_xlim([xs[0], xs[-1]])
    ax3D.set_ylim([ys[0], ys[-1]])
    ax3D.set_zlim([zs[0], zs[-1]])
    num_particles = len(points_3D_seq[0])
    im = ax2D.imshow(observation_grids[0], cmap='gray', interpolation='none', vmin=0, vmax=1)
    particles_3D = [ax3D.plot([], [], [], 'o')[0] for _ in range(num_particles)]
    tails_3D = [ax3D.plot([], [], [], '-', color=cmap[i], alpha=0.3, linewidth=.5)[0] for i in range(num_particles)]
    def update(frame):
        print(frame)
        im.set_array(observation_grids[frame])
        for i, (particle, (x, y, z)) in enumerate(zip(particles_3D, points_3D_seq[frame])):
            particle.set_data([x], [y])
#            particle.set_alpha(float(jnp.exp(float(scores[frame][i]))**.1))
            particle.set_3d_properties([z])
            particle.set_color(cmap[i])
            if plot_history:
                history = np.array(points_3D_seq[:frame+1])[:, i, :]
                tails_3D[i].set_data(history[:, 0], history[:, 1])
                tails_3D[i].set_3d_properties(history[:, 2])
        return [im] + particles_3D + tails_3D
    
    anim = FuncAnimation(fig, update, frames=len(observations), interval=interval, blit=True)
    return anim

def plot_spike_dictionary_studio(spiketimes_dict):
    num_components = len(spiketimes_dict)
    plot_objects = []
    component_labels = []
    for neuron_id, spikes_and_label in enumerate(spiketimes_dict.values()):
        spikes, label = spikes_and_label
        component_labels.append(label)
        neuron_y = num_components - (neuron_id + 1)
        for sp in spikes:
            plot_objects.append({"spiketimes": sp, "neuron_id": neuron_y, "component": label})
    raster = Plot.tickX(plot_objects, x="spiketimes", y="neuron_id", stroke="component") + Plot.axisY(
        {"tickSize": 1, "tickFormat": None, "tickRotate": 1, "label": component_labels})
    return raster

init_states_and_scores, first_step_states_and_scores, unrolled_pf = (
    gjutils.smc.run_particle_filter(
        obs_traces,
        num_particles,
        len_sim,
        genfns,
        key,
        model.translate_proposal_cm_to_model,
    )
)
# keep print of scores or not? its useful for debugging.
xyz, pscores = xyz_and_particle_scores(init_states_and_scores, 
                                       first_step_states_and_scores,
                                       unrolled_pf)
#pscores = [jnp.zeros(num_particles) for i in range(len_sim+1)]
an = animate_current_particle_locs(vis_angle_observations[0:len_sim], 
                              xyz, 
                              pscores, True, model.visual_angles, (model.xs, model.ys, model.zs))

latent_variables = [
    {
        "variable": "v3d",
        "q_id": ("dot", "v3d"),
        "q_parents": [],
        "p_parents": [],
        "support": model.xyz_vels,
        "type": "distribution",
    },
    {
        "variable": "xyz",
        "q_id": ("dot", "xyz"),
        "q_parents": [("ego_pos", "ego_matter")],
        "p_parents": [],
        "support": model.xyz_point_cloud,
        "type": "distribution",
    },
    {
        "variable": ("ego_pos", "ego_matter"),
        "q_id": ("dot", "ego_pos", "ego_matter"),
        "q_parents": [],
        "p_parents": ["xyz"],
        "support": model.bool_support,
        "type": "probmap",
    },
    {
        "variable": "lights",
        "q_id": ("dot", "lights"),
        "p_parents": [],
        "q_parents": [],
        "support": model.bool_support,
        "type": "distribution",
    },
    {
        "variable": "diam",
        "q_id": ("dot", "diam"),
        "p_parents": [],
        "q_parents": [],
        "support": model.diams,
        "type": "distribution",
    },
]

obs_variables = [
    {
        "variable": ("obs", "pix"),
        "parents": [],
        "support": model.bool_support,
        "type": "probmap",
    }
]

results = run_smcnn_particle_filter(
    (latent_variables, obs_variables),
    model.initial_model,
    model.step_model,
    model.initial_proposal,
    model.step_proposal,
    model.obs_model,
    5,
    2,
    vis_angle_observations[0:35],
)

xyz_spikes = gather_spikes_from_single_particle_samplescore(results, 0, "xyz", range(len_sim))

xyz_filtered = {k: v for k, v in xyz_spikes[0].items() if len(v[0]) != 0}

pa = plot_spike_dictionary_studio(xyz_filtered)
pa


