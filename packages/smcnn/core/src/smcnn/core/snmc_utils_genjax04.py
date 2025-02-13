import asyncio
import re
import time
import traceback
import genstudio.plot as Plot
from genstudio.plot import Row, Column
import jax
import jax.numpy as jnp
import numpy as np
from genjax import ChoiceMapBuilder as CMB
from smcnn.core.plot_snmc_run import (
    merge_and_roll_spikes,
    snmc_spikes_wrapper,
)
from smcnn.core.snmc_interpreter_analog04 import (
    Particle_Analog,
    Resampler_Analog,
    ClampedSampleScore,
)
from smcnn.core.snmc_interpreter_digital04 import Particle, Resampler
from smcnn.core.plot_snmc_run import (
    invert_spiketime_labels,
    get_spikes_from_samplescore,
)
from IPython.display import display

np.seterr(divide="ignore")
html = Plot.Hiccup
js = Plot.js

# Notes MapCombinator handling:
# in new setup, MapCombinator isn't the obs_model itself, but is called by a wrapper
# generative function that creates an entire image instead of a set of pixels.
# this is the format required by the grid library. so write a wrapper for each model
# that you want to run in SMC and exact inference, but keep the MapCombinator separate
# so you can give it to SNMC. SNMC deals in pixel space only. keep the latent_transform_fn for now.
# its easier to map directly from the MapCombinator to PixelBasedLikelihoods.

key = jax.random.PRNGKey(10000)
key, subkey = jax.random.split(key, 2)
# map_trace_type = genjax._src.generative_functions.combinators.vector.map_combinator.MapTrace


def get_categorical_probs_obs(key, genfunc_sim, v, args):
    trace = genfunc_sim(key, args)
    if isinstance(v, tuple):
        probs, support = trace.get_subtrace((v[0],)).inner.get_subtrace((v[1],)).args
    else:
        probs, support = trace.get_subtrace((v,)).args
    return probs, support


def get_categorical_probs(key, genfunc_imp, v, args, constraints):
    trace, w = genfunc_imp(key, constraints, args)
    if isinstance(v, tuple):
        probs, support = (
            trace.get_subtrace((v[0],))
            .subtraces[trace.get_subtrace((v[0],)).args[0]]
            .get_subtrace((v[1],))
            .args
        )
    else:
        probs, support = trace.get_subtrace((v,)).args
    return probs, support


def get_variable_id(variable_dict, prop_model_obs):
    var = variable_dict["variable"]
    for pmo in variable_dict["subtraced"]:
        if pmo[0] == prop_model_obs:
            var = (pmo[1], variable_dict["variable"])
    return var


def filter_variable(v, variables):
    return list(filter(lambda x: x["variable"] == v, variables))[0]


def snmc_particle_filter_step_variables(
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

    # map this over: proposal_args, subkeys, particles
    # think more deeply about timing here. run_time is too simple.

    def sample_full_proposal(key, particle, sampled_list, prop_args):
        if set([v["variable"] for v in proposal_variables]) == set(sampled_list):
            return particle
        if sampled_list == []:
            empty_cm = CMB.d({})
            for prop_v in proposal_variables:
                if prop_v["parents"] == []:
                    q_probs, _ = get_categorical_probs(
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
                    q_probs, _ = get_categorical_probs(
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
            p_probs, _ = get_categorical_probs(
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
def snmc_particle_filter_score_obs(
    key, observation, obs_model, obs_args, obs_variables, model_variables, particles
):
    # print("scoring observations")
    obs_indexer = 0
    for obs_variable in obs_variables:
        subkey = jax.random.split(key, len(particles))
        for obs_arg, particle in zip(obs_args, particles):
            key, subkey = jax.random.split(key, 2)
            probs, _ = get_categorical_probs_obs(
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


def initialize_snmc_particle_filter(
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
    # print("initializing particle filter")
    model_variables, proposal_variables, obs_variables = variables
    key, subkey = jax.random.split(key, 2)
    if dig_or_analog == "digital":
        particles = [
            Particle(neurons_per_assembly, model_variables, obs_variables)
            for i in range(num_particles)
        ]
    elif dig_or_analog == "analog":
        particles = [
            Particle_Analog(neurons_per_assembly, model_variables, obs_variables)
            for i in range(num_particles)
        ]
    # this is correct. format so initial model always takes no arguments, proposal only takes
    # the first observation.
    model_args = [() for i in range(num_particles)]
    proposal_args = [first_observation for i in range(num_particles)]
    particles = snmc_particle_filter_step_variables(
        key,
        initial_proposal,
        proposal_args,
        proposal_variables,
        initial_model,
        model_args,
        model_variables,
        particles,
        "init",
    )
    obs_args = [
        tuple([v["support"][p.choicemap[v["variable"]]] for v in model_variables])
        for p in particles
    ]
    # print(obs_args)
    particles = snmc_particle_filter_score_obs(
        subkey,
        first_observation,
        obs_model,
        obs_args,
        obs_variables,
        model_variables,
        particles,
    )
    particles = list(map(lambda p: p.score_particle(), particles))
    return particles


def category_label(category):
    return category.rsplit("_", 1)[0]


def render_observation_entry(i, v, pos, set_pos):
    # can change this to onMouseEnter (only called when it enters the element)
    props = {"onClick": lambda _: set_pos(i)} if set_pos is not None else {}
    if v == 0:
        return ["div.w3.h3.bg-white", props]
    if v == 1:
        return [
            "div.w3.h3.flex.items-center.justify-center.bg-black",
            props,
            ["div.bg-green-5.w2.h2.br8.o8"] if i == pos else None,
        ]
    elif v == 2:
        return [
            "div.w3.h3.flex.items-center.justify-center.bg-white",
            props,
            ["div.bg-green-5.w2.h2.br8"],
        ]


def render_observation(observation, pos=None, set_pos=None):
    return [
        "div.flex.flex-column-reverse.ba.gray-10.g1.bg-gray-10.mt3",
        *[
            render_observation_entry(i, v, pos, set_pos)
            for i, v in enumerate(observation)
        ],
    ]


def component_key(component_name):
    return re.sub(r"_?\d+$", "", component_name)


component_labels = {
    "assemblies_p": "model probabilities",
    "assemblies_q": "proposal probabilities",
    "wta": "sampled state",
    "state_buffer": "previous state",
    "mux_p": "scoring sample under model",
    "mux_q": "scoring sample under proposal",
    "accum": "proposal score numerator",
    "tik_p": "model scoring complete",
    "tik_q": "proposal scoring complete",
}

component_colors = {
    "assemblies_p": "#efb118",
    "assemblies_q": "#ff725c",
    "wta": "#9c6b4e",
    "state_buffer": "#ff8ab7",
    "mux_p": "#efb118",
    "mux_q": "#ff725c",
    "accum": "#4269d0",
    "tik_p": "#efb118",
    "tik_q": "#ff725c",
}


def set_toggle(the_set, item):
    """Toggle the presence of an item in a set."""
    if item in the_set:
        the_set.remove(item)
    else:
        the_set.add(item)


def components_toggle_around(active_components, component):
    """Toggle between a single item and all other items in the component set."""
    all_items = set(component_labels.keys())
    if active_components == {component}:
        active_components.clear()
        active_components.update(all_items - {component})
    else:
        active_components.clear()
        active_components.add(component)


def render_legend(active_components: set):
    return Plot.Hiccup(
        [
            "div.flex.flex-wrap.g2",
            {"style": {"marginLeft": 85}},
            *[
                [
                    "div.p2.flex.items-center.g1",
                    {
                        "key": component,
                        "onClick": (
                            lambda e, c=component: components_toggle_around(
                                active_components, c
                            )
                            if e["altKey"]
                            else set_toggle(active_components, c)
                        ),
                        "style": {
                            "opacity": 1 if component in active_components else 0.5
                        },
                    },
                    [
                        "div.w1.h1.flex.items-center.justify-center.f2.mono",
                        {"style": {"background-color": color}},
                        "/" if component not in active_components else None,
                    ],
                    component_labels[component],
                ]
                for component, color in component_colors.items()
            ],
        ]
    )


def js_array(items):
    """Convert a Python list to a JavaScript array string."""
    return "[" + (",".join(f"'{item}'" for item in items)) + "]"


def js_obj(d):
    """Convert a Python dictionary to a JavaScript object string."""
    return "{" + ",".join(f"'{k}': '{v}'" for k, v in d.items()) + "}"


component_labels_transform = Plot.js(
    """(data, facets) => {
    const MIN_GAP = 3;
    const LABELS = COMPONENT_LABELS;
    const COLORS = COMPONENT_COLORS;

    const componentExtremes = data.reduce((acc, [id, _, component]) => {
        if (!(component in acc)) acc[component] = [id, id];
        else {
            acc[component][0] = Math.min(acc[component][0], id);
            acc[component][1] = Math.max(acc[component][1], id);
        }
        return acc;
    }, {});

    let lastPosition = -Infinity;
    const result = Object.entries(componentExtremes)
        .sort(([_, [minA]], [__, [minB]]) => minA - minB)
        .map(([component, [minId, maxId]]) => {
            const position = Math.max(Math.round((minId + maxId) / 2), lastPosition + MIN_GAP);
            lastPosition = position;
            return [position, LABELS[component], COLORS[component]];
        });

    return {data: result, facets};
}
""".replace("COMPONENT_LABELS", js_obj(component_labels)).replace(
        "COMPONENT_COLORS", js_obj(component_colors)
    )
)


def render_spike_plot(curr_ss_spikes, active_components=None):
    data = Plot.cache(
        [
            [id, time, component_key(component)]
            for spikes in curr_ss_spikes
            for id, (times, component) in spikes.items()
            for time in times
        ]
    )
    return (
        Plot.tickX(
            data,
            {
                "x": "1",
                "y": "0",
                "stroke": "2",
                "filter": Plot.js(
                    f"(d) => {js_array(active_components)}.includes(d[2])"
                )
                if active_components is not None
                else None,
            },
            tip=True,
        )
        + Plot.domainY(range(0, 133 + 3))
        + Plot.text(
            data,
            {
                "frameAnchor": "left",
                "transform": component_labels_transform,
                "textAnchor": "end",
                "lineAnchor": "middle",
                "dx": -6,
                "dy": -2,
                "y": "0",
                "text": "1",
                "fill": "2",
                "lineWidth": 20,
            },
        )
        + Plot.axisX(
            {
                "label": None,
                "ticks": 4,
                "anchor": "top",
                "tickFormat": js("(t)=> t + 'ms'"),
            }
        )
        + Plot.axisY({"tickSize": 1, "tickFormat": None, "label": None})
        + Plot.color_map(component_colors)
        + {"height": 500, "width": 600, "y": {"axis": None}, "marginLeft": 180}
    )


def render_plot(curr_ss_spikes, observation, pos, set_pos, active_components):
    return Plot.Hiccup(
        "div",
        ["div.ml6", render_legend(active_components)],
        [
            "div.flex.items-start",
            render_observation(observation, pos, set_pos),
            render_spike_plot(curr_ss_spikes, active_components),
        ],
    )


async def run_snmc_livedemo(
    variables,
    initial_model,
    step_model,
    initial_proposal,
    step_proposal,
    obs_model,
    assembly_size,
    num_particles,
    first_observation,
    positions,
    occluder_varbs,
):
    try:
        initial_model = jax.jit(initial_model.importance)
        step_model = jax.jit(step_model.importance)
        initial_proposal = jax.jit(initial_proposal.importance)
        step_proposal = jax.jit(step_proposal.importance)
        obs_model = jax.jit(obs_model.simulate)
        occ_loc, occ_width = occluder_varbs
        model_variables, proposal_variables, obs_variables = variables
        components = ()

        def index_to_obs(index):
            obs = jnp.zeros(9)
            occ_indices = jnp.arange(occ_loc, occ_loc + occ_width).astype(int)
            obs_w_target = obs.at[index].set(2)
            obs_w_target_and_occ = obs_w_target.at[occ_indices].set(jnp.ones(occ_width))
            return obs_w_target_and_occ

        def curr_ms_time():
            return time.time() * 1000

        async def snmc_worker(particles, observation):
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
            particles = snmc_particle_filter_step_variables(
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
                tuple(
                    [v["support"][p.choicemap[v["variable"]]] for v in model_variables]
                )
                for p in particles
            ]
            particles = snmc_particle_filter_score_obs(
                subkey,
                observation,
                obs_model,
                obs_args,
                obs_variables,
                model_variables,
                particles,
            )
            particles = list(map(lambda p: p.score_particle(), particles))
            resampler = Resampler_Analog(particles)
            resampler.norm_and_resample()
            return particles, resampler

        key = jax.random.PRNGKey(5000)
        particles = initialize_snmc_particle_filter(
            key,
            variables,
            initial_model,
            initial_proposal,
            obs_model,
            assembly_size,
            num_particles,
            first_observation,
            "analog",
        )
        resampler = Resampler_Analog(particles)
        resampler.norm_and_resample()

        curr_ss_spikes, curr_rs_spikes = snmc_spikes_wrapper(
            [(), [particles], [resampler]],
            "y",
            [0],
            range(len(particles)),
            components,
            False,
        )

        active_components = set(component_labels.keys())

        slider_pos = 1

        def set_slider_pos(i):
            nonlocal slider_pos
            slider_pos = i

        spike_plot = render_plot(
            curr_ss_spikes[:1],
            first_observation[0],
            slider_pos,
            set_slider_pos,
            active_components,
        )
        display(spike_plot)

        particles = resampler.particles
        time_win = 1000
        start_run_timepoint = curr_ms_time()
        num_passes = 0

        curr_ss_spikes = curr_ss_spikes[0]

        worker_task = None
        try:
            for i in range(200000):
                observation = index_to_obs(slider_pos)
                key, subkey = jax.random.split(key, 2)

                if num_passes == 0 or worker_task is None:
                    worker_task = asyncio.ensure_future(
                        snmc_worker(particles, (observation,))
                    )

                try:
                    particles, resampler = await asyncio.wait_for(
                        worker_task, timeout=0.01
                    )
                    ss_spikes, rs_spikes = snmc_spikes_wrapper(
                        [(), [particles], [resampler]],
                        "y",
                        [0],
                        range(len(particles)),
                        components,
                        False,
                    )
                    run_timepoint = curr_ms_time() - start_run_timepoint
                    curr_ss_spikes = merge_and_roll_spikes(
                        curr_ss_spikes,
                        ss_spikes[0],
                        run_timepoint,
                        [run_timepoint - time_win, run_timepoint + time_win],
                    )

                    particles = resampler.particles

                    worker_task = asyncio.ensure_future(
                        snmc_worker(particles, (observation,))
                    )
                except asyncio.TimeoutError:
                    run_timepoint = curr_ms_time() - start_run_timepoint
                    curr_ss_spikes = merge_and_roll_spikes(
                        curr_ss_spikes,
                        {},
                        run_timepoint,
                        [run_timepoint - time_win, run_timepoint + time_win],
                    )

                num_passes += 1
                spike_plot.reset(
                    render_plot(
                        [curr_ss_spikes],
                        observation,
                        slider_pos,
                        set_slider_pos,
                        active_components,
                    )
                )

                await asyncio.sleep(0)  # Allow other tasks to run
        except Exception as e:
            print(f"An error occurred in the main loop: {str(e)}")
            print("Stacktrace:")
            traceback.print_exc()
        finally:
            # Cancel any pending tasks
            if worker_task is not None and not worker_task.done():
                worker_task.cancel()
    except Exception as e:
        print(f"An error occurred in run_snmc_livedemo: {str(e)}")
        print("Stacktrace:")
        traceback.print_exc()


def histogram(title, values, ground_truth):
    if values is not None:
        return (
            Plot.rectY(values, Plot.binX({"y": "count"}))
            + Plot.ruleX(
                [ground_truth], stroke=Plot.constantly("Ground truth"), strokeWidth=2
            )
            + Plot.color_map({"Ground truth": "blue"})
            + Plot.color_legend()
            + Plot.title(title)
        )
    return None


def constrained_step_demo(
    step_model,
    step_proposal,
    assembly_size,
    num_particles,
    observation,
    start_kp,
    start_kq,
    clamped_state_y,
):
    # can also add observation as an argument to this function
    step_model = jax.jit(step_model.importance)
    step_proposal = jax.jit(step_proposal.importance)
    key = jax.random.PRNGKey(1000)
    # ball was previously at position 1, with velocity 1
    vy = 1
    vy_constraint = CMB.d({"vy": vy})
    p_args = (clamped_state_y - vy, vy)
    q_args = p_args + (observation,)
    p_probs, _ = get_categorical_probs(key, step_model, "y", p_args, vy_constraint)
    q_probs, _ = get_categorical_probs(
        key, step_proposal, ("target", "y"), q_args, vy_constraint
    )
    # _, true_p = step_model(key, vy_constraint, p_args)
    # _, true_q = step_proposal(key, vy_constraint, q_args)
    true_p = p_probs[clamped_state_y]
    true_q = q_probs[clamped_state_y]

    print(true_p, true_q)
    # STATIC PLOT THE OBSERVATION AS AN ARRAY JUST LIKE IN LIVE DEMO.

    params = {"kp": start_kp, "kq": start_kq}

    plot = Plot.new()
    display(plot)

    def render():
        kp, kq, p, one_over_q, spikes = (
            params.get("kp"),
            params.get("kq"),
            params.get("p"),
            params.get("one_over_q"),
            params.get("spikes"),
        )

        plot.reset(
            Column(
                Row(
                    [
                        "div.flex.items-center.mr4",
                        ["label.mr2", "kp:"],
                        [
                            "input",
                            {
                                "type": "range",
                                "min": 1,
                                "max": 1000,
                                "defaultValue": kp,
                                "onChange": lambda e: params.update(
                                    {"kp": int(e["value"])}
                                )
                                or render(),
                            },
                        ],
                        ["span.ml2", kp],
                    ],
                    [
                        "div.flex.items-center.mr4",
                        ["label.mr2", "kq:"],
                        [
                            "input",
                            {
                                "type": "range",
                                "min": 1,
                                "max": 1000,
                                "defaultValue": kq,
                                "onChange": lambda e: params.update(
                                    {"kq": int(e["value"])}
                                )
                                or render(),
                            },
                        ],
                        ["span.ml2", kq],
                    ],
                    [
                        "button.br2.bg-blue-5.white.pa2.outline-0.bn.px4.py3.glow.o11",
                        {"onClick": lambda e: launch_particles_buttonpressed()},
                        "Launch!",
                    ],
                ),
                html(
                    "div.flex",
                    ["div.mr4", render_observation(observation)],
                    Column(
                        Row(
                            histogram("p", p, np.log(true_p)),
                            histogram("1/q", one_over_q, np.log(1 / true_q)),
                        ),
                        render_spike_plot([spikes]) if spikes is not None else None,
                    ),
                ),
            )
        )

    render()

    def score_constrained_step(q_probs, p_probs):
        css = ClampedSampleScore(
            assembly_size, clamped_state_y, q_probs, p_probs, params["kq"], params["kp"]
        )
        css.run_scoring()
        return css

    def launch_particles_buttonpressed():
        nonlocal params
        one_over_q = []
        p = []
        spikes = {}
        for i in range(num_particles):
            clamped_ss = score_constrained_step(q_probs, p_probs)
            one_over_q.append(clamped_ss.one_over_q)
            p.append(clamped_ss.p)
            if i == 0:
                spikes = invert_spiketime_labels(
                    get_spikes_from_samplescore(clamped_ss)
                )
        params.update({"p": p, "one_over_q": one_over_q, "spikes": spikes})
        render()
        return p, one_over_q, spikes

    p, one_over_q, spikes = launch_particles_buttonpressed()
    return p, one_over_q, spikes


def run_snmc_particle_filter(
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
    particles = initialize_snmc_particle_filter(
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
    if dig_or_analog == "digital":
        resampler = Resampler(particles)
    elif dig_or_analog == "analog":
        resampler = Resampler_Analog(particles)
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
        particles = snmc_particle_filter_step_variables(
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
        particles = snmc_particle_filter_score_obs(
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
        if dig_or_analog == "digital":
            resampler = Resampler(particles)
        elif dig_or_analog == "analog":
            resampler = Resampler_Analog(particles)
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
