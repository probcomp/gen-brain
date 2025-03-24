import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import seaborn as sns
import copy
from astropy.convolution import convolve_fft, Gaussian1DKernel
np.seterr(divide="ignore")

""" ANALOG PLOT LIB """


def gather_spikes_from_single_particle_samplescore(
    pf_results, particle_id, v, step_range
):
    particles, resamplers = pf_results[1:3]
    particle_over_time = [
        p[particle_id] for p in particles[step_range[0] : step_range[-1] + 1]
    ]
    resamplers = resamplers[step_range[0] : step_range[-1] + 1]
    resampler_start_and_end_times = [
        (r.resampler_start_time, r.resampler_end_time) for r in resamplers
    ]
    offset_time = 0
    sss = [p.samplescores[v] for p in particle_over_time]
    samplescore_spiketimes = {}
    resampler_spiketimes = {}
    for step, ss in enumerate(sss):
        new_ss_spikes = get_spikes_from_samplescore(ss)
        new_resampler_spikes = get_spikes_from_resampler(resamplers[step])
        offset_ss_spikes = add_offset_to_spikes(new_ss_spikes, offset_time)
        resampler_offset = offset_time + resampler_start_and_end_times[step][0]
        offset_resampler_spikes = add_offset_to_spikes(
            new_resampler_spikes, resampler_offset
        )
        if step == 0:
            samplescore_spiketimes = offset_ss_spikes
            resampler_spiketimes = offset_resampler_spikes
        else:
            new_ss_spiketimes = {}
            new_resampler_spiketimes = {}
            for neuron_id, (spikes, label) in samplescore_spiketimes.items():
                extended_ss_spikes = np.concatenate(
                    (spikes, offset_ss_spikes[neuron_id][0])
                )
                new_ss_spiketimes[neuron_id] = (extended_ss_spikes, label)
            for neuron_id, (spikes, label) in resampler_spiketimes.items():
                extended_rs_spikes = np.concatenate(
                    (spikes, offset_resampler_spikes[neuron_id][0])
                )
                new_resampler_spiketimes[neuron_id] = (extended_rs_spikes, label)
            samplescore_spiketimes = new_ss_spiketimes
            resampler_spiketimes = new_resampler_spiketimes
        offset_time += resampler_start_and_end_times[step][1]
    return samplescore_spiketimes, resampler_spiketimes


def add_offset_to_spikes(spiketimes, offset):
    new_spiketimes = {}
    for key, (spike_array, label) in spiketimes.items():
        new_spikes = spike_array + offset
        new_spiketimes[key] = (new_spikes, label)
    return new_spiketimes


def organize_labels_into_layers(ss_spiketimes):
    order = {
        1: ["wta", "state_buffer"],
        2: ["assemblies_q", "assemblies_p"],
        3: ["mux", "accum", "tik"],
    }

    def assign_spiketrains(st, new_st, layercounter, neuroncounter):
        if layercounter > np.max([k for k in order.keys()]):
            return new_st
        else:
            curr_labels = order[layercounter]
            for c in curr_labels:
                for s_times, label in st.values():
                    if label[0:3] == "ass":
                        if label[0:12] == c:
                            new_st[neuroncounter] = (s_times, label)
                            neuroncounter += 1
                    elif label[0:3] == c[0:3]:
                        new_st[neuroncounter] = (s_times, label)
                        neuroncounter += 1
            return assign_spiketrains(st, new_st, layercounter + 1, neuroncounter)

    new_spiketimes = assign_spiketrains(ss_spiketimes, {}, 1, 0)
    return new_spiketimes


def invert_spiketime_labels(d):
    max_keys = max(d.keys())
    new_d = {max_keys - k: v for k, v in d.items()}
    return dict(sorted(new_d.items()))


def get_spikes_from_samplescore(ss):
    spiketimes = {}
    n_counter = 0
    for key, item in ss.component_dict.items():
        # print(key)
        for k, activity in item.items():
            label = key + "_" + k
            if key != "assemblies":
                spiketimes[n_counter] = (np.array(activity), label)
                n_counter += 1
            else:
                for neuron in activity:
                    spiketimes[n_counter] = (np.array(neuron), label)
                    n_counter += 1
    return invert_spiketime_labels(spiketimes)


def get_spikes_from_resampler(resampler):
    resampler_spiketimes = {}
    n_counter = 0
    for key, item in resampler.component_dict.items():
        for k, activity in item.items():
            label = key + "_" + k
            resampler_spiketimes[n_counter] = (np.array(activity), label)
            n_counter += 1
    return resampler_spiketimes


# MODEL AGNOSTIC PLOTTING BACKEND.


def my_tab20(num_colors):
    if num_colors <= 16:
        num_colors = 17
    particle_colors = [9, 13, 16]
    non_particle_colors = [c for c in range(num_colors) if c not in particle_colors]
    new_20 = [
        sns.color_palette("tab20b", num_colors)[i]
        for i in particle_colors + non_particle_colors
    ]
    return new_20


# static_plot is entirely correct unless you add a
# second spiketimes to the list. one is perfect.
# figure out why


def static_plot_snmc(ss_spiketimes_list_input, *resampler):
    ss_spiketimes_list = copy.deepcopy(ss_spiketimes_list_input)

    def particle_id_decode(label):
        if label[-2] == "_":
            return int(label[-1])
        else:
            return int(label[-2:])

    fig, ax = plt.subplots()
    #    cpal = sns.color_palette("tab20b", 20)
    cpal = my_tab20(100)

    if resampler != ():
        resampler_spikedict = resampler[0]
        indexed_resampler_spikes = {
            i + len(ss_spiketimes_list[0]): v for i, v in resampler_spikedict.items()
        }
        #        return indexed_resampler_spikesmerged_dict = {**dict1, **dict2}
        ss_spiketimes_list[0] = {**ss_spiketimes_list[0], **indexed_resampler_spikes}

    # this is impervious to whether resampler has been added or not.
    num_components = len(ss_spiketimes_list[0])

    for c, spiketimes in enumerate(ss_spiketimes_list):
        for neuron_id, spikes_and_label in spiketimes.items():
            spikes, label = spikes_and_label
            neuron_y = num_components - (neuron_id + 1)
            if label[0:3] in ["res", "nor"]:
                color_id = particle_id_decode(label)
            else:
                color_id = c
            ax.vlines(
                spikes, neuron_y, neuron_y + 0.8, color=cpal[color_id], linewidth=1.0
            )

    comp_labels = [v[1] for k, v in ss_spiketimes_list[0].items()]
    #    xlim = np.max(np.concatenate([v[0] for v in ss_spiketimes_list.values()])) + 1
    comp_labels.reverse()
    #    ax.set_xlim(-5, xlim)
    ax.set_ylim(0.5, len(comp_labels) + 0.5)
    ax.set_yticks(range(len(comp_labels)))
    ax.set_yticklabels(comp_labels, fontdict={"fontsize": 5})
    ax.set_xlabel("Time")
    ax.set_ylabel("Neuron ID")
    ax.set_title("Raster Plot")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def animate_snmc_spikes(spiketimes):
    # Create the initial plot with empty vlines for each neuron
    num_neurons = len(spiketimes)
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = [
        ax.vlines([], [], [], color="darkcyan", linewidth=1.5)
        for _ in range(num_neurons)
    ]
    ax.set_xlim(-5, 1000)  # Initially display the most recent 10 seconds
    ax.set_ylim(0.5, num_neurons + 0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Neuron ID")
    ax.set_title("Raster Plot")
    ax.grid(False)
    plt.tight_layout()
    win = 2000

    # Function to update the plot for each frame
    def update(frame):
        frame = (frame - 1) * 6
        #        Calculate the time window to display (most recent 10 seconds)
        time_window_start = frame
        time_window_end = frame + win
        for neuron_id, spikes_and_label in spiketimes.items():
            # Calculate the visible spikes within the time window
            spikes, _ = spikes_and_label
            neuron_y = num_neurons - (neuron_id + 1)
            visible_spikes = spikes[
                (spikes >= time_window_start) & (spikes <= time_window_end)
            ]
            # Update the data in vlines for the corresponding neuron
            lines[neuron_id].set_segments(
                [
                    [
                        [spike - time_window_start, neuron_y],
                        [spike - time_window_start, neuron_y + 0.8],
                    ]
                    for spike in visible_spikes
                ]
            )

        # Update x-axis limits to show the most recent 10 seconds
        ax.set_xlim(0, win)
        ax.set_title("Frame: {}".format(frame))
        return lines

    # Create the animation
    frames_total = np.max(
        np.concatenate([v[0] for v in spiketimes.values() if not len(v[0]) == 0])
    ).astype(int)
    ani = FuncAnimation(
        fig,
        update,
        interval=40,
        frames=np.arange(0, (frames_total + win) / 10, 1),
        repeat=True,
        blit=False,
    )
    return ani


def snmc_spikes_wrapper(
    pf_results, v, step_range, particles_to_plot, components_to_plot, plotit
):
    all_ss_component_spiketimes = []
    for p_i, p in enumerate(particles_to_plot):
        ss_raw, rs = gather_spikes_from_single_particle_samplescore(
            pf_results, p, v, step_range
        )

        ss = organize_labels_into_layers(ss_raw)
        if components_to_plot != ():
            ss_spikearrays = [
                v for i, (k, v) in enumerate(ss.items()) if v[1] in components_to_plot
            ]
            ss_component_spiketimes = {i: v for i, v in enumerate(ss_spikearrays)}
            if p_i == 0:
                rs_spikearrays = [
                    v
                    for i, (k, v) in enumerate(rs.items())
                    if v[1] in components_to_plot
                ]
                rs_component_spiketimes = {i: v for i, v in enumerate(rs_spikearrays)}

        else:
            ss_component_spiketimes = ss
            if p_i == 0:
                rs_component_spiketimes = rs
        all_ss_component_spiketimes.append(ss_component_spiketimes)

    if plotit:
        if not rs_component_spiketimes == {}:
            static_plot_snmc(all_ss_component_spiketimes, rs_component_spiketimes)
        else:
            static_plot_snmc(all_ss_component_spiketimes)
    return all_ss_component_spiketimes, rs_component_spiketimes


def plot_particle_weight_and_state(pf_results, v, step_range):
    particles = pf_results[1][step_range[0] : step_range[-1]]
    resamplers = pf_results[2][step_range[0] : step_range[-1]]
    marker_colors = my_tab20(100)[0 : len(pf_results[1][0])]
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for step, (particles_at_step, resampler_at_step) in enumerate(
        zip(particles, resamplers)
    ):
        particle_state = np.array([p.choicemap[v] for p in particles_at_step])
        particle_probs = resampler_at_step.resampler_probs
        offset = 0.1
        size_scale = 50
        for p, (size, y, color) in enumerate(
            zip(particle_probs, particle_state, marker_colors)
        ):
            ax.scatter(step + offset * p, y, s=size * size_scale, color=color)
    plt.show()


# this will combine all neurons in a spiketimes dictionary
# that have the same component id (i.e. assembly_p0)
def merge_component_dicts(spiketimes, component_order):
    merged_dict = {}
    for i, curr_component in enumerate(component_order):
        curr_spiketimes = np.array([])
        for k, (spikes, component) in spiketimes.items():
            if component == curr_component:
                curr_spiketimes = np.sort(np.concatenate((curr_spiketimes, spikes)))
                continue
        merged_dict[i] = (curr_spiketimes, curr_component)
    return merged_dict


# this will combine all spikes across particles
def merge_particle_dicts(spikes_by_particle):

    def add_scalar_to_keys(scalar, d):
        new_d = {}
        for k, v in d.items():
            new_d[k + scalar] = v
        return new_d
    
    final_spiketimes = {}
    max_key = 0
    for st in spikes_by_particle:
        new_st = add_scalar_to_keys(max_key, st)
        max_key += max(st.keys())
        final_spiketimes.update(new_st)
    return final_spiketimes


# the values are already in milliseconds that come out of spiketimes.
# the bin size can be 1 or 10.


def merge_and_roll_spikes(orig_sts, new_sts, new_spike_offset, rolling_window):
    merged_dict = {}
    if new_sts != {}:
        for neuron_id, (spikes, component_id) in orig_sts.items():
            new_spikes = new_sts[neuron_id][0] + new_spike_offset
            merged_spikes = np.concatenate((spikes, new_spikes), axis=0)
            ms_clip_top = merged_spikes[merged_spikes < rolling_window[1]]
            ms_clip_bottom = ms_clip_top[ms_clip_top > rolling_window[0]]
            merged_dict[neuron_id] = (ms_clip_bottom, component_id)
    else:
        merged_dict = orig_sts
    return merged_dict


def eeg(spiketimes):
    def count_values_in_bins(arr, bin_size):
        num_bins = int((np.max(arr) - 0) / bin_size) + 1
        bin_indices = (arr / bin_size).astype(int)
        bin_counts = np.bincount(bin_indices, minlength=num_bins)
        return bin_counts

    #    resfactor = 1
    binsize = 1
    spikes = [
        np.round(v[0], binsize) for k, v in spiketimes.items()
    ]  # if v[1] in filterfunc]
    all_spikes = np.sort(np.concatenate(spikes))
    #   bcs = count_values_in_bins(all_spikes, 1/(10**resfactor))
    bcs = count_values_in_bins(all_spikes, binsize)
    σ = 10
    kernel = Gaussian1DKernel(σ)
    eeg_signal = convolve_fft(bcs, kernel)
    #    plt.plot(eeg_signal)
    #    plt.show()
    return eeg_signal


# this is going to show a physics observation where the first N steps have occluded targets.
def variability_quench(pf_results, variable, components):
    num_total_steps = len(pf_results[2])
    num_particles = len(pf_results[1][0])
    fano_by_step = []
    all_spikearrays = []
    for step in range(num_total_steps):
        ss_spiketimes, res_spiketimes = snmc_spikes_wrapper(
            pf_results,
            variable,
            range(step, step + 1),
            range(num_particles),
            components,
        )
        for ss in ss_spiketimes:
            for v in ss.values():
                all_spikearrays.append(v[0])
        fano_by_step.append(fanofactor(all_spikearrays))
    return fano_by_step


def get_components(variable_support, particle_range, ctx_or_bg):
    components = []
    if "ctx" in ctx_or_bg:
        for i, vs in enumerate(variable_support):
            components.append("wta_" + str(int(i)))
            components.append("state_buffer_" + str(int(i)))
            components.append("assemblies_p" + str(int(i)))
            components.append("assemblies_q" + str(int(i)))
            components.append("mux_p" + str(int(i)))
            components.append("mux_q" + str(int(i)))
        for i in range(10):
            components.append("accum_" + str(i))
        components.append("tik_p")
        components.append("tik_q")

    if "bg" in ctx_or_bg:
        for p in particle_range:
            components.append("norm_neurons_" + str(p))
            components.append("resampler_wta_" + str(p))
    return components


# make sure you invert this so the LFP is downward. see Bartosz Teleńczuk paper.
# spikes induce a downward LFP deflection. i think this will still look
# like the wikipedia entry. i can even just use layer 4
# as the LFP so its more convincing that the spikes are causing the LFP.


def lfp_and_spikes(spiketimes, lfp, plot_now):
    spike_height = 1
    lfp_height = 4
    neuron_y = lfp_height + 0.2
    lfp_scaled = (lfp_height / np.max(lfp)) * lfp
    cpal = my_tab20(5)
    components = []
    if plot_now:
        fig, ax = plt.subplots()
        for i, (k, (spikes, component)) in enumerate(spiketimes.items()):
            neuron_y += spike_height * 1.1
            components.append(component)
            if component[0:5] == "assem":
                clr = cpal[0]
            elif component[0:3] in ["wta", "sta"]:
                clr = cpal[1]
            else:
                clr = cpal[2]
            ax.vlines(
                spikes,
                neuron_y,
                neuron_y + 0.8 * spike_height,
                color=clr,
                linewidth=1.0,
            )
        #        sns.histplot(x=spikes, binwidth=10, ax=ax_hist[i])
        #        ax_hist[i].set_xlim([0, 800])
        # ax_hist[i].set_ylim([0, 300])
        ax.legend(components)
        ax.plot(lfp_scaled, color="k")
        plt.show()
    return lfp_scaled


def fanofactor(spiketrains):
    """
    Evaluates the empirical Fano factor F of an array of spike counts

    Given the vector v containing the observed spike counts (one per
    spike train) in the time window [t0, t1], F is defined as:

    .. math::
        F := \frac{var(v)}{mean(v)}

    The Fano factor is typically computed for spike trains representing the
    activity of the same neuron over different trials. The higher F, the
    larger the cross-trial non-stationarity. In theory for a time-stationary
    Poisson process, F=1.

    Parameters
    ----------
    spiketrains : list
        List of spike times for which to compute the Fano factor of spike counts.

    Returns
    -------
    fano : float
        The Fano factor of the spike counts of the input spike trains.
        Returns np.NaN if an empty list is specified, or if all spike trains
        are empty.
    """
    # Build array of spike counts (one per spike train)
    spike_counts = np.array([len(st) for st in spiketrains])

    # Compute FF
    if all(count == 0 for count in spike_counts):
        # empty list of spiketrains reaches this branch, and NaN is returned
        return np.nan
    fano = spike_counts.var() / spike_counts.mean()
    return fano


def traveling_wave_from_sts(sts):
    fig, ax = plt.subplots()
    cpal = sns.color_palette("Set2")
    for i, st in enumerate(sts):
        eeg_ = eeg(st)
        ax.plot(eeg_, color=cpal[i])
    plt.show()


def selectivity_index(pf_results, particle_id, variable):
    # first find the step where the neuron is most selective.
    # then make a rate out of the number of spikes divided by
    # score_complete_time -race_start_time.
    # we also want a direction selectivity index.
    # you do this by having two pf_results, one going one way, one going the other,
    # and then inverting one pf (i.e. you step forward on one, and backward
    # on the other pf_results[1][10] matched with pf_results2[1][0]
    # only do 3 steps, and make the normalizer the average of the non-max
    # rates.
    samplescores_per_step = [
        pf_results[1][step][particle_id].samplescores[variable]
        for step in range(len(pf_results[1]))
    ]
    spikes_per_step = [get_spikes_from_samplescore(ss) for ss in samplescores_per_step]
    num_neurons = len(spikes_per_step[1])
    single_neuron_over_time = [
        [spikes[neuron] for spikes in spikes_per_step] for neuron in range(num_neurons)
    ]
    labels_and_counts_for_neurons_that_spiked = [
        (spikes[0][1], [len(sp[0]) for sp in spikes])
        for spikes in single_neuron_over_time
        if np.concatenate([s[0] for s in spikes]).size > 0
    ]

    # return labels_and_counts_for_neurons_that_spiked
    # pref - mean / pref + mean
    def p_i(x):
        _, spikecounts = x
        maxarg = np.argmax(spikecounts)
        max_count = spikecounts[maxarg]
        mean_rest = np.mean(np.delete(spikecounts, maxarg))
        pi = (max_count - mean_rest) / (max_count + mean_rest)
        return pi

    preference_index = list(
        map(lambda x: p_i(x), labels_and_counts_for_neurons_that_spiked)
    )
    labels_only = [lb for (lb, spikes) in labels_and_counts_for_neurons_that_spiked]
    print("LABELS ONLY")
    print(labels_only)
    layer = {
        "wta": 2,
        "state_buffer": 2,
        "assemblies": 4,
        "mux": 5,
        "accum": 5,
        "tik": 5,
    }
    layer_keys = list(layer.keys())

    def check_label(lab):
        for layer_key in layer_keys:
            if layer_key in lab:
                return layer_key

    si_by_layer = {2: [], 4: [], 5: []}
    for label, pi in zip(labels_only, preference_index):
        print(label)
        print(check_label(label))
        print(pi)
        si_layer_assignment = layer[check_label(label)]
        si_by_layer[si_layer_assignment].append(pi)

    return si_by_layer, preference_index, labels_and_counts_for_neurons_that_spiked


def direction_selectivity(pf_res1, pf_res2, step, variable):
    samplescore_l = pf_res1[1][step][0].samplescores[variable]
    samplescore_r = pf_res2[1][step][0].samplescores[variable]
    spikes_l = get_spikes_from_samplescore(samplescore_l)
    spikes_r = get_spikes_from_samplescore(samplescore_r)
    labels = []
    ds_indices = []
    for neuron_l, neuron_r in zip(spikes_l.values(), spikes_r.values()):
        labels.append(neuron_l[1])
        if neuron_l[0].size == 0 and neuron_r[0].size == 0:
            ds_index = np.nan
        else:
            ds_index = (len(neuron_l[0]) - len(neuron_r[0])) / (
                len(neuron_l[0]) + len(neuron_r[0])
            )
        ds_indices.append(ds_index)
    return labels, ds_indices


# i think this could be framed as "number inputs entering" and "number of outputs exciting". but it feels like it could be something for the biorealistic paper,
# since the models should be generating LFPs.


def source_sink(pf_results):
    return


def lfp_and_spikes_animated(spiketimes, lfp, interval=10):
    spike_height = 1
    lfp_height = 20
    neuron_y = lfp_height + 0.2
    lfp_scaled = (lfp_height / np.max(lfp)) * lfp
    # Create two subplots: one for the spike/LFP plot, one for the ball/squares
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.01)
    components = [sp[1] for sp in spiketimes.values()]
    cpal = my_tab20(len(spiketimes))
    ax.set_xlim(0, len(lfp_scaled))
    ax.set_ylim(0, neuron_y + 2)
    spike_lines = []
    for i, (k, (spikes, component)) in enumerate(spiketimes.items()):
        if component[0:5] == "assem":
            clr = cpal[1]
        elif component[0:3] in ["wta", "sta"]:
            clr = cpal[0]
        else:
            clr = cpal[2]
        neuron_y += spike_height * 1.1
        spike_lines.append((spikes, neuron_y, clr))
    (lfp_line,) = ax.plot([], [], color="k")

    # Define colors and labels
    colors = cpal[0:3]
    labels = ["State", "Probabilities", "Scores"]
    # Create custom legend handles
    legend_handles = [
        mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))
    ]
    lfp_label = ["LFP" for i in range(lfp_height)]

    def update(frame):
        ax.clear()
        neuron_y_temp = lfp_height + 0.2
        for spikes, neuron_y, color in spike_lines:
            active_spikes = spikes[spikes <= frame]
            ax.vlines(
                active_spikes,
                neuron_y,
                neuron_y + 0.8 * spike_height,
                color=color,
                linewidth=1.0,
            )
            neuron_y_temp += spike_height * 1.1
        ax.plot(np.arange(0, frame), lfp_scaled[:frame], color="k")
        ax.set_xlim(0, len(lfp_scaled))
        ax.set_ylim(0, neuron_y_temp + 2)
        ax.legend(handles=legend_handles)
        ax.set_yticks(1.1 * np.arange(len(components) + lfp_height))
        ax.set_yticklabels(lfp_label + components, fontdict={"fontsize": 4})

    # Create animation with synchronized updates
    ani = FuncAnimation(
        fig, update, frames=np.arange(0, len(lfp_scaled)), interval=interval, blit=False
    )
    return ani


# x_p_assemblies_particle_0 = snmc_spikes_wrapper(pf_results, 'x', range(num_snmc_steps), range(0,1), ["assemblies_p13", "assemblies_p14", "assemblies_p15"])[0]

# there are 3 neuroscience results i think we can explain from stryker.
# one, the average selectivity index of neurons in the different layers
# two, the flow of activity (source / sink) from layer to layer.
# three, the presence of direction selective neurons (state buffer).

# mountcastle contains a second analysis method – what if you go off
# by just a bit, do the receptive fields change?

# you'll have a separate diagram for excitatory neurons vs inhibitory neurons.
# you'll do a drawing of what the circuit looks like with inhibitory neurons
# added.

# ctx = merge_particle_dicts(snmc_spikes_wrapper

# bg = merge_particle_dicts(snmc_spikes_wrapper(pf_results, 'x', range(num_snmc_steps), range(num_particles), get_components(positions, range(num_particles), ['bg'])))

# x_spikes =  merge_particle_dicts(snmc_spikes_wrapper(pf_results, 'x', range(num_snmc_steps), range(num_particles), get_components(positions, range(num_particles), ['ctx']))[0])

# vx_spikes = merge_particle_dicts(snmc_spikes_wrapper(pf_results, 'vx', range(num_snmc_steps), range(num_particles), get_components(positions, range(num_particles), ['ctx'])))

# to  all spikes at once, just merge the return vals, which are
# all spiketimes for ss and all spiketimes for rs. also use all particles.

# merged_assemblies = merge_component_dicts(x_p_assemblies_particle_0[0], ["assemblies_p13", "assemblies_p14", "assemblies_p15"])

# wta_components = ['wta_' + str(int(i)) for i in positions]

# this is for the traveling wave.
# merged_wtas = invert_spiketimes(merge_component_dicts(x_spikes, wta_components))

# lfp_and_spikes(invert_spiketime_labels(merged_assemblies), eeg(x_spikes))

# have to incorporate multiple levels of the bayes net.
# so even if you're querying on a single variable, have to check
# the last spike time for ALL variables per step for ALL particles. That's when
# the resampler starts.

# for any given step, times within a samplescore are all correct and synched.
# the resampler should start at the very end of all samplescore times.
# first go through all the samplescores and save their spiketimes by step.
