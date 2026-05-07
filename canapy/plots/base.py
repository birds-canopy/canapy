# Author: Nathan Trouvain at 11/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: BSD-3-Clause
# Copyright: Nathan Trouvain
import pathlib
import itertools

import numpy as np
import librosa as lbr
import matplotlib.pyplot as plt
import matplotlib.figure

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models import LinearColorMapper, BasicTicker
from bokeh.models import ColorBar
from bokeh.models import WheelZoomTool

from ..timings import seconds_to_audio, audio_to_frames


def plot_bokeh_confusion_matrix(cm, classes=None, title=None):

    if classes is None:
        classes = [str(i) for i in range(cm.shape[0])]
    else:
        classes = list(classes)

    if "TRASH" in classes:
        idx = classes.index("TRASH")
        new_order = [i for i in range(len(classes)) if i != idx] + [idx]
        classes = [classes[i] for i in new_order]
        cm = cm[np.ix_(new_order, new_order)]

    n = len(classes)

    data = ColumnDataSource(
        {
            "xlabel": [u for u, v in itertools.product(classes, classes)],
            "ylabel": [v for u, v in itertools.product(classes, classes)],
            "confusion": cm.T.ravel(),
        }
    )

    colormap = LinearColorMapper(palette="Magma256", low=0.1, high=1)

    p = figure(
        title=title,
        x_axis_location="above",
        tools="hover,save,wheel_zoom,box_zoom,pan,reset",
        x_range=classes,
        y_range=list(reversed(classes)),
        tooltips=[("True", "@ylabel"), ("Predicted", "@xlabel"), ("Recall", "@confusion{0.000}")],
        active_scroll="wheel_zoom",
        active_drag="box_zoom",
        sizing_mode="stretch_both"
    )

    p.grid.grid_line_color = None
    p.axis.major_label_text_font_size = "10px"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi / 2
    p.xaxis.axis_label = "Predicted"
    p.yaxis.axis_label = "True label"
    p.axis.axis_label_text_font_size = "11px"
    p.axis.axis_label_standoff = 8

    p.rect(
        "xlabel",
        "ylabel",
        1,
        1,
        source=data,
        line_color=None,
        color={"field": "confusion", "transform": colormap},
    )

    color_bar = ColorBar(
        color_mapper=colormap,
        ticker=BasicTicker(),
        label_standoff=12,
        border_line_color=None,
        location=(0, 0),
    )

    p.add_layout(color_bar, "right")

    return p


def plot_bokeh_label_count(df):

    s = df.groupby("label")["label"].count()
    s.sort_values(ascending=False, inplace=True)

    data = ColumnDataSource({"x": s.index.values.tolist(), "top": s})

    p = figure(
        title="Misclassified samples count",
        tools="hover,save",
        x_range=s.index.values.tolist(),
        height=350,
        tooltips=[("class", "@x"), ("", "@top")],
    )

    p.xaxis.major_label_orientation = np.pi / 3

    p.vbar(x="x", top="top", width=0.9, source=data)

    return p, s


def plot_segment_melspectrogram(
    notated_path,
    onset_s,
    offset_s,
    sampling_rate=16000,
    hop_length=0.01,
    n_fft=2048,
    win_length=0.02,
    fmin=500,
    fmax=8000,
    return_audio=False,
    show_ticks=False,
    x_tick_ref_s=None,
):


    audio_file = pathlib.Path(notated_path)
    if audio_file.suffix == ".npy":
        y = np.load(str(audio_file))
    else:
        y, sampling_rate = lbr.load(audio_file, sr=sampling_rate)

    max_length = 1 * sampling_rate
    d = offset_s * sampling_rate - onset_s * sampling_rate
    delta = max(0, (max_length - d)) / 2

    onset_audio = seconds_to_audio(onset_s, sampling_rate)
    offset_audio = seconds_to_audio(offset_s, sampling_rate)

    s_delta = onset_audio - delta
    e_delta = offset_audio + delta

    if s_delta < 0:
        e_delta += -s_delta
    elif e_delta > len(y):
        s_delta -= e_delta - len(y)

    s_delta = max(0, round(s_delta))
    e_delta = min(len(y), round(e_delta))

    full = y[s_delta:e_delta]

    hop_length_samples = seconds_to_audio(hop_length, sampling_rate)
    win_length_samples = min(seconds_to_audio(win_length, sampling_rate), n_fft)

    spec = lbr.feature.melspectrogram(
        y=full,
        sr=sampling_rate,
        n_fft=n_fft,
        win_length=win_length_samples,
        hop_length=hop_length_samples,
        fmin=fmin,
        fmax=fmax,
    )

    spec = lbr.power_to_db(spec)

    fig = matplotlib.figure.Figure(figsize=(4, 1.5))
    ax = fig.subplots()

    sample_onset = audio_to_frames(onset_audio - s_delta, hop_length_samples)
    sample_offset = spec.shape[1] - audio_to_frames(e_delta - offset_audio, hop_length_samples)

    ax.imshow(spec, origin="lower", cmap="magma", aspect="auto")

    ax.axvline(sample_onset, color="white", linestyle="--", lw=1)
    ax.axvline(sample_offset, color="white", linestyle="--", lw=1)

    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.15)

    if show_ticks:
        n_mels, n_frames = spec.shape

        tick_h = n_mels * 0.12
        tick_fontsize = 7

        mel_freqs = lbr.mel_frequencies(n_mels, fmin=fmin, fmax=fmax)

        for yp in np.linspace(0, n_mels - 1, 3):
            idx = min(int(round(yp)), n_mels - 1)
            freq_hz = float(mel_freqs[idx])

            if freq_hz >= 1000:
                lbl = f"{freq_hz / 1000:.1f}k"
                if lbl.endswith(".0k"):
                    lbl = lbl[:-3] + "k"
            else:
                lbl = f"{int(round(freq_hz))}"

            ax.plot([0, n_frames * 0.025], [yp, yp], color="white", lw=1.0)

            ax.text(
                n_frames * 0.025 + n_frames * 0.01,
                yp,
                lbl,
                color="white",
                fontsize=tick_fontsize,
                va="center",
                ha="left",
                clip_on=False,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.75, ec="none"),
            )

        t_start = s_delta / sampling_rate
        margin = max(2, int(n_frames * 0.01))

        ticks = [
            (0, 0, "left"),
            (sample_onset, onset_s - t_start, "center"),
            (sample_offset, offset_s - t_start, "right"),
        ]

        ticks_vis = []
        for xp, t_val, align in ticks:
            xp_vis = np.clip(xp, margin, n_frames - 1 - margin)
            ticks_vis.append([xp, xp_vis, t_val, align])

        # gestion chevauchement labels
        min_dx = n_frames * 0.12
        ticks_vis.sort(key=lambda x: x[1])

        for i in range(1, len(ticks_vis)):
            prev = ticks_vis[i - 1]
            curr = ticks_vis[i]

            if curr[1] - prev[1] < min_dx:
                shift = (min_dx - (curr[1] - prev[1])) / 2
                prev[1] -= shift
                curr[1] += shift

        for t in ticks_vis:
            t[1] = np.clip(t[1], margin, n_frames - 1 - margin)

        for i, (xp_true, xp_vis, t_val, align) in enumerate(ticks_vis):
            ax.plot([xp_true, xp_true], [0, tick_h], color="white", lw=1.0)

            if i == 0:
                continue

            ax.text(
                xp_vis,
                tick_h + n_mels * 0.04,
                f"{t_val:.2f}s",
                color="white",
                fontsize=tick_fontsize,
                va="bottom",
                ha=align,
                clip_on=False,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.75, ec="none"),
            )

    if return_audio:
        sample_only = y[max(0, onset_audio): min(len(y), offset_audio)]
        return (
            fig,
            np.int16(full * np.iinfo(np.int16).max),
            np.int16(sample_only * np.iinfo(np.int16).max),
        )
    else:
        return fig