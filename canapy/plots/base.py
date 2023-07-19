# Author: Nathan Trouvain at 11/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pathlib
import itertools

import numpy as np
import librosa as lbr
import matplotlib.pyplot as plt
import matplotlib.figure

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models import LogColorMapper, LogTicker
from bokeh.models import ColorBar

from ..timings import seconds_to_audio, audio_to_frames


def plot_bokeh_confusion_matrix(cm, classes=None, title=None):

    if classes is None:
        classes = [str(i) for i in range(cm.shape[0])]

    data = ColumnDataSource(
        {
            "xlabel": [u for u, v in itertools.product(classes, classes)],
            "ylabel": [v for u, v in itertools.product(classes, classes)],
            "confusion": cm.T.ravel(),
        }
    )

    colormap = LogColorMapper(palette="Magma256", low=1e-3, high=1.0)

    p = figure(
        title=title,
        x_axis_location="above",
        tools="hover,save",
        x_range=classes,
        y_range=list(reversed(classes)),
        tooltips=[("class", "@ylabel, @xlabel"), ("", "@confusion")],
        sizing_mode='stretch_both'
    )

    # p.width = 800
    # p.height = 800
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "12px"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi / 3

    p.rect(
        "xlabel",
        "ylabel",
        1.0,
        1.0,
        source=data,
        line_color="gray",
        color={"field": "confusion", "transform": colormap},
        hover_line_color="white",
    )

    color_bar = ColorBar(
        color_mapper=colormap,
        ticker=LogTicker(),
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
        return_audio=False
    ):

    audio_file = pathlib.Path(notated_path)
    if audio_file.suffix == ".npy":
        y = np.load(str(audio_file))
    else:
        y, sampling_rate = lbr.load(audio_file, sr=sampling_rate)

    max_length = 1 * sampling_rate
    d = offset_s * sampling_rate - onset_s * sampling_rate
    delta = max(0, (max_length - d)) / 2

    onset_audio  = seconds_to_audio(onset_s, sampling_rate)
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
    spec = lbr.feature.melspectrogram(
        y=full,
        sr=sampling_rate,
        n_fft=n_fft,
        win_length=seconds_to_audio(win_length, sampling_rate),
        hop_length=seconds_to_audio(hop_length, sampling_rate),
        fmin=fmin,
        fmax=fmax,
        )

    spec = lbr.power_to_db(spec)

    fig = matplotlib.figure.Figure(figsize=(1.5, 0.5))
    ax = fig.subplots()
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    sample_onset = audio_to_frames(onset_audio - s_delta,
                    seconds_to_audio(hop_length, sampling_rate)),

    sample_offset = spec.shape[1] - audio_to_frames(e_delta - offset_audio,
                    seconds_to_audio(hop_length, sampling_rate)),

    ax.imshow(spec, origin="lower", cmap="magma", aspect="auto")
    ax.axvline(
        sample_onset,
        color="white",
        linestyle="--",
        lw=1,
        marker=">",
        markevery=0.01,
        markersize=5,
        )
    ax.axvline(
        sample_offset,
        color="white",
        linestyle="--",
        lw=1,
        marker="<",
        markevery=0.01,
        markersize=5,
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
