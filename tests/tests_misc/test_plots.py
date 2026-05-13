# Author: Nathan Trouvain / Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Nathan Trouvain / Axel Arnaud
"""Tests for canapy/plots/base.py."""

import numpy as np
import pandas as pd
import pytest
import scipy.io.wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure

from canapy.plots import plot_bokeh_confusion_matrix, plot_bokeh_label_count
from canapy.plots.base import plot_segment_melspectrogram


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wav_path(tmp_path_factory):
    rate = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(rate * duration))
    wav = np.sin(2 * np.pi * 440 * t)
    wav_i16 = (wav * np.iinfo(np.int16).max).astype(np.int16)
    path = tmp_path_factory.mktemp("audio") / "tone.wav"
    scipy.io.wavfile.write(str(path), rate, wav_i16)
    return str(path)


# ---------------------------------------------------------------------------
# plot_bokeh_confusion_matrix / plot_bokeh_label_count
# ---------------------------------------------------------------------------

def test_plot_bokeh_confusion_matrix():
    cm = np.random.uniform(size=(10, 10))
    classes = list("abcdefghij")

    fig = plot_bokeh_confusion_matrix(cm, classes)
    assert fig.title is None

    fig = plot_bokeh_confusion_matrix(cm, None, title="foo")
    assert fig.title.text == "foo"


def test_plot_bokeh_label_count():
    df = pd.DataFrame({"label": list("aaabbcdefghabfjkdaz")})
    fig, counts = plot_bokeh_label_count(df)
    assert counts.loc["a"] == 5
    assert counts.loc["z"] == 1


# ---------------------------------------------------------------------------
# plot_segment_melspectrogram
# ---------------------------------------------------------------------------

class TestPlotSegmentMelspectrogram:
    def test_returns_figure(self, wav_path):
        fig = plot_segment_melspectrogram(wav_path, onset_s=0.1, offset_s=0.5)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) > 0

    def test_return_audio_true_gives_three_values(self, wav_path):
        result = plot_segment_melspectrogram(wav_path, onset_s=0.1, offset_s=0.5, return_audio=True)
        assert len(result) == 3
        fig, full_audio, sample_audio = result
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(full_audio, np.ndarray)
        assert isinstance(sample_audio, np.ndarray)
        # the segment extract must be shorter than the full recording
        assert len(sample_audio) < len(full_audio)

    def test_return_audio_false_returns_figure_only(self, wav_path):
        result = plot_segment_melspectrogram(wav_path, onset_s=0.1, offset_s=0.5, return_audio=False)
        assert isinstance(result, matplotlib.figure.Figure)

    def test_show_ticks_true_does_not_raise(self, wav_path):
        fig = plot_segment_melspectrogram(wav_path, onset_s=0.2, offset_s=0.6, show_ticks=True)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_onset_at_zero(self, wav_path):
        fig = plot_segment_melspectrogram(wav_path, onset_s=0.0, offset_s=0.3)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_offset_near_end_of_file(self, wav_path):
        fig = plot_segment_melspectrogram(wav_path, onset_s=1.7, offset_s=2.0)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_sampling_rate(self, wav_path):
        fig = plot_segment_melspectrogram(wav_path, onset_s=0.1, offset_s=0.5, sampling_rate=22050)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_audio_arrays_are_int16(self, wav_path):
        _, full, sample = plot_segment_melspectrogram(wav_path, onset_s=0.2, offset_s=0.5, return_audio=True)
        assert full.dtype == np.int16
        assert sample.dtype == np.int16
