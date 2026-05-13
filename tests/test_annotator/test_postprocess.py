# Author: Nathan Trouvain at 06/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: BSD-3-Clause
# Copyright: Nathan Trouvain
import numpy as np
import pytest

from canapy.timings import frames_to_timed_df, frames_to_seconds
from canapy.annotator.commons.postprocess import frame_df_to_annots_df


def test_frames_to_timed_df(cls_frame_predictions):
    df = frames_to_timed_df(
        cls_frame_predictions,
        notated_path="foo/baz",
        frame_size=256,
        sampling_rate=44100,
    )
    assert len(df) == len(cls_frame_predictions)
    assert {"label", "onset_s", "offset_s", "notated_path"}.issubset(df.columns)
    assert (df["notated_path"] == "foo/baz").all()
    assert df["label"].tolist() == list(cls_frame_predictions)


def test_frame_df_to_annots_df(cls_frame_predictions):
    df = frames_to_timed_df(
        cls_frame_predictions,
        notated_path="foo/baz",
        frame_size=100,
        sampling_rate=10000,
    )
    result = frame_df_to_annots_df(df, min_label_duration=0.025, min_silence_gap=0.001)

    # consecutive same-label frames must be merged → fewer rows than original
    assert len(result) < len(df)
    assert "label" in result.columns
    # all labels in result come from the original predictions
    assert set(result["label"]).issubset(set(cls_frame_predictions))


def test_frame_to_seconds():
    frames_indices = np.arange(300)
    onsets, offsets = frames_to_seconds(frames_indices, 256, 44100)
    assert len(onsets) == 300
    assert len(offsets) == 300
    assert np.all(np.diff(onsets) >= 0)   # monotone
    assert np.all(offsets >= onsets)       # offset always after onset
