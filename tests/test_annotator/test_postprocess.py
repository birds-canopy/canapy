# Author: Nathan Trouvain at 06/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: BSD-3-Clause
# Copyright: Nathan Trouvain
import numpy as np
import pandas as pd
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


@pytest.mark.parametrize(
    "min_label_duration, expected_rows", [(0.02, 1), (0.004, 3)]
)
def test_min_label_duration_decides_the_annotation_level(
    min_label_duration, expected_rows
):
    # three 40 ms syllables of the same label, 5 ms apart: silences shorter
    # than min_label_duration get absorbed and the syllables glue back into
    # one phrase
    rows, cursor = [], 0.0
    for i in range(3):
        for label, duration in (("a", 0.040), ("SIL", 0.005))[: 1 if i == 2 else 2]:
            for _ in range(int(duration * 1000)):
                rows.append({"label": label, "onset_s": cursor,
                             "offset_s": cursor + 0.001, "notated_path": "foo/baz"})
                cursor += 0.001

    result = frame_df_to_annots_df(
        pd.DataFrame(rows),
        min_label_duration=min_label_duration,
        min_silence_gap=0.001,
        silence_tag="SIL",
    )

    assert len(result) == expected_rows
    assert set(result["label"]) == {"a"}


def test_frame_to_seconds():
    frames_indices = np.arange(300)
    onsets, offsets = frames_to_seconds(frames_indices, 256, 44100)
    assert len(onsets) == 300
    assert len(offsets) == 300
    assert np.all(np.diff(onsets) >= 0)   # monotone
    assert np.all(offsets >= onsets)       # offset always after onset
