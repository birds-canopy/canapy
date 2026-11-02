# Author: Nathan Trouvain at 06/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: BSD-3-Clause
# Copyright: Nathan Trouvain
import numpy as np
import pytest

from canapy.timings import frames_to_timed_df, frames_to_seconds
from canapy.annotator.commons.postprocess import (
    frame_labels_to_annots,
    frames_to_annots,
    predictions_to_corpus,
)

HOP = 0.01
SIL = "SIL"


def frames(*parts):
    """("A", 3), ("SIL", 10)... to one label per frame."""
    return np.array([label for label, n in parts for _ in range(n)], dtype=object)


def segments(*parts, min_label_duration=0.03, min_silence_duration=0.02, **kwargs):
    """Post-processed segments, as (label, first frame, exclusive last frame).

    Defaults to a 3 frame minimum for labels, 2 for silences. A segment ending
    on the very last frame of the file would read one frame short, since the
    last offset is not padded: end the sequences on a silence.
    """
    kwargs = {"min_silence_gap": 0.001, "lonely_labels": ["TRASH"], **kwargs}
    annots = frame_labels_to_annots(
        frames(*parts),
        notated_path="foo/baz",
        hop_length=HOP,
        sampling_rate=1000,
        time_precision=0.001,
        min_label_duration=min_label_duration,
        min_silence_duration=min_silence_duration,
        silence_tag=SIL,
        **kwargs,
    )
    return [
        (row.label,
         int(np.floor((row.onset_s + HOP / 2) / HOP + 1e-6)),
         int(np.floor((row.offset_s + HOP / 2) / HOP + 1e-6)))
        for row in annots.itertuples()
    ]


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


def test_frame_to_seconds():
    frames_indices = np.arange(300)
    onsets, offsets = frames_to_seconds(frames_indices, 256, 44100)
    assert len(onsets) == 300
    assert len(offsets) == 300
    assert np.all(np.diff(onsets) >= 0)   # monotone
    assert np.all(offsets >= onsets)       # offset always after onset


def test_annotations_come_from_the_predicted_labels(cls_frame_predictions):
    annots = frame_labels_to_annots(
        cls_frame_predictions,
        notated_path="foo/baz",
        hop_length=0.01,
        sampling_rate=10000,
        time_precision=0.001,
        min_label_duration=0.025,
        min_silence_duration=0.025,
        min_silence_gap=0.001,
    )

    assert {"label", "onset_s", "offset_s", "notated_path"}.issubset(annots.columns)
    assert len(annots) < len(cls_frame_predictions)
    assert set(annots["label"]).issubset(set(cls_frame_predictions))
    assert (annots["offset_s"] > annots["onset_s"]).all()


@pytest.mark.parametrize("sequence", [[], [(SIL, 50)]])
def test_nothing_to_annotate_gives_an_empty_result(sequence):
    assert segments(*sequence) == []


def test_silences_are_left_out():
    assert segments((SIL, 20), ("A", 10), (SIL, 20)) == [("A", 20, 30)]


# -- short runs ----------------------------------------------------------------
def test_a_stray_frame_does_not_rename_a_long_run():
    result = segments((SIL, 20), ("A", 10), ("B", 1), ("C", 1), ("B", 1), (SIL, 20))
    assert [label for label, _, _ in result] == ["A"]
    assert result[0][1] == 20


def test_a_short_run_between_two_labels_is_shared_out():
    result = segments((SIL, 20), ("A", 10), ("C", 2), ("B", 10), (SIL, 20))
    assert [label for label, _, _ in result] == ["A", "B"]
    assert result[0][2] == result[1][1]


def test_a_zone_filled_by_one_label_becomes_a_segment():
    result = segments(
        (SIL, 20), ("B", 2), ("C", 1), ("B", 2), ("C", 1), ("B", 2), (SIL, 20)
    )
    assert result == [("B", 20, 28)]


def test_zone_factor_sets_when_a_zone_becomes_a_segment():
    sequence = ((SIL, 20), ("B", 2), ("C", 1), ("B", 2), ("C", 1), ("B", 2), (SIL, 20))
    assert segments(*sequence, zone_factor=3.0) == []


# -- silences ------------------------------------------------------------------
def test_a_silence_below_its_own_threshold_is_absorbed():
    sequence = ((SIL, 20), ("A", 10), (SIL, 1), ("A", 10), (SIL, 20))
    assert segments(*sequence) == [("A", 20, 41)]


def test_a_silence_at_its_threshold_is_kept():
    sequence = ((SIL, 20), ("A", 10), (SIL, 2), ("A", 10), (SIL, 20))
    assert segments(*sequence) == [("A", 20, 30), ("A", 32, 42)]


@pytest.mark.parametrize(
    "min_label_duration, min_silence_duration, expected_rows",
    [(0.02, 0.02, 1), (0.004, 0.004, 3), (0.004, 0.02, 1)],
)
def test_the_thresholds_decide_the_annotation_level(
    min_label_duration, min_silence_duration, expected_rows
):
    # three 40 ms syllables of the same label, 5 ms apart, with a 1 ms frame:
    # what keeps them apart is the threshold on silences
    parts = (("a", 40), (SIL, 5), ("a", 40), (SIL, 5), ("a", 40), (SIL, 1))
    annots = frame_labels_to_annots(
        frames(*parts),
        notated_path="foo/baz",
        hop_length=0.001,
        sampling_rate=10000,
        time_precision=0.001,
        min_label_duration=min_label_duration,
        min_silence_duration=min_silence_duration,
        min_silence_gap=0.001,
        silence_tag=SIL,
    )

    assert len(annots) == expected_rows
    assert set(annots["label"]) == {"a"}


def test_a_large_gap_is_not_merged():
    sequence = (("A", 10), (SIL, 50), ("A", 10), (SIL, 5))
    assert segments(*sequence) == [("A", 0, 10), ("A", 60, 70)]


def test_a_gap_within_min_silence_gap_is_merged():
    sequence = ((SIL, 20), ("A", 10), (SIL, 3), ("A", 10), (SIL, 20))
    assert segments(*sequence, min_silence_gap=0.03) == [("A", 20, 43)]


def test_lonely_labels_are_not_merged():
    sequence = ((SIL, 20), ("TRASH", 10), (SIL, 3), ("TRASH", 10), (SIL, 20))
    assert segments(*sequence, min_silence_gap=0.03) == [
        ("TRASH", 20, 30), ("TRASH", 33, 43),
    ]


# -- start of the file ---------------------------------------------------------
def test_a_stray_frame_at_the_start_does_not_cover_the_silence():
    sequence = (("A", 1), (SIL, 200), ("B", 20), (SIL, 10))
    assert segments(*sequence) == [("B", 201, 221)]


def test_a_short_segment_at_the_start_is_silenced():
    sequence = (("A", 5), (SIL, 50), ("B", 20), (SIL, 10))
    assert segments(*sequence) == [("B", 55, 75)]


def test_head_factor_zero_keeps_it():
    sequence = (("A", 5), (SIL, 50), ("B", 20), (SIL, 10))
    assert segments(*sequence, head_factor=0.0) == [("A", 0, 5), ("B", 55, 75)]


def test_a_long_segment_at_the_start_is_kept():
    sequence = (("A", 20), (SIL, 50), ("B", 20), (SIL, 10))
    assert segments(*sequence) == [("A", 0, 20), ("B", 70, 90)]


def test_a_short_segment_further_in_is_kept():
    sequence = ((SIL, 20), ("A", 5), (SIL, 50))
    assert segments(*sequence) == [("A", 20, 25)]


# -- config --------------------------------------------------------------------
def _annots_config(config, **values):
    config.data["transforms"]["annots"].update(values)
    config.data["transforms"]["audio"].update(hop_length=HOP, sampling_rate=1000)
    return config


SHORT_SILENCE = frames((SIL, 20), ("A", 10), (SIL, 2), ("A", 10), (SIL, 20))


def test_the_silence_threshold_falls_back_on_min_label_duration(config):
    config = _annots_config(config, min_label_duration=0.03)
    config.data["transforms"]["annots"].pop("min_silence_duration", None)
    # 2 frames of silence are short against a 3 frame minimum
    assert len(frames_to_annots(SHORT_SILENCE, "foo/baz", config)) == 1


def test_the_config_can_set_the_silence_threshold_on_its_own(config):
    config = _annots_config(config, min_label_duration=0.03, min_silence_duration=0.02)
    assert len(frames_to_annots(SHORT_SILENCE, "foo/baz", config)) == 2


def test_the_config_can_disable_the_head_rule(config):
    head = frames(("A", 5), (SIL, 50), ("B", 20), (SIL, 10))
    config = _annots_config(config, min_label_duration=0.03, head_factor=0.0)
    assert frames_to_annots(head, "foo/baz", config)["label"].tolist() == ["A", "B"]


def test_a_corpus_is_built_even_with_a_silent_file(config):
    config = _annots_config(config, min_label_duration=0.03)
    corpus = predictions_to_corpus(
        notated_paths=["silent.wav", "song.wav"],
        cls_preds=[frames((SIL, 50)), SHORT_SILENCE],
        frame_size=10,
        sampling_rate=1000,
        time_precision=0.001,
        min_label_duration=0.03,
        min_silence_gap=0.001,
        silence_tag=SIL,
        lonely_labels=[],
        config=config,
    )

    assert set(corpus.dataset["notated_path"]) == {"song.wav"}
    assert len(corpus.data_resources["frames_predictions"]) == 50 + len(SHORT_SILENCE)
