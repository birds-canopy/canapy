# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from canapy.corpus import Corpus
from dashboard.controler.segments import fetch_misclassified_samples

# ---------------------------------------------------------------------------
# Constants
# All tests use hop_length=0.01s, sampling_rate=44100Hz → frame_size=441.
# Annotation "a": 0.0s–0.2s → onset_frame=1, offset_frame=21 (21 frames).
# Annotation "b": 0.3s–0.5s → onset_frame=31, offset_frame=51 (21 frames).
# ---------------------------------------------------------------------------

HOP_LENGTH = 0.01
SAMPLING_RATE = 44100
THRESHOLD = 0.5
NOTATED_PATH = "foo.wav"
N_FRAMES = 60


def _make_gold_frames(annot_frames, notated_path=NOTATED_PATH, n=N_FRAMES):
    """Build the DataFrame that as_frame_comparison would return.

    annot_frames: dict mapping frame index → label, e.g. {1: "a", ..., 21: "a"}
    Everything else gets "SIL".
    """
    labels = ["SIL"] * n
    for idx, lbl in annot_frames.items():
        labels[idx] = lbl
    return pd.DataFrame(
        {
            "label": labels,
            "onset_s": [i * HOP_LENGTH for i in range(n)],
            "offset_s": [(i + 1) * HOP_LENGTH for i in range(n)],
            "notated_path": [notated_path] * n,
        }
    )  # index = 0..n-1, representing frame numbers


def _make_pred_corpus(pred_labels, gold_df, notated_path=NOTATED_PATH):
    """Build a prediction Corpus with a frames_predictions data resource."""
    n = len(pred_labels)
    frames_preds = pd.DataFrame(
        {
            "label": pred_labels,
            "onset_s": [i * HOP_LENGTH for i in range(n)],
            "notated_path": [notated_path] * n,
        }
    )  # index = 0..n-1, aligned with gold_frames
    corpus = Corpus.from_df(gold_df)
    corpus.register_data_resource("frames_predictions", frames_preds)
    return corpus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def gold_df_one_annot():
    return pd.DataFrame(
        {
            "label": ["a"],
            "onset_s": [0.0],
            "offset_s": [0.2],
            "notated_path": [NOTATED_PATH],
        }
    )


@pytest.fixture()
def gold_df_two_annots():
    return pd.DataFrame(
        {
            "label": ["a", "b"],
            "onset_s": [0.0, 0.3],
            "offset_s": [0.2, 0.5],
            "notated_path": [NOTATED_PATH, NOTATED_PATH],
        }
    )


@pytest.fixture()
def gold_df_with_silence():
    return pd.DataFrame(
        {
            "label": ["SIL", "a"],
            "onset_s": [0.0, 0.3],
            "offset_s": [0.2, 0.5],
            "notated_path": [NOTATED_PATH, NOTATED_PATH],
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(gold_corpus, predictions, gold_frames):
    with patch(
        "dashboard.controler.segments.as_frame_comparison",
        return_value=gold_frames,
    ):
        return fetch_misclassified_samples(
            gold_corpus,
            predictions,
            hop_length=HOP_LENGTH,
            sampling_rate=SAMPLING_RATE,
            min_segment_proportion_agreement=THRESHOLD,
        )


# ---------------------------------------------------------------------------
# Correctly classified segment — should not appear in result
# ---------------------------------------------------------------------------

def test_correct_segment_not_in_result(gold_df_one_annot):
    gold_corpus = Corpus.from_df(gold_df_one_annot)
    # frames 1..21 are "a", model predicts "a" everywhere
    gold_frames = _make_gold_frames({i: "a" for i in range(1, 22)})
    pred_labels = ["a"] * N_FRAMES
    predictions = {"model1": _make_pred_corpus(pred_labels, gold_df_one_annot)}

    result = _run(gold_corpus, predictions, gold_frames)

    assert result.empty


def test_correct_segment_returns_empty_dataframe(gold_df_one_annot):
    gold_corpus = Corpus.from_df(gold_df_one_annot)
    gold_frames = _make_gold_frames({i: "a" for i in range(1, 22)})
    pred_labels = ["a"] * N_FRAMES
    predictions = {"model1": _make_pred_corpus(pred_labels, gold_df_one_annot)}

    result = _run(gold_corpus, predictions, gold_frames)

    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Misclassified segment — should appear in result
# ---------------------------------------------------------------------------

def test_misclassified_segment_in_result(gold_df_one_annot):
    gold_corpus = Corpus.from_df(gold_df_one_annot)
    gold_frames = _make_gold_frames({i: "a" for i in range(1, 22)})
    # model always predicts "b" → score = 0.0 < threshold
    pred_labels = ["b"] * N_FRAMES
    predictions = {"model1": _make_pred_corpus(pred_labels, gold_df_one_annot)}

    result = _run(gold_corpus, predictions, gold_frames)

    assert len(result) == 1


def test_misclassified_result_contains_gold_columns(gold_df_one_annot):
    gold_corpus = Corpus.from_df(gold_df_one_annot)
    gold_frames = _make_gold_frames({i: "a" for i in range(1, 22)})
    pred_labels = ["b"] * N_FRAMES
    predictions = {"model1": _make_pred_corpus(pred_labels, gold_df_one_annot)}

    result = _run(gold_corpus, predictions, gold_frames)

    for col in ("label", "onset_s", "offset_s", "notated_path"):
        assert col in result.columns


def test_misclassified_result_contains_model_column(gold_df_one_annot):
    # The function prefixes model names with "pred_" in the output columns.
    gold_corpus = Corpus.from_df(gold_df_one_annot)
    gold_frames = _make_gold_frames({i: "a" for i in range(1, 22)})
    pred_labels = ["b"] * N_FRAMES
    predictions = {"model1": _make_pred_corpus(pred_labels, gold_df_one_annot)}

    result = _run(gold_corpus, predictions, gold_frames)

    assert "pred_model1" in result.columns


def test_misclassified_result_stores_predicted_label(gold_df_one_annot):
    gold_corpus = Corpus.from_df(gold_df_one_annot)
    gold_frames = _make_gold_frames({i: "a" for i in range(1, 22)})
    pred_labels = ["b"] * N_FRAMES
    predictions = {"model1": _make_pred_corpus(pred_labels, gold_df_one_annot)}

    result = _run(gold_corpus, predictions, gold_frames)

    assert result["pred_model1"].iloc[0] == "b"


# ---------------------------------------------------------------------------
# Multi-model: only bad if ALL models fail
# ---------------------------------------------------------------------------

def test_one_model_passes_segment_not_in_result(gold_df_one_annot):
    gold_corpus = Corpus.from_df(gold_df_one_annot)
    gold_frames = _make_gold_frames({i: "a" for i in range(1, 22)})
    predictions = {
        "model1": _make_pred_corpus(["b"] * N_FRAMES, gold_df_one_annot),  # fails
        "model2": _make_pred_corpus(["a"] * N_FRAMES, gold_df_one_annot),  # passes
    }

    result = _run(gold_corpus, predictions, gold_frames)

    assert result.empty


def test_all_models_fail_segment_in_result(gold_df_one_annot):
    gold_corpus = Corpus.from_df(gold_df_one_annot)
    gold_frames = _make_gold_frames({i: "a" for i in range(1, 22)})
    predictions = {
        "model1": _make_pred_corpus(["b"] * N_FRAMES, gold_df_one_annot),
        "model2": _make_pred_corpus(["c"] * N_FRAMES, gold_df_one_annot),
    }

    result = _run(gold_corpus, predictions, gold_frames)

    assert len(result) == 1


def test_multi_model_result_has_one_column_per_model(gold_df_one_annot):
    gold_corpus = Corpus.from_df(gold_df_one_annot)
    gold_frames = _make_gold_frames({i: "a" for i in range(1, 22)})
    predictions = {
        "model1": _make_pred_corpus(["b"] * N_FRAMES, gold_df_one_annot),
        "model2": _make_pred_corpus(["c"] * N_FRAMES, gold_df_one_annot),
    }

    result = _run(gold_corpus, predictions, gold_frames)

    assert "pred_model1" in result.columns
    assert "pred_model2" in result.columns


# ---------------------------------------------------------------------------
# Multiple annotations: only misclassified ones appear
# ---------------------------------------------------------------------------

def test_only_misclassified_annot_in_result(gold_df_two_annots):
    gold_corpus = Corpus.from_df(gold_df_two_annots)
    # "a" at frames 1..21, "b" at frames 31..51
    gold_frames = _make_gold_frames(
        {**{i: "a" for i in range(1, 22)}, **{i: "b" for i in range(31, 52)}}
    )
    # model predicts "a" everywhere: correct for "a", wrong for "b"
    pred_labels = ["a"] * N_FRAMES
    predictions = {"model1": _make_pred_corpus(pred_labels, gold_df_two_annots)}

    result = _run(gold_corpus, predictions, gold_frames)

    assert len(result) == 1
    assert result["label"].iloc[0] == "b"


# ---------------------------------------------------------------------------
# Silence tag — silences must be skipped
# ---------------------------------------------------------------------------

def test_silence_segment_not_in_result(gold_df_with_silence):
    gold_corpus = Corpus.from_df(gold_df_with_silence)
    # "SIL" at frames 1..21, "a" at frames 31..51
    gold_frames = _make_gold_frames(
        {**{i: "SIL" for i in range(1, 22)}, **{i: "a" for i in range(31, 52)}}
    )
    # model predicts "b" everywhere → only the non-silence "a" annotation is checked
    pred_labels = ["b"] * N_FRAMES
    predictions = {"model1": _make_pred_corpus(pred_labels, gold_df_with_silence)}

    result = _run(gold_corpus, predictions, gold_frames)

    # "SIL" annotation skipped; "a" is misclassified → only 1 row
    assert len(result) == 1
    assert result["label"].iloc[0] == "a"


def test_custom_silence_tag_skipped(gold_df_one_annot):
    gold_df = pd.DataFrame(
        {
            "label": ["NOISE"],
            "onset_s": [0.0],
            "offset_s": [0.2],
            "notated_path": [NOTATED_PATH],
        }
    )
    gold_corpus = Corpus.from_df(gold_df)
    gold_frames = _make_gold_frames({i: "NOISE" for i in range(1, 22)})
    pred_labels = ["b"] * N_FRAMES
    predictions = {"model1": _make_pred_corpus(pred_labels, gold_df)}

    with patch(
        "dashboard.controler.segments.as_frame_comparison",
        return_value=gold_frames,
    ):
        result = fetch_misclassified_samples(
            gold_corpus,
            predictions,
            hop_length=HOP_LENGTH,
            sampling_rate=SAMPLING_RATE,
            min_segment_proportion_agreement=THRESHOLD,
            silence_tag="NOISE",
        )

    assert result.empty


# ---------------------------------------------------------------------------
# File absent from gold_frames — must be skipped gracefully
# ---------------------------------------------------------------------------

def test_missing_file_in_gold_frames_skipped():
    gold_df = pd.DataFrame(
        {
            "label": ["a"],
            "onset_s": [0.0],
            "offset_s": [0.2],
            "notated_path": ["bar.wav"],  # different file
        }
    )
    gold_corpus = Corpus.from_df(gold_df)
    # gold_frames only contains "foo.wav" → "bar.wav" absent from frames_by_file
    gold_frames = _make_gold_frames({i: "a" for i in range(1, 22)}, notated_path="foo.wav")
    pred_labels = ["b"] * N_FRAMES
    predictions = {"model1": _make_pred_corpus(pred_labels, gold_df, notated_path="bar.wav")}

    result = _run(gold_corpus, predictions, gold_frames)

    assert result.empty


# ---------------------------------------------------------------------------
# Threshold boundary
# ---------------------------------------------------------------------------

def test_score_exactly_at_threshold_not_misclassified(gold_df_one_annot):
    """Score == threshold means the condition `< threshold` is False → not bad."""
    gold_corpus = Corpus.from_df(gold_df_one_annot)
    gold_frames = _make_gold_frames({i: "a" for i in range(1, 22)})
    # 21 frames; predict "a" for exactly 50% of them → score = 0.5 (not < 0.5)
    half = N_FRAMES // 2
    pred_labels = ["a"] * half + ["b"] * (N_FRAMES - half)
    predictions = {"model1": _make_pred_corpus(pred_labels, gold_df_one_annot)}

    result = _run(gold_corpus, predictions, gold_frames)

    assert result.empty


def test_score_just_below_threshold_is_misclassified(gold_df_one_annot):
    """Score strictly below threshold → segment flagged."""
    gold_corpus = Corpus.from_df(gold_df_one_annot)
    # frames 1..21 → 21 frames. Predict "a" for 10 out of 21 → score ≈ 0.476 < 0.5
    gold_frames = _make_gold_frames({i: "a" for i in range(1, 22)})
    pred_labels = ["b"] * N_FRAMES
    # Override frames 1..10 to "a" (10 correct out of 21)
    pred_labels[1:11] = ["a"] * 10
    predictions = {"model1": _make_pred_corpus(pred_labels, gold_df_one_annot)}

    result = _run(gold_corpus, predictions, gold_frames)

    assert len(result) == 1
