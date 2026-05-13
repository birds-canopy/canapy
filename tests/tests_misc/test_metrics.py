# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
import pandas as pd
import pytest

from canapy import metrics
from canapy.corpus import Corpus
from canapy.metrics.base import _check_corpus_comparison, compute_sklearn_metrics
from canapy.metrics.utils import as_frame_comparison
from canapy.utils.exceptions import MissingData


# ---------------------------------------------------------------------------
# segment_error_rate / sklearn_confusion_matrix / sklearn_classification_report
# ---------------------------------------------------------------------------

def test_segment_error_rate(prediction_corpus):
    ser = metrics.segment_error_rate(prediction_corpus, prediction_corpus)
    assert isinstance(ser, pd.DataFrame)
    assert "ser" in ser.columns
    assert ser["ser"].sum() == 0.0


def test_sklearn_confusion_matrix(prediction_corpus):
    classes = list("abcdefgh")
    cm = metrics.sklearn_confusion_matrix(prediction_corpus, prediction_corpus, classes=classes)
    assert cm.shape == (len(classes), len(classes))


def test_sklearn_classification_report(prediction_corpus):
    report = metrics.sklearn_classification_report(prediction_corpus, prediction_corpus)
    assert isinstance(report, dict)


# ---------------------------------------------------------------------------
# _check_corpus_comparison
# ---------------------------------------------------------------------------

def test_check_corpus_comparison_matching_does_not_raise(prediction_corpus):
    _check_corpus_comparison(prediction_corpus, prediction_corpus)


def test_check_corpus_comparison_mismatched_paths_raises(prediction_corpus, config):
    df = pd.DataFrame({
        "label": ["a"],
        "onset_s": [0.0],
        "offset_s": [0.1],
        "notated_path": ["__other__.wav"],
    })
    other = Corpus.from_df(df, config=config)
    with pytest.raises(ValueError):
        _check_corpus_comparison(other, prediction_corpus)


def test_check_corpus_comparison_accepts_corpus_with_frames_predictions(prediction_corpus):
    _check_corpus_comparison(prediction_corpus, prediction_corpus)


# ---------------------------------------------------------------------------
# as_frame_comparison
# ---------------------------------------------------------------------------

def test_as_frame_comparison_returns_dataframe(prediction_corpus):
    result = as_frame_comparison(prediction_corpus, prediction_corpus)
    assert isinstance(result, pd.DataFrame)


def test_as_frame_comparison_has_expected_columns(prediction_corpus):
    result = as_frame_comparison(prediction_corpus, prediction_corpus)
    for col in ("label", "onset_s", "offset_s", "notated_path"):
        assert col in result.columns


def test_as_frame_comparison_length_equals_pred_frames(prediction_corpus):
    result = as_frame_comparison(prediction_corpus, prediction_corpus)
    expected = len(prediction_corpus.data_resources["frames_predictions"])
    assert len(result) == expected


def test_as_frame_comparison_raises_missing_data_without_frames(config):
    df = pd.DataFrame({
        "label": ["a"],
        "onset_s": [0.0],
        "offset_s": [0.1],
        "notated_path": ["foo.wav"],
    })
    pred = Corpus.from_df(df, config=config)
    gold = Corpus.from_df(df, config=config)
    with pytest.raises(MissingData):
        as_frame_comparison(gold, pred)


# ---------------------------------------------------------------------------
# compute_sklearn_metrics
# ---------------------------------------------------------------------------

def test_compute_sklearn_metrics_returns_cm_and_report(prediction_corpus):
    classes = list("abcdefgh")
    cm, report = compute_sklearn_metrics(prediction_corpus, prediction_corpus, classes=classes)
    assert cm.shape == (len(classes), len(classes))
    assert isinstance(report, dict)


def test_compute_sklearn_metrics_report_has_accuracy(prediction_corpus):
    _, report = compute_sklearn_metrics(prediction_corpus, prediction_corpus)
    assert "accuracy" in report


def test_compute_sklearn_metrics_cm_rows_sum_to_one_or_zero(prediction_corpus):
    classes = list("abcdefgh")
    cm, _ = compute_sklearn_metrics(prediction_corpus, prediction_corpus, classes=classes)
    for row in cm:
        s = row.sum()
        assert s == pytest.approx(1.0, abs=0.01) or s == pytest.approx(0.0, abs=0.01)
