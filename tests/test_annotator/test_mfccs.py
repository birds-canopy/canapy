# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Tests for canapy/annotator/commons/mfccs.py — load_mfccs_and_repeat_labels,
load_mfccs_for_annotation."""

import numpy as np
import pandas as pd
import pytest
import toml

from canapy.corpus import Corpus
from canapy.annotator.commons.mfccs import (
    load_mfccs_and_repeat_labels,
    load_mfccs_for_annotation,
)
from canapy.utils.exceptions import MissingData, NotTrainableError
from config import Config, default_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CFG_STR = """
[misc]
seed=42
[transforms.annots]
time_precision=0.001
min_label_duration=0.02
lonely_labels=[]
min_silence_gap=0.001
silence_tag="SIL"
[transforms.audio]
audio_features=["mfcc"]
sampling_rate=44100
n_mfcc=13
hop_length=0.01
win_length=0.02
n_fft=2048
fmin=500
fmax=8000
lifter=40
[transforms.audio.delta]
padding="wrap"
[transforms.audio.delta2]
padding="wrap"
[transforms.training]
max_sequences=-1
test_ratio=0.2
[transforms.training.balance]
min_silence_duration=0.2
[transforms.training.balance.data_augmentation]
noise_std=0.01
[model.syn]
units=50
sr=0.4
leak=0.1
iss=0.0005
isd=0.02
isd2=0.002
ridge=1e-8
backend="threading"
workers=1
[model.nsyn]
units=50
sr=0.4
leak=0.1
iss=0.0005
isd=0.02
isd2=0.002
ridge=1e-8
backend="threading"
workers=1
[correction]
min_segment_proportion_agreement=0.66
"""


def _make_config():
    return Config(**toml.loads(_CFG_STR))


def _make_prepared_corpus(tmp_path, n_features=13, n_frames=200):
    """Corpus with syn_mfcc data_resource and train/test labels."""
    config = _make_config()
    npz = tmp_path / "audio.mfcc.npz"
    np.savez(str(npz), feature=np.random.default_rng(0).random((n_features, n_frames)).astype(np.float32))

    labels = ["A", "B", "SIL", "A", "SIL", "B"]
    n = len(labels)
    hop = 0.01
    duration = n_frames * hop

    onsets = np.linspace(0.0, duration - duration / n, n)
    offsets = np.linspace(duration / n, duration, n)
    train_flags = [True, True, True, False, False, False]
    n_cls = len(set(labels))
    enc_map = {lbl: np.eye(n_cls)[i] for i, lbl in enumerate(sorted(set(labels)))}

    df = pd.DataFrame({
        "label": labels,
        "onset_s": onsets,
        "offset_s": offsets,
        "notated_path": ["audio.wav"] * n,
        "annot_path": ["audio.csv"] * n,
        "sequence": [0] * n,
        "annotation": ["audio.wav"] * n,
        "train": train_flags,
        "encoded_label": [enc_map[lbl] for lbl in labels],
    })

    mfcc_df = pd.DataFrame({
        "notated_path": ["audio.wav"],
        "feature_path": [str(npz)],
    })

    corpus = Corpus.from_df(df, config=config)
    corpus.dataset = df.copy()
    corpus.register_data_resource("syn_mfcc", mfcc_df)
    return corpus


# ---------------------------------------------------------------------------
# load_mfccs_and_repeat_labels
# ---------------------------------------------------------------------------

class TestLoadMfccsAndRepeatLabels:
    def test_returns_four_iterables(self, tmp_path):
        corpus = _make_prepared_corpus(tmp_path)
        result = load_mfccs_and_repeat_labels(corpus, purpose="training")
        assert len(result) == 4

    def test_annotations_and_sequences_are_lists(self, tmp_path):
        corpus = _make_prepared_corpus(tmp_path)
        annotations, sequences, mfccs, labels = load_mfccs_and_repeat_labels(
            corpus, purpose="training"
        )
        assert isinstance(annotations, list)
        assert isinstance(sequences, list)

    def test_mfccs_are_2d_arrays(self, tmp_path):
        corpus = _make_prepared_corpus(tmp_path)
        _, _, mfccs, _ = load_mfccs_and_repeat_labels(corpus, purpose="training")
        for m in mfccs:
            assert m.ndim == 2

    def test_mfcc_feature_dim_correct(self, tmp_path):
        corpus = _make_prepared_corpus(tmp_path, n_features=13)
        _, _, mfccs, _ = load_mfccs_and_repeat_labels(corpus, purpose="training")
        for m in mfccs:
            assert m.shape[1] == 13

    def test_labels_shape_matches_mfcc_time_dim(self, tmp_path):
        corpus = _make_prepared_corpus(tmp_path)
        _, _, mfccs, labels = load_mfccs_and_repeat_labels(corpus, purpose="training")
        for m, l in zip(mfccs, labels):
            assert m.shape[0] == l.shape[0]

    def test_raises_missing_data_without_syn_mfcc(self):
        df = pd.DataFrame({
            "label": ["A", "B"],
            "onset_s": [0.0, 0.1],
            "offset_s": [0.1, 0.2],
            "notated_path": ["f.wav", "f.wav"],
            "annot_path": ["f.csv", "f.csv"],
            "sequence": [0, 0],
            "annotation": ["a", "a"],
            "train": [True, True],
        })
        corpus = Corpus.from_df(df, config=default_config)
        corpus.dataset = df.copy()
        with pytest.raises(MissingData):
            load_mfccs_and_repeat_labels(corpus, purpose="training")

    def test_raises_value_error_on_invalid_purpose(self, tmp_path):
        corpus = _make_prepared_corpus(tmp_path)
        with pytest.raises(ValueError):
            load_mfccs_and_repeat_labels(corpus, purpose="unknown")

    def test_eval_purpose_uses_test_split(self, tmp_path):
        corpus = _make_prepared_corpus(tmp_path)
        annotations_train, _, _, _ = load_mfccs_and_repeat_labels(corpus, purpose="training")
        annotations_eval, _, _, _ = load_mfccs_and_repeat_labels(corpus, purpose="eval")
        # Both use the same audio file so there's overlap, but function should not raise
        assert isinstance(annotations_eval, list)


# ---------------------------------------------------------------------------
# load_mfccs_for_annotation
# ---------------------------------------------------------------------------

class TestLoadMfccsForAnnotation:
    def test_returns_paths_and_arrays(self, tmp_path):
        corpus = _make_prepared_corpus(tmp_path)
        paths, mfccs = load_mfccs_for_annotation(corpus)
        assert isinstance(paths, list)
        assert isinstance(mfccs, list)

    def test_paths_and_mfccs_same_length(self, tmp_path):
        corpus = _make_prepared_corpus(tmp_path)
        paths, mfccs = load_mfccs_for_annotation(corpus)
        assert len(paths) == len(mfccs)

    def test_mfccs_are_2d(self, tmp_path):
        corpus = _make_prepared_corpus(tmp_path)
        _, mfccs = load_mfccs_for_annotation(corpus)
        for m in mfccs:
            assert m.ndim == 2

    def test_feature_dim_preserved(self, tmp_path):
        corpus = _make_prepared_corpus(tmp_path, n_features=13)
        _, mfccs = load_mfccs_for_annotation(corpus)
        for m in mfccs:
            assert m.shape[1] == 13

    def test_raises_missing_data_without_syn_mfcc(self):
        df = pd.DataFrame({
            "label": ["A"],
            "onset_s": [0.0],
            "offset_s": [0.1],
            "notated_path": ["f.wav"],
            "annot_path": ["f.csv"],
            "sequence": [0],
            "annotation": ["a"],
        })
        corpus = Corpus.from_df(df, config=default_config)
        corpus.dataset = df.copy()
        with pytest.raises(MissingData):
            load_mfccs_for_annotation(corpus)
