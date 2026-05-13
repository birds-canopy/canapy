# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Tests for canapy/transforms/commons/training.py — split_train_test, encode_labels, DatasetTransform."""

import numpy as np
import pandas as pd
import pytest
import toml

from canapy.corpus import Corpus
from canapy.transforms.commons.training import (
    split_train_test,
    encode_labels,
    DatasetTransform,
)
from config import Config


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


def _make_config(**overrides):
    cfg = Config(**toml.loads(_CFG_STR))
    for key, val in overrides.items():
        setattr(cfg.transforms.training, key, val)
    return cfg


def _make_corpus(n_per_class=5, labels=None, config=None):
    if labels is None:
        labels = list("ABCDE")
    if config is None:
        config = _make_config()

    rows = []
    for i, lbl in enumerate(labels * n_per_class):
        # Each (label, index) gets its own annotation file to create distinct seqids
        rows.append({
            "label": lbl,
            "onset_s": 0.0,
            "offset_s": 0.1,
            "notated_path": f"file_{i:04d}.wav",
            "annot_path": f"file_{i:04d}.csv",
            "sequence": 0,
            "annotation": f"file_{i:04d}.wav",
        })

    df = pd.DataFrame(rows)
    return Corpus.from_df(df, config=config)


class TestSplitTrainTest:
    def test_adds_train_column(self):
        corpus = _make_corpus(n_per_class=6)
        result = split_train_test(corpus)
        assert "train" in result.dataset.columns

    def test_train_column_is_bool(self):
        corpus = _make_corpus(n_per_class=6)
        result = split_train_test(corpus)
        assert result.dataset["train"].dtype == bool

    def test_all_classes_present_in_train(self):
        labels = list("ABCDE")
        corpus = _make_corpus(n_per_class=6, labels=labels)
        result = split_train_test(corpus)
        train_labels = set(result.dataset.loc[result.dataset["train"], "label"].unique())
        for lbl in labels:
            assert lbl in train_labels, f"Class {lbl!r} missing from train split"

    def test_test_set_is_nonempty(self):
        corpus = _make_corpus(n_per_class=10)
        result = split_train_test(corpus)
        assert (~result.dataset["train"]).sum() > 0

    def test_already_split_dataset_not_redone(self):
        corpus = _make_corpus(n_per_class=6)
        once = split_train_test(corpus)
        first_split = once.dataset["train"].copy()
        twice = split_train_test(once)
        assert twice.dataset["train"].equals(first_split)

    def test_redo_flag_forces_new_split(self):
        corpus = _make_corpus(n_per_class=10)
        once = split_train_test(corpus)
        # With redo=True the column should be rewritten (no error)
        twice = split_train_test(once, redo=True)
        assert "train" in twice.dataset.columns

    def test_invalid_max_sequences_raises_value_error(self):
        import toml as _toml
        # Build config with max_sequences > corpus size from the start
        cfg_data = _toml.loads(_CFG_STR)
        cfg_data["transforms"]["training"]["max_sequences"] = 9999
        cfg = Config(**cfg_data)
        corpus = _make_corpus(n_per_class=1, labels=list("AB"), config=cfg)
        # 2 sequences total but max_sequences=9999 → should raise ValueError
        with pytest.raises(ValueError):
            split_train_test(corpus)


class TestEncodeLabels:
    def _split_corpus(self, n_per_class=4, labels=None):
        corpus = _make_corpus(n_per_class=n_per_class, labels=labels)
        corpus.dataset["train"] = True
        return corpus

    def test_adds_encoded_label_column(self):
        corpus = self._split_corpus()
        result = encode_labels(corpus, resource_name=None)
        assert "encoded_label" in result.dataset.columns

    def test_one_hot_length_equals_n_classes(self):
        labels = list("ABC")
        corpus = self._split_corpus(labels=labels)
        result = encode_labels(corpus, resource_name=None)
        for enc in result.dataset["encoded_label"]:
            assert len(enc) == len(labels)

    def test_one_hot_sums_to_one(self):
        corpus = self._split_corpus()
        result = encode_labels(corpus, resource_name=None)
        for enc in result.dataset["encoded_label"]:
            assert abs(enc.sum() - 1.0) < 1e-6

    def test_one_hot_binary_values(self):
        corpus = self._split_corpus()
        result = encode_labels(corpus, resource_name=None)
        for enc in result.dataset["encoded_label"]:
            assert set(enc.tolist()).issubset({0.0, 1.0})

    def test_different_labels_get_different_encodings(self):
        labels = list("AB")
        corpus = self._split_corpus(n_per_class=3, labels=labels)
        result = encode_labels(corpus, resource_name=None)
        enc_by_label = {
            lbl: result.dataset.loc[result.dataset["label"] == lbl, "encoded_label"].iloc[0]
            for lbl in labels
        }
        assert not np.array_equal(enc_by_label["A"], enc_by_label["B"])


class TestDatasetTransform:
    def test_instantiates_without_error(self):
        t = DatasetTransform()
        assert t is not None

    def test_has_annots_transforms(self):
        t = DatasetTransform()
        assert len(t.annots_transforms) > 0

    def test_has_training_data_transforms(self):
        t = DatasetTransform()
        assert len(t.training_data_transforms) > 0

    def test_split_train_test_in_training_transforms(self):
        t = DatasetTransform()
        assert split_train_test in t.training_data_transforms

    def test_encode_labels_in_training_transforms(self):
        t = DatasetTransform()
        assert encode_labels in t.training_data_transforms
