# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Tests for canapy/annotator/base.py — to_disk / from_disk serialization."""

import pickle
from pathlib import Path

import pytest

from canapy.annotator.base import Annotator
from canapy.annotator.synannotator import SynAnnotator
from canapy.annotator.nsynannotator import NSynAnnotator
from config import default_config


class TestToDisk:
    def test_creates_file_on_disk(self, tmp_path):
        ann = SynAnnotator(config=default_config)
        path = str(tmp_path / "model.pkl")
        ann.to_disk(path)
        assert Path(path).exists()

    def test_file_is_valid_pickle(self, tmp_path):
        ann = SynAnnotator(config=default_config)
        path = str(tmp_path / "model.pkl")
        ann.to_disk(path)
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, SynAnnotator)

    def test_nsyn_annotator_serializable(self, tmp_path):
        ann = NSynAnnotator(config=default_config)
        path = str(tmp_path / "nsyn.pkl")
        ann.to_disk(path)
        assert Path(path).exists()


class TestFromDisk:
    def test_loads_syn_annotator(self, tmp_path):
        ann = SynAnnotator(config=default_config)
        path = str(tmp_path / "syn.pkl")
        ann.to_disk(path)
        loaded = Annotator.from_disk(path, config=default_config)
        assert isinstance(loaded, SynAnnotator)

    def test_loads_nsyn_annotator(self, tmp_path):
        ann = NSynAnnotator(config=default_config)
        path = str(tmp_path / "nsyn.pkl")
        ann.to_disk(path)
        loaded = Annotator.from_disk(path, config=default_config)
        assert isinstance(loaded, NSynAnnotator)

    def test_annotator_with_vocab_is_marked_trained(self, tmp_path):
        ann = SynAnnotator(config=default_config)
        ann._vocab = ["SIL", "A", "B"]
        path = str(tmp_path / "syn_vocab.pkl")
        ann.to_disk(path)
        loaded = Annotator.from_disk(path, config=default_config)
        assert loaded.trained is True

    def test_vocab_preserved_after_roundtrip(self, tmp_path):
        ann = SynAnnotator(config=default_config)
        ann._vocab = ["SIL", "A", "B", "C"]
        path = str(tmp_path / "syn_vocab.pkl")
        ann.to_disk(path)
        loaded = Annotator.from_disk(path, config=default_config)
        assert loaded.vocab == ["SIL", "A", "B", "C"]

    def test_invalid_pickle_raises_not_implemented(self, tmp_path):
        path = str(tmp_path / "bad.pkl")
        with open(path, "wb") as f:
            pickle.dump({"not": "an annotator"}, f)
        with pytest.raises(NotImplementedError):
            Annotator.from_disk(path)

    def test_roundtrip_preserves_config_seed(self, tmp_path):
        ann = SynAnnotator(config=default_config)
        path = str(tmp_path / "syn.pkl")
        ann.to_disk(path)
        loaded = Annotator.from_disk(path, config=default_config)
        assert loaded.config.misc.seed == default_config.misc.seed
