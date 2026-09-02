# Licence: BSD-3-Clause
"""Annotating an external dataset one model at a time, as the Annotate page does."""
import types

import pytest

import dashboard.controler.base as base
from dashboard.controler.base import Controler

VOCAB = ["SIL", "A", "B"]


class FakeAnnotator:
    """A sub-annotator that predicts a marker and knows its vocabulary."""

    def __init__(self, vocab=VOCAB):
        self._vocab = list(vocab)
        self.trained = True

    @property
    def vocab(self):
        return self._vocab

    def predict(self, corpus, return_raw=False):
        return f"raw:{id(self)}"


class FakeEnsemble:
    """Records what the controler hands it before asking for a prediction."""

    last = None

    def __init__(self, config, *args, **kwargs):
        self.config = config
        self._vocab = []
        self._trained = False
        FakeEnsemble.last = self

    @property
    def vocab(self):
        return self._vocab

    def predict(self, corpora, return_raw=False):
        if not self._trained:
            raise RuntimeError("Call .fit on annotated data (Corpus) before .predict.")
        return f"voted:{len(corpora)}"


def _controler(tmp_path, **kwargs):
    controler = types.SimpleNamespace(
        model_root=tmp_path, output_directory=tmp_path, config=object(),
        annotators=["syn-esn", "nsyn-esn", "ensemble"], _annotators={},
        _external_accum={}, _external_vocab=None,
    )
    controler.__dict__.update(kwargs)
    for method in ("_find_model_on_disk", "_load_annotator_from_disk"):
        setattr(controler, method,
                types.MethodType(getattr(Controler, method), controler))
    return controler


def test_a_sub_annotator_leaves_its_vocabulary_behind(tmp_path):
    controler = _controler(tmp_path, _annotators={"syn-esn": FakeAnnotator()})

    Controler.annotate_external(controler, corpus="corpus",
                                model_sources={"syn-esn": "path"})

    assert list(controler._external_accum) == ["syn-esn"]
    assert controler._external_vocab == VOCAB


def test_the_ensemble_uses_the_vocabulary_of_an_earlier_call(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "get_annotator", lambda name: FakeEnsemble)
    controler = _controler(
        tmp_path,
        _external_accum={"syn-esn": "raw1", "nsyn-esn": "raw2"},
        _external_vocab=VOCAB,
    )

    results = Controler.annotate_external(controler, corpus="corpus",
                                          model_sources={"ensemble": "path"})

    assert results["ensemble"] == "voted:2"
    assert FakeEnsemble.last.vocab == VOCAB
    assert FakeEnsemble.last._trained
    assert controler._external_accum == {}
    assert controler._external_vocab is None


def test_an_ensemble_without_a_vocabulary_does_not_reach_predict(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "get_annotator", lambda name: FakeEnsemble)
    controler = _controler(
        tmp_path,
        _external_accum={"syn-esn": "raw1", "nsyn-esn": "raw2"},
        _external_vocab=None,
    )

    results = Controler.annotate_external(controler, corpus="corpus",
                                          model_sources={"ensemble": "path"})

    assert "ensemble" not in results
