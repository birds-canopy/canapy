# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Tests for canapy/log.py — the @log corpus-transform decorator."""

import pandas as pd
import numpy as np
import pytest

from canapy.log import log
from canapy.corpus import Corpus
from config import default_config


def _minimal_corpus():
    df = pd.DataFrame({
        "label": ["A", "B"],
        "onset_s": [0.0, 0.1],
        "offset_s": [0.1, 0.2],
        "notated_path": ["f.wav", "f.wav"],
        "annot_path": ["f.csv", "f.csv"],
        "sequence": [0, 0],
        "annotation": ["a", "a"],
    })
    return Corpus.from_df(df, config=default_config)


class TestLogDecorator:
    def test_decorated_function_returns_corpus(self):
        @log(fn_type="test")
        def identity(corpus, **kwargs):
            return corpus

        result = identity(_minimal_corpus())
        assert isinstance(result, Corpus)

    def test_decorated_function_executes_body(self):
        called = []

        @log(fn_type="test")
        def side_effect(corpus, **kwargs):
            called.append(True)
            return corpus

        side_effect(_minimal_corpus())
        assert called == [True]

    def test_kwargs_forwarded_to_function(self):
        received = {}

        @log(fn_type="test")
        def capture_kwargs(corpus, **kwargs):
            received.update(kwargs)
            return corpus

        capture_kwargs(_minimal_corpus(), redo=True, resource_name="mfcc")
        assert received["redo"] is True
        assert received["resource_name"] == "mfcc"

    def test_multiple_fn_types_do_not_raise(self):
        for fn_type in ("corpus transform", "audio transform", "training data tranform"):
            @log(fn_type=fn_type)
            def fn(corpus, **kwargs):
                return corpus

            fn(_minimal_corpus())

    def test_corpus_data_unchanged(self):
        corpus = _minimal_corpus()
        original_labels = corpus.dataset["label"].tolist()

        @log(fn_type="test")
        def passthrough(c, **kwargs):
            return c

        result = passthrough(corpus)
        assert result.dataset["label"].tolist() == original_labels

    def test_exception_in_body_propagates(self):
        @log(fn_type="test")
        def failing(corpus, **kwargs):
            raise ValueError("intentional error")

        with pytest.raises(ValueError, match="intentional error"):
            failing(_minimal_corpus())
