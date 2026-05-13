# Author: Nathan Trouvain at 07/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: BSD-3-Clause
# Copyright: Nathan Trouvain
import pytest

from canapy.annotator.ensemble import Ensemble


def test_ensemble_fit_sets_vocab(prediction_corpus, config):
    annotator = Ensemble(config).fit(prediction_corpus)
    assert annotator.vocab == ["SIL"] + list("abcdefgh")


def test_ensemble_predict_returns_nonempty_corpus(prediction_corpus, prediction_corpus2, config):
    annotator = Ensemble(config).fit(prediction_corpus)
    corpus = annotator.predict([prediction_corpus, prediction_corpus2])
    assert len(corpus.dataset) > 0


def test_ensemble_predict_before_fit_raises(prediction_corpus, prediction_corpus2, config):
    annotator = Ensemble(config)
    with pytest.raises(Exception):
        annotator.predict([prediction_corpus, prediction_corpus2])


def test_ensemble_hard_vote_raises(prediction_corpus_no_raw, prediction_corpus2, config):
    annotator = Ensemble(config).fit(prediction_corpus2)
    with pytest.raises(KeyError):
        annotator.predict([prediction_corpus_no_raw, prediction_corpus2])
