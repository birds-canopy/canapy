# Author: Nathan Trouvain at 07/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pytest

from canapy.annotator.ensemble import Ensemble


def test_ensemble(prediction_corpus, prediction_corpus2, config):

    annotator = Ensemble(config).fit(prediction_corpus)
    corpus = annotator.predict([prediction_corpus, prediction_corpus2])

    assert len(corpus.dataset) > 0
    assert annotator.vocab == list("abcdefgh")

    annotator = Ensemble(config)

    with pytest.raises(Exception):
        annotator.predict([prediction_corpus, prediction_corpus2])


def test_ensemble_hard_vote_raises(prediction_corpus_no_raw, prediction_corpus2, config):

    annotator = Ensemble(config).fit(prediction_corpus2)

    with pytest.raises(KeyError):
        annotator.predict([prediction_corpus_no_raw, prediction_corpus2])
