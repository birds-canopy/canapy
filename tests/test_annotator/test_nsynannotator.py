# Author: Nathan Trouvain at 06/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: BSD-3-Clause
# Copyright: Nathan Trouvain
from canapy.annotator.nsynannotator import NSynAnnotator


def test_nsynannotator(corpus, output_directory):
    annotator = NSynAnnotator(config=corpus.config)
    annotator.fit(corpus)

    assert annotator.trained is True
    assert len(annotator.vocab) > 0

    pred_corpus = annotator.predict(corpus)

    assert len(pred_corpus.dataset) > 0
    assert "frames_predictions" in pred_corpus.data_resources

    pred_corpus.to_directory(output_directory)
    assert any(output_directory.glob("*.csv"))
