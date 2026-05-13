# Author: Nathan Trouvain at 06/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: BSD-3-Clause
# Copyright: Nathan Trouvain
from canapy.annotator.nsynannotator import NSynAnnotator


def test_nsynannotator(corpus, output_directory):
    annotator = NSynAnnotator(
        config=corpus.config,
    )
    annotator.fit(corpus)

    pred_corpus = annotator.predict(corpus)

    pred_corpus.to_directory(output_directory)
