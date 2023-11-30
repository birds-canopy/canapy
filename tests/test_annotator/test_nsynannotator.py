# Author: Nathan Trouvain at 06/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from canapy.annotator.nsynannotator import NSynAnnotator


def test_nsynannotator(corpus, spec_directory, output_directory):
    annotator = NSynAnnotator(
        config=corpus.config,
        spec_directory=spec_directory,
    )
    annotator.fit(corpus)

    pred_corpus = annotator.predict(corpus)

    pred_corpus.to_directory(output_directory)
