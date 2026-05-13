# Author: Nathan Trouvain at 04/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: BSD-3-Clause
# Copyright: Nathan Trouvain
from canapy.annotator.synannotator import SynAnnotator

def test_synannotator(corpus, output_directory):
    annotator = SynAnnotator(
        config=corpus.config,
    )

    annotator.fit(corpus)

    pred_corpus = annotator.predict(corpus)

    pred_corpus.to_directory(output_directory)
