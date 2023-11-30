# Author: Nathan Trouvain at 04/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from canapy.annotator.synannotator import SynAnnotator

def test_synannotator(corpus, spec_directory, output_directory):
    annotator = SynAnnotator(
        config=corpus.config,
        spec_directory=spec_directory,
    )

    annotator.fit(corpus)

    pred_corpus = annotator.predict(corpus)

    pred_corpus.to_directory(output_directory)
