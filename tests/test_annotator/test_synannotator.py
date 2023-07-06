# Author: Nathan Trouvain at 04/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from canapy.annotator.synannotator import SynAnnotator
from canapy.corpus import Corpus


def test_synannotator():

    corpus = Corpus.from_directory(
        audio_directory="/home/nathan/Documents/Code/canapy-test/data/",
        annots_directory="/home/nathan/Documents/Code/canapy-test/data/",
    )

    # df = c.dataset
    #
    # df["train"] = False
    #
    # samples = df.sample(50).index
    # df.loc[samples, "train"] = True

    annotator = SynAnnotator(
        config=corpus.config,
        spec_directory="/home/nathan/Documents/Code/canapy-reborn/tests/tests/output/transforms",
    )

    annotator.fit(corpus)

    pred_corpus = annotator.predict(corpus)

    pred_corpus.to_directory("output_preds")
