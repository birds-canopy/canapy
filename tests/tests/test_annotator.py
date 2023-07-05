# Author: Nathan Trouvain at 04/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from canapy.annotator.synannotator import SynAnnotator
from canapy.corpus import Corpus


def test_synannotator():
    # crowsetta.register_format(Marron1CSV)
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

    annotator = SynAnnotator(config=corpus.config, transforms_output_directory="./output/transforms")
    annotator.fit(corpus)


if __name__ == "__main__":

    test_synannotator()
