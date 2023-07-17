# Author: Nathan Trouvain at 06/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from canapy.annotator.nsynannotator import NSynAnnotator
from canapy.corpus import Corpus


def test_nsynannotator():
    # corpus = Corpus.from_directory(
    #     audio_directory="/home/nathan/Documents/Code/canapy-test/data/",
    #     annots_directory="/home/nathan/Documents/Code/canapy-test/data/",
    # )
    corpus = Corpus.from_directory(
        audio_directory="/home/nathan/Documents/Code/canapy-test/data",
        annots_directory="/home/nathan/Documents/Code/canapy-test/data",
    )
    annotator = NSynAnnotator(
        config=corpus.config,
        spec_directory="/home/nathan/Documents/Code/canapy-test/data",
    )
    annotator.fit(corpus)

    corpus = annotator.predict(corpus)

    # print(corpus.dataset.query("notated_path == @n").label)
