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
        audio_directory="/home/vincent/Documents/Travail/Stage_L3/canapy-master_github/data_alizee/fast_train",
        annots_directory="/home/vincent/Documents/Travail/Stage_L3/canapy-master_github/data_alizee/fast_train",
    )
    annotator = NSynAnnotator(
        config=corpus.config,
        spec_directory="/home/vincent/Documents/Travail/Stage_L3/canapy-reborn/tests/tests/output/transforms",
    )
    annotator.fit(corpus)

    n, _, cls_pred, _ = annotator.predict(corpus)

    n = n[0]

    print(n)
    print(cls_pred[0])
    print(corpus.dataset.query("notated_path == @n").label)


