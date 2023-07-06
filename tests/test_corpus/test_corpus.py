# Author: Nathan Trouvain at 06/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from canapy.corpus import Corpus


def test_corpus_to_directory(corpus):
    corpus.to_directory("output")


def test_corpus_from_df(df, predictions_df):
    corpus = Corpus.from_df(df)
    print(corpus)
    corpus = Corpus.from_df(predictions_df)
    print(corpus)

    corpus.to_directory("output2")
    corpus = Corpus.from_directory(audio_directory="output", annots_directory="output2")
    corpus
