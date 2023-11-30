# Author: Nathan Trouvain at 06/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pathlib

import pytest

import pandas as pd

from canapy.corpus import Corpus


def test_to_directory(corpus, output_directory, annot_directory):
    corpus.to_directory(str(output_directory))
    out_files = [f.name for f in sorted(output_directory.glob("*.csv"))]
    in_files = [f.name for f in sorted(annot_directory.glob("*.csv"))]
    assert out_files == in_files


def test_from_df(corpus):
    new_corpus = Corpus.from_df(corpus.dataset)

    assert new_corpus.dataset[["onset_s", "offset_s", "label"]].equals(corpus.dataset[["onset_s", "offset_s", "label"]])

    new_corpus = Corpus.from_df(corpus.dataset, annots_directory=corpus.annots_directory)

    assert str(pathlib.Path(new_corpus.dataset.loc[0, "annot_path"]).parents) == str(pathlib.Path(corpus.dataset.loc[0, "annot_path"]).parents)


def test_clone_with_df(corpus):

    new_corpus = corpus.clone_with_df(corpus.dataset)
    assert new_corpus.dataset[["onset_s", "offset_s", "label", "annot_path"]].equals(corpus.dataset[["onset_s", "offset_s", "label", "annot_path"]])
