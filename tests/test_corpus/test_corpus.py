# Author: Nathan Trouvain / Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Nathan Trouvain / Axel Arnaud
import pathlib

import numpy as np
import pandas as pd
import pytest

from canapy.corpus import Corpus


# ---------------------------------------------------------------------------
# to_directory / from_df / clone_with_df
# ---------------------------------------------------------------------------

def test_to_directory(corpus, output_directory, annot_directory):
    corpus.to_directory(str(output_directory))
    out_files = [f.name for f in sorted(output_directory.glob("*.csv"))]
    in_files = [f.name for f in sorted(annot_directory.glob("*.csv"))]
    assert out_files == in_files


def test_from_df(corpus):
    new_corpus = Corpus.from_df(corpus.dataset)

    assert new_corpus.dataset[["onset_s", "offset_s", "label"]].equals(
        corpus.dataset[["onset_s", "offset_s", "label"]]
    )

    new_corpus = Corpus.from_df(corpus.dataset, annots_directory=corpus.annots_directory)

    assert str(pathlib.Path(new_corpus.dataset.loc[0, "annot_path"]).parents) == str(
        pathlib.Path(corpus.dataset.loc[0, "annot_path"]).parents
    )


def test_clone_with_df(corpus):
    new_corpus = corpus.clone_with_df(corpus.dataset)
    assert new_corpus.dataset[["onset_s", "offset_s", "label", "annot_path"]].equals(
        corpus.dataset[["onset_s", "offset_s", "label", "annot_path"]]
    )


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------

def test_len_returns_int(corpus):
    assert isinstance(len(corpus), int)


def test_len_matches_unique_annotation_groups(corpus):
    assert len(corpus) == corpus.dataset["annotation"].nunique()


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------

def test_getitem_boolean_mask_filters_rows(corpus):
    sub = corpus[corpus.dataset["label"] == "a"]
    assert set(sub.dataset["label"].unique()) == {"a"}


def test_getitem_returns_corpus_instance(corpus):
    sub = corpus[corpus.dataset["label"].isin(["a", "b"])]
    assert isinstance(sub, Corpus)


def test_getitem_preserves_config(corpus):
    sub = corpus[corpus.dataset["label"] == "a"]
    assert sub.config is corpus.config


def test_getitem_does_not_modify_original(corpus):
    original_len = len(corpus.dataset)
    _ = corpus[corpus.dataset["label"] == "a"]
    assert len(corpus.dataset) == original_len


def test_getitem_empty_result_is_corpus(corpus):
    sub = corpus[corpus.dataset["label"] == "NONEXISTENT"]
    assert isinstance(sub, Corpus)
    assert len(sub.dataset) == 0


# ---------------------------------------------------------------------------
# register_data_resource
# ---------------------------------------------------------------------------

def test_register_data_resource_stores_value(corpus):
    resource = pd.DataFrame({"x": [1, 2, 3]})
    corpus.register_data_resource("test_df", resource)
    assert "test_df" in corpus.data_resources


def test_register_data_resource_returns_self(corpus):
    result = corpus.register_data_resource("r", {"a": 1})
    assert result is corpus


def test_register_data_resource_overwrites(corpus):
    corpus.register_data_resource("r", "first")
    corpus.register_data_resource("r", "second")
    assert corpus.data_resources["r"] == "second"


def test_register_data_resource_numpy_array(corpus):
    arr = np.ones((5, 13))
    corpus.register_data_resource("features", arr)
    np.testing.assert_array_equal(corpus.data_resources["features"], arr)


# ---------------------------------------------------------------------------
# from_directory
# ---------------------------------------------------------------------------

def test_from_directory_raises_without_audio_or_spec():
    with pytest.raises(ValueError):
        Corpus.from_directory(audio_directory=None, spec_directory=None)


def test_from_directory_with_audio_files(audio_directory):
    corpus = Corpus.from_directory(audio_directory=str(audio_directory))
    assert corpus.audio_directory == audio_directory


def test_from_directory_spec_directory_auto_created(audio_only_corpus):
    assert audio_only_corpus.spec_directory is not None


def test_from_directory_loads_annotations(audio_directory, annot_directory):
    corpus = Corpus.from_directory(
        audio_directory=str(audio_directory),
        annots_directory=str(annot_directory),
    )
    assert len(corpus.dataset) > 0


def test_from_directory_without_annotations_dataset_not_none(audio_only_corpus):
    assert audio_only_corpus.dataset is not None


def test_from_directory_redo_transforms_does_not_raise(spec_directory):
    corpus = Corpus.from_directory(
        spec_directory=str(spec_directory),
        redo_transforms=True,
    )
    assert corpus is not None
