# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
import pandas as pd
import pytest

from canapy.corpus import Corpus
from dashboard.controler.corpusutils import mark_whole_corpus_as_train, query_split


@pytest.fixture()
def simple_corpus():
    df = pd.DataFrame(
        {
            "label": ["a", "b", "c", "d"],
            "onset_s": [0.0, 0.1, 0.2, 0.3],
            "offset_s": [0.1, 0.2, 0.3, 0.4],
            "notated_path": ["foo.wav"] * 4,
        }
    )
    return Corpus.from_df(df)


@pytest.fixture()
def split_corpus(simple_corpus):
    df = simple_corpus.dataset.copy()
    df["train"] = [True, True, False, False]
    simple_corpus.dataset = df
    return simple_corpus


# ---------------------------------------------------------------------------
# mark_whole_corpus_as_train
# ---------------------------------------------------------------------------

def test_mark_whole_corpus_as_train_adds_column(simple_corpus):
    result = mark_whole_corpus_as_train(simple_corpus)
    assert "train" in result.dataset.columns


def test_mark_whole_corpus_as_train_all_true(simple_corpus):
    result = mark_whole_corpus_as_train(simple_corpus)
    assert result.dataset["train"].all()


def test_mark_whole_corpus_as_train_overwrites_existing_flags(split_corpus):
    result = mark_whole_corpus_as_train(split_corpus)
    assert result.dataset["train"].all()


def test_mark_whole_corpus_as_train_returns_corpus(simple_corpus):
    result = mark_whole_corpus_as_train(simple_corpus)
    assert isinstance(result, Corpus)


def test_mark_whole_corpus_as_train_preserves_row_count(simple_corpus):
    n = len(simple_corpus.dataset)
    result = mark_whole_corpus_as_train(simple_corpus)
    assert len(result.dataset) == n


# ---------------------------------------------------------------------------
# query_split
# ---------------------------------------------------------------------------

def test_query_split_all_returns_full_corpus(split_corpus):
    result = query_split(split_corpus, split="all")
    assert len(result.dataset) == len(split_corpus.dataset)


def test_query_split_train_returns_only_train_rows(split_corpus):
    result = query_split(split_corpus, split="train")
    assert result.dataset["train"].all()
    assert len(result.dataset) == split_corpus.dataset["train"].sum()


def test_query_split_test_returns_only_test_rows(split_corpus):
    result = query_split(split_corpus, split="test")
    assert not result.dataset["train"].any()
    assert len(result.dataset) == (~split_corpus.dataset["train"]).sum()


def test_query_split_invalid_value_raises(split_corpus):
    with pytest.raises(ValueError):
        query_split(split_corpus, split="invalid")


def test_query_split_returns_corpus(split_corpus):
    for split in ("all", "train", "test"):
        result = query_split(split_corpus, split=split)
        assert isinstance(result, Corpus)
