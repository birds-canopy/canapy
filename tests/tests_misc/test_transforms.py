# Author: Nathan Trouvain at 29/06/2023 <nathan.trouvain@inria.fr>
# Licence: BSD-3-Clause
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import crowsetta
import numpy as np

from canapy.corpus import Corpus
from canapy.transforms.commons.training import encode_labels
from canapy.transforms.nsynesn import *
from canapy.transforms.synesn import *


def test_balance_labels(corpus):
    df = corpus.dataset
    df["train"] = False
    samples = df.sample(50).index
    df.loc[samples, "train"] = True

    c = balance_labels_duration(corpus, resource_name="balanced_df")

    bdf = c.data_resources["balanced_df"]
    bdf["duration"] = bdf["offset_s"] - bdf["onset_s"]
    durations = bdf.groupby("label")["duration"].sum()

    # All classes should reach at least the median total duration of the TRAINING set
    # (balance_labels_duration targets the training split, not the full corpus)
    train_df = df.query("train").copy()
    train_df["duration"] = train_df["offset_s"] - train_df["onset_s"]
    expected_min = train_df.groupby("label")["duration"].sum().median()
    assert durations.min() >= expected_min


def test_compute_mfcc_for_balanced_dataset(corpus):
    df = corpus.dataset
    df["train"] = False
    samples = df.sample(50).index
    df.loc[samples, "train"] = True

    df["augmented"] = False

    # c = balance_labels_duration(corpus, resource_name="balanced_df")

    corpus.register_data_resource("balanced_df", df)

    bdf = corpus.data_resources["balanced_df"]
    bdf["duration"] = bdf["offset_s"] - bdf["onset_s"]
    durations = bdf.groupby("label")["duration"].sum()

    corpus = compute_mfcc_for_balanced_dataset(corpus, resource_name="mfccs")

    assert "mfccs" in corpus.data_resources


def test_encode_labels(corpus):
    df = corpus.dataset
    df["train"] = False
    samples = df.sample(50).index
    df.loc[samples, "train"] = True

    result = encode_labels(corpus, resource_name="encoded_labels")

    assert "encoded_label" in result.dataset.columns


if __name__ == "__main__":
    pass
