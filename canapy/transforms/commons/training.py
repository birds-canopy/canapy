# Author: Nathan Trouvain at 28/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
# Author: Nathan Trouvain at 28/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


from .annots import (
    sort_annotations,
    tag_silences,
    remove_short_labels,
    merge_labels,
)
from ..base import Transform
from ...log import log


logger = logging.getLogger("canapy")


@log(fn_type="training data tranform")
def split_train_test(corpus, *, redo=False, **kwargs):
    """Build train and test sets from data for syntactic training.
    Ensure that at least one example of each syllable is present in train set.
    """
    df = corpus.dataset

    # Already split!
    if "train" in df and not redo:
        return corpus

    config = corpus.config.transforms.training

    rs = np.random.default_rng(corpus.config.misc.seed)

    df["seqid"] = df["annotation"].astype(str) + df["sequence"].astype(str)

    n_sequences = len((df["seqid"]).unique())

    max_sequences = config.max_sequences
    if max_sequences == -1:
        max_sequences = n_sequences

    if max_sequences > n_sequences:
        raise ValueError(
            f"Can't select {max_sequences} training sequences "
            f"in a dataset of {n_sequences} sequences."
        )

    # Train dataframe
    train_df = pd.DataFrame(columns=df.columns)
    class_min_occurence = df.groupby("label")["label"].count().min()
    n_classes = len(df["label"].unique())
    while len(train_df.groupby("label")) < n_classes:
        class_min_occurence += 1
        min_occurences = (
            df.groupby("label")["label"]
            .count()
            .index[df.groupby("label")["label"].count() < class_min_occurence]
        )
        min_occurences_seqs = df.query("label in @min_occurences")["seqid"].unique()
        train_df = df.query("seqid in @min_occurences_seqs")

    already_picked = train_df["seqid"].unique()
    left_to_pick = df.query("seqid not in @already_picked")["seqid"].unique()

    logger.info(
        f"Min. number of sequences to train over all classes : "
        f"{len(already_picked)}"
    )

    if max_sequences < len(already_picked):
        logger.warning(
            f"Only {max_sequences} sequences will be selected (from max_sequence "
            f"config parameter) but {len(already_picked)} sequences are necessary "
            f"to train over all existing label classes."
        )

    # Add data to train_df up to test_ratio
    test_ratio = config.test_ratio

    more_size = np.floor(
        (1 - test_ratio) * len(left_to_pick) - test_ratio * len(already_picked)
    ).astype(int)

    some_more_seqs = rs.choice(left_to_pick, size=more_size, replace=False)
    some_more_data = df.query("seqid in @some_more_seqs")

    # Split
    train_df = pd.concat([train_df, some_more_data])
    test_df = df.query("seqid not in @train_df.seqid.unique()")

    # Reduce sequence number up to max_sequences
    n_train_seqs = len(train_df["seqid"].unique())
    if max_sequences < n_train_seqs:
        sequences = train_df["seqid"].unique()
        selection = rs.choice(sequences, size=max_sequences, replace=False)
        train_df = train_df.query("seqid in @selection")

    df["train"] = False
    df.loc[train_df.index, "train"] = True

    # Time stats
    train_time = (train_df["offset_s"] - train_df["onset_s"]).sum()
    test_time = (test_df["offset_s"] - test_df["onset_s"]).sum()

    silence_tag = corpus.config.transforms.annots.silence_tag
    train_no_silence = train_df.query("label != @silence_tag")
    test_no_silence = test_df.query("label != @silence_tag")

    train_nosilence_time = (
        train_no_silence["offset_s"] - train_no_silence["onset_s"]
    ).sum()
    test_nosilence_time = (
        test_no_silence["offset_s"] - test_no_silence["onset_s"]
    ).sum()

    logger.info(
        f"Final repartition of data - "
        f"\nTrain : {len(train_df['seqid'].unique())} ({len(train_df)} labels "
        f"- {train_time:.3f} s - {train_nosilence_time:.3f} s (w/o silence)"
        f"\nTest: {len(test_df.groupby('seqid'))} ({len(test_df)} labels) "
        f"- {test_time:.3f} s - {test_nosilence_time:.3f} s (w/o silence)"
    )

    df.drop("seqid", axis=1, inplace=True)

    return corpus


@log(fn_type="training data tranform")
def encode_labels(corpus, *, resource_name, **kwargs):
    df = corpus.dataset
    df["encoded_label"] = np.nan

    all_dfs = []
    if "balanced_dataset-train" in corpus.data_resources:
        balanced = corpus.data_resources["balanced_dataset-train"]
        balanced["encoded_label"] = np.nan

        all_dfs.append(balanced)

    all_dfs += [df]

    categories = np.sort(df["label"].unique()).reshape(1, -1).tolist()

    labels = df["label"].values.reshape(-1, 1)
    encoder = OneHotEncoder(categories=categories, sparse_output=False).fit(labels)

    for one_df in all_dfs:
        labels = one_df["label"].values.reshape(-1, 1)
        encoded_labels = encoder.transform(labels)
        one_df["encoded_label"] = [e for e in encoded_labels]

    return corpus


def prepare_dataset_for_training(corpus, **kwargs):
    if "dataset" in corpus.data_resources:
        return corpus
    else:
        transform = DatasetTransform()
        return transform(corpus, purpose="training")


class DatasetTransform(Transform):
    def __init__(self):
        super().__init__(
            annots_transforms=[
                sort_annotations,
                merge_labels,
                sort_annotations,
                tag_silences,
                sort_annotations,
                remove_short_labels,
                sort_annotations,
            ],
            training_data_transforms=[split_train_test, encode_labels],
            training_data_resource_name=[None, None],
        )
