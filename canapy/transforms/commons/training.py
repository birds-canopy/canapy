# Author: Nathan Trouvain at 28/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: BSD-3-Clause
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
    Robust handling for many classes / small corpora.
    """
    df = corpus.dataset

    # Already split!
    if "train" in df and not redo:
        return corpus

    config = corpus.config.transforms.training
    rs = np.random.default_rng(corpus.config.misc.seed)

    df["seqid"] = df["annotation"].astype(str) + df["sequence"].astype(str)

    n_sequences = len(df["seqid"].unique())
    max_sequences = config.max_sequences
    if max_sequences == -1:
        max_sequences = n_sequences

    if max_sequences > n_sequences:
        raise ValueError(
            f"Can't select {max_sequences} training sequences "
            f"in a dataset of {n_sequences} sequences."
        )

    train_df = pd.DataFrame(columns=df.columns)
    class_min_occurence = df.groupby("label")["label"].count().min()
    n_classes = len(df["label"].unique())

    while len(train_df.groupby("label")) < n_classes:
        class_min_occurence += 1
        counts = df.groupby("label")["label"].count()
        min_occurences = counts.index[counts < class_min_occurence]
        min_occurences_seqs = df.query("label in @min_occurences")["seqid"].unique()
        train_df = df.query("seqid in @min_occurences_seqs")

    already_picked = np.sort(train_df["seqid"].unique())
    left_to_pick = np.sort(df.query("seqid not in @already_picked")["seqid"].unique())

    logger.info(f"Min. number of sequences to train over all classes: {len(already_picked)}")

    if max_sequences < len(already_picked):
        logger.warning(
            f"Only {max_sequences} sequences will be selected (from max_sequences config) "
            f"but {len(already_picked)} sequences are necessary to cover all label classes."
        )

    test_ratio = config.test_ratio
    desired_train_seqs = int(np.round((1.0 - test_ratio) * n_sequences))

    if n_sequences >= 2:
        desired_train_seqs = min(desired_train_seqs, n_sequences - 1)

    desired_train_seqs = min(desired_train_seqs, max_sequences)

    logger.info(
        f"Total sequences: {n_sequences}. Desired train sequences: {desired_train_seqs}."
    )

    current_train_seqs = len(train_df["seqid"].unique())
    if current_train_seqs > desired_train_seqs:
        sequences = np.sort(train_df["seqid"].unique())
        selection = rs.choice(sequences, size=desired_train_seqs, replace=False)
        train_df = train_df.query("seqid in @selection")
        logger.info(
            f"Reduced train sequences from {current_train_seqs} to {desired_train_seqs} to respect desired ratio."
        )
    else:
        need = desired_train_seqs - current_train_seqs
        if need > 0 and len(left_to_pick) > 0:
            add_n = min(need, len(left_to_pick))
            some_more_seqs = rs.choice(left_to_pick, size=add_n, replace=False)
            some_more_data = df.query("seqid in @some_more_seqs")
            train_df = pd.concat([train_df, some_more_data])
            logger.info(f"Added {add_n} sequences to train (now {len(train_df['seqid'].unique())}).")

    test_df = df.query("seqid not in @train_df.seqid.unique()")

    if len(test_df["seqid"].unique()) == 0 and n_sequences >= 2:
        train_seqs = np.sort(train_df["seqid"].unique())
        chosen = rs.choice(train_seqs, size=1, replace=False)
        train_df = train_df.query("seqid not in @chosen")
        test_df = df.query("seqid in @chosen")
        logger.warning("Test set was empty; moved one sequence from train to test to ensure non-empty test set.")


    n_train_seqs = len(train_df["seqid"].unique())
    if max_sequences < n_train_seqs:
        sequences = np.sort(train_df["seqid"].unique())
        selection = rs.choice(sequences, size=max_sequences, replace=False)
        train_df = train_df.query("seqid in @selection")
        logger.info(f"Applied max_sequences cap: train sequences reduced to {max_sequences}.")

    df["train"] = False
    df.loc[train_df.index, "train"] = True

    train_time = (train_df["offset_s"] - train_df["onset_s"]).sum()
    test_time = (test_df["offset_s"] - test_df["onset_s"]).sum()
    silence_tag = corpus.config.transforms.annots.silence_tag
    train_no_silence = train_df.query("label != @silence_tag")
    test_no_silence = test_df.query("label != @silence_tag")
    train_nosilence_time = (train_no_silence["offset_s"] - train_no_silence["onset_s"]).sum()
    test_nosilence_time = (test_no_silence["offset_s"] - test_no_silence["onset_s"]).sum()

    logger.info(
        f"Final repartition of data - "
        f"\nTrain : {len(train_df['seqid'].unique())} ({len(train_df)} labels - "
        f"{train_time:.3f} s - {train_nosilence_time:.3f} s w/o silence)"
        f"\nTest  : {len(test_df.groupby('seqid'))} ({len(test_df)} labels - "
        f"{test_time:.3f} s - {test_nosilence_time:.3f} s w/o silence)"
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
