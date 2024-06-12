# Author: Nathan Trouvain at 05/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
"""
Transform MFCC data and labels to train ESN models.
"""

import logging

import numpy as np
import pandas as pd

from ...timings import seconds_to_frames, seconds_to_audio
from ...utils.exceptions import NotTrainableError, MissingData

logger = logging.getLogger("canapy")


def load_mfccs_and_repeat_labels(corpus, purpose="training"):
    """Load precomputed MFCC and repeat labels along time axis
    to get a suitable training/testing dataset for a recurrent neural
    network.

    Parameters
    ----------
    corpus : Corpus
        An annotated corpus of vocalizations. Must contain MFCC data
        in data_resources.
    purpose : ["training", "eval"], default to "training"
        Purpose of dataset. If "training", will check that
        Corpus has annotations, and load the "train" split
        of the Corpus. Else, the test split will be loaded.

    Returns
    -------
    list of np.ndarray, list of np.ndarray, list of int, list of int
        MFCCs as list of arrays of shape (timeframes, mfcc), one-hot
        encoded labels as list of arrays of shape (timeframes, classes),
        sequence id and annotation id as defined by crowsetta.
    """
    if purpose == "training":
        split = "train"

        if len(corpus.dataset.query(split)) == 0:
            raise NotTrainableError(
                "Training data was not provided, or corpus was "
                "not properly divided between train and test data."
            )

    elif purpose == "eval":
        split = "not train"

        if len(corpus.dataset.query(split)) == 0:
            raise NotTrainableError(
                "Test data was not provided, or corpus was "
                "not properly divided between train and test data."
            )
    else:
        raise ValueError("'purpose' should be either 'training' or 'eval'.")

    # load data
    df = corpus.dataset.query(split).copy()

    if "syn_mfcc" not in corpus.data_resources:
        raise MissingData(
            "'syn_mfcc' were never computed or can't be found in Corpus. "
            "Maybe provide and audio/spec directory to the Corpus ?"
        )

    mfcc_paths = corpus.data_resources["syn_mfcc"]

    df["seqid"] = df["sequence"].astype(str) + df["annotation"].astype(str)

    sampling_rate = corpus.config.transforms.audio.sampling_rate
    hop_length = seconds_to_audio(
        corpus.config.transforms.audio.hop_length, sampling_rate
    )

    df["onset_spec"] = seconds_to_frames(df["onset_s"], hop_length, sampling_rate)
    df["offset_spec"] = seconds_to_frames(df["offset_s"], hop_length, sampling_rate)

    n_classes = len(df["label"].unique())

    mfccs = []
    labels = []
    sequences = []
    annotations = []
    for seqid in df["seqid"].unique():
        seq_annots = df.query("seqid == @seqid")

        notated_audio = seq_annots["notated_path"].unique()[0]
        notated_spec = mfcc_paths.query("notated_path == @notated_audio")[
            "feature_path"
        ].unique()[0]

        seq_end = seq_annots["offset_spec"].iloc[-1]
        mfcc = np.load(notated_spec)

        # MFCC may be stored as archive arrays for convenience
        # (see transforms/commons/audio.py)
        if hasattr(mfcc, "keys") and "feature" in mfcc:
            mfcc = mfcc["feature"].squeeze()
        else:
            raise KeyError("No key named 'feature' in mfcc archive file.")

        if seq_end > mfcc.shape[1]:
            logger.warning(
                f"Found inconsistent sequence length: "
                f"audio {notated_audio} was converted to "
                f"{mfcc.shape[1]} timesteps but last annotation is at "
                f"timestep {seq_end}. Annotation will be trimmed."
            )

        seq_end = min(seq_end, mfcc.shape[1])

        mfcc = mfcc[:, :seq_end]

        # repeat labels along time axis
        repeated_labels = np.zeros((seq_end, n_classes))
        for row in seq_annots.itertuples():
            onset = row.onset_spec
            offset = min(row.offset_spec, seq_end)
            label = row.encoded_label

            repeated_labels[onset:offset] = label

        mfccs.append(mfcc.T)
        labels.append(repeated_labels)
        sequences.append(seq_annots["sequence"].unique()[0])
        annotations.append(seq_annots["annotation"].unique()[0])

    return annotations, sequences, mfccs, labels


def load_mfccs_for_annotation(corpus):
    """Load MFCC for audio-only Corpus annotation.

    Parameters
    ----------
    corpus : Corpus
        Corpus to annotate. Must have been preprocessed and
        contain MFCC data in data_resources.

    Returns
    -------
    list of str, list of np.ndarray
        List of audio filename and corresponding list of MFCC as arrays
        of shape (timeframes, mfcc).
    """
    if "syn_mfcc" not in corpus.data_resources:
        raise MissingData(
            "'syn_mfcc' were never computed or can't be found in Corpus. "
            "Maybe provide and audio/spec directory to the Corpus ?"
        )

    # selected_paths = corpus.dataset.notated_path.unique().tolist()
    mfcc_paths = corpus.data_resources["syn_mfcc"]
    mfccs = []
    notated_paths = []
    for row in mfcc_paths.itertuples():
        spec_path = row.feature_path

        if row.notated_path == np.nan:
            notated_path = spec_path
        else:
            notated_path = row.notated_path

        # We might not want to load all data, but only the subset
        # actually present in the corpus
        # if notated_path not in selected_paths:
        #    continue

        mfcc = np.load(spec_path)

        # MFCC may be stored as structured arrays for convenience
        # (see transforms/commons/audio.py)
        if hasattr(mfcc, "keys") and "feature" in mfcc:
            mfcc = mfcc["feature"].squeeze()

        mfccs.append(mfcc.T)
        notated_paths.append(notated_path)

    return notated_paths, mfccs
