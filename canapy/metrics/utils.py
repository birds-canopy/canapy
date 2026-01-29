# Author: Nathan Trouvain at 07/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import numpy as np
import pandas as pd

from ..timings import seconds_to_frames, frames_to_timed_df, seconds_to_audio
from ..utils.exceptions import MissingData


def as_frame_comparison(gold_corpus, corpus):
    """Expand gold_corpus dataframe from annotation format (one row per segment) to
    frame format (one row per annotated frame of data), following frames defined by
    another corpus.

    This is used to compare two corpora at frame level, using frame accuracy for
    instance."""
    if "frames_predictions" not in corpus.data_resources:
        raise MissingData(
            "Corpus has no 'frames_predictions' data resource. Was this "
            "Corpus created from an Annotator ?"
        )

    gold_df = gold_corpus.dataset.copy()
    frame_df = corpus.data_resources["frames_predictions"]

    sampling_rate = corpus.config.transforms.audio.sampling_rate
    hop_length = corpus.config.transforms.audio.hop_length
    time_precision = corpus.config.transforms.annots.time_precision

    hop_length = seconds_to_audio(hop_length, sampling_rate)

    gold_df["onset_spec"] = seconds_to_frames(
        gold_df["onset_s"], hop_length, sampling_rate
    )
    gold_df["offset_spec"] = seconds_to_frames(
        gold_df["offset_s"], hop_length, sampling_rate
    )

    silence_tag = corpus.config.transforms.annots.silence_tag

    gold_frames = []
    for seqid in gold_df["notated_path"].unique():
        seq_annots = gold_df.query("notated_path == @seqid")
        frame_annots = frame_df.query("notated_path == @seqid")

        n_frames = len(frame_annots)

        # repeat labels along time axis
        repeated_labels = np.array([silence_tag] * n_frames, dtype=object)

        for row in seq_annots.itertuples():
            onset = row.onset_spec
            offset = min(row.offset_spec, n_frames)
            repeated_labels[onset:offset] = row.label

        notated_path = seq_annots["notated_path"].unique()[0]

        gold_frames.append(
            frames_to_timed_df(
                repeated_labels,
                notated_path=notated_path,
                frame_size=hop_length,
                sampling_rate=sampling_rate,
                time_precision=time_precision,
            )
        )

    gold_frames = pd.concat(gold_frames)

    return gold_frames
