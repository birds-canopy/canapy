# Author: Nathan Trouvain at 07/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import numpy as np
import pandas as pd

from ..timings import seconds_to_frames, frames_to_timed_df
from ..utils.exceptions import MissingData


def as_frame_comparison(gold_corpus, corpus):
    if "frames_predictions" not in corpus.data_resources:
        raise MissingData(
            "Corpus has no 'frames_predictions' data resource. Was this "
            "Corpus created from an Annotator ?"
        )

    gold_df = gold_corpus.dataset.copy()
    frame_df = corpus.data_resources["frames_predictions"]

    sampling_rate = corpus.config.transforms.audio.sampling_rate
    hop_length = corpus.config.transforms.audio.as_fftwindow("hop_length")
    time_precision = corpus.config.transforms.annots.time_precision

    gold_df["onset_spec"] = seconds_to_frames(
        gold_df["onset_s"], hop_length, sampling_rate
    )
    gold_df["offset_spec"] = seconds_to_frames(
        gold_df["offset_s"], hop_length, sampling_rate
    )

    gold_df["seqid"] = str(gold_df["sequence"]) + str(gold_df["annotation"])
    frame_df["seqid"] = str(frame_df["sequence"]) + str(frame_df["annotation"])

    silence_tag = corpus.config.transforms.annots.silence_tag

    gold_frames = []
    for seqid in gold_df["seqid"].unique():
        seq_annots = gold_df.query("seqid == @seqid")
        frame_annots = frame_df.query("seqid == @seqid")

        n_frames = len(frame_annots)

        # repeat labels along time axis
        repeated_labels = [silence_tag] * n_frames

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
                time_precision=round(-np.log10(time_precision)),
            )
        )

    gold_frames = pd.concat(gold_frames)

    return gold_frames
