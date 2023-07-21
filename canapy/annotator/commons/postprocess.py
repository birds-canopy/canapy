# Author: Nathan Trouvain at 05/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import numpy as np
import pandas as pd
import scipy

from ...corpus import Corpus
from ...timings import frames_to_timed_df
from ...utils.arrays import to_structured


def extract_vocab(corpus, silence_tag):
    vocab = corpus.dataset["label"].unique().tolist()
    if silence_tag not in vocab:
        vocab += [silence_tag]
    return sorted(vocab)


def remove_silence(annots_df: pd.DataFrame, silence_tag):
    label_idxs = annots_df["label"] != silence_tag
    return annots_df[label_idxs]


def frame_df_to_annots_df(
    frame_df, min_label_duration, min_silence_gap, silence_tag="SIL", lonely_labels=None
):
    # Identify all phrases of identical labels
    gemini_groups = (
        frame_df["label"].shift(fill_value=str(np.nan)) != frame_df["label"]
    ).cumsum()

    # Aggregate: define phrase label and onset/offset
    # "A (0) A (1) A (2) A (3)" -> "A (0, 3)"
    df = (
        frame_df.groupby(gemini_groups, as_index=False)
        .agg(
            {
                "label": "first",
                "onset_s": min,
                "offset_s": max,
                "notated_path": "first",
            }
        )
        .sort_values(by=["onset_s"])
    )

    # Identify all sequences of very short label occurences
    short_samples = (df["offset_s"] - df["onset_s"] >= min_label_duration).cumsum()

    # Aggregate: take label mode (majority vote) for all consecutive
    # very short occurences
    # "A (0, 4), [B (4, 5), C (5, 6), B (6, 7)], C (7, 10)"
    # -> very short: [B C B] -> majority: B
    # -> "A (0, 4), B (4, 7), C (7, 10)"
    df = (
        df.groupby(short_samples, as_index=False)
        .agg(
            {
                "label": lambda x: scipy.stats.mode(x)[0][0],
                "onset_s": min,
                "offset_s": max,
                "notated_path": "first",
            }
        )
        .sort_values(by=["onset_s"])
    )

    # Finally, remove all singletons based on surrounding labels
    short_samples = (
        (df["offset_s"] - df["onset_s"] >= min_label_duration)
        & (
            df["offset_s"].shift(fill_value=np.inf)
            - df["onset_s"].shift(fill_value=0.0)
            >= min_label_duration
        )
        & (
            df["offset_s"].shift(-1, fill_value=np.inf)
            - df["onset_s"].shift(-1, fill_value=0.0)
            >= min_label_duration
        )
    ).cumsum()

    df = (
        df.groupby(short_samples, as_index=False)
        .agg(
            {
                "label": lambda x: scipy.stats.mode(x)[0][0],
                "onset_s": min,
                "offset_s": max,
                "notated_path": "first",
            }
        )
        .sort_values(by=["onset_s"])
    )

    df = remove_silence(df, silence_tag=silence_tag)

    if lonely_labels is None:
        lonely_labels = list()

    # Merge repeated labels
    gemini_groups = (
        (df["label"].shift(fill_value=str(np.nan)) != df["label"])
        & ~(df["label"].isin(lonely_labels))
        & (df["onset_s"].shift(fill_value=0.0) - df["offset_s"] <= min_silence_gap)
    ).cumsum()

    df = (
        df.groupby(gemini_groups, as_index=False)
        .agg(
            {
                "label": "first",
                "onset_s": min,
                "offset_s": max,
                "notated_path": "first",
            }
        )
        .sort_values(by="onset_s")
    )

    return df


def predictions_to_corpus(
    notated_paths,
    cls_preds,
    frame_size,
    sampling_rate,
    time_precision,
    min_label_duration,
    min_silence_gap,
    silence_tag,
    lonely_labels,
    config=None,
    raw_preds=None,
):
    frame_dfs = []
    annot_dfs = []
    for y_pred, notated_path in zip(cls_preds, notated_paths):
        frames = frames_to_timed_df(
            y_pred,
            notated_path=notated_path,
            frame_size=frame_size,
            sampling_rate=sampling_rate,
            time_precision=time_precision,
        )

        annots = frame_df_to_annots_df(
            frames,
            min_label_duration,
            min_silence_gap,
            silence_tag=silence_tag,
            lonely_labels=lonely_labels,
        )

        frame_dfs.append(frames)
        annot_dfs.append(annots)

    frame_df = pd.concat(frame_dfs)
    annots_df = pd.concat(annot_dfs)

    corpus = Corpus.from_df(annots_df, config=config)

    corpus.register_data_resource("frames_predictions", frame_df)

    if raw_preds is not None:
        if not isinstance(raw_preds, dict):
            raw_preds = dict(zip(notated_paths, raw_preds))
        # Save raw neural network outputs (logits, softmax activity...) as
        # structured array for convenience. This allows for easy memory mapping to avoid
        # memory issues.
        corpus.register_data_resource("nn_output", to_structured(raw_preds))

    return corpus
