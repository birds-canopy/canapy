# Author: Nathan Trouvain at 05/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import numpy as np
import pandas as pd
import scipy

from ...corpus import Corpus


def frames_to_seconds(
    frame_indices, frame_size, sampling_rate, center=True, time_precision=0.001
):
    frame_positions = frame_indices * frame_size

    if center:
        # Frames are usually centered around (frame_size * frame_index) in original
        # audio
        onsets_f = frame_positions - frame_size / 2
        offsets_f = frame_positions + frame_size / 2

        # Correct at edges to remove padding
        onsets_f[0] = 0.0
        offsets_f[-1] = offsets_f[-1] - frame_size / 2

        # Round to the closest sample position
        onsets_f = np.ceil(onsets_f)
        offsets_f = np.floor(offsets_f)

    else:
        # Frames start at (frame_size * frame_index)
        onsets_f = frame_positions
        offsets_f = frame_positions + frame_size

    # Convert to seconds
    decimals = round(-np.log10(time_precision))

    onsets_f = np.around(onsets_f / sampling_rate, decimals=decimals)
    offsets_f = np.around(offsets_f / sampling_rate, decimals=decimals)

    return onsets_f, offsets_f


def remove_silence(annots_df: pd.DataFrame, silence_tag):
    label_idxs = annots_df["label"] != silence_tag
    return annots_df[label_idxs]


def frames_to_timed_df(
    frame_predictions,
    notated_path,
    frame_size,
    sampling_rate,
    center=True,
    time_precision=0.001,
):
    n_frames = len(frame_predictions)

    if n_frames == 0:
        return pd.DataFrame(columns=["label", "onset_s", "offset_s", "notated_path"])

    frame_indices = np.arange(n_frames)

    onset_s, offset_s = frames_to_seconds(
        frame_indices,
        frame_size=frame_size,
        sampling_rate=sampling_rate,
        center=center,
        time_precision=time_precision,
    )

    df = pd.DataFrame(
        {
            "label": frame_predictions,
            "onset_s": onset_s,
            "offset_s": offset_s,
            "notated_path": [notated_path] * n_frames,
        }
    )

    return df


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
    center,
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
            center=center,
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

    corpus.register_data_resource("frames", frame_df)

    if raw_preds is not None:
        corpus.register_data_resource("nn_output", raw_preds)

    return corpus
