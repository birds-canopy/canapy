# Author: Nathan Trouvain at 05/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
"""
Postprocess annotations. Mainly used to group predictions
made at the timeframe level, and convert them to
onsets, offsets in seconds and single labels per
annotated segments.
"""

import numpy as np
import pandas as pd

from config import default_config
from ...corpus import Corpus
from ...timings import frames_to_timed_df
from ...utils.arrays import to_structured


def extract_vocab(corpus, silence_tag):
    """Get vocabulary (list of classes) from
    a Corpus, by selecting all unique labels
    in Corpus annotations.
    Specify the silence label.
    """
    vocab = corpus.dataset["label"].unique().tolist()
    if silence_tag not in vocab:
        vocab += [silence_tag]
    return sorted(vocab)


def remove_silence(annots_df: pd.DataFrame, silence_tag):
    """Remove annotations of silences in an annotation
    dataframe."""
    label_idxs = annots_df["label"] != silence_tag
    return annots_df[label_idxs]


def frame_df_to_annots_df(
    frame_df, min_label_duration, min_silence_gap, silence_tag="SIL", lonely_labels=None
):
    """Group annotations made at the timeframe level (one annotation per time bin)
    to create annotations at the segment level (one annotation per segment with an
    onset and an offset in seconds).

    Apply correction to avoid creating segments inferiors to `min_label_duration` in
    length, or silences inferiors to `min_silence_gap` when grouping consecutive
    segments with the same label.

    Transform:

    AAAAABCBBDAAAA -> A B A

    Parameters
    ----------
    frame_df : pandas.DataFrame
        Dataframe with frame level annotations (one annotation per timebin).
    min_label_duration : float
        Minimum admissible duration for a segment. Any segment with a smaller
        duration will be dropped or grouped with consecutive segments.
    min_silence_gap : float
        Minimum admissible duration for a silence between segments with the
        same label. If to identical segments are too close, will merge them,
        except if they are in lonely_labels.
    silence_tag : str, default to "SIL"
        Label used to tag silent segments.
    lonely_labels : list of str, optional
        Define segments labels that must never be merged, even if the rule
        defined by min_silence_gap is not respected.

    Returns
    -------
    pandas.DataFrame
        Segment level annotations.
    """
    # Identify all phrases of identical labels
    gemini_groups = (
        frame_df["label"].shift(fill_value=str(np.nan)) != frame_df["label"]
    ).cumsum()

    # Aggregate: define phrase label and onset/offset
    # "A (0) A (1) A (2) A (3)" -> "A (0, 3)"
    df = (
        frame_df.groupby(gemini_groups)
        .agg(
            {
                "label": "first",
                "onset_s": "min",
                "offset_s": "max",
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
    def mode(x):
        return max(x.values.tolist(), key=x.values.tolist().count)

    df = (
        df.groupby(short_samples)
        .agg(
            {
                "label": mode,
                "onset_s": "min",
                "offset_s": "max",
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
        df.groupby(short_samples)
        .agg(
            {
                "label": mode,  # mode of labels
                "onset_s": "min",
                "offset_s": "max",
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
        df.groupby(gemini_groups)
        .agg(
            {
                "label": "first",
                "onset_s": "min",
                "offset_s": "max",
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
    config=default_config,
    raw_preds=None,
):
    """Transform timeframe annotations from
    an annotator into a new Corpus object.

    Will group and filter annotations to do so.

    Parameters
    ----------
    notated_paths : list of str
        Audio filenames.
    cls_preds : list of np.ndarray
        List of annotated sequences, as arrays of shape (timeframes, )
    frame_size : float
        Duration of one timeframe in seconds.
    sampling_rate : float
        Audio sampling rate in Hz.
    time_precision : float
        Minimum admissible duration, in seconds. All timestamps
        will be rounded to the same decimal.
    min_label_duration : float
        Minimum admissible duration for a segment. Any segment with a smaller
        duration will be dropped or grouped with consecutive segments.
    min_silence_gap : float
        Minimum admissible duration for a silence between segments with the
        same label. If to identical segments are too close, will merge them,
        except if they are in lonely_labels.
    silence_tag : str, default to "SIL"
        Label used to tag silent segments.
    lonely_labels : list of str, optional
        Define segments labels that must never be merged, even if the rule
        defined by min_silence_gap is not respected.
    config : Config, default to default_config
        A configuration object. If None, use default_config.
    raw_preds : dict or list of np.ndarray, optional
        Raw outputs of neural networks. If provided, will be saved in
        the new Corpus "nn_output" key of its data_resources.
        Needed for Ensemble.

    Returns
    -------
    Corpus
        A newly created corpus from predictions.
    """
    frame_dfs = []
    annot_dfs = []
    for y_pred, notated_path in zip(cls_preds, notated_paths):
        # Convert index to timestamps
        frames = frames_to_timed_df(
            y_pred,
            notated_path=notated_path,
            frame_size=frame_size,
            sampling_rate=sampling_rate,
            time_precision=time_precision,
        )
        # Group frame labels to form segments.
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
