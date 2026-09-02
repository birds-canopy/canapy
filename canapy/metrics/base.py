# Author: Nathan Trouvain at 07/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: BSD-3-Clause
# Copyright: Nathan Trouvain
import pandas as pd
import Levenshtein

from sklearn.metrics import classification_report, confusion_matrix

from .utils import as_frame_comparison


def _check_corpus_comparison(gold_corpus, corpus):
    gold_notated = set(gold_corpus.dataset["notated_path"].unique())
    # Files predicted as entirely silent are absent from corpus.dataset but
    # still present in frames_predictions.
    if "frames_predictions" in corpus.data_resources:
        df_notated = set(
            corpus.data_resources["frames_predictions"]["notated_path"].unique()
        )
    else:
        df_notated = set(corpus.dataset["notated_path"].unique())

    if gold_notated != df_notated:
        raise ValueError(
            f"Ground truth corpus and predicted corpus do not match: "
            f"different audio have been annotated. "
            f"Mismatched: {gold_notated.symmetric_difference(df_notated)}"
        )


def compute_sklearn_metrics(gold_corpus, corpus, classes=None):
    """Confusion matrix and classification report in a single pass.

    Shares one ``as_frame_comparison`` expansion between both metrics.
    """
    _check_corpus_comparison(gold_corpus, corpus)

    gold_frames = as_frame_comparison(gold_corpus, corpus)
    pred_frames = corpus.data_resources["frames_predictions"]

    gold_labels = gold_frames.sort_values(by=["notated_path", "onset_s"])["label"]
    pred_labels = pred_frames.sort_values(by=["notated_path", "onset_s"])["label"]

    cm = confusion_matrix(gold_labels, pred_labels, labels=classes, normalize="true")
    report = classification_report(
        gold_labels,
        pred_labels,
        target_names=classes,
        labels=classes,
        zero_division=0,
        output_dict=True,
    )
    return cm, report


def sklearn_classification_report(gold_corpus, corpus, classes=None):
    _check_corpus_comparison(gold_corpus, corpus)

    gold_frames = as_frame_comparison(gold_corpus, corpus)
    pred_frames = corpus.data_resources["frames_predictions"]

    return classification_report(
        gold_frames.sort_values(by=["notated_path", "onset_s"])["label"],
        pred_frames.sort_values(by=["notated_path", "onset_s"])["label"],
        target_names=classes,
        labels=classes,
        zero_division=0,
        output_dict=True,
    )


def sklearn_confusion_matrix(gold_corpus, corpus, classes=None):
    _check_corpus_comparison(gold_corpus, corpus)

    gold_frames = as_frame_comparison(gold_corpus, corpus)
    pred_frames = corpus.data_resources["frames_predictions"]

    return confusion_matrix(
        gold_frames.sort_values(by=["notated_path", "onset_s"])["label"],
        pred_frames.sort_values(by=["notated_path", "onset_s"])["label"],
        labels=classes,
        normalize="true",
    )


_FRAME_KEYS = ["notated_path", "onset_s"]


def _gold_frames(gold_corpus, corpus):
    """Gold annotations expanded onto the frame grid of ``corpus``.

    Frames rather than ``dataset`` rows: post-processing strips the silence rows
    of a predicted corpus while the gold corpus keeps them, so the two row
    sequences are not comparable. On the frame grid every gap carries the
    silence tag on both sides alike.
    """
    return as_frame_comparison(gold_corpus, corpus).sort_values(by=_FRAME_KEYS)


def _raw_prediction_frames(corpus):
    """What the model emitted, frame by frame, before post-processing."""
    return corpus.data_resources["frames_predictions"].sort_values(by=_FRAME_KEYS)


def _annotated_prediction_frames(corpus):
    """The post-processed annotations of ``corpus``, back on its frame grid.

    Differs from ``_raw_prediction_frames``: post-processing merges runs, drops
    short labels and removes silence.
    """
    return as_frame_comparison(corpus, corpus).sort_values(by=_FRAME_KEYS)


def _label_sequence(labels, silence_tag):
    """One token per continuous run of a label, silence dropped.

    The run is cut *before* the silence is dropped, never after: two segments
    carrying the same label on either side of a silence are two tokens, and
    dropping first would merge them and hide a segment from the edit distance.
    """
    runs = labels.loc[labels.shift() != labels]
    return runs[runs != silence_tag].tolist()


def segment_error_rate(gold_corpus, corpus):
    """Syllable error rate: edit distance normalised by the reference length.

    ``levenshtein(predicted, reference) / len(reference)`` over sequences of
    segment labels. Insertions, deletions and substitutions all count, and the
    normalisation is by the reference alone, so the rate is asymmetric and may
    exceed 1.

    The predicted sequence is read off the **post-processed** annotations, the
    ones canapy publishes; reading it off the raw frames would score a corpus
    against itself well above 0. ``frame_error_rate`` keeps the raw frames, so
    the two rates look at different sides of post-processing.

    Returns one row per audio file.
    """
    _check_corpus_comparison(gold_corpus, corpus)

    silence_tag = corpus.config.transforms.annots.silence_tag
    gold_frames = _gold_frames(gold_corpus, corpus)
    pred_frames = _annotated_prediction_frames(corpus)

    ser = []
    for notated_path, gold_group in gold_frames.groupby("notated_path"):
        pred_group = pred_frames[pred_frames["notated_path"] == notated_path]

        reference = _label_sequence(gold_group["label"], silence_tag)
        predicted = _label_sequence(pred_group["label"], silence_tag)

        if reference:
            rate = Levenshtein.distance(predicted, reference) / len(reference)
        else:
            # undefined against an empty reference
            rate = 0.0 if not predicted else float("nan")

        ser.append({"notated_path": notated_path, "ser": rate})

    return pd.DataFrame(ser)


def frame_error_rate(gold_corpus, corpus):
    """Share of frames whose label differs from the reference.

    Silence frames count like any other: the rate runs over every frame of the
    recording, not only over the annotated ones. Read off the raw frame
    predictions, so this is the complement of the ``accuracy`` row of
    ``sklearn_classification_report``.
    """
    _check_corpus_comparison(gold_corpus, corpus)

    gold_frames = _gold_frames(gold_corpus, corpus)
    pred_frames = _raw_prediction_frames(corpus)
    gold_labels = gold_frames["label"].to_numpy()
    pred_labels = pred_frames["label"].to_numpy()

    # truncate both to the shorter one rather than failing
    n_frames = min(len(gold_labels), len(pred_labels))
    if n_frames == 0:
        return float("nan")

    return float((gold_labels[:n_frames] != pred_labels[:n_frames]).mean())
