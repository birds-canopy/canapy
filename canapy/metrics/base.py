# Author: Nathan Trouvain at 07/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: BSD-3-Clause
# Copyright: Nathan Trouvain
import pandas as pd
import Levenshtein

from sklearn.metrics import classification_report, confusion_matrix

from .utils import as_frame_comparison


def _check_corpus_comparison(gold_corpus, corpus):
    gold_notated = set(gold_corpus.dataset["notated_path"].unique())
    # Use frames_predictions to determine annotated files: files predicted as
    # entirely silent are absent from corpus.dataset (silence rows are removed
    # during post-processing) but are still present in frames_predictions.
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
    """Compute confusion matrix and classification report in a single pass.

    Calling sklearn_confusion_matrix and sklearn_classification_report separately
    triggers two identical as_frame_comparison expansions. This function computes
    gold_frames once and shares it across both metrics.
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


def segment_error_rate(gold_corpus, corpus):
    _check_corpus_comparison(gold_corpus, corpus)

    gold_df = gold_corpus.dataset
    pred_df = corpus.dataset

    gold_sequences = gold_df.groupby("notated_path")
    pred_sequences = pred_df.groupby("notated_path")

    ser = []
    for seqid, gold_seq in gold_sequences:
        if seqid in pred_sequences.groups:
            pred_seq = pred_sequences.get_group(seqid)
        else:
            # File was predicted as entirely silent: no annotations in pred corpus.
            pred_seq = pred_df.iloc[0:0]  # empty df, SER = 1.0

        notated_path = gold_seq["notated_path"].unique()[0]

        ser.append({
            "notated_path": notated_path,
            "ser": 1.0 - Levenshtein.ratio(gold_seq.label.values, pred_seq.label.values)
            })

    ser = pd.DataFrame(ser)

    return ser
