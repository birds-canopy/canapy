# Author: Nathan Trouvain at 07/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pandas as pd
import Levenshtein

from sklearn.metrics import classification_report, confusion_matrix

from .utils import as_frame_comparison


def _check_corpus_comparison(gold_corpus, corpus):
    gold_notated = set(gold_corpus.dataset["notated_path"].unique())
    df_notated = set(corpus.dataset["notated_path"].unique())

    if gold_notated != df_notated:
        raise ValueError(
            "Ground truth corpus and predicted corpus do not match: "
            "different audio have been annotated."
        )


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
        pred_seq = pred_sequences.get_group(seqid)

        notated_path = gold_seq["notated_path"].unique()[0]

        ser.append({
            "notated_path": notated_path,
            "ser": 1.0 - Levenshtein.ratio(gold_seq.label.values, pred_seq.label.values)
            })

    ser = pd.DataFrame(ser)

    return ser
