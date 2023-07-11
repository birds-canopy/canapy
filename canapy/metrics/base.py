# Author: Nathan Trouvain at 07/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pandas as pd
import sklearn
import Levenshtein

from .utils import as_frame_comparison


def _check_corpus_comparison(gold_corpus, corpus):
    gold_notated = set(gold_corpus.dataset["notated_path"].unique())
    df_notated = set(corpus.dataset["notated_path"].unique())

    if gold_notated != df_notated:
        raise ValueError(
            "Ground truth corpus and predicted corpus do not match: "
            "different audio have been annotated."
        )


def classification_report(gold_corpus, corpus, classes=None):
    _check_corpus_comparison(gold_corpus, corpus)

    gold_frames = as_frame_comparison(gold_corpus, corpus)
    pred_frames = corpus.data_resources["frames_predictions"]

    return sklearn.metrics.classification_report(
        gold_frames.sort_values(by=["notated_path", "onset_s"])["label"],
        pred_frames.sort_values(by=["notated_path", "onset_s"])["label"],
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )


def confusion_matrix(gold_corpus, corpus, classes=None):
    _check_corpus_comparison(gold_corpus, corpus)

    gold_frames = as_frame_comparison(gold_corpus, corpus)
    pred_frames = corpus.data_resources["frames_predictions"]

    return sklearn.metrics.confusion_matrix(
        gold_frames.sort_values(by=["notated_path", "onset_s"])["label"],
        pred_frames.sort_values(by=["notated_path", "onset_s"])["label"],
        labels=classes,
        normalize="true",
    )


def segment_error_rate(gold_corpus, corpus):
    _check_corpus_comparison(gold_corpus, corpus)

    gold_df = gold_corpus.dataset
    pred_df = corpus.dataset

    gold_df["seqid"] = str(gold_df["sequence"]) + str(gold_df["annotation"])
    pred_df["seqid"] = str(pred_df["sequence"]) + str(pred_df["annotation"])

    gold_sequences = gold_df.groupby("seqid")
    pred_sequences = pred_df.groupby("seqid")

    ser = []
    for seqid, gold_seq in gold_sequences:
        pred_seq = pred_sequences.get_group(seqid)

        notated_path = gold_seq["notated_path"].unique()[0]

        ser.append({
            "notated_path": notated_path,
            "seqid": seqid,
            "ser": 1.0 - Levenshtein.ratio(gold_seq.label.values, pred_seq.label.values)
            })

    ser = pd.DataFrame(ser)

    return ser
