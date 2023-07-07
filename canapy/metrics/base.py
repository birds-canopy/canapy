# Author: Nathan Trouvain at 07/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import sklearn

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


def word_error_rate(gold_corpus, corpus):
    ...
