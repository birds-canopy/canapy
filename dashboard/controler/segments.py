# Author: Nathan Trouvain at 11/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from typing import Dict
import pandas as pd
import numpy as np

from canapy.timings import seconds_to_frames, seconds_to_audio
from canapy.metrics.utils import as_frame_comparison
from canapy.corpus import Corpus

def fetch_misclassified_samples(
    gold_corpus: Corpus,
    predictions: Dict[str, Corpus],
    hop_length: float,
    sampling_rate: int,
    min_segment_proportion_agreement: float,
    silence_tag: str = "SIL",
):

    # Retrieve all data as framed data (one annotation per timestep)
    gold_frames = as_frame_comparison(gold_corpus, predictions[list(predictions.keys())[0]])
    gold_frames = gold_frames.sort_values(by=["notated_path", "onset_s"])

    pred_frames = {
        f"pred_{n}": p.data_resources["frames_predictions"].sort_values(by=["notated_path", "onset_s"])["label"]
        for n, p in predictions.items()
    }

    pred_frames = pd.DataFrame(pred_frames)
    frames = pd.concat([gold_frames, pred_frames], axis="columns")

    gold_df = gold_corpus.dataset.copy()

    gold_df["onset_frame"] = seconds_to_frames(
        gold_df["onset_s"], seconds_to_audio(hop_length, sampling_rate), sampling_rate
    )
    gold_df["offset_frame"] = seconds_to_frames(
        gold_df["offset_s"], seconds_to_audio(hop_length, sampling_rate), sampling_rate
    )

    bad_ones = {}
    for notated_path, annots in gold_df.groupby("notated_path"):
        for one_annot in annots.itertuples():
            if one_annot.label == silence_tag:
                continue

            onset_frame = one_annot.onset_frame
            offset_frame = one_annot.offset_frame

            annot_frames = frames.query("notated_path == @notated_path").loc[
                onset_frame:offset_frame
            ]

            preds = annot_frames.filter(regex="pred_.*")

            # Get: all labels found by models, for every frame of current annotation,
            # the counts of unique labels and the accuracy score of these predictions
            # against the annotated frames true value (from gold corpus).
            counts = preds.apply(
                lambda col: pd.Series(
                    np.unique(col, return_counts=True) + ((col == annot_frames["label"]).sum() / len(preds), ),
                    index=["label", "count", "score"]
                    ),
                axis="rows").T

            # If all models achieve less than min_segment_proportion_agreement accuracy,
            # then this segment should be considered as problematic
            if (counts["score"] < min_segment_proportion_agreement).all():
                bad_ones[one_annot.Index] = {r.Index: r.label[np.argmax(r.count)] for r in counts.itertuples()}

    bad_ones = pd.DataFrame.from_dict(bad_ones, orient="index")

    misclassified = pd.concat([gold_df.loc[bad_ones.index], bad_ones], axis="columns")

    return misclassified
