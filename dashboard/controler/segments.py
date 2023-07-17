# Author: Nathan Trouvain at 11/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pandas as pd

from canapy.timings import seconds_to_frames, seconds_to_audio
from canapy.metrics.utils import as_frame_comparison


def fetch_misclassified_samples(gold_corpus, predictions,  hop_length, sampling_rate, min_segment_proportion_agreement, silence_tag="SIL"):

    gold_frames = as_frame_comparison(gold_corpus, predictions[0])

    pred_frames = {f"pred_{n}": p.data_resource["frame_predictions"]["label"] for n, p in predictions.items()}
    pred_frames = pd.DataFrame(pred_frames)

    frames = pd.concat([gold_frames, pred_frames], axis="columns")

    gold_df = gold_corpus.dataset.copy()

    gold_df["onset_frame"] = seconds_to_frames(gold_df["onset_s"], seconds_to_audio(hop_length, sampling_rate), sampling_rate)
    gold_df["offset_frame"] = seconds_to_frames(gold_df["onset_s"], seconds_to_audio(hop_length, sampling_rate),sampling_rate)

    for annot in gold_df.itertuples():
        if annot.label == silence_tag:
            continue

        notated_path = annot["notated_path"].unique()[0]

        onset_frame = annot.onset_frames
        offset_frame = annot.offset_frames

        annot_frames = frames.query("notated_path == @notated_path").loc[onset_frame:offset_frame]

        comp = annot_frames.apply()

    annots = self.corpus.annotations
    df = self.corpus.df
    bad_ones = {}
    for rep in tqdm(
            df[df["syll"] != "SIL"].itertuples(), "Fetching all misclassified samples"
            ):
        song = rep.wave
        start_y = self.config.frames(rep.start)
        end_y = self.config.frames(rep.end)

        scores = []
        detected_labels = []
        for m in annots.keys():
            if m != "truth":
                preds = np.array(annots[m][song][start_y:end_y])
                truth = np.array(annots["truth"][song][start_y:end_y])
                scores.append(np.sum(preds == truth) / len(truth))
                label_freqs = np.unique(preds, return_counts=True)
                if len(label_freqs[1]) != 0:
                    detected_labels.append(label_freqs[0][label_freqs[1].argmax()])

        if (
                np.sum(np.array(scores) > min_segment_proportion_agreement)
                == 0
        ):
            bad_ones[rep.Index] = detected_labels

    self.misclass_indexes = bad_ones
    self.misclass_df = self.corpus.df.iloc[list(self.misclass_indexes.keys())]
    self.misclassified_counts_plot()
