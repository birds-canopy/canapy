# Author: Nathan Trouvain at 11/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
def fetch_misclassified_samples(self):
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
                np.sum(np.array(scores) > self.config.correction.min_segment_proportion_agreement)
                == 0
        ):
            bad_ones[rep.Index] = detected_labels

    self.misclass_indexes = bad_ones
    self.misclass_df = self.corpus.df.iloc[list(self.misclass_indexes.keys())]
    self.misclassified_counts_plot()


def load_repertoire(selected_samples):
    with Parallel() as parallel:
        specs = parallel(
            delayed(load_sample)(s) for s in selected_samples.itertuples()
        )
    return specs
