# Author: Nathan Trouvain at 28/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pathlib
import logging
import numpy as np

from .base import Annotator
from .commons import init_esn_model
from ..transforms.synesn import SynESNTransform

logger = logging.getLogger("canapy")


class SynAnnotator(Annotator):
    def __init__(self, config, transforms_output_directory):
        self.config = config
        self.transforms = SynESNTransform()
        self.transforms_output_directory = transforms_output_directory
        self.rpy_model = self.initialize()

    def initialize(self):
        return init_esn_model(
            self.config.model.syn,
            self.config.transforms.audio.n_mfcc,
            self.config.transforms.audio.audio_features,
            self.config.misc.seed,
        )

    def fit(self, corpus):
        corpus = self.transforms(
            corpus,
            purpose="training",
            output_directory=self.transforms_output_directory,
        )

        # load data
        df = corpus.dataset
        spec_paths = corpus.data_resources["syn_mfcc"]

        df["seqid"] = df["sequence"].astype(str) + df["annotation"].astype(str)

        sampling_rate = self.config.transforms.audio.sampling_rate
        hop_length = self.config.transforms.audio.hop_length

        df["onset_spec"] = np.round(df["onset_s"] * sampling_rate / hop_length).astype(int)
        df["offset_spec"] = np.round(df["offset_s"] * sampling_rate / hop_length).astype(int)

        df = df.query("train")

        n_classes = len(df["label"].unique())

        train_mfcc = []
        train_labels = []
        for seqid in df["seqid"].unique():
            seq_annots = df.query("seqid == @seqid")

            notated_audio = pathlib.Path(seq_annots["notated_path"].unique()[0]).name
            notated_spec = spec_paths.query("notated_path == @notated_audio")["feature_path"].unique()[0]

            seq_end = seq_annots["offset_spec"].iloc[-1]
            mfcc = np.load(notated_spec)

            if seq_end > mfcc.shape[1]:
                logger.warning(
                    f"Found inconsistent sequence length: "
                    f"audio {notated_audio} was converted to "
                    f"{mfcc.shape[1]} timesteps but last annotation is at "
                    f"timestep {seq_end}. Annotation will be trimmed."
                )

            seq_end = min(seq_end, mfcc.shape[1])

            mfcc = mfcc[:, :seq_end]

            # repeat labels along time axis
            repeated_labels = np.zeros((seq_end, n_classes))
            for row in seq_annots.itertuples():
                onset = row.onset_spec
                offset = min(row.offset_spec, seq_end)
                label = row.encoded_label

                repeated_labels[onset:offset] = label

            train_mfcc.append(mfcc.T)
            train_labels.append(repeated_labels)

        # train
        self.rpy_model.fit(train_mfcc, train_labels)

        corpus.dataset.drop(
            ["seqid", "onset_spec", "offset_spec"], axis=1, inplace=True
        )

        self._trained = True

        return self

    def predict(self, corpus):
        corpus = self.transforms(
            corpus,
            purpose="annotation",
            output_directory=self.transforms_output_directory,
        )

    def eval(self, corpus):
        ...
