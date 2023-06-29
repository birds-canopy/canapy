# Author: Nathan Trouvain at 29/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging
import numpy as np

from reservoirpy.nodes import Reservoir, Ridge, ESN

from .base import Model
from ..transforms.synesn import SynESNTransform

logger = logging.getLogger("canapy")


class NSynModel(Model):

    def __init__(self, config, transform_output_directory):
        self.config = config
        self.transform = SynESNTransform()
        self.transform_output_directory = transform_output_directory
        self.rpy_model = self.initialize()

    def initialize(self):
        scalings = []
        if "mfcc" in self.input_list:
            iss = np.ones((self.input_dim,)) * self.config.iss
            scalings.append(iss)
        if "delta" in self.input_list:
            isd = np.ones((self.input_dim,)) * self.config.isd
            scalings.append(isd)
        if "delta2" in self.input_list:
            isd2 = np.ones((self.input_dim,)) * self.config.isd2
            scalings.append(isd2)

        input_scaling = np.concatenate(scalings, axis=0)
        bias_scaling = self.config.iss

        reservoir = Reservoir(
            self.config.N,
            sr=self.config.sr,
            lr=self.config.leak,
            input_scaling=input_scaling,
            bias_scaling=bias_scaling,
            W=fast_spectral_initialization,
            seed=self.seed,
            )

        readout = Ridge(ridge=1e-8)

        return ESN(reservoir=reservoir, readout=readout, workers=-1, backend="loky")

    def fit(self, corpus):

        corpus = self.transform(
            corpus, purpose="training", output_directory=self.transform_output_directory
            )

        # load data
        df = corpus.dataset.query("train == True")
        df["seqid"] = df["sequence"].astype(str) + df["annotation"].astype(str)

        sampling_rate = self.config.transforms.audio.sampling_rate
        hop_length = self.config.transforms.audio.mfcc.hop_length

        df["onset_spec"] = np.round(df["onset_s"] * sampling_rate / hop_length)
        df["offset_spec"] = np.round(df["offset_s"] * sampling_rate / hop_length)

        n_classes = len(df["label"].unique())

        train_mfcc = []
        train_labels = []
        for seqid in df["seqid"].unique():

            seq_annots = df.query("seqid == @seqid")
            notated_audio = seq_annots.loc[0, "notated_path"]

            seq_end = df.loc["offset_spec", -1]
            mfcc = np.load(notated_audio)

            if seq_end > mfcc.shape[1]:
                logger.warning(f"Found inconsistent sequence length: "
                               f"audio {notated_audio} was converted to "
                               f"{mfcc.shape[1]} timesteps but last annotation is at "
                               f"timestep {seq_end}. Annotation will be trimmed.")

            seq_end = min(seq_end, mfcc.shape[1])

            mfcc = mfcc[:, :seq_end]

            # repeat labels along time axis
            repeated_labels = np.zeros((seq_end, n_classes))
            for row in seq_annots.itertuples():
                onset = row.onset_spec
                offset = min(row.offset_spec, seq_end)
                label = row.encoded_label

                repeated_labels[onset: offset] = label

            train_mfcc.append(mfcc)
            train_labels.append(repeated_labels)

        # train
        self.rpy_model.fit(train_mfcc, train_labels)

        return self

    def predict(self, corpus):
        ...
