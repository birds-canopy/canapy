# Author: Nathan Trouvain at 29/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging
import numpy as np

from reservoirpy.nodes import Reservoir, Ridge, ESN

from .base import Model
from ..transforms.nsynesn import NSynESNTransform


logger = logging.getLogger("canapy")


class NSynModel(Model):

    def __init__(self, config, transform_output_directory):
        self.config = config
        self.transform = NSynESNTransform()
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
        df = corpus.data_ressource["mfcc_dataset"]

        n_classes = len(df["label"].unique())

        train_mfcc = []
        train_labels = []

        for row in df.itertuples():

            train_mfcc.append(row.mfcc.T)
            train_labels.append(np.repeat(row.encoded_label.reshape(1,-1), row.mfcc.shape[1], axis=0))

        # train
        self.rpy_model.fit(train_mfcc, train_labels)

        return self

    def predict(self, corpus):
        ...
