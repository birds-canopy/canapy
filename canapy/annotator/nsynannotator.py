# Author: Nathan Trouvain at 29/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging
import numpy as np

from reservoirpy.nodes import Reservoir, Ridge, ESN

from .base import Annotator
from .commons import init_esn_model
from ..transforms.nsynesn import NSynESNTransform


logger = logging.getLogger("canapy")


class NSynAnnotator(Annotator):
    def __init__(self, config, transforms_output_directory):
        self.config = config
        self.transforms = NSynESNTransform()
        self.transforms_output_directory = transforms_output_directory
        self.rpy_model = self.initialize()

    def initialize(self):
        return init_esn_model(
            self.config.model.nsyn,
            self.config.transforms.audio.n_mfcc,
            self.config.transforms.audio.audio_features,
            self.config.misc.seed,
        )
        # scalings = []
        # if "mfcc" in self.input_list:
        #     iss = np.ones((self.input_dim,)) * self.config.iss
        #     scalings.append(iss)
        # if "delta" in self.input_list:
        #     isd = np.ones((self.input_dim,)) * self.config.isd
        #     scalings.append(isd)
        # if "delta2" in self.input_list:
        #     isd2 = np.ones((self.input_dim,)) * self.config.isd2
        #     scalings.append(isd2)
        #
        # input_scaling = np.concatenate(scalings, axis=0)
        # bias_scaling = self.config.iss
        #
        # reservoir = Reservoir(
        #     self.config.N,
        #     sr=self.config.sr,
        #     lr=self.config.leak,
        #     input_scaling=input_scaling,
        #     bias_scaling=bias_scaling,
        #     W=fast_spectral_initialization,
        #     seed=self.seed,
        # )
        #
        # readout = Ridge(ridge=1e-8)
        #
        # return ESN(reservoir=reservoir, readout=readout, workers=-1, backend="loky")

    def fit(self, corpus):
        corpus = self.transforms(
            corpus,
            purpose="training",
            output_directory=self.transforms_output_directory,
        )

        # load data
        df = corpus.data_resources["mfcc_dataset"]

        train_mfcc = []
        train_labels = []

        for row in df.itertuples():
            if isinstance(row.mfcc, np.ndarray):
                train_mfcc.append(row.mfcc.T)
                train_labels.append(
                    np.repeat(row.encoded_label.reshape(1, -1), row.mfcc.shape[1], axis=0)
                )

        # train
        self.rpy_model.fit(train_mfcc, train_labels)

        self._trained = True

        return self

    def predict(self, corpus):
        corpus = self.transforms(
            corpus,
            purpose="annotation",
            output_directory=self.transforms_output_directory,
        )
