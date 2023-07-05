# Author: Nathan Trouvain at 29/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging
import numpy as np

from .base import Annotator
from .commons.esn import predict_with_esn, init_esn_model
from ..transforms.nsynesn import NSynESNTransform


logger = logging.getLogger("canapy")


class NSynAnnotator(Annotator):
    def __init__(self, config, spec_directory):
        self.config = config
        self.transforms = NSynESNTransform()
        self.spec_directory = spec_directory
        self.rpy_model = self.initialize()

    def initialize(self):
        return init_esn_model(
            self.config.model.nsyn,
            self.config.transforms.audio.n_mfcc,
            self.config.transforms.audio.audio_features,
            self.config.misc.seed,
        )

    def fit(self, corpus):
        corpus = self.transforms(
            corpus,
            purpose="training",
            output_directory=self.spec_directory,
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

    def predict(
        self,
        corpus,
        return_classes=True,
        return_group=False,
        return_raw=False,
        redo_transforms=False,
    ):
        return predict_with_esn(
            self,
            corpus,
            return_classes=return_classes,
            return_group=return_group,
            return_raw=return_raw,
            redo_transforms=redo_transforms,
        )

    def eval(self, corpus):
            pass
