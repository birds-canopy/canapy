# Author: Nathan Trouvain at 28/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging
import numpy as np

from .base import Annotator
from .commons.esn import init_esn_model, predict_with_esn
from .commons.mfccs import load_mfccs_and_repeat_labels
from ..transforms.synesn import SynESNTransform

logger = logging.getLogger("canapy")


class SynAnnotator(Annotator):
    def __init__(self, config, spec_directory):
        self.config = config
        self.transforms = SynESNTransform()
        self.spec_directory = spec_directory
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
            output_directory=self.spec_directory,
        )

        _, _, train_mfcc, train_labels = load_mfccs_and_repeat_labels(
            corpus, purpose="training"
        )

        self.rpy_model.fit(train_mfcc, train_labels)

        self._trained = True

        self._vocab = np.sort(corpus.dataset["label"].unique()).tolist()

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
        ...
