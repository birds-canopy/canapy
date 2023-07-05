# Author: Nathan Trouvain at 28/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging
import numpy as np

from .base import Annotator
from .commons.esn import init_esn_model
from .commons.mfccs import load_mfccs_and_repeat_labels, load_mfccs_for_annotation
from .commons.postprocess import group_frames, maximum_a_posteriori
from .commons.exceptions import NotTrainedError
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

        # train reservoirpy model
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
        if not self.trained:
            raise NotTrainedError(
                "Call .fit on annotated data (Corpus) before calling " ".predict."
            )

        corpus = self.transforms(
            corpus,
            purpose="annotation",
            output_directory=self.spec_directory,
        )

        notated_paths, mfccs = load_mfccs_for_annotation(corpus)

        raw_preds = self.rpy_model.run(mfccs)

        cls_preds = None
        group_preds = None
        if return_classes or return_group:
            cls_preds = []
            for y in raw_preds:
                y_map = maximum_a_posteriori(y, classes=self.vocab)
                cls_preds.append(y_map)

            if return_group:
                group_preds = []
                for y_cls in cls_preds:
                    seq = group_frames(y_cls)
                    group_preds.append(seq)

        if not return_raw:
            raw_preds = None
        if not return_classes:
            cls_preds = None
        if not return_group:
            group_preds = None

        return notated_paths, group_preds, cls_preds, raw_preds

    def eval(self, corpus):
        ...
