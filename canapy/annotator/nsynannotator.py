# Author: Nathan Trouvain at 29/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging
import numpy as np
import math

from .base import Annotator
from .commons.esn import predict_with_esn, init_esn_model
from .commons.postprocess import predictions_to_corpus
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

        error_audio_path = set()
        error_step = 0

        for row in df.itertuples():
            if isinstance(row.mfcc, np.ndarray):
                train_mfcc.append(row.mfcc.T)
                len_mfcc = row.mfcc.shape[1]
            else:
                len_mfcc = math.ceil(
                    (row.offset_s - row.onset_s)
                    * corpus.config.transforms.audio.sampling_rate
                    / corpus.config.transforms.audio.as_fftwindow("hop_length")
                )
                error_step += 1
                error_audio_path.add(row.notated_path)
                train_mfcc.append(np.zeros((len_mfcc, 39)))
            train_labels.append(
                np.repeat(row.encoded_label.reshape(1, -1), len_mfcc, axis=0)
            )
        if len(error_audio_path) != 0:
            str_base = "\n\t"
            logger.error(
                f"{error_step} failure(s) during mfcc transformation (replaced by 0). "
                f"\nConcerned audio(s) are : \n\t{str_base.join(error_audio_path)}"
            )
        # train
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
        notated_paths, cls_preds, raw_preds = predict_with_esn(
            self,
            corpus,
            return_raw=return_raw,
            redo_transforms=redo_transforms,
        )

        config = self.config

        frame_size = config.transforms.audio.as_fftwindow("hop_length")
        sampling_rate = config.transforms.audio.sampling_rate
        time_precision = config.transforms.annots.time_precision
        min_label_duration = config.transforms.annots.min_label_duration
        min_silence_gap = config.transforms.annots.min_silence_gap
        silence_tag = config.transforms.annots.silence_tag
        lonely_labels = config.transforms.annots.lonely_labels

        return predictions_to_corpus(
            notated_paths=notated_paths,
            cls_preds=cls_preds,
            frame_size=frame_size,
            sampling_rate=sampling_rate,
            time_precision=time_precision,
            min_label_duration=min_label_duration,
            min_silence_gap=min_silence_gap,
            silence_tag=silence_tag,
            lonely_labels=lonely_labels,
            config=config,
            raw_preds=raw_preds,
        )

    def eval(self, corpus):
        pass
