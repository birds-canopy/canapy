# Author: Nathan Trouvain at 07/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging
from collections import OrderedDict

import numpy as np
import scipy.special

from .base import Annotator
from canapy.utils.exceptions import NotTrainedError
from .commons.postprocess import predictions_to_corpus
from ..corpus import Corpus

logger = logging.getLogger("canapy")


def _check_prediction_lengths(predictions):
    if not all([set(predictions[0].keys()) == set(p.keys()) for p in predictions]):
        raise ValueError(
            "Corpora have different number of annotations, or do not represent "
            "annotations for the same corpus."
        )

    keys = predictions[0].keys()
    if not all(
        [
            all([predictions[0][k].shape[0] == p[k].shape[0] for p in predictions])
            for k in keys
        ]
    ):
        raise ValueError("Mismatch in annotation length in Corpora. Can't do Ensemble.")

    keys = predictions[0].keys()
    if not all(
        [
            all([predictions[0][k].shape[1] == p[k].shape[1] for p in predictions])
            for k in keys
        ]
    ):
        raise ValueError(
            "Mismatch in annotation vocabulary size. Can't do hard vote "
            "Ensemble on annotations with different number of classes."
        )


def hard_vote(corpora, classes=None):
    raw_predictions = []
    for corpus in corpora:
        if "nn_output" not in corpus.data_resources:
            raise KeyError(
                "'nn_output' not found in Corpus.data_resources. Try using "
                "Annotator.predict with 'return_raw=True' on some data first, and feed "
                "the result to Ensemble."
            )

        raw_output = corpus.data_resources["nn_output"]
        raw_predictions.append(raw_output)

    _check_prediction_lengths(raw_predictions)

    notated_paths = sorted(raw_predictions[0].keys())

    votes = OrderedDict()
    for notated_path in notated_paths:
        maxs, argmaxs = [], []
        for prediction in raw_predictions:
            if classes is None:
                n_classes = prediction[notated_path].shape[1]
                classes = list(range(n_classes))

            preds = scipy.special.softmax(prediction[notated_path], axis=1)

            max_response = np.max(preds, axis=1)
            argmax_response = np.argmax(preds, axis=1)

            maxs.append(max_response)
            argmaxs.append(argmax_response)

        maxs = np.asarray(maxs)
        argmaxs = np.asarray(argmaxs)

        vote = [classes[argmaxs[c, i]] for i, c in enumerate(np.argmax(maxs, axis=0))]

        votes[notated_path] = vote

    return list(votes.keys()), list(votes.values())


MODE_REGISTRY = dict(
    hard_vote=hard_vote,
)


class Ensemble(Annotator):
    def __init__(self, config, spec_directory=None, mode="hard_vote"):
        if mode not in MODE_REGISTRY:
            raise ValueError(
                f"'mode' should be one of {', '.join(list(MODE_REGISTRY.keys()))}."
            )

        self.vote_mode = MODE_REGISTRY[mode]
        self.config = config
        self.spec_directory = spec_directory

        self._mode = mode

    @property
    def mode(self):
        return self._mode

    def fit(self, corpus):
        self._vocab = np.sort(corpus.dataset["label"].unique()).tolist()
        self._trained = True
        return self

    def predict(
        self,
        corpora,
        return_raw=False,
        redo_transforms=False,
    ):
        if not self.trained:
            raise NotTrainedError(
                "Call .fit on annotated data (Corpus) before calling " ".predict."
            )

        if isinstance(corpora, Corpus):
            logger.warning(
                "Only one Corpus was provided to Ensemble.predict. "
                "Ensemble can only predict based on several Corpus notated by "
                "different annotators."
            )
            return corpora

        notated_paths, cls_preds = self.vote_mode(corpora, classes=self.vocab)

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
            raw_preds=None,
        )
