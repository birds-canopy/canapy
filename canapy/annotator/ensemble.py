# Author: Nathan Trouvain at 07/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging
from collections import OrderedDict

import numpy as np
import scipy.special

from .base import Annotator
from canapy.utils.exceptions import NotTrainedError
from .commons.postprocess import predictions_to_corpus, extract_vocab
from ..corpus import Corpus
from ..timings import seconds_to_audio

logger = logging.getLogger("canapy")


def _check_prediction_lengths(predictions):
    """
    Check if the predictions have consistent lengths.

    Parameters
    ----------
    predictions : list[dict]
        List of prediction dictionaries.

    Raises
    ------
    ValueError
        If the corpora have different numbers of annotations or do not represent annotations for the same corpus.
    ValueError
        If there is a mismatch in the annotation length in the corpora.
    ValueError
        If there is a mismatch in the annotation vocabulary size.

    Returns
    -------
    None
        Meaning all tests have succeeded.

    """
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
    """
    Perform hard voting ensemble on the given corpora.

    Parameters
    ----------
    corpora : list[Corpus]
        List of Corpus objects.
    classes : list, default=None
        List of classes.

    Returns
    -------
    tuple
        A tuple containing the keys and values of the ensemble votes.

    Raises
    ------
    KeyError
        If 'nn_output' is not found in Corpus.data_resources.
    ValueError
        If there is a mismatch in annotation length or vocabulary size in the corpora.

    """
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
    """
    Ensemble Annotator for audio classification, using Echo State Network (ESN).
    'Ensemble' refers to the predicting approach of the annotator, where predictions are made by combining the results
    of others annotators' predictions on the same songs to determine the best one.

    The basic usage of an annotator involves three steps:
        1. Load your annotator (using the 'from_disk' method or by creating a new one)
        2. Train your annotator (using the 'fit' method).
        3. Make predictions with the annotator (using the 'predict' method).



    Parameters
    ----------
    config : object
        The configuration object for the annotator.
    spec_directory : str
        The path to the spectra files.
    mode : str, default='hard_vote'
        Type of vote to determine the best prediction

    Attributes
    ----------
    config : Config (from config)
        The configuration object for the annotator.
    spec_directory : str
        The path to the spectrogram files.
    _mode : str
        Type of vote to determine the best prediction

    Methods
    -------
    fit(corpus)
        Fit the annotator on the provided training corpus.
    predict(corpus, return_raw=False, redo_transforms=False)
        Predict annotations for the given corpus.

    """

    def __init__(self, config, spec_directory=None, mode="hard_vote"):
        """
        Initialization method.

        Parameters
        ----------
        config : object
            The configuration object for the annotator.
        spec_directory : str
            The path to the spectra files.
        mode : str, default='hard_vote'
            Type of vote to determine the best prediction
        """
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
        """
        Get the mode of the annotator.

        Returns
        -------
        str
            The mode of the annotator.

        Example
        -------
            >>> from canapy.annotator.ensemble import Ensemble
            >>> from config import default_config
            >>> # Ensemble and default_config are imported to create a new ensemble annotator
            >>> my_annotator = Ensemble(default_config, "/path/to/spec")
            >>> my_annotator.mode()
            hard_vote

        """
        return self._mode

    def fit(self, corpus):
        """
        Fit the annotator to the given corpus.

        Parameters
        ----------
        corpus : Corpus
            The corpus object used for training the annotator.

        Returns
        -------
        Ensemble
            The trained annotator itself.

        Notes
        ----
        Even though this function returns the trained annotator, the annotator itself is trained.

        The Ensemble annotator does not directly use an ESN. Consequently, the training process involves
        creating the adapted vocabulary for the annotator.

        Example
        -------
            >>> from canapy.annotator.ensemble import Ensemble
            >>> from config import default_config
            >>> # Ensemble and default_config are imported to create a new annotator
            >>> my_annotator = Ensemble(default_config, "/path/to/spec")
            >>> from canapy.corpus import Corpus
            >>> corpus = Corpus.from_directory(audio_directory="/path/to/audio", annots_directory="/path/to/annotation")
            >>> # A new corpus is created to train the annotator
            >>> my_annotator_trained = my_annotator.fit(corpus)
            >>> # The annotator is now trained with the given corpus

        """
        self._vocab = extract_vocab(
            corpus, silence_tag=self.config.transforms.annots.silence_tag
            )

        self._trained = True
        return self

    def predict(self, corpora, return_raw=False, redo_transforms=False):
        """
        Determine the best prediction using predictions from different annotators on the same original corpus

        Parameters
        ----------
        corpora : list[Corpus]
            The corpuses list that have been created by different annotators predictions (using 'return_raw' = True)
        return_raw : bool, optional
            If True, raw annotations are added into the 'data_resources'
            Raw outputs are necessary to train an 'EnsemleAnnotator' on a corpus
        redo_transforms : bool, optional
            If True, redo the transformations on the corpus before predicting.

        Returns
        -------
        Corpus
            The corpus object with the best annotations.

        Note
        ----
        Annotators need to be trained before being able to make predictions

        Example
        -------
            >>> import canapy.annotator as ant
            >>> from config import default_config
            >>> # Annotators and default_config are imported to create a new annotator
            >>> my_annotator = ant.ensemble.Ensemble(default_config)
            >>> from canapy.corpus import Corpus
            >>> corpus = Corpus.from_directory(audio_directory="/path/to/audio", annots_directory="/path/to/annotation")
            >>> # A new corpus is created to train the annotator
            >>> my_annotator_trained = my_annotator.fit(corpus)
            >>> # The annotator is now trained with the given corpus
            >>> syn_annotator = ant.synannotator.SynAnnotator(default_config)
            >>> nsyn_annotator = ant.nsynannotator.NSynAnnotator(default_config)
            >>> # Two more annotators are created
            >>> syn_annotator.fit(corpus)
            >>> nsyn_annotator.fit(corpus)
            >>> # And trained on a corpus
            >>> unannotated_corpus =  Corpus.from_directory(audio_directory="/path/to/audio")
            >>> # A new corpus is created with potentially unannotated songs
            >>> annotated_raw_syn_corpus = syn_annotator.predict(unannotated_corpus, return_raw=True)
            >>> annotated_raw_nsyn_corpus =  nsyn_annotator.predict(unannotated_corpus, return_raw=True)
            >>> # Those corpus contains predictions for each annotator and raw output of their model
            >>> annotated_corpus = my_annotator.predict([annotated_raw_syn_corpus, annotated_raw_nsyn_corpus])
            >>> # 'annotated_corpus' contains the annotation made by the Ensemble annotator
            >>> annotated_corpus.to_disk("/path/to/new/annotations")
            >>> # Those annotations can be stored on the disk using Corpus 'to_disk' method

        """
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

        hop_length = config.transforms.audio.hop_length
        sampling_rate = config.transforms.audio.sampling_rate
        time_precision = config.transforms.annots.time_precision
        min_label_duration = config.transforms.annots.min_label_duration
        min_silence_gap = config.transforms.annots.min_silence_gap
        silence_tag = config.transforms.annots.silence_tag
        lonely_labels = config.transforms.annots.lonely_labels

        frame_size = seconds_to_audio(hop_length, sampling_rate)

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
