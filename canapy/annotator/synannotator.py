# Author: Nathan Trouvain at 28/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging
import numpy as np

from .base import Annotator
from .commons.esn import init_esn_model, predict_with_esn
from .commons.mfccs import load_mfccs_and_repeat_labels
from .commons.postprocess import predictions_to_corpus, extract_vocab
from ..transforms.synesn import SynESNTransform
from ..timings import seconds_to_audio

from config import default_config


logger = logging.getLogger("canapy")


class SynAnnotator(Annotator):
    """
    Syntaxic Annotator for audio classification, using Echo State Network (ESN).
    'Syntaxic' refers to the training approach of the annotator where the model is trained on songs where phrases
    follow a specific order chosen by the canary.

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

    Attributes
    ----------
    config : Config (from config)
        The configuration object for the annotator.
    transforms : NSynESNTransform (from canapy.transforms.nsynesn)
        The NSynESNTransform object used for transforming the data.
    spec_directory : str
        The path to the spectrogram files.
    rpy_model : ESN (from reservoirpy)
        The Echo State Network (ESN) model.

    Methods
    -------
    fit(corpus)
        Fit the annotator on the provided training corpus.
    predict(corpus, return_raw=False, redo_transforms=False)
        Predict annotations for the given corpus.
    """

    def __init__(
        self,
        config=default_config,
    ):  # spec_directory):
        """
        Initialization method.

        Parameters
        ----------
        config : Config (from config)
            Contains parameters of the corpus.
        spec_directory
            The path to the spectrogram files.

        """
        self.config = config
        self.transforms = SynESNTransform()
        # self.spec_directory = spec_directory
        self.rpy_model = self.initialize()

    def initialize(self):
        """
        Initialize annotator's model

        Returns
        -------
        ESN (from reservoirpy)
            The Echo State Network (ESN) model.

        Note
        ----
        This method should not be called manually

        """
        return init_esn_model(
            self.config.model.syn,
            self.config.transforms.audio.n_mfcc,
            self.config.transforms.audio.audio_features,
            self.config.misc.seed,
        )

    def fit(self, corpus):
        """
        Fit the annotator to the given corpus.

        Parameters
        ----------
        corpus : Corpus
            The corpus object used for training the annotator.

        Returns
        -------
        SynAnnotator
            The trained annotator itself.

        Note
        ----
        Even though this function returns the trained annotator, the annotator itself is trained.

        Example
        -------
            >>> from canapy.annotator.synannotator import SynAnnotator
            >>> from config import default_config
            >>> # SynAnnotator and default_config are imported to create a new annotator
            >>> my_annotator = SynAnnotator(default_config, "/path/to/spec")
            >>> from canapy.corpus import Corpus
            >>> corpus = Corpus.from_directory(audio_directory="/path/to/audio", annots_directory="/path/to/annotation")
            >>> # A new corpus is created to train the annotator
            >>> my_annotator_trained = my_annotator.fit(corpus)
            >>> # The annotator is now trained with the given corpus

        """
        corpus = self.transforms(
            corpus,
            purpose="training",
            output_directory=corpus.spec_directory,
        )

        _, _, train_mfcc, train_labels = load_mfccs_and_repeat_labels(
            corpus, purpose="training"
        )

        self.rpy_model.fit(train_mfcc, train_labels)

        self._trained = True

        self._vocab = extract_vocab(
            corpus, silence_tag=self.config.transforms.annots.silence_tag
        )

        return self

    def predict(self, corpus, return_raw=False, redo_transforms=False):
        """
        Predict annotations for the given corpus.

        Parameters
        ----------
        corpus : Corpus
            The corpus object for which to predict annotations.
        return_raw : bool, optional
            If True, raw annotations are added into the 'data_resources'
            Raw outputs are necessary to train an 'EnsemleAnnotator' on a corpus
        redo_transforms : bool, optional
            If True, redo the transformations on the corpus before predicting.

        Returns
        -------
        Corpus
            The corpus object with predicted annotations.

        Note
        ----
        Annotators need to be trained before being able to make predictions

        Example
        -------
            >>> from canapy.annotator.synannotator import SynAnnotator
            >>> from config import default_config
            >>> # SynAnnotator and default_config are imported to create a new annotator
            >>> my_annotator = SynAnnotator(default_config, "/path/to/spec")
            >>> from canapy.corpus import Corpus
            >>> corpus = Corpus.from_directory(audio_directory="/path/to/audio", annots_directory="/path/to/annotation")
            >>> # A new corpus is created to train the annotator
            >>> my_annotator_trained = my_annotator.fit(corpus)
            >>> # The annotator is now trained with the given corpus
            >>> unannotated_corpus =  Corpus.from_directory(audio_directory="/path/to/audio")
            >>> # A new corpus iis created with potentially unannotated songs
            >>> annotated_corpus = my_annotator_trained.predict(unannotated_corpus)
            >>> # 'annotated_corpus' contains the annotation made by the annotator
            >>> annotated_corpus.to_disk("/path/to/new/annotations")
            >>> # Those annotations can be stored on the disk using Corpus 'to_disk' method

        """
        notated_paths, cls_preds, raw_preds = predict_with_esn(
            self,
            corpus,
            return_raw=return_raw,
            redo_transforms=redo_transforms,
        )

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
            raw_preds=raw_preds,
        )
