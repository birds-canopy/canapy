# Authors: Nathan Trouvain at 29/06/2023 <nathan.trouvain<at>inria.fr>
#          Vincent Gardies at 10/07/2023 <vincent.gardies<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain

import abc
import pickle
import logging

from pathlib import Path
from ..config import default_config
from .commons.compat import _Compat, _CompatModelUnpickler

logger = logging.getLogger("canapy")


class Annotator(abc.ABC):
    """
    Base abstract class for annotators

    The basic usage of an annotator involves three steps:
        1. Load your annotator (using the 'from_disk' method or by creating a new one).

        2. Train your annotator (using the 'fit' method).

        3. Make predictions with the annotator (using the 'predict' method).

    """

    _trained: bool = False
    _vocab: list = list()

    @classmethod
    def from_disk(cls, path, config=default_config, spec_directory=None):
        """
        Load an annotator object from disk.

        Parameters
        ----------
        path : str
            Path to the file containing the annotator object.
        config : Config, default=default_config
            The configuration object for the annotator.
        spec_directory : str, optional
            The directory containing the spectrogram files.

        Returns
        -------
        Annotator
            The loaded annotator object.

        Raises
        ------
        NotImplementedError
            If the loaded annotator object is of an unsupported type.

        Note
        ----
        Annotators that have been created and saved in the disk with the former version of canapy are also loadable.
        Make sure to give the configuration of the model if you had made some changes, and a spec_directory.

        Examples
        --------
            >>> from canapy.annotator.base import Annotator

            >>> my_annotator = Annotator.from_disk("/path/to/annotator")
            >>> # The annotator saved in the 'saved_annotator' file is now loaded into 'my_annotator'

            >>> from canapy.config import default_config
            >>> my_old_config = default_config  # Get the config that the model was saved with
            >>> my_annotator_old = Annotator.from_disk("/path/to/old/model",
            >>>     config=my_old_config  # If no changes were made to the config, default_config will work
            >>>     spec_directory="/path/to/spec"
            >>> )
            >>> # The annotator saved in the 'syn' file is loaded into my_annotator_old

        """

        from . import get_annotator

        with Path(path).open("rb") as file:
            try:
                loaded_annot = pickle.load(file)
            except ModuleNotFoundError:
                loaded_annot = _CompatModelUnpickler(file).load()

        # This case concerns Annotators made with this version of Canapy
        if isinstance(loaded_annot, Annotator):
            if spec_directory is not None:
                loaded_annot.spec_directory = spec_directory
            return loaded_annot

        # This case concerns Annotators made with a previous version of Canapy
        elif isinstance(loaded_annot, _Compat):
            # The name of the file is used to determine the type of annotator.
            # If you need to change the model's name, make sure it ends with its type (syn / nsyn).
            annotator_type = get_annotator(
                "nsyn-esn" if len(path) > 3 and path[-4] == "n" else "syn-esn"
            )

            new_annotator = annotator_type(config, spec_directory)
            new_annotator._trained = True
            new_annotator.rpy_model = loaded_annot.esn
            new_annotator._vocab = loaded_annot.vocab
            return new_annotator
        else:
            raise NotImplementedError

    def to_disk(self, path):
        """
        Save the annotator object to disk.

        Parameters
        ----------
        path : str or Path
            Path to the file where the annotator object will be saved.

        Example
        -------
            >>> from canapy.annotator.synannotator import SynAnnotator
            >>> from canapy.config import default_config
            >>> my_annotator = SynAnnotator(default_config, "/home/vincent/Documents/data_canary/spec")
            >>> my_annotator.to_disk("/home/vincent/Documents/data_canary/annotators/my_annotator")
            >>> # The annotator is now saved in the disk

        """
        with Path(path).open("wb+") as file:
            pickle.dump(self, file)

    @property
    def trained(self):
        """
        Property indicating if the annotator is trained.

        Returns
        -------
        bool
            True if the annotator is trained, False otherwise.

        Example
        -------
            >>> from canapy.annotator.ensemble import Ensemble
            >>> from canapy.config import default_config
            >>> my_ensemble_annotator = Ensemble (default_config, None)
            >>> # For example, we are using an Ensemble annotator
            >>> print(f"My annotator is trained : {my_ensemble_annotator.trained()}")
            My annotator is trained : False
            >>> from canapy.corpus import Corpus
            >>> corpus = Corpus.from_directory(audio_directory="/path/to/audio", annots_directory="/path/to/annotation")
            >>> my_ensemble_annotator.fit(corpus)
            >>> # We create a corpus from some files and then train the annotator on it
            >>> print(f"My annotator is trained : {my_ensemble_annotator.trained()}")
            My annotator is trained : True

        """
        return self._trained

    @property
    def vocab(self):
        """
        Property containing the vocabulary of the annotator.

        Returns
        -------
        list
            The vocabulary of the annotator.

        Note
        ----
        The vocabulary of the annotator is determined during the 'fit' method.
        Trying to access the vocabulary of an annotator before training it is useless
        """
        return self._vocab

    def fit(self, corpus):
        """
        Fit the annotator to the given corpus.

        Parameters
        ----------
        corpus : Corpus
            The corpus object used for training the annotator.

        Returns
        -------
        Annotator
            The trained annotator itself.

        Note
        ----
        Even though this function returns the trained annotator, the annotator itself is trained.

        Example
        -------
            >>> from canapy.annotator.nsynannotator import NSynAnnotator
            >>> from canapy.config import default_config
            >>> my_annotator = SynAnnotator(default_config, "/path/to/spec")
            >>> # A not-syntaxic annotator is used in this example
            >>> from canapy.corpus import Corpus
            >>> corpus = Corpus.from_directory(audio_directory="/path/to/audio", annots_directory="/path/to/annotation")
            >>> my_annotator_trained = my_annotator.fit(corpus)
            >>> # The annotator is now trained with the given corpus
            >>> my_annotator_trained == my_annotator
            True

        """
        raise NotImplementedError

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

        Examples
        --------
        Find examples of use in annotators subclasses

        """
        raise NotImplementedError
