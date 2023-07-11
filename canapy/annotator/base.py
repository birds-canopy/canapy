# Author: Nathan Trouvain at 29/06/2023 <nathan.trouvain<at>inria.fr>
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
    _trained: bool = False
    _vocab: list = list()

    @classmethod
    def from_disk(cls, path, config=default_config, spec_directory=None):

        from . import get_annotator

        with Path(path).open('rb') as file:
            try:
                loaded_annot = pickle.load(file)
            except ModuleNotFoundError:
                loaded_annot = _CompatModelUnpickler(file).load()

        if isinstance(loaded_annot, Annotator):
            if spec_directory is not None:
                loaded_annot.spec_directory = spec_directory
            return loaded_annot

        elif isinstance(loaded_annot, _Compat):
            if spec_directory is None:
                logger.warning("Annotator do not have a spec_directory")
            annotator_type = get_annotator("nsyn-esn" if len(path) > 3 and path[-4] == 'n' else "syn-esn")
            new_annotator = annotator_type(config, spec_directory)
            new_annotator._trained = True
            new_annotator.rpy_model = loaded_annot.esn
            new_annotator._vocab = loaded_annot.vocab
            return new_annotator
        else:
            raise NotImplementedError

    def to_disk(self, path):
        with Path(path).open("wb+") as file:
            pickle.dump(self, file)

    @property
    def trained(self):
        return self._trained

    @property
    def vocab(self):
        return self._vocab

    def fit(self, corpus):
        raise NotImplementedError

    def predict(self, corpus):
        raise NotImplementedError

    def eval(self, corpus):
        pass
