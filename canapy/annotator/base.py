# Author: Nathan Trouvain at 29/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import abc


class Annotator(abc.ABC):
    _trained: bool = False
    _vocab: list = list()

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
