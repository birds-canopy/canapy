# Author: Nathan Trouvain at 29/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import abc


class Annotator(abc.ABC):
    _trained: bool = False

    @property
    def trained(self):
        return self._trained

    def fit(self, corpus):
        raise NotImplementedError

    def predict(self, corpus):
        raise NotImplementedError

    def eval(self, corpus):
        pass
