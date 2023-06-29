# Author: Nathan Trouvain at 29/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import abc


class Model(abc.ABC):

    def fit(self, corpus):
        pass

    def predict(self, corpus):
        pass
