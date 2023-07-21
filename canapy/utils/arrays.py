# Author: Nathan Trouvain at 21/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging
from collections import UserDict
from typing import List

import numpy as np

from .tempstorage import tempfile


MAX_ARRAY_SIZE = 1e8

logger = logging.getLogger("canapy")


class _StructuredContainer(UserDict):

    def __init__(self, arrays, dtype, shapes):
        self._arrays = arrays
        self._dtype = dtype
        self._shapes = shapes

        super().__init__()

    def __getitem__(self, item):
        shape = self._shapes[item]
        return self._arrays[item].reshape(*shape)

    def __setitem__(self, key, value):
        raise NotImplementedError(f"Can't set item of {self}.")

    def __contains__(self, item):
        return item in self.keys()

    def items(self):
        for k in self._dtype.fields.keys():
            yield k, self[k]

    def keys(self):
        return self._dtype.fields.keys()

    def values(self):
        for k in self._dtype.fields.keys():
            yield self[k]


def to_structured(dict_of_arrays):
    dtype = np.dtype(
        [(k, v.dtype, v.shape) for k, v in dict_of_arrays.items()]
    )
    shapes = {k: v.shape for k, v in dict_of_arrays.items()}

    if total_size(dict_of_arrays.values()) > MAX_ARRAY_SIZE:
        logger.info(f"Saving data resource as memory map in temporary file: "
                    f"arrays exceed {MAX_ARRAY_SIZE} bytes."
        )
        array = np.memmap(tempfile(), dtype=dtype, shape=(1, ))
    else:
        array = np.zeros(1, dtype=dtype)

    for k, v in dict_of_arrays.items():
        array[k] = v

    return _StructuredContainer(array, dtype, shapes)


def total_size(arrays: List[np.ndarray]):
    return sum([arr.nbytes for arr in arrays])
