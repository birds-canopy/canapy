"""
Utilities to load canapy annotators from old discontinued
versions.
"""

import pickle
from types import SimpleNamespace


class _CompatModelUnpickler(pickle._Unpickler):
    """
    Utility class to unpickle annotators trained using
    previous versions of canapy.
    """

    def __init__(self, fp):
        super().__init__(fp)
        self._magic_classes = {}

    def find_class(self, module, name):
        if "canapy" in module.split(".") and module.split(".")[1] in [
            "dataset",
            "processor",
            "sequence",
            "model",
            "config",
        ]:
            return _Compat
        else:
            return super().find_class(module, name)


class _Compat(SimpleNamespace):
    """Compatibility wrapper from pickled object.
    Expose all attributes without the help of the original
    class.
    """

    def __setitem__(self, key, value):
        pass
