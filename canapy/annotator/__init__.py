import logging

from .base import Annotator
from .synannotator import SynAnnotator
from .nsynannotator import NSynAnnotator
from .ensemble import Ensemble


logger = logging.getLogger("canapy")


class _Registry:

    def __init__(self):

        self._registry = {
            "syn-esn": SynAnnotator,
            "nsyn-esn": NSynAnnotator,
            "ensemble": Ensemble,
        }

    def __getitem__(self, item):
        return self._registry[item]

    def register_annotator(self, name, cls):
        if name in cls:
            logger.warning(f"'{name}' is already registered. Skipping.")
            return
        self._registry[name] = cls


registry = _Registry()


def get_annotator(name):
    return registry[name]


def get_annotator_names():
    return sorted(registry._registry.keys())
