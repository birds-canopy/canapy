# Author: Nathan Trouvain at 21/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from tempfile import TemporaryFile

_registry = list()


def tempfile():
    global _registry
    fp = TemporaryFile()
    _registry.append(fp)
    return fp


def close_tempfiles():
    global _registry
    for fp in _registry:
        fp.close()
    _registry = list()
