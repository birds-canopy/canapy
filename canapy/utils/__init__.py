# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
from pathlib import Path


def as_path(path_or_none):
    if path_or_none is not None:
        return Path(path_or_none)
    else:
        return path_or_none
