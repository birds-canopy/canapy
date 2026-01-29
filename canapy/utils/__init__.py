from pathlib import Path


def as_path(path_or_none):
    if path_or_none is not None:
        return Path(path_or_none)
    else:
        return path_or_none
