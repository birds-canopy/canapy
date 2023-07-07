# Author: Nathan Trouvain at 07/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from datetime import datetime
from pathlib import Path


def as_path(path_or_none):
    if path_or_none is not None:
        return Path(path_or_none)
    else:
        return path_or_none


#
#
# def name_directory_now():
#     return datetime.now().strftime("%Y%m%d%H%M%S-%s")
