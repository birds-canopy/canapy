# Author: Nathan Trouvain at 27/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from .base import Transform
from .commons.audio import compute_mfcc
from .commons.training import split_train_test


class SynESNTransform(Transform):
    training_data_transforms = [split_train_test]
    training_data_resource_name = ["dataset"]

    audio_transforms = [compute_mfcc]
    audio_resource_names = ["syn_mfcc"]
