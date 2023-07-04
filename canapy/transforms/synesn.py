# Author: Nathan Trouvain at 27/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from .base import Transform
from .commons.audio import compute_mfcc
from .commons.training import encode_labels, prepare_dataset_for_training


class SynESNTransform(Transform):
    def __init__(self):
        super().__init__(
            training_data_transforms=[prepare_dataset_for_training, encode_labels],
            audio_transforms=[compute_mfcc],
            audio_resource_names=["syn_mfcc"],
        )
