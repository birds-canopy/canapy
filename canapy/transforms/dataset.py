# Author: Nathan Trouvain at 28/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from .base import Transform
from .commons.annots import (
    sort_annotations,
    tag_silences,
    remove_short_labels,
    merge_labels,
)
from .commons.training import split_train_test


class CorpusDatasetTransform(Transform):
    annots_transforms = [
        sort_annotations,
        tag_silences,
        sort_annotations,
        merge_labels,
        sort_annotations,
        remove_short_labels,
        sort_annotations,
    ]

    training_data_transforms = [split_train_test]
    training_data_resource_name = ["dataset"]
