# Author: Nathan Trouvain at 27/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import abc
import pathlib

from typing import Callable, Mapping, Iterable, Optional


class Transform(abc.ABC):
    def __init__(
        self,
        annots_transforms=None,
        training_data_transforms=None,
        training_data_resource_name=None,
        audio_transforms=None,
        audio_resource_names=None,
    ):
        self.annots_transforms = (
            list() if annots_transforms is None else annots_transforms
        )
        self.training_data_transforms = (
            list() if training_data_transforms is None else training_data_transforms
        )
        self.training_data_resource_name = (
            list()
            if training_data_resource_name is None
            else training_data_resource_name
        )
        self.audio_transforms = list() if audio_transforms is None else audio_transforms
        self.audio_resource_names = (
            list() if audio_resource_names is None else audio_resource_names
        )

    def __call__(
        self,
        corpus,
        purpose="training",
        redo_annots=False,
        redo_audio=False,
        redo_training=False,
        output_directory=None,
        **kwargs
    ):
        if output_directory is not None:
            output_dir = pathlib.Path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = None

        if purpose == "training":
            corpus = self.transform_annots(corpus, redo=redo_annots)
            corpus = self.transform_audio(
                corpus, redo=redo_audio, output_directory=output_dir
            )
            corpus = self.transform_training_data(corpus, redo=redo_training)

        elif purpose == "annotation":
            corpus = self.transform_annots(corpus, redo=redo_annots)
            corpus = self.transform_audio(
                corpus, redo=redo_audio, output_directory=output_dir
            )

        return corpus

    def transform_annots(self, corpus, redo=False):
        for transform in self.annots_transforms:
            corpus = transform(corpus, redo=redo)
        return corpus

    def transform_training_data(self, corpus, redo=False):
        if len(self.training_data_resource_name) == 0:
            resource_names = [None] * len(self.training_data_transforms)
        else:
            resource_names = self.training_data_resource_name

        for transform, resource_name in zip(
            self.training_data_transforms, resource_names
        ):
            corpus = transform(corpus, resource_name=resource_name, redo=redo)
        return corpus

    def transform_audio(self, corpus, redo=False, output_directory=None):
        if len(self.audio_resource_names) == 0:
            resource_names = [None] * len(self.audio_transforms)
        else:
            resource_names = self.audio_resource_names

        for transform, resource_name in zip(self.audio_transforms, resource_names):
            corpus = transform(
                corpus,
                output_directory=output_directory,
                resource_name=resource_name,
                redo=redo,
            )
        return corpus
