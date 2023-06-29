# Author: Nathan Trouvain at 28/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import abc
import pathlib
import pickle

from typing import Any, Optional

import attr

from ..transforms.base import Transform
from ..transforms import SynESNTransform, NSynESNTransform


@attr.define
class Annotator(abc.ABC):
    model: Any = attr.field()
    transform: Transform = attr.field()
    path_on_disk: Optional[pathlib.Path] = attr.field(converter=pathlib.Path)

    @classmethod
    def from_disk(cls, annotator_path):
        annotator_path = pathlib.Path(annotator_path)

        with annotator_path.open("rb") as fp:
            annotator = pickle.load(fp)

        if isinstance(annotator, cls):
            annotator.path_on_disk = annotator_path
            return annotator
        else:
            if "nsyn" in annotator_path.stem:
                transform = NSynESNTransform()
            else:
                transform = SynESNTransform()

            return cls(
                model=annotator, transform=transform, path_on_disk=annotator_path
            )

    def run(
        self,
        corpus=None,
        output_directory=None,
        to_group=False,
        return_truths=False,
        vectors=False,
        models_vectors=None,
        csv_directory: str = None,
        **kwargs
    ):
        corpus = self.transform(
            corpus,
            purpose="annotation",
            redo_annots=False,
            redo_audio=False,
            redo_training=False,
            output_directory=output_directory,
        )

        outputs = self.model.run(
            corpus,
            vectors=vectors,
            to_group=to_group,
            return_truths=return_truths,
        )

        if model == "all":
            if M.name == "ensemble":
                outputs = outs
                break
            else:
                annots = {M.name: outs[0]}
                outputs = (annots, *outs[1:])
        else:
            annots = {M.name: outs[0]}
            outputs = (annots, *outs[1:])

        if csv_directory is not None and not vectors:
            for model_name, annotations in outputs[0].items():
                directory = Path(csv_directory) / model_name
                if not directory.exists():
                    directory.mkdir(parents=True)

                self.to_csv(
                    annotations,
                    directory,
                    model_name=model_name,
                    config=dataset.config,
                )
        elif csv_directory and vectors:
            raise ValueError(
                "Impossible to export vectors to csv. vectors should be False."
            )

        if not hasattr(outputs, "items") and len(outputs[0]) == 1:
            outputs = (outputs[0][list(outputs[0].keys())[0]], *outputs[1:])

        return outputs

    # def to_csv(self, annotations, directory, model_name=None, config=None):
    #     config = config if config is not None else self.dataset.config
    #     model_name = "" if model_name is None else model_name
    #
    #     sr = config.sampling_rate
    #     hop = config.as_fftwindow("hop_length")
    #
    #     songs = list(annotations.keys())
    #
    #     @delayed
    #     def export_one(song, annotation):
    #         seq = group(annotation)
    #
    #         durations = {"start": [], "end": [], "syll": [], "frames": []}
    #         onset = 0.0
    #         for s in seq:
    #             end = onset + (s[1] * (hop / sr))
    #             if s[0] != "SIL":
    #                 durations["syll"].append(s[0])
    #                 durations["end"].append(round(end, 3))
    #                 durations["start"].append(round(onset, 3))
    #                 durations["frames"].append(s[1])
    #             onset = end
    #
    #         df = pd.DataFrame(durations)
    #
    #         file_name = Path(directory) / Path(song)
    #         df.to_csv(file_name.with_suffix(".csv"), index=False)
    #
    #     with Parallel(n_jobs=-1) as parallel:
    #         parallel(
    #             export_one(song, annotations[song])
    #             for song in tqdm(songs, f"Exporting to .csv - {model_name}")
    #         )
