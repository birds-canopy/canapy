from pathlib import Path

import click
from .. import Corpus
from ..annotator import Annotator, Ensemble


@click.command("annotate")
@click.option(
    "-s",
    "--audio-dir",
    "audio_directory",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Audio directory, containing only audio files. Use "
    "this option in conjunction with '-a' instead of '-d' "
    "if your annotations and audio files are in "
    "separate directories.",
)
@click.option(
    "-m",
    "--annotator",
    "annotator_paths",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a trained canapy annotator. May be "
    "used several times to load several annotators.",
)
@click.option(
    "-o",
    "--output-dir",
    "output_directory",
    required=True,
    type=click.Path(),
    help="Output directory of models and dataset checkpoints.",
)
@click.option(
    "--spec-dir",
    "spec_directory",
    type=click.Path(),
    help="Directory where preprocessed audio data will be stored. "
    "By default, preprocessed data will be stored in the same "
    "directory as '--output-directory', in the 'spectrograms/' subdir.",
)
@click.option(
    "-c",
    "--config-path",
    type=click.Path(exists=True, dir_okay=False, allow_dash=True),
    help="Path to a configuration file in TOML format. May come from standard "
    "input (using dash).",
)
@click.option(
    "--annot-format",
    type=str,
    default="marron1csv",
    help="Annotation format. See crowsetta documentation to "
    "learn about supported annotation formats. "
    "(https://crowsetta.readthedocs.io/en/latest/). "
    "Default is 'marron1csv', canapy own annotation "
    "format, based on CSV files.",
)
@click.option(
    "--audio-ext",
    type=click.Choice([".wav", ".npy"]),
    default=".wav",
    help="Audio data format. Either 'wav' for WAV audio files or 'npy' "
    "for Numpy .npy files storing arrays. If 'npy', sampling rate "
    "should be explicitely specified.",
)
@click.option(
    "--ensemble",
    is_flag=True,
    default=False,
    help="If True and several annotators has been provided, "
    "will output hard voted annotations between annotators "
    "in addition to single-annotators annotations.",
)
def annotate(
    audio_directory,
    annotator_paths,
    output_directory,
    spec_directory,
    config_path,
    annot_format,
    audio_ext,
    ensemble,
):
    corpus = Corpus.from_directory(
        audio_directory=audio_directory,
        audio_ext=audio_ext,
        annot_format=annot_format,
        config_path=config_path,
        spec_directory=spec_directory,
    )

    if len(annotator_paths) == 1:
        annotator = Annotator.from_disk(annotator_paths[0])
        annotations: Corpus = annotator.predict(corpus)
        annotations.to_directory(output_directory)
        return

    corpora = {}
    annotators = []
    # Create sub directories if annotating with several models
    for path in annotator_paths:
        annotator = Annotator.from_disk(path)
        # If ensembling is requested, need to return predictions
        # from each class neuron ("raw predictions")
        annotations = annotator.predict(corpus, return_raw=ensemble)
        model_name = Path(path).stem
        corpora[model_name] = annotations
        annotators.append(annotator)

    if ensemble:
        ensemble = Ensemble.from_annotators(*annotators)
        corpora["ensemble"] = ensemble.predict(list(corpora.values()))

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    for name, annotations in corpora.items():
        subdir = Path(output_directory) / name
        subdir.mkdir()
        annotations.to_directory(subdir)


if __name__ == "__main__":
    annotate()
