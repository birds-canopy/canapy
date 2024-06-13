# Author: Nathan Trouvain at 18/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
"""Python tool for canary vocalization dataset correction and
Reservoir Computing model training.
"""

import click
import panel as pn

from .app import CanapyDashboard


_annotators = ["syn-esn", "nsyn-esn", "ensemble"]


@click.group(
    name="canapy",
    help="Python tool for birdsong "
    "vocalization dataset correction and "
    "Reservoir Computing annotation models training.",
)
def cli():
    pass


@click.command("dash", help="Launch canapy dashboard.")
@click.option(
    "-d",
    "--data-dir",
    "data_directory",
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing data source with audio files and "
    "annotations files. Replace '-a' and '-s'.",
)
@click.option(
    "-a",
    "--annots-dir",
    "annots_directory",
    type=click.Path(exists=True, file_okay=False),
    help="Annotations directory, containing only annotation files. Use "
    "this option in conjunction with '-s' instead of '-d' "
    "if your annotations and audio files are in "
    "separate directories.",
)
@click.option(
    "-s",
    "--audio-dir",
    "audio_directory",
    type=click.Path(exists=True, file_okay=False),
    help="Audio directory, containing only audio files. Use "
    "this option in conjunction with '-a' instead of '-d' "
    "if your annotations and audio files are in "
    "separate directories.",
)
@click.option(
    "-o",
    "--output-dir",
    "output_directory",
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
    "-p",
    "--port",
    default=9321,
    help="Port use by the Bokeh server. By default, 9321.",
)
@click.option("--annot-format", type=str, default="marron1csv")
@click.option(
    "--audio-ext",
    type=click.Choice([".wav", ".npy"]),
    default=".wav",
    help="Audio data format. Either 'wav' for WAV audio files or 'npy' "
    "for Numpy .npy files storing arrays. If 'npy', sampling rate "
    "should be explicitely specified.",
)
@click.option(
    "--annotators",
    type=click.Choice(_annotators),
    default=_annotators,
    multiple=True,
    help="Select which annotator to run within canapy.",
)
def display_dashboard(
    data_directory=None, annots_directory=None, audio_directory=None, *args, **kwargs
):
    if data_directory is None and (annots_directory is None or audio_directory is None):
        raise NotADirectoryError("If -d is unset, then -a AND -s must be set!")

    if data_directory is not None and (
        annots_directory is not None or audio_directory is not None
    ):
        raise ValueError("If -d is set, then -a AND -s must not be used.")

    if data_directory is not None:
        annots_directory = data_directory
        audio_directory = data_directory

    pn.extension()
    dashboard = CanapyDashboard(audio_directory, annots_directory, *args, **kwargs)
    dashboard.show()
    return 0


cli.add_command(display_dashboard)


if __name__ == "__main__":
    cli()
