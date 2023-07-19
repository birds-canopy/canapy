# Author: Nathan Trouvain at 18/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
"""Python tool for canary vocalization dataset correction and
Reservoir Computing model training.
"""
import argparse

import panel as pn

from .app import CanapyDashboard


parser = argparse.ArgumentParser(prog="canapy", description=__doc__)

parser.add_argument(
    "data_directory",
    type=str,
    help="Directories containing data source with .wav audio "
    "file and .csv annotations files, and repertoire "
    "directory.",
)
parser.add_argument(
    "output_directory", type=str, help="Output directory of models and checkpoints."
)
parser.add_argument(
    "-c", "--config_path", type=str,
    )

parser.add_argument(
    "--port",
    "-p",
    type=int,
    nargs=1,
    default=9321,
    help="Port use by the Bokeh server. By default, 9321.",
)
parser.add_argument(
    "--annot_format", type=str, default="marron1csv"
    )
parser.add_argument(
    "--audio_ext",
    type=str,
    choices=[".wav", ".npy"],
    default=".wav",
    help="Audio data format. Either 'wav' for WAV audio files or 'npy' "
    "for Numpy .npy files storing arrays. If 'npy', sampling rate "
    "should be explicitely specified.",
)
parser.add_argument(
    "--annotators",
    action="append",
    nargs="+",
    default=["syn-esn", "nsyn-esn", "ensemble"]
)


def display_dashboard(**kwargs):

    pn.extension()
    dashboard = CanapyDashboard(**kwargs)
    dashboard.show()

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    # print("Starting...")
    display_dashboard(**vars(args))
