"""Python tool for canary vocalization dataset correction and
Reservoir Computing model training.
"""
import argparse

import panel as pn

from .dashboard.app import Canapy


parser = argparse.ArgumentParser(prog="canapy", description=__doc__)

parser.add_argument(
    "data",
    type=str,
    help="Directories containing data source with .wav audio "
    "file and .csv annotations files, and repertoire "
    "directory.",
)
parser.add_argument(
    "output", type=str, help="Output directory of models and checkpoints."
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
    "--audioformat",
    "-a",
    type=str,
    choices=["wav", "npy"],
    default="wav",
    help="Audio data format. Either 'wav' for WAV audio files or 'npy' "
    "for Numpy .npy files storing arrays. If 'npy', sampling rate "
    "should be explicitely specified.",
)
parser.add_argument(
    "--rate",
    "-r",
    type=float,
    help="Sampling rate of audio files. If WAV files are used, can be "
    "inferred from audio files. If a config file is used, config "
    "sampling rate will be used and this parameter overwritten.",
)


def display_dashboard(**args):

    pn.extension()
    dashboard = Canapy(**args)
    dashboard.serve()

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    print("Starting...")
    display_dashboard(**vars(args))
