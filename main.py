"""Python tool for canary vocalization dataset correction and
Reservoir Computing model training.
"""
import argparse

import panel as pn

from canapy.dashboard.app import Canapy


parser = argparse.ArgumentParser(prog="canapy", description=__doc__)

parser.add_argument(
    "data",
    type=str,
    default="../canapy-test/data/data",
    help="Directories containing data source with .wav audio "
    "file and .csv annotations files, and repertoire "
    "directory.",
)
parser.add_argument(
    "output",
    type=str,
    default="../canapy-test/data/output",
    help="Output directory of models and checkpoints.",
)
parser.add_argument(
    "--port",
    "-p",
    type=int,
    nargs=1,
    help="Port use by the Bokeh server. By default, 9321.",
)


def display_dashboard(**args):

    pn.extension()

    dashboard = Canapy(**args)
    dashboard.serve()

    return 0


if __name__ == "__main__":
    # args = parser.parse_args()

    display_dashboard(
        data="../canapy-test/data/data", output="../canapy-test/data/output"
    )
    # display_dashboard(data="./tests/data/test-r6-short", output="./tests/data/output")
