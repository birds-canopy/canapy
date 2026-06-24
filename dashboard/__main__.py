# Author: Nathan Trouvain at 18/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: BSD-3-Clause
# Copyright: Nathan Trouvain
"""Python tool for canary vocalization dataset correction and
Reservoir Computing model training.
"""

import click
import panel as pn
from pathlib import Path
import logging
import sys
import warnings

warnings.filterwarnings("ignore")

from .app import CanapyDashboard
from .controler import Controler
from canapy.utils.tempstorage import close_tempfiles

class _NoWarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.WARNING

def _suppress_warnings():
    """
    Suppress WARNING-level log messages for the entire process lifetime.

    Two layers:
    1. Patch existing loggers (panel/bokeh/tornado created during import).
    2. Monkey-patch logging.getLogger so every logger created later also
       receives the filter automatically (Bokeh server, panel extensions…).
    """
    _filter = _NoWarningFilter()
    _fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def _apply(lg):
        if not any(isinstance(f, _NoWarningFilter) for f in lg.filters):
            lg.addFilter(_filter)
        for h in lg.handlers:
            if not any(isinstance(f, _NoWarningFilter) for f in h.filters):
                h.addFilter(_filter)

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        root.removeHandler(h)
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(_fmt)
    _handler.addFilter(_filter)
    root.addHandler(_handler)
    root.addFilter(_filter)

    # Patch all loggers already created during imports
    for _lg in logging.Logger.manager.loggerDict.values():
        if isinstance(_lg, logging.Logger):
            _apply(_lg)

    # Raise threshold on the noisiest libraries
    for _name in ("panel", "bokeh", "tornado", "param", "asyncio", "websocket"):
        logging.getLogger(_name).setLevel(logging.ERROR)

    # Monkey-patch logging.getLogger so every future logger is also covered
    _orig_getLogger = logging.getLogger
    def _patched_getLogger(name=None):
        lg = _orig_getLogger(name)
        _apply(lg)
        return lg
    logging.getLogger = _patched_getLogger

_suppress_warnings()
logger = logging.getLogger("canapy-cli")

_annotators = ["syn-esn", "nsyn-esn", "ensemble"]


@click.group(
    name="canapy",
    help="Python tool for birdsong vocalization dataset correction and Reservoir Computing annotation models training.",
)
def cli():
    pass



@click.command("dash", help="Launch canapy dashboard.")
@click.option(
    "-d",
    "--data-dir",
    "data_directory",
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing data source with audio files and annotations files.",
)
@click.option(
    "-a",
    "--annots-dir",
    "annots_directory",
    type=click.Path(exists=True, file_okay=False),
    help="Annotations directory.",
)
@click.option(
    "-s",
    "--audio-dir",
    "audio_directory",
    type=click.Path(exists=True, file_okay=False),
    help="Audio directory.",
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
    help="Directory where preprocessed audio data will be stored.",
)
@click.option(
    "-c",
    "--config-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to a configuration file in TOML/YAML format.",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to a directory containing pre-trained models to load.",
)
@click.option("-p", "--port", default=9321, help="Port use by the Bokeh server.")
@click.option("--annot-format", type=str, default="marron1csv")
@click.option("--audio-ext", type=click.Choice([".wav", ".npy"]), default=".wav")
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
        annots_directory = None
        audio_directory = None

    if data_directory is not None and (
        annots_directory is not None or audio_directory is not None
    ):
        raise ValueError("If -d is set, then -a AND -s must not be used.")

    if data_directory is not None:
        annots_directory = data_directory
        audio_directory = data_directory

    # Expert direct-launch: when data directories are passed on the command line,
    # we deduce the working directory (so the Home selection step is skipped).
    # It is the parent of the output directory when given, else the current dir.
    working_directory = None
    if audio_directory is not None and annots_directory is not None:
        out = kwargs.get("output_directory")
        working_directory = Path(out).resolve().parent if out else Path.cwd()

    # Otherwise, if Canapy is launched from a directory that already looks like a
    # working directory (it contains both canapy_output/ and config/), reuse it
    # directly and skip the working-directory selection on the Home page.
    if working_directory is None:
        cwd = Path.cwd()
        if (cwd / "canapy_output").is_dir() and (cwd / "config").is_dir():
            working_directory = cwd

    pn.extension(notifications=True)

    dashboard = CanapyDashboard(
        annots_directory=annots_directory,
        audio_directory=audio_directory,
        working_directory=working_directory,
        *args,
        **kwargs
    )
    dashboard.show()
    return 0



@click.command("train", help="Train models on a dataset and export them directly.")
@click.option(
    "-d", "--data-dir", "data_directory",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing audio and annotations (labels)."
)
@click.option(
    "-o", "--output-dir", "output_directory",
    required=True,
    type=click.Path(),
    help="Output directory where models will be saved."
)
@click.option(
    "-c", "--config-path",
    type=click.Path(exists=True),
    help="Optional path to TOML config."
)
def train_cli(data_directory, output_directory, config_path):
    logger.info("Initializing Headless Training...")
    
    output_directory = Path(output_directory)
    spec_directory = output_directory / "spectrograms"
    

    ctrl = Controler(
        annots_directory=data_directory,
        audio_directory=data_directory,
        output_directory=output_directory,
        spec_directory=spec_directory,
        config_path=config_path,
        dashboard=None,  
        annotators=_annotators
    )
    
    if not ctrl.is_ready:
        logger.error("Controller failed to initialize properly. Check input paths.")
        return

    try:
        logger.info(f"Training models: {_annotators}")
        for annot_name in _annotators:
            logger.info(f"Processing {annot_name}...")
            ctrl.train(annot_name, export=True, save=True)
            
        logger.info(f"All models trained and exported to {output_directory / 'model'}")

    except Exception as e:
        logger.critical(f"Training failed: {e}")
        raise e
    finally:
        close_tempfiles()



@click.command("annotate", help="Annotate audio files using trained models.")
@click.option(
    "-d", "--data-dir", "audio_directory",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing raw audio files to annotate."
)
@click.option(
    "-m", "--models-dir", "models_directory",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing trained models.",
)
@click.option(
    "-o", "--output-dir", "output_directory",
    required=True,
    type=click.Path(),
    help="Output directory for annotation CSV files."
)
def annotate_cli(audio_directory, models_directory, output_directory):
    logger.info("Initializing Headless Annotation...")
    
    output_directory = Path(output_directory)
    spec_directory = output_directory / "spectrograms"
    

    models_path = Path(models_directory)
    

    ctrl = Controler(
        annots_directory=None,
        audio_directory=None,
        output_directory=output_directory,
        spec_directory=spec_directory,
        config_path=None,          
        model_root=models_path,    
        dashboard=None,
        annotators=_annotators
    )

    try:
        corpus = ctrl.load_external_corpus(audio_directory)
        logger.info(f"Loaded {len(corpus.dataset)} audio segments from {audio_directory}")

        logger.info("Running inference...")
        results = ctrl.annotate_external(corpus, use_in_memory=False)

        if "ensemble" in results:
            logger.info("Exporting 'ensemble' predictions...")
            ctrl.export_predictions(results["ensemble"], output_directory)
        else:
            logger.warning("'ensemble' model not found. Exporting individual model predictions.")
            for name, pred_corpus in results.items():
                sub_dir = output_directory / name
                ctrl.export_predictions(pred_corpus, sub_dir)
        
        logger.info(f"Annotations exported to {output_directory}")

    except Exception as e:
        logger.critical(f"Annotation failed: {e}")
        raise e
    finally:
        close_tempfiles()


cli.add_command(display_dashboard)
cli.add_command(train_cli)
cli.add_command(annotate_cli)


if __name__ == "__main__":
    cli()