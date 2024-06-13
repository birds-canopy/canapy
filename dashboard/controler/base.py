# Author: Nathan Trouvain at 10/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging
from pathlib import Path
from typing import Dict, Optional, List, Mapping
from collections import defaultdict

import attr
import joblib
import pandas as pd
import panel

from canapy.corpus import Corpus
from canapy.annotator import get_annotator, get_annotator_names, Annotator
from canapy.correction import Corrector
from canapy.metrics import (
    sklearn_confusion_matrix,
    sklearn_classification_report,
    segment_error_rate,
)
from canapy.plots import plot_segment_melspectrogram
from canapy.utils import as_path
from canapy.annotator.commons.postprocess import extract_vocab
from canapy.transforms.commons.training import split_train_test
from canapy.utils.tempstorage import close_tempfiles

from .segments import fetch_misclassified_samples
from .corpusutils import mark_whole_corpus_as_train, query_split


logger = logging.getLogger("canapy")


def _sort_annotators(annotators: List):
    # Ensemble should always be last
    if "ensemble" in annotators:
        annotators.remove("ensemble")
        sorted_annots = ["ensemble"]
    else:
        sorted_annots = []
    sorted_annots = annotators.copy() + sorted_annots
    return sorted_annots


@attr.define
class Controler:
    annots_directory: Path = attr.field(converter=as_path)
    audio_directory: Path = attr.field(converter=as_path)
    output_directory: Path = attr.field(converter=as_path)
    spec_directory: Path = attr.field(converter=as_path)
    config_path: Optional[Path] = attr.field(converter=as_path)
    dashboard: panel.viewable.Viewer = attr.field()
    annot_format: str = attr.field(default="marron1csv")
    audio_ext: str = attr.field(default=".wav")
    annotators: List[str] = attr.field(default=["syn-esn", "nsyn-esn", "ensemble"], converter=list)

    corpus: Optional[Corpus] = attr.field(default=None)
    config: Optional[Mapping] = attr.field(default=None)
    corrector: Optional[Corrector] = attr.field(default=None)
    _iter: Optional[int] = attr.field(alias="_iter", default=1)
    _step: Optional[str] = attr.field(alias="_step", default="train")
    _annotators: Optional[Dict[str, Annotator]] = attr.field(
        alias="_annotators", default=dict()
    )
    _pred_corpora: Optional[Dict[str, Dict]] = attr.field(
        alias="_pred_corpora", default=dict()
    )
    _metrics_store: Optional[Dict[str, Dict]] = attr.field(
        alias="_metrics_store", default=dict()
    )
    _correction_store: Optional[Dict] = attr.field(
        alias="_correctoin_store", default=dict()
    )
    _classes: Optional[List[str]] = attr.field(alias="_classes", default=None)

    def __attrs_post_init__(self):
        self.corpus = Corpus.from_directory(
            audio_directory=self.audio_directory,
            spec_directory=self.spec_directory,
            annots_directory=self.annots_directory,
            config_path=self.config_path,
            annot_format=self.annot_format,
            audio_ext=self.audio_ext,
        )

        self.config = self.corpus.config
        self.corrector = Corrector(
            self.output_directory / "checkpoints", [{"class": dict(), "annot": dict()}]
        )
        self._iter = 1
        self._step = "train"

        self.annotators = _sort_annotators(self.annotators)

        self.initialize_output()
        self.initialize_models()
        self.initialize_annots()

        self.corpus = split_train_test(self.corpus, redo=True)

    @property
    def iter(self):
        return self._iter

    @property
    def step(self):
        return self._step

    @property
    def metrics(self):
        return {k: v for k, v in self._metrics_store.items() if k in ["train", "test"]}

    @property
    def misclassified_segments(self):
        return self._metrics_store["misclass"]

    @property
    def classes(self):
        return self._classes

    def initialize_models(self):
        for name in self.annotators:
            try:
                annot_cls = get_annotator(name)
                annot_obj = annot_cls(self.config)
                self._annotators[name] = annot_obj

                if name == "ensemble":
                    # Ensemble does not really require training but will grasp the
                    # labels from .fit
                    annot_obj.fit(self.corpus)

            except KeyError:
                logger.warning(
                    f"Annotator model '{name}' not found in registry. Skipping."
                )

        if len(self._annotators) == 0:
            logger.error(
                f"No Annotator provided! All requested Annotator models "
                f"({self.annotators}) are probably invalid, or the list is"
                f" empty. Valid annotators are: {get_annotator_names()}."
            )

    def initialize_output(self, output=None):
        try:
            self.output_directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.critical(e)

        try:
            (self.output_directory / "spectro").mkdir(parents=True, exist_ok=True)

            logger.info(
                f"All results will be stored in {self.output_directory}. "
                f"Audio transformations (spectrograms, MFCC...) will be "
                f"stored in "
                f"{self.output_directory / 'spectro'}."
            )
        except OSError as e:
            logger.critical(e)

    def initialize_annots(self):
        self._pred_corpora = dict(all=dict(), train=dict(), test=dict())
        logger.info("Annotators predictions (re)initialized.")
        self._metrics_store = dict(
            all=defaultdict(dict), train=defaultdict(dict), test=defaultdict(dict)
        )
        logger.info("Metrics and scores (re)initialized.")
        self._correction_store = dict({"class": dict(), "annot": dict()})
        logger.info("Current corrections (re)initialized.")

    def next_iter(self):
        self.initialize_models()
        self.initialize_annots()

        self.corpus = split_train_test(self.corpus, redo=True)

        self._iter += 1
        logger.info(f"Current training iteration : {self.iter}")

    def checkpoint(self):
        try:
            ckpt_dir = self.output_directory / "model" / str(self.iter)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            for name, model in self._annotators.items():
                model.to_disk(ckpt_dir / name)
                logger.info(f"Annotator checkpoint created at {ckpt_dir / name}.")
        except OSError as e:
            logger.critical("Failed to create checkpoint.")
            logger.critical(e)

    def upload_corrections(self, corrections, target):
        self._correction_store[target].update(corrections)
        logger.info(f"Uploaded corrections {target}: {corrections} to store.")

    def apply_corrections(self):
        class_corrections = self._correction_store["class"]
        annot_corrections = self._correction_store["annot"]
        self.corpus = self.corrector.correct(
            self.corpus,
            class_corrections=class_corrections,
            annot_corrections=annot_corrections,
            checkpoint=True,
        )
        logger.info(
            f"Applied corrections on {self.corpus}:"
            f"\nClass merge:\n{class_corrections}"
            f"\nAnnotation correction:\n{annot_corrections}"
        )

    def export_corpus(self):
        try:
            export_dir = self.output_directory / "corrected_annotations"
            export_dir.mkdir(parents=True, exist_ok=True)
            self.corpus.to_directory(str(export_dir))
        except OSError as e:
            logger.critical("Failed to create export directory for annotations:")
            logger.critical(e)

    def next_step(self, export=False):
        if self.step == "train":
            self._step = "eval"
            self.get_metrics()
            logger.info('Setting current dashboard to "eval".')

        elif self.step == "eval" and not export:
            self._step = "train"
            self.checkpoint()
            self.apply_corrections()
            self.next_iter()
            logger.info("Setting current dashboard to 'train'.")

        elif export:
            self._step = "export"
            self.checkpoint()
            self.apply_corrections()
            self.next_iter()
            self.export_corpus()
            logger.info("Setting current dashboard to 'export'.")

        self.dashboard.switch_panel()

    def train(self, annotator_name, export=False, save=False):
        annotator = self._annotators[annotator_name]

        if export:
            logger.info("Preparing for exportation.")
            corpus = mark_whole_corpus_as_train(self.corpus)
            logger.info(f"Training Annotator '{annotator_name}'.")
            annotator.fit(corpus)
            logger.info(f"Done!")
        else:
            annotator.fit(self.corpus)

        if save:
            try:
                (self.output_directory / "models").mkdir(parents=True, exist_ok=True)
                annotator.to_disk(self.output_directory / "model" / annotator_name)
                logger.info(
                    f"Saved Annotator '{annotator_name}' in file "
                    f"{self.output_directory / 'model' / annotator_name}."
                )
            except OSError as e:
                logger.critical(e)

    def annotate(self, annotator_name, split="all"):
        if annotator_name == "ensemble":
            # Get all previous predictions
            corpora = [c for c in self._pred_corpora[split].values()]
        else:
            corpora = query_split(self.corpus, split)

        annotator = self._annotators[annotator_name]
        pred_corpus = annotator.predict(
            corpora, return_raw="ensemble" in self.annotators
        )

        self._pred_corpora[split][annotator_name] = pred_corpus

    def train_syn(self):
        self.train("syn-esn")

    def train_syn_export(self):
        self.train("syn-esn", export=True, save=True)

    def train_nsyn(self):
        self.train("nsyn-esn")

    def train_nsyn_export(self):
        self.train("nsyn-esn", export=True, save=True)

    def annotate_syn(self):
        self.annotate("syn-esn", split="train")
        self.annotate("syn-esn", split="test")

    def annotate_nsyn(self):
        self.annotate("nsyn-esn", split="train")
        self.annotate("nsyn-esn", split="test")

    def annotate_ensemble(self):
        self.annotate("ensemble", split="train")
        self.annotate("ensemble", split="test")

    def get_metrics(self):
        bad_ones = []
        for split in ["train", "test"]:
            predictions = self._pred_corpora[split]

            for annot_name, pred_corpus in predictions.items():
                logger.info(
                    f"Calculating metrics for split: "
                    f"{split} | annotator: {annot_name}."
                )

                gold_corpus = query_split(self.corpus, split)
                classes = extract_vocab(
                    self.corpus, silence_tag=self.config.transforms.annots.silence_tag
                )

                self._classes = classes

                cm = sklearn_confusion_matrix(gold_corpus, pred_corpus, classes=classes)
                report = sklearn_classification_report(
                    gold_corpus, pred_corpus, classes=classes
                )
                ser = segment_error_rate(gold_corpus, pred_corpus)

                def float_format(x):
                    return f"{x:.3f}"

                logger.info(
                    f"Report <{split}|{annot_name}>:"
                    f"\n{pd.DataFrame(report).to_string(float_format=float_format)}"
                )
                logger.info(
                    f"Segment error rate: {ser['ser'].mean():.3f} (mean) "
                    f"± {ser['ser'].std():.3f} (std)"
                )

                self._metrics_store[split]["cm"][annot_name] = cm
                self._metrics_store[split]["report"][annot_name] = report
                self._metrics_store[split]["ser"][annot_name] = ser

            sampling_rate = self.config.transforms.audio.sampling_rate
            hop_length = self.config.transforms.audio.hop_length
            silence_tag = self.config.transforms.annots.silence_tag
            min_segment_proportion_agreement = (
                self.config.correction.min_segment_proportion_agreement
            )

            gold_corpus = query_split(self.corpus, split)

            misclassified = fetch_misclassified_samples(
                gold_corpus,
                predictions,
                hop_length,
                sampling_rate,
                min_segment_proportion_agreement,
                silence_tag=silence_tag,
            )
            bad_ones.append(misclassified)

        misclassified = pd.concat(bad_ones)

        logger.info(f"Found {len(misclassified)} potentially misclassified samples.")

        self._metrics_store["misclass"] = misclassified

    def load_repertoire(self, selected_samples):
        logger.info(
            f"Loading repertoire samples for label(s): {selected_samples.label.unique()}"
        )
        audio_conf = self.config.transforms.audio
        with joblib.Parallel(backend="multiprocessing", n_jobs=-1) as parallel:
            specs = parallel(
                joblib.delayed(plot_segment_melspectrogram)(
                    s.notated_path,
                    s.onset_s,
                    s.offset_s,
                    sampling_rate=audio_conf.sampling_rate,
                    hop_length=audio_conf.hop_length,
                    n_fft=audio_conf.n_fft,
                    win_length=audio_conf.win_length,
                    fmin=audio_conf.fmin,
                    fmax=audio_conf.fmax,
                    return_audio=True,
                )
                for s in selected_samples.itertuples()
            )
        return specs

    def stop_app(self):
        close_tempfiles()
        self.dashboard.stop()
