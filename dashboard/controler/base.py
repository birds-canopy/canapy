# Author: Nathan Trouvain at 10/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import itertools
import gc
import logging
from pathlib import Path
from typing import Union, Optional, List

import attr
import numpy as np
import panel

from sklearn import metrics
from tqdm import tqdm
from canapy import plots
from canapy.corpus import Corpus
from canapy.annotator import get_annotator, Annotator, get_annotator_names
from canapy.correction import Corrector
from canapy.metrics import confusion_matrix, classification_report, segment_error_rate

from .segments import load_repertoire, fetch_misclassified_samples
from .corpusutils import mark_whole_corpus_as_train, query_split


logger = logging.getLogger("canapy")


def _sort_annotators(annotators: List):
    # Ensemble should always be last
    sorted_annots = []
    if "ensemble" in annotators:
        annotators.remove("ensemble")
        sorted_annots = ["ensemble"]
    else:
        sorted_annots = []
    sorted_annots = annotators.copy() + sorted_annots
    return sorted_annots


@attr.define
class Controler(object):
    data_directory: Path = attr.field(converter=Path)
    output_directory: Path = attr.field(converter=Path)
    config_path: Path = attr.field(converter=Path)
    dashboard: panel.viewable.Viewable = attr.field()
    audio_rate: Optional[Union[int, float]] = attr.field()
    annot_format: str = attr.field(default="marron1csv")
    audio_ext: str = attr.field(default=".wav")
    annotators: List = attr.field(default=["syn-esn", "nsyn-esn", "ensemble"])

    def __attrs_post_init__(self):
        self.corpus = Corpus.from_directory(
            audio_directory=self.data_directory,
            spec_directory=self.output_directory / "spectro",
            annots_directory=self.data_directory,
            config_path=self.config_path,
            annot_format=self.annot_format,
            audio_ext=self.audio_ext,
        )

        self.config = self.corpus.config
        self.corrector = Corrector(self.output_directory / "checkpoints", list())
        self._iter = 1
        self._step = "train"

        self.annotators = _sort_annotators(self.annotators)

        self._annotators = dict()
        self._pred_corpora = None
        self._metrics_store = None

        self.initialize_output()
        self.initialize_models()
        self.initialize_annots()

    @property
    def iter(self):
        return self._iter

    @property
    def step(self):
        return self._step

    def initialize_models(self):
        for name in self.annotators:
            try:
                annot_cls = get_annotator(name)
                annot_obj = annot_cls(self.config, self.output_directory / "spectro")
                self._annotators[name] = annot_obj

                if name == "ensemble":
                    # Ensemble does not really require training but will grasp the
                    # labels from .fit
                    annot_obj.fit(self.corpus)

            except KeyError:
                logger.warning(
                    f"Annotator model '{name}' not found in registry. " f"Skipping."
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
        self._metrics_store = dict(all=dict(), train=dict(), test=dict())
        logger.info("Annotators predictions (re)initialized.")
        logger.info("Metrics and scores (re)initialized.")

    def next_iter(self):
        self.initialize_models()
        self.initialize_annots()
        self._iter += 1
        logger.info(f"Current training iteration : {self.iter}")

    def checkpoint(self):
        try:
            (self.output_directory / "model" / str(self.iter)).mkdir(
                parents=True, exist_ok=True
            )
            for name, model in self._annotators:
                model.to_disk(self.output_directory / str(self.iter) / "model" / name)
        except OSError as e:
            logger.critical("Failed to create checkpoint.")
            logger.critical(e)

    def next_step(self, export=False):
        if self.step == "train":
            self._step = "eval"
            print("Fetching metrics...")
            self.get_metrics()
            print('Setting current dashboard to "eval".')

        elif self.step == "eval" and not export:
            self._step = "train"
            print("Updating corpus...")
            self.corpus.update(iteration=self.iter, corrections=self.new_corrections)
            print("Saving corpus chekpoint...")
            self.checkpoint()
            print('Setting current dashboard to "train".')

            self.next_iter()

        elif export:
            self._step = "export"

            print("Updating corpus...")
            self.corpus.update(iteration=self.iter, corrections=self.new_corrections)
            print("Saving corpus chekpoint...")
            self.corpus.checkpoint(self.output)
            print('Setting current dashboard to "export".')
            self.next_iter()

        # del_temp()
        # print("All temporary files deleted.")
        # print("Switching panel...")
        self.dashboard.switch_panel()
        print("Done.")

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
            corpora = [
                query_split(c, split) for c in self._pred_corpora[split].values()
            ]
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
        self.annotate("syn-esn", split="all")
        # syn, truth, vect = self.annotator.run(
        #     model="syn", vectors=True, return_truths=True
        # )
        # self.corpus.annotations["syn"] = create_memmap(syn, "syn_annots")
        # self.corpus.annotations["truth"] = create_memmap(truth, "truth_annots")
        # self.syn_annots_vectors = create_memmap(vect, "syn_vect")
        #
        # del syn
        # del truth
        # del vect
        # gc.collect()

    def annotate_nsyn(self):
        self.annotate("nsyn-esn", split="all")
        # nsyn, vect = self.annotator.run(model="nsyn", vectors=True)
        # self.corpus.annotations["nsyn"] = create_memmap(nsyn, "nsyn_annots")
        # self.nsyn_annots_vectors = create_memmap(vect, "nsyn_vect")
        #
        # del nsyn
        # del vect
        # gc.collect()

    def annotate_ensemble(self):
        self.annotate("ensemble", split="all")
        # ens = self.annotator.run(
        #     models_vectors=[self.syn_annots_vectors, self.nsyn_annots_vectors],
        #     model="ensemble",
        # )
        # self.corpus.annotations["ensemble"] = create_memmap(ens, "ens_annots")
        #
        # del ens
        # del self.syn_annots_vectors
        # close_memmap("syn_vect")
        # del self.nsyn_annots_vectors
        # close_memmap("nsyn_vect")
        #
        # gc.collect()

    def get_metrics(self, split="all"):
        predictions = self._pred_corpora[split]

        for annot_name, pred_corpus in predictions.items():
            gold_corpus = query_split(self.corpus, split)
            classes = np.sort(self.corpus["label"].unique()).tolist()

            cm = confusion_matrix(gold_corpus, pred_corpus, classes=classes)
            report = classification_report(gold_corpus, pred_corpus, classes=classes)
            # ser = segment_error_rate(gold_corpus, pred_corpus)

            self._metrics_store[split]["cm"] = cm
            self._metrics_store[split]["report"] = report
            # self._metrics_store[split]["ser"] = ser

        # flat_annots = {k: np.concatenate([*s.values()]) for k, s in annots.items()}
        #
        # truth = flat_annots["truth"]
        # labels = self.corpus.vocab
        # cms = {}
        # reports = {}
        # for model, values in tqdm(
        #     flat_annots.items(), "Computing metrics for annotations"
        # ):
        #
        #     if model != "truth":
        #         cms[model] = metrics.confusion_matrix(
        #             truth, values, labels=labels, normalize="true"
        #         )
        #
        #         reports[model] = metrics.classification_report(
        #             truth, values, digits=3, output_dict=True, zero_division=0
        #         )
        #
        # self.cms = cms
        # self.reports = reports

        self.fetch_misclassified_samples()
        self.misclassified_counts_plot()

    def fetch_misclassified_samples(self):
        return fetch_misclassified_samples(df)

    def misclassified_counts_plot(self):
        p, s = plots.plot_bokeh_label_count(df)

    def confusion_matrix(self, model):
        p = plots.plot_bokeh_confusion_matrix(cm, classes, title)

    def load_sample(self, sample):
        return load_sample(s)

    def load_repertoire(self, selected_samples):
        return load_repertoire(selected)

    def stop_app(self):
        del_temp()
        self.dash.stop()
        print("Server shutdown.")


def make_directory(directory):
    if not directory.exists():
        directory.mkdir(parents=True)
    return directory
