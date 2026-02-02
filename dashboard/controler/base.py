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
    # Modif 19/01 rendre optionnel les dossiers audio/annots
    dashboard: panel.viewable.Viewer = attr.field()
    annot_format: str = attr.field(default="marron1csv")
    audio_ext: str = attr.field(default=".wav")
    annotators: List[str] = attr.field(default=["syn-esn", "nsyn-esn", "ensemble"], converter=list)
    annots_directory: Optional[Path] = attr.field(default=None, converter=as_path)
    audio_directory: Optional[Path] = attr.field(default=None, converter=as_path)
    output_directory: Optional[Path] = attr.field(default=None, converter=as_path)
    spec_directory: Optional[Path] = attr.field(default=None, converter=as_path)
    config_path: Optional[Path] = attr.field(default=None, converter=as_path)

    corpus: Optional[Corpus] = attr.field(default=None)
    config: Optional[Mapping] = attr.field(default=None)
    # Modif 14/01 : Ajout model_root pour chargement modèle externe (séparer file et dir choisi avec config path)
    model_root: Optional[Path] = attr.field(default=None)
    # Modif Axel 11/12 : Ajout d'un flag pour savoir si le preprocess a été fait et quel chemin emprunter depuis home
    home_path: Optional[str] = attr.field(default=None)
    preprocess_done: Optional[bool] = attr.field(default=False)
    corrector: Optional[Corrector] = attr.field(default=None)
    _iter: Optional[int] = attr.field(alias="_iter", default=1)
    _step: Optional[str] = attr.field(alias="_step", default="train")
    # Modif 19/01
    _is_ready: bool = attr.field(default=False, init=False)
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
        alias="_correction_store", default=dict()
    )
    _classes: Optional[List[str]] = attr.field(alias="_classes", default=None)
    _external_accum: Optional[Dict[str, Corpus]] = attr.field(factory=dict, init=False)
    # Modif Axel 14/01 
    def __attrs_post_init__(self):
        # Modif 19/01
        if (
            self.audio_directory is None
            or self.annots_directory is None
            or self.output_directory is None
            or self.spec_directory is None
        ):
            self.annotators = _sort_annotators(self.annotators)
            self._step = "home"
            self._is_ready = False
            return

        self.model_root = None

        config_path = Path(self.config_path) if self.config_path is not None else None

        if config_path is not None and config_path.exists():
            if config_path.is_file():
                self.corpus = Corpus.from_directory(
                    audio_directory=self.audio_directory,
                    spec_directory=self.spec_directory,
                    annots_directory=self.annots_directory,
                    config_path=config_path,
                    annot_format=self.annot_format,
                    audio_ext=self.audio_ext,
                )
                self.config = self.corpus.config

            elif config_path.is_dir():
                self.model_root = config_path

                self.corpus = Corpus.from_directory(
                    audio_directory=self.audio_directory,
                    spec_directory=self.spec_directory,
                    annots_directory=self.annots_directory,
                    config_path=None,  
                    annot_format=self.annot_format,
                    audio_ext=self.audio_ext,
                )
                self.config = self.corpus.config

            else:
                raise ValueError(f"Invalid --config-path: {config_path}")

        else:
            # Aucun -c fourni : comportement par défaut
            self.corpus = Corpus.from_directory(
                audio_directory=self.audio_directory,
                spec_directory=self.spec_directory,
                annots_directory=self.annots_directory,
                config_path=None,
                annot_format=self.annot_format,
                audio_ext=self.audio_ext,
            )
            self.config = self.corpus.config

        self.corrector = Corrector(
            self.output_directory / "checkpoints",
            [{"class": dict(), "annot": dict()}],
        )

        self.home_path = None
        self.preprocess_done = False
        self._iter = 1
        self._step = "home"
        self._external_accum = {}

        self.annotators = _sort_annotators(self.annotators)

        self.initialize_output()
        self.initialize_models()
        self.initialize_annots()

        self.corpus = split_train_test(self.corpus, redo=True)
        self.compute_classes()
        # Modif 19/01
        self._is_ready = True


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

    # Modif 19/01 : rendre les arg optionnels
    @property
    def is_ready(self) -> bool:
        return self._is_ready

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
            (self.output_directory / "spectrograms").mkdir(parents=True, exist_ok=True)

            logger.info(
                f"All results will be stored in {self.output_directory}. "
                f"Audio transformations (spectrograms, MFCC...) will be "
                f"stored in "
                f"{self.output_directory / 'spectrograms'}."
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

    # #26/01 Modif Axel MAJ pages
    def apply_live_corrections(self, corrections, target):
        """Applies corrections immediately to the in-memory dataframe and updates the classes."""
        # 1. Store corrections for future disk checkpoints
        self.upload_corrections(corrections, target)

        # 2. Apply to current RAM dataset
        if target == "class":
            # Renaming classes in the dataframe
            self.corpus.dataset['label'] = self.corpus.dataset['label'].replace(corrections)
        
        elif target == "annot":
            # Corrections is a dict {index: new_label}
            for idx, new_label in corrections.items():
                if idx in self.corpus.dataset.index:
                    self.corpus.dataset.at[idx, 'label'] = new_label
        
        # 3. Recompute class list (vocabulary) so other views (like selectors) are aware
        self.compute_classes()
        logger.info(f"Applied live corrections ({target}) and updated class registry.")

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
            logger.info(f"Dataset exported to: {export_dir}")
        except OSError as e:
            logger.critical("Failed to create export directory for annotations:")
            logger.critical(e)

    def load_page(self, page_name: str):
        """Force manual navigation to a specific page."""
        logger.info(f"Manual navigation request to: {page_name}")
        
        if page_name == "preprocess":
            self._step = "preprocess"
            if self.home_path == "edit":
                self.compute_classes()
        
        elif page_name == "home":
             self._step = "home"
             self.home_path = None # Reset context
        
        # Update the view
        self.dashboard.switch_panel()

    def next_step(self, export=False):

        if self.step == "preprocess":
            if self.home_path == "edit":
                logger.info("Quick Edit mode: Applying corrections and exporting.")
                self.apply_corrections() 
                self.export_corpus()
                
                self._step = "home"
                self.home_path = None
                self.dashboard.switch_panel()
                return 

            self._step = "train"
            self.apply_corrections()
            self.preprocess_done = True
            logger.info('Setting current dashboard to "train".')
            
        if self.step == "train":
            if self.preprocess_done:
                logger.info("Preprocess was just completed. Preparing train/test split.")
                self.preprocess_done = False
                
                self.corpus = split_train_test(self.corpus, redo=True)
                
                self.dashboard.switch_panel()
                return
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

        elif self.step == "home":
            if self.home_path == "preprocess":
                self._step = "preprocess"
            elif self.home_path == "annotate":
                self._step = "annotate"
            logger.info(f'Setting current dashboard to "{self._step}".')
        
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
                (self.output_directory / "model").mkdir(parents=True, exist_ok=True)
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

    def compute_classes(self):
        self._classes = extract_vocab(
            self.corpus, silence_tag=self.config.transforms.annots.silence_tag
        )

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
    
    def load_external_corpus(self, folder: str) -> "Corpus":
        logger.info(f"Loading external corpus from {folder}")
        return Corpus.from_directory(
            audio_directory=folder,
            spec_directory=self.spec_directory,
            annots_directory=None,
            config_path=self.config_path,
            annot_format=self.annot_format,
            audio_ext=self.audio_ext,
        )



    def _load_annotator_from_disk(self, name: str, model_path: Path):
        """Load an annotator using the official API Annotator.from_disk(path, config).

        All annotators in Canapy expose `from_disk(path, config)`,
        which handles both new and legacy saved models.
        """
        logger.info(f"Loading annotator '{name}' from {model_path}")

        annot_cls = get_annotator(name)

        try:
            annot = annot_cls.from_disk(model_path, config=self.config)
            logger.info(f"Successfully loaded annotator '{name}' from disk.")
            return annot
        except Exception as e:
            logger.error(f"Failed loading annotator '{name}' from {model_path}: {e}")
            raise

    def annotate_external(self, corpus, model_sources=None,
                                use_in_memory=True, return_raw=False):
            model_sources = model_sources or {}
            all_names = list(model_sources.keys()) if model_sources else self.annotators
            logger.info(f"Processing models: {all_names}")
            
            results = {}
            
            for name in all_names:
                if name == "ensemble":
                    continue
                    
                annot = None
                if use_in_memory and name in self._annotators:
                    annot = self._annotators[name]
                else:
                    model_root = self.model_root or (self.output_directory / "model")
                    direct_path = model_root / name
                    if not direct_path.exists() and model_root.exists():
                        iters = sorted([d for d in model_root.iterdir() if d.is_dir()], key=lambda p: p.name, reverse=True)
                        for itdir in iters:
                            if (itdir / name).exists():
                                direct_path = itdir / name
                                break
                    
                    if direct_path.exists():
                        annot = self._load_annotator_from_disk(name, direct_path)

                if annot is not None:
                    if hasattr(annot, 'rpy_model'):
                        annot._trained = True
                        if hasattr(annot.rpy_model, 'readout') and annot.vocab:
                            annot.rpy_model.readout.output_dim = len(annot.vocab)

                    logger.info(f"Running prediction for {name}...")
                    pred_corpus = annot.predict(corpus, return_raw=True)
                    results[name] = pred_corpus
                    self._external_accum[name] = pred_corpus 
                    logger.info(f"Stored {name} in persistent accumulator.")

            if any("ensemble" in n.lower() for n in all_names):
                ens_name = [n for n in all_names if "ensemble" in n.lower()][0]
                ens_annot = None
                
                model_root = self.model_root or (self.output_directory / "model")
                

                search_paths = [
                    model_root / ens_name,
                    model_root / "1" / ens_name
                ]
                
                for path in search_paths:
                    if path.exists():
                        logger.info(f"Ensemble found at: {path}")
                        ens_annot = self._load_annotator_from_disk(ens_name, path)
                        break

                if ens_annot:
                    raw_inputs = list(self._external_accum.values())
                    logger.info(f"Ensemble voting with {len(raw_inputs)} accumulated corpora.")
                    if len(raw_inputs) >= 2:
                        results[ens_name] = ens_annot.predict(raw_inputs)
                        logger.info("Ensemble prediction successful.")
                        self._external_accum = {} 
                    else:
                        logger.error(f"Ensemble needs 2 sub-predictions, has {len(raw_inputs)}.")
                else:
                    logger.error(f"Ensemble model NOT FOUND. Path attempted: {model_root / ens_name}")

            return results

    def export_predictions(self, pred_corpus, out_dir: Path):
        try:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_corpus.to_directory(str(out_dir))
            logger.info(f"Exported predictions to {out_dir}")
        except Exception as e:
            logger.critical(f"Failed to export predictions to {out_dir}: {e}")
            raise

    def stop_app(self):
        close_tempfiles()
        self.dashboard.stop()