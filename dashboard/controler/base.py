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
    compute_sklearn_metrics,
)
from canapy.plots import plot_segment_melspectrogram
from canapy.utils import as_path
from canapy.annotator.commons.postprocess import extract_vocab
from canapy.transforms.commons.training import split_train_test, encode_labels
from canapy.utils.tempstorage import close_tempfiles
from canapy.transforms.commons.annots import sort_annotations, merge_labels, tag_silences, remove_short_labels
from canapy.transforms.commons.audio import compute_mfcc
from .segments import fetch_misclassified_samples
from .corpusutils import mark_whole_corpus_as_train, query_split
from canapy.optimization import optimize_hyperparameters_isolated

logger = logging.getLogger("canapy")


def _sort_annotators(annotators: List):
    if "ensemble" in annotators:
        annotators.remove("ensemble")
        sorted_annots = ["ensemble"]
    else:
        sorted_annots = []
    sorted_annots = annotators.copy() + sorted_annots
    return sorted_annots


VALID_STEPS = {"home", "preprocess", "train", "eval", "export", "annotate", "loaddata", "settings"}


@attr.define
class Controler:
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
    model_root: Optional[Path] = attr.field(default=None)
    home_path: Optional[str] = attr.field(default=None)
    preprocess_done: Optional[bool] = attr.field(default=False)
    corrector: Optional[Corrector] = attr.field(default=None)
    _iter: Optional[int] = attr.field(alias="_iter", default=1)
    _step: Optional[str] = attr.field(alias="_step", default="train")
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
    _repertoire_cache: Dict = attr.field(factory=dict, init=False)
    _settings_return_step: Optional[str] = attr.field(alias="_settings_return_step", default="home")
    fit_done: bool = attr.field(default=False)
    eval_done: bool = attr.field(default=False)
    export_done: bool = attr.field(default=False)
    opt_parallel: bool = attr.field(default=False)
    opt_max_percentage: float = attr.field(default=0.3)
    opt_n_jobs: int = attr.field(default=4)
    opt_max_evals: int = attr.field(default=100)
    opt_hp_val_ratio: float = attr.field(default=0.2)
    opt_seed: int = attr.field(default=42)
    min_class_count: int = attr.field(default=10)

    def __attrs_post_init__(self):
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

        # config_path is always a file (TOML/YAML); model_root is always a directory.
        config_path = Path(self.config_path) if self.config_path is not None else None
        if config_path is not None and not config_path.is_file():
            raise ValueError(f"Invalid -c value: expected a config file, got: {config_path}")

        self.corpus = Corpus.from_directory(
            audio_directory=self.audio_directory,
            spec_directory=self.spec_directory,
            annots_directory=self.annots_directory,
            config_path=config_path,
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
        self._settings_return_step = "home"

        self.annotators = _sort_annotators(self.annotators)

        self.initialize_output()
        self.initialize_models()
        self.initialize_annots()

        self.corpus = split_train_test(self.corpus, redo=True)
        self.compute_classes()
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

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def go_settings(self):
        self._settings_return_step = self._step
        self._step = "settings"
        self.dashboard.switch_panel()

    def leave_settings(self):
        self._step = self._settings_return_step
        self.dashboard.switch_panel()

    def initialize_models(self):
        for name in self.annotators:
            try:
                annot_cls = get_annotator(name)
                annot_obj = annot_cls(self.config)
                self._annotators[name] = annot_obj

                if name == "ensemble":
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
        self.fit_done = False
        self.eval_done = False
        self.export_done = False
        logger.info(f"Current training iteration : {self.iter}")

    def checkpoint(self):
        try:
            ckpt_dir = self.output_directory / "model" / "checkpoints" / str(self.iter)
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

    def _compute_silence_ratio(self):
        """Return (ratio, pct_str) of silence (explicit SIL + implicit gaps) in current corpus."""
        df = self.corpus.dataset.copy()
        silence_tag = self.config.transforms.annots.silence_tag
        df["_duration"] = df["offset_s"] - df["onset_s"]
        sil_duration = df.loc[df["label"] == silence_tag, "_duration"].sum()
        df_sorted = df.sort_values(["notated_path", "onset_s"])
        prev_offset = df_sorted.groupby("notated_path")["offset_s"].shift(fill_value=0.0)
        gap_duration = (df_sorted["onset_s"] - prev_offset).clip(lower=0).sum()
        total_sil = sil_duration + gap_duration
        total_dur = df["_duration"].sum() + gap_duration
        ratio = total_sil / total_dur if total_dur > 0 else 0.0
        return ratio, f"{ratio * 100:.1f}%"

    def check_silence_ratio(self, max_silence_ratio=0.2):
        """Show a warning notification if silence ratio exceeds threshold."""
        ratio, pct_str = self._compute_silence_ratio()
        if ratio > max_silence_ratio:
            panel.state.notifications.warning(
                f"High silence ratio detected ({pct_str} of total duration). "
                "Consider trimming silence before training.",
                duration=8000,
            )
            logger.warning(f"High silence ratio: {pct_str}")
        return ratio

    def apply_live_corrections(self, corrections, target):
        self.upload_corrections(corrections, target)

        if target == "class":
            self.corpus.dataset['label'] = self.corpus.dataset['label'].replace(corrections)
            self.corpus = merge_labels(self.corpus)

        elif target == "annot":
            for idx, new_label in corrections.items():
                if idx in self.corpus.dataset.index:
                    self.corpus.dataset.at[idx, 'label'] = new_label

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

    def clean_corpus_pre_training(self):
        df = self.corpus.dataset.copy()
        silence_tag = self.config.transforms.annots.silence_tag

        # Remove under-represented classes (occurrence-based)
        counts = df[~df["label"].isin([silence_tag, "TRASH"])].groupby("label").size()
        classes_to_remove = counts[counts < self.min_class_count].index.tolist()

        if classes_to_remove:
            df.loc[df["label"].isin(classes_to_remove), "label"] = "TRASH"
            msg = f"Not enough samples for {classes_to_remove}, classes have been removed"
            panel.state.notifications.warning(msg, duration=0)
            logger.warning(msg)

        self.corpus = self.corpus.clone_with_df(df)

    def export_corpus(self):
        try:
            if hasattr(self, '_correction_store') and self._correction_store:
                self.apply_corrections()

            export_dir = self.output_directory / "corrected_annotations"
            export_dir.mkdir(parents=True, exist_ok=True)
            self.corpus.to_directory(str(export_dir))
            logger.info(f"Dataset exported to: {export_dir}")
        except OSError as e:
            logger.critical("Failed to create export directory for annotations:")
            logger.critical(e)
        except Exception as e:
            logger.critical(f"Failed to export corpus: {e}")
            raise

    def load_page(self, page_name: str):
        logger.info(f"Manual navigation request to: {page_name}")

        if page_name not in VALID_STEPS:
            logger.error(f"Unknown page: {page_name}")
            return

        if page_name == "preprocess":
            self._step = "preprocess"
            if self.home_path == "edit":
                self.compute_classes()
            self.check_silence_ratio()

        elif page_name == "home":
            self._step = "home"
            self.home_path = None

        else:
            self._step = page_name

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
            self.clean_corpus_pre_training()
            self.compute_classes()
            self.preprocess_done = True
            logger.info('Setting current dashboard to "train".')

        if self.step == "train":
            if self.preprocess_done:
                logger.info("Preprocess was just completed.")
                logger.info("Preparing train/test split.")

                self.preprocess_done = False
                self.corpus = split_train_test(self.corpus, redo=True)

                self.dashboard.switch_panel()
                return

            self._step = "eval"
            self.get_metrics()
            self.eval_done = True
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
                (self.output_directory / "model" / "exported_models").mkdir(parents=True, exist_ok=True)
                annotator.to_disk(self.output_directory / "model" / "exported_models" / annotator_name)
                logger.info(
                    f"Saved Annotator '{annotator_name}' in file "
                    f"{self.output_directory / 'model' / annotator_name}."
                )
            except OSError as e:
                logger.critical(e)

    def annotate(self, annotator_name, split="all"):
        if annotator_name == "ensemble":
            corpora = [c for name, c in self._pred_corpora[split].items() if name != "ensemble"]
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
        # Sync ensemble vocab with the preprocessed vocab from a trained sub-annotator.
        # Ensemble._vocab is set at initialize_models() time from the raw corpus, but
        # SynAnnotator._vocab is set after preprocessing (remove_short_labels may drop
        # classes). If they diverge, hard_vote maps argmax indices to wrong labels.
        ensemble = self._annotators.get("ensemble")
        if ensemble is not None:
            for ref_name in ("syn-esn", "nsyn-esn"):
                ref = self._annotators.get(ref_name)
                if ref is not None and ref.trained:
                    ensemble._vocab = list(ref.vocab)
                    logger.info(
                        f"Ensemble vocab synced from '{ref_name}': {ensemble._vocab}"
                    )
                    break
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

                cm, report = compute_sklearn_metrics(gold_corpus, pred_corpus, classes=classes)
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
        if selected_samples.empty:
            return []
        cache_key = (
            selected_samples["label"].iloc[0],
            tuple(selected_samples.index.tolist()),
        )
        if cache_key in self._repertoire_cache:
            logger.info(f"Repertoire cache hit: {cache_key[0]}")
            return self._repertoire_cache[cache_key]
        logger.info(
            f"Loading repertoire samples for label(s): {selected_samples.label.unique()}"
        )
        audio_conf = self.config.transforms.audio
        df = self.corpus.dataset
        silence_tag = self.config.transforms.annots.silence_tag
        non_sil = df[df["label"] != silence_tag]
        seg_durs = non_sil["offset_s"] - non_sil["onset_s"]
        x_tick_ref_s = float(seg_durs[seg_durs > 0].max()) if not seg_durs.empty else 1.0
        specs = [
            plot_segment_melspectrogram(
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
                show_ticks=True,
                x_tick_ref_s=x_tick_ref_s,
            )
            for s in selected_samples.itertuples()
        ]
        self._repertoire_cache[cache_key] = specs
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
        _sub_annot_vocab = None  # vocab from first successful sub-annotator (for ensemble sync)

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
                    iters = sorted(
                        [d for d in model_root.iterdir() if d.is_dir()],
                        key=lambda p: p.name,
                        reverse=True,
                    )
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
                if _sub_annot_vocab is None and annot.vocab:
                    _sub_annot_vocab = list(annot.vocab)

        if any("ensemble" in n.lower() for n in all_names):
            ens_name = [n for n in all_names if "ensemble" in n.lower()][0]
            ens_annot = None

            model_root = self.model_root or (self.output_directory / "model")

            search_paths = [
                model_root / ens_name,
                model_root / "1" / ens_name,
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
                    # Sync ensemble vocab with the preprocessed vocab from a sub-annotator
                    # so hard_vote uses the correct label→index mapping.
                    if _sub_annot_vocab is not None:
                        ens_annot._vocab = _sub_annot_vocab
                        logger.info(f"Ensemble (external) vocab synced: {_sub_annot_vocab}")
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

    def export_config(self) -> str:
        project_root = Path(__file__).resolve().parents[2]
        user_config_dir = project_root / "config" / "user"
        user_config_dir.mkdir(parents=True, exist_ok=True)
        config_path = user_config_dir / "user.config.toml"
        self.config.to_disk(config_path, format="toml")
        logger.info(f"Config exported to {config_path}")
        return str(config_path)

    def stop_app(self):
        close_tempfiles()
        self.dashboard.stop()

    def trim_silences_hard(self, target_ratio: float, progress_callback=None):
        """
        Hard-trim silence regions (explicit SIL annotations + implicit gaps) from
        audio files by center-cropping each region so that the total silence reaches
        target_ratio.  Writes new WAVs and annotation CSVs under
        output_directory/audio_trimmed/ and output_directory/annots_trimmed/, clears
        the MFCC cache, then reloads the corpus from these new paths.

        Original audio files are never modified.
        """
        import soundfile as sf
        import numpy as np

        if self.audio_directory is None:
            panel.state.notifications.error(
                "No audio directory configured. Cannot trim silences.", duration=5000
            )
            logger.error("trim_silences_hard: audio_directory is None.")
            return

        silence_tag = self.config.transforms.annots.silence_tag

        # --- Compute keep_fraction ---
        ratio, _ = self._compute_silence_ratio()
        if ratio <= target_ratio:
            panel.state.notifications.warning(
                f"Silence ratio ({ratio*100:.1f}%) is already at or below "
                f"target ({target_ratio*100:.1f}%). Nothing to do.",
                duration=5000,
            )
            return

        df = self.corpus.dataset.copy()
        df["_dur"] = df["offset_s"] - df["onset_s"]
        sil_dur = df.loc[df["label"] == silence_tag, "_dur"].sum()
        non_sil_dur = df["_dur"].sum() - sil_dur

        df_sorted = df.sort_values(["notated_path", "onset_s"])
        prev_offset = df_sorted.groupby("notated_path")["offset_s"].shift(fill_value=0.0)
        gap_dur = (df_sorted["onset_s"] - prev_offset).clip(lower=0).sum()
        total_sil = sil_dur + gap_dur

        # target_ratio = keep_sil / (non_sil + keep_sil)  =>  keep_sil = target_ratio * non_sil / (1 - target_ratio)
        keep_sil = (target_ratio * non_sil_dur / (1 - target_ratio)) if target_ratio < 1 else float("inf")
        keep_fraction = min(1.0, keep_sil / total_sil) if total_sil > 0 else 1.0

        logger.info(
            f"trim_silences_hard: ratio={ratio*100:.1f}% → target={target_ratio*100:.1f}%, "
            f"keep_fraction={keep_fraction:.3f}"
        )

        # --- Output directories ---
        trimmed_audio_dir = self.output_directory / "audio_trimmed"
        trimmed_annots_dir = self.output_directory / "annots_trimmed"

        # Clear any leftover files from a previous dataset before writing new ones
        if trimmed_audio_dir.exists():
            for f in trimmed_audio_dir.glob("*"):
                f.unlink(missing_ok=True)
        if trimmed_annots_dir.exists():
            for f in trimmed_annots_dir.glob("*"):
                f.unlink(missing_ok=True)

        trimmed_audio_dir.mkdir(parents=True, exist_ok=True)
        trimmed_annots_dir.mkdir(parents=True, exist_ok=True)

        audio_files = list(df["notated_path"].unique())
        n_files = len(audio_files)

        for file_idx, audio_path_str in enumerate(audio_files):
            if progress_callback:
                progress_callback(file_idx + 1, n_files)

            audio_path = Path(audio_path_str)
            file_df = df[df["notated_path"] == audio_path_str].sort_values("onset_s")

            audio_data, sr = sf.read(str(audio_path))
            audio_total_samples = len(audio_data)
            audio_duration_s = audio_total_samples / sr

            # Build ordered segment list: (start_s, end_s, is_silence, row_or_None)
            segments = []
            prev_end_s = 0.0
            for _, row in file_df.iterrows():
                if row["onset_s"] > prev_end_s + 1e-6:
                    segments.append((prev_end_s, row["onset_s"], True, None))
                segments.append((row["onset_s"], row["offset_s"], row["label"] == silence_tag, row))
                prev_end_s = row["offset_s"]
            if prev_end_s < audio_duration_s - 1e-6:
                segments.append((prev_end_s, audio_duration_s, True, None))

            # Reconstruct audio and collect updated annotation rows
            audio_chunks = []
            updated_annot_rows = []
            current_pos_s = 0.0

            for start_s, end_s, is_silence, row in segments:
                start_sample = int(round(start_s * sr))
                end_sample = min(int(round(end_s * sr)), audio_total_samples)
                chunk = audio_data[start_sample:end_sample]

                if is_silence:
                    n = len(chunk)
                    keep_n = max(0, min(int(round(n * keep_fraction)), n))
                    if keep_n == 0:
                        chunk = chunk[0:0]
                    elif keep_n < n:
                        # Keep start & end, trim middle
                        keep_start = keep_n // 2
                        keep_end = keep_n - keep_start
                        chunk = np.concatenate([chunk[:keep_start], chunk[n - keep_end:]])

                chunk_dur_s = len(chunk) / sr

                if len(chunk) > 0:
                    audio_chunks.append(chunk)

                if row is not None:
                    new_row = row.copy()
                    new_row["onset_s"] = current_pos_s
                    new_row["offset_s"] = current_pos_s + chunk_dur_s
                    updated_annot_rows.append(new_row)

                current_pos_s += chunk_dur_s

            # Write new WAV
            if audio_chunks:
                new_audio = np.concatenate(audio_chunks, axis=0) if len(audio_chunks) > 1 else audio_chunks[0]
                sf.write(str(trimmed_audio_dir / audio_path.name), new_audio, sr)

            # Write updated annotation CSV (marron1csv: wave, start, end, syll)
            if updated_annot_rows:
                rows_df = pd.DataFrame(updated_annot_rows)
                csv_out = pd.DataFrame({
                    "wave": audio_path.name,
                    "start": rows_df["onset_s"].values,
                    "end": rows_df["offset_s"].values,
                    "syll": rows_df["label"].values,
                })
                orig_annot_name = Path(str(file_df["annot_path"].iloc[0])).name
                csv_out.to_csv(str(trimmed_annots_dir / orig_annot_name), index=False)

            logger.info(f"Trimmed {audio_path.name} ({file_idx + 1}/{n_files})")

        if progress_callback:
            progress_callback(n_files, n_files)

        # --- Clear MFCC cache ---
        if self.spec_directory and self.spec_directory.exists():
            for f in self.spec_directory.glob("*.npz"):
                f.unlink(missing_ok=True)
            logger.info("MFCC cache cleared.")

        # --- Reload corpus from trimmed directories ---
        self.audio_directory = trimmed_audio_dir
        self.annots_directory = trimmed_annots_dir

        # Preserve HP params before corpus reload (Corpus.from_directory resets config from disk)
        _hp_attrs = ("sr", "leak", "ridge", "iss", "isd", "isd2")
        # Reading via attribute access is fine for scalars (copy has same scalar values).
        _saved_hp = {k: self.config.data["model"]["syn"].get(k) for k in _hp_attrs}

        config_path = Path(self.config_path) if self.config_path is not None else None
        self.corpus = Corpus.from_directory(
            audio_directory=trimmed_audio_dir,
            spec_directory=self.spec_directory,
            annots_directory=trimmed_annots_dir,
            config_path=config_path,
            annot_format=self.annot_format,
            audio_ext=self.audio_ext,
        )
        self.config = self.corpus.config

        # Reapply HP params that were set by optimization (lost during corpus reload)
        for k, v in _saved_hp.items():
            if v is not None:
                self.config.data["model"]["syn"][k] = v
        self.corpus = split_train_test(self.corpus, redo=True)
        self.compute_classes()
        self.initialize_annots()
        self.preprocess_done = False

        logger.info(f"trim_silences_hard complete. Corpus reloaded from {trimmed_audio_dir}.")

    def optimize_models(self, progress_callback=None):
        logger.info("Starting optimization from Controller...")

        target_name = "syn-esn"
        if target_name not in self._annotators:
            logger.error(f"Annotator '{target_name}' not found.")
            return

        syn_annotator = self._annotators[target_name]

        # Shallow-copy the corpus so annotation transforms and MFCC registration
        # don't affect the live corpus used by the rest of the dashboard.
        # Only dataset (modified in-place by transforms) and data_resources (new
        # entries added) need their own copies; config and annotations are read-only.
        import copy
        c = copy.copy(self.corpus)
        c.dataset = self.corpus.dataset.copy()
        c.data_resources = dict(self.corpus.data_resources)

        logger.info("Step 1: Stabilizing labels and split...")
        c = sort_annotations(c)
        c = merge_labels(c)
        c = tag_silences(c, silence_tag=syn_annotator.config.transforms.annots.silence_tag)
        c = remove_short_labels(c, min_label_duration=syn_annotator.config.transforms.annots.min_label_duration)
        # redo=False : preserve the original split to stay consistent with the main pipeline
        c = split_train_test(c, redo=False)
        # encode_labels is required by load_mfccs_and_repeat_labels (encoded_label column)
        c = encode_labels(c, resource_name=None)

        # Dedicated temp directory so we don't overwrite shared .npz files
        import tempfile
        optim_spec_dir = Path(tempfile.mkdtemp(prefix="canapy_optim_"))
        logger.info(f"Step 2: Computing MFCCs in isolated directory: {optim_spec_dir}")

        c = compute_mfcc(
            c,
            output_directory=optim_spec_dir,
            resource_name="syn_mfcc",
            redo=True,
        )

        if "syn_mfcc" not in c.data_resources:
            logger.error("Critical: 'syn_mfcc' still missing from resources!")
            return

        import shutil
        logger.info("Step 3: Launching optimization (isolated subprocess)...")
        try:
            best_params = optimize_hyperparameters_isolated(
                c,
                syn_annotator.config,
                annotator_type="syn",
                n_iter=self.opt_max_evals,
                max_percentage=self.opt_max_percentage,
                parallel=self.opt_parallel,
                n_jobs=self.opt_n_jobs,
                hp_val_ratio=self.opt_hp_val_ratio,
                seed=self.opt_seed,
                progress_callback=progress_callback,
            )
        finally:
            # Always clean up temp MFCC files, even if optimization raised.
            shutil.rmtree(optim_spec_dir, ignore_errors=True)

        if not best_params:
            logger.warning(
                "Optimization returned no results — subprocess may have crashed or timed out."
            )
            return None

        logger.info(f"Optimization finished. Best params: {best_params}")
        # _BaseConfig.__getattr__ returns a COPY on every attribute access,
        # so `setattr(config.model.syn, k, v)` writes to a throwaway object.
        # Write directly into the underlying dict to actually persist the values.
        for k, v in best_params.items():
            syn_annotator.config.data["model"]["syn"][k] = v

        # Rebuild the ESN model with the new HP params.
        # The rpy_model was created at annotator init time (before optimization),
        # so updating the config alone has no effect on the existing model.
        syn_annotator.rpy_model = syn_annotator.initialize()
        logger.info("ESN model reinitialized with optimized hyperparameters.")

        return best_params

