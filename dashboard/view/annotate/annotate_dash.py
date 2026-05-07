# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
import logging
import time
from pathlib import Path
import panel as pn
import pandas as pd
import soundfile as sf
from ..helpers import SubDash, SideBar, pick_directory
from canapy.corpus import Corpus

logger = logging.getLogger("canapy-dashboard")

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")
KNOWN_ANNOTATORS = ["syn-esn", "nsyn-esn", "ensemble"]

ANNOTATE_CSS = """
.annotate-card {
    background-color: #ffffff;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    padding: 18px 20px;
    border: 1px solid #e5e7eb;
    box-sizing: border-box;
}
.section-header {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #6b7280;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 6px;
    margin-bottom: 10px;
}
"""


def _spinner_col(label, indicator, status):
    return pn.Column(
        pn.pane.HTML(f"<span style='font-size:12px; color:#6b7280;'>{label}</span>"),
        indicator,
        status,
        sizing_mode="stretch_width",
        align="center",
    )


def _idle():
    return "<p style='margin:4px 0 0 0; color:#9ca3af; font-size:12px;'>Idle</p>"


def _set_status(obj, status, duration=0.0):
    if status == "annotating":
        obj.object = (
            "<p style='color:#2563eb; font-weight:600; margin:4px 0 0 0; font-size:12px;'>Annotating...</p>"
        )
    elif status == "done":
        obj.object = (
            f"<p style='color:#16a34a; font-weight:600; margin:4px 0 0 0; font-size:12px;'>"
            f"Done {round(duration, 1)} s</p>"
        )
    elif status == "skipped":
        obj.object = "<p style='margin:4px 0 0 0; color:#9ca3af; font-size:12px;'>Skipped</p>"
    elif status == "idle":
        obj.object = _idle()


class AnnotateDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)
        pn.config.raw_css.append(ANNOTATE_CSS)

        self.sidebar = SideBar(self, "Annotation")
        self.parent = parent
        self.controler = self.parent.controler

        self._external_corpus = None
        self._predictions = {}
        self._available_models = self._scan_models()

        config_card = self._build_config_panel()
        monitor_card = self._build_monitor_panel()

        header = pn.Column(
            pn.pane.Markdown("# Annotate", css_classes=["page-title"], margin=0),
            pn.pane.Markdown(
                "Annotate an external audio dataset using trained models.",
                css_classes=["page-subtitle"],
                margin=0,
            ),
            margin=(0, 0, 12, 0),
        )

        self.layout = pn.Row(
            self.sidebar,
            pn.Column(
                header,
                pn.Row(
                    config_card,
                    pn.Spacer(width=16),
                    monitor_card,
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_both",
                margin=(20, 10),
            ),
            sizing_mode="stretch_both",
            background="#f8fafc",
        )

        self._validate_audio_on_init()

    def _build_config_panel(self):
        self.dataset_input = pn.widgets.TextInput(
            placeholder="/path/to/audio/folder",
            sizing_mode="stretch_width",
            height=38,
            margin=0,
        )
        self.dataset_browse_btn = pn.widgets.Button(
            name="Browse", button_type="default", width=90, height=38, margin=0
        )
        self.dataset_browse_btn.on_click(self._browse_dataset)

        self.dataset_load_btn = pn.widgets.Button(
            name="Load Dataset",
            button_type="primary",
            sizing_mode="stretch_width",
            height=38,
        )
        self.dataset_load_btn.on_click(self._load_external_dataset)

        self.dataset_status = pn.pane.HTML(
            "<span style='color:#6b7280;font-size:12px;'>No dataset loaded.</span>",
            sizing_mode="stretch_width",
        )

        if self.controler.audio_directory:
            self.dataset_input.value = str(self.controler.audio_directory)

        self.toggle_syn = pn.widgets.Toggle(
            name="Syn-ESN", value=False, button_type="default", sizing_mode="stretch_width"
        )
        self.toggle_nsyn = pn.widgets.Toggle(
            name="NSyn-ESN", value=False, button_type="default", sizing_mode="stretch_width"
        )
        self.toggle_ens = pn.widgets.Toggle(
            name="Ensemble", value=False, button_type="default", sizing_mode="stretch_width"
        )
        for t in [self.toggle_syn, self.toggle_nsyn, self.toggle_ens]:
            t.param.watch(self._on_toggle_change, "value")

        self.btn_run = pn.widgets.Button(
            name="Start Annotation",
            button_type="primary",
            height=45,
            sizing_mode="stretch_width",
            margin=(4, 0, 0, 0),
        )
        self.btn_run.on_click(self._on_annotate)

        default_export_dir = str(Path(self.controler.output_directory) / "annotated_external")
        self.export_dir_input = pn.widgets.TextInput(
            value=default_export_dir,
            placeholder="/path/to/output",
            sizing_mode="stretch_width",
            height=38,
            margin=0,
            disabled=True,
        )
        self.export_dir_browse_btn = pn.widgets.Button(
            name="Browse", button_type="default", width=90, height=38, margin=0, disabled=True
        )
        self.export_dir_browse_btn.on_click(self._browse_export_dir)

        self.btn_export = pn.widgets.Button(
            name="Export Results",
            button_type="success",
            height=38,
            sizing_mode="stretch_width",
            disabled=True,
            margin=(6, 0, 0, 0),
        )
        self.btn_export.on_click(self._on_export)

        self.info_msg = pn.pane.HTML("", sizing_mode="stretch_width")

        return pn.Column(
            pn.pane.HTML("<div class='section-header'>Dataset to Annotate</div>"),
            pn.Row(
                self.dataset_input,
                self.dataset_browse_btn,
                sizing_mode="stretch_width",
                align="center",
                margin=0,
            ),
            pn.Spacer(height=6),
            self.dataset_load_btn,
            self.dataset_status,
            pn.Spacer(height=8),
            pn.pane.HTML("<div class='section-header'>Select Models</div>"),
            self.toggle_syn,
            pn.Spacer(height=4),
            self.toggle_nsyn,
            pn.Spacer(height=4),
            self.toggle_ens,
            pn.Spacer(height=12),
            self.btn_run,
            pn.Spacer(height=8),
            pn.pane.HTML("<div class='section-header'>Export Directory</div>"),
            pn.Row(
                self.export_dir_input,
                self.export_dir_browse_btn,
                sizing_mode="stretch_width",
                align="center",
                margin=0,
            ),
            self.btn_export,
            pn.Spacer(height=6),
            self.info_msg,
            css_classes=["annotate-card"],
            sizing_mode="stretch_width",
            min_width=240,
            max_width=360,
        )

    def _build_monitor_panel(self):
        self.syn_indicator = pn.indicators.LoadingSpinner(value=False, width=60, height=60)
        self.nsyn_indicator = pn.indicators.LoadingSpinner(value=False, width=60, height=60)
        self.ens_indicator = pn.indicators.LoadingSpinner(value=False, width=60, height=60)

        self.syn_status = pn.pane.HTML(_idle())
        self.nsyn_status = pn.pane.HTML(_idle())
        self.ens_status = pn.pane.HTML(_idle())

        return pn.Column(
            pn.pane.HTML("<div class='section-header'>Process Monitor</div>"),
            pn.Row(
                _spinner_col("Syn-ESN",  self.syn_indicator,  self.syn_status),
                _spinner_col("NSyn-ESN", self.nsyn_indicator, self.nsyn_status),
                _spinner_col("Ensemble", self.ens_indicator,  self.ens_status),
                sizing_mode="stretch_width",
            ),
            css_classes=["annotate-card"],
            sizing_mode="stretch_both",
        )

    def _browse_dataset(self, event):
        directory = pick_directory("Select Audio Folder to Annotate")
        if directory:
            self.dataset_input.value = directory

    def _load_external_dataset(self, event):
        folder_str = self.dataset_input.value.strip()
        if not folder_str:
            self.dataset_status.object = (
                "<span style='color:#dc2626;font-size:12px;'>Please enter or browse a folder path.</span>"
            )
            return
        folder_path = Path(folder_str)
        if not folder_path.exists():
            self.dataset_status.object = (
                f"<span style='color:#dc2626;font-size:12px;'>Folder not found: {folder_str}</span>"
            )
            return

        self.dataset_status.object = "<span style='color:#d97706;font-size:12px;'>Loading...</span>"

        audio_files = sorted(
            [f for f in folder_path.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS]
        )
        if not audio_files:
            self.dataset_status.object = (
                "<span style='color:#dc2626;font-size:12px;'>No audio files found in folder.</span>"
            )
            return

        durations = []
        valid_files = []
        for p in audio_files:
            try:
                info = sf.info(str(p))
                durations.append(info.duration)
                valid_files.append(p)
            except Exception:
                pass

        if not valid_files:
            self.dataset_status.object = (
                "<span style='color:#dc2626;font-size:12px;'>No readable audio files found.</span>"
            )
            return

        data = pd.DataFrame({
            "label": ["UNLABELED"] * len(valid_files),
            "onset_s": [0.0] * len(valid_files),
            "offset_s": durations,
            "notated_path": [str(p.resolve()) for p in valid_files],
            "annot_path": [str(p.with_suffix(".csv").resolve()) for p in valid_files],
            "sequence": [0] * len(valid_files),
            "annotation": [p.stem for p in valid_files],
        })
        self._external_corpus = Corpus.from_df(
            df=data,
            annots_directory=None,
            config=self.controler.config,
            seq_ids=data["notated_path"],
        )
        self._external_corpus.audio_directory = str(folder_path.resolve())
        self._external_corpus.spec_directory = str(folder_path.resolve())
        self._external_corpus.spec_ext = ".mfcc.npz"

        total_s = sum(durations)
        self.dataset_status.object = (
            f"<span style='color:#16a34a;font-weight:600;font-size:12px;'>"
            f"✓ {len(valid_files)} file(s) — {total_s:.1f} s ({total_s/3600:.2f} h) total</span>"
        )
        logger.info(f"External corpus loaded: {len(valid_files)} files from {folder_path}")

    def _on_toggle_change(self, event):
        event.obj.button_type = "primary" if event.new else "default"

    def _on_annotate(self, event):
        if not self._external_corpus:
            self.info_msg.object = (
                "<span style='color:#dc2626;font-size:12px;'>No corpus loaded. Load a dataset first.</span>"
            )
            return

        chosen = []
        if self.toggle_syn.value:  chosen.append("syn-esn")
        if self.toggle_nsyn.value: chosen.append("nsyn-esn")
        if self.toggle_ens.value:  chosen.append("ensemble")

        if not chosen:
            self.info_msg.object = (
                "<span style='color:#d97706;font-size:12px;'>Please select at least one model.</span>"
            )
            return

        if "ensemble" in chosen:
            if not self.toggle_syn.value:  self.toggle_syn.value = True
            if not self.toggle_nsyn.value: self.toggle_nsyn.value = True
            if "syn-esn"  not in chosen: chosen.append("syn-esn")
            if "nsyn-esn" not in chosen: chosen.append("nsyn-esn")

        self._available_models = self._scan_models()
        missing = [m for m in chosen if m not in self._available_models]
        if missing:
            self.info_msg.object = (
                "<span style='color:#dc2626;font-size:12px;'>"
                + " &nbsp;·&nbsp; ".join(f"No <b>{m}</b> model found" for m in missing)
                + "</span>"
            )
            return

        models_to_run = {k: v for k, v in self._available_models.items() if k in chosen}

        self.btn_run.disabled = True
        self.info_msg.object = "<span style='color:#6b7280;font-size:12px;'>Running...</span>"

        for stat_obj in [self.syn_status, self.nsyn_status, self.ens_status]:
            _set_status(stat_obj, "idle")

        try:
            all_preds = {}

            if "syn-esn" in chosen:
                _set_status(self.syn_status, "annotating")
                self.syn_indicator.value = True
                tic = time.time()
                preds = self.controler.annotate_external(
                    self._external_corpus,
                    model_sources={"syn-esn": models_to_run["syn-esn"]},
                    use_in_memory=False,
                )
                all_preds.update(preds)
                _set_status(self.syn_status, "done", time.time() - tic)
                self.syn_indicator.value = False
            else:
                _set_status(self.syn_status, "skipped")

            if "nsyn-esn" in chosen:
                _set_status(self.nsyn_status, "annotating")
                self.nsyn_indicator.value = True
                tic = time.time()
                preds = self.controler.annotate_external(
                    self._external_corpus,
                    model_sources={"nsyn-esn": models_to_run["nsyn-esn"]},
                    use_in_memory=False,
                )
                all_preds.update(preds)
                _set_status(self.nsyn_status, "done", time.time() - tic)
                self.nsyn_indicator.value = False
            else:
                _set_status(self.nsyn_status, "skipped")

            if "ensemble" in chosen:
                _set_status(self.ens_status, "annotating")
                self.ens_indicator.value = True
                tic = time.time()
                ens_preds = self.controler.annotate_external(
                    self._external_corpus,
                    model_sources={"ensemble": models_to_run["ensemble"]},
                    use_in_memory=False,
                )
                all_preds.update(ens_preds)
                _set_status(self.ens_status, "done", time.time() - tic)
                self.ens_indicator.value = False
            else:
                _set_status(self.ens_status, "skipped")

            self._predictions = all_preds
            self.btn_export.disabled = False
            self.export_dir_input.disabled = False
            self.export_dir_browse_btn.disabled = False
            self.info_msg.object = (
                "<span style='color:#16a34a;font-weight:600;font-size:12px;'>All tasks completed.</span>"
            )
            logger.info("Annotation sequence finished.")

        except Exception as e:
            logger.exception("Annotation Error")
            self.info_msg.object = f"<span style='color:#dc2626;font-size:12px;'>Error: {str(e)}</span>"
            self.syn_indicator.value = False
            self.nsyn_indicator.value = False
            self.ens_indicator.value = False
        finally:
            self.btn_run.disabled = False

    def _browse_export_dir(self, event):
        directory = pick_directory("Select Export Directory")
        if directory:
            self.export_dir_input.value = directory

    def _on_export(self, event):
        if not self._predictions:
            return
        try:
            out_dir = Path(self.export_dir_input.value.strip()) if self.export_dir_input.value.strip() else Path(self.controler.output_directory) / "annotated_external"
            from datetime import datetime
            ts = datetime.now().strftime("%Y-%m-%d_%Hh%Mmin%S")
            for name, pred in self._predictions.items():
                target = out_dir / ts / name
                self.controler.export_predictions(pred, target)
            self.info_msg.object = (
                f"<span style='color:#16a34a;font-size:12px;'>Exported to: {out_dir.name}/{ts}</span>"
            )
        except Exception as e:
            self.info_msg.object = f"<span style='color:#dc2626;font-size:12px;'>Export error: {e}</span>"

    def _scan_models(self):
        found = {}
        model_root = getattr(self.controler, "model_root", None)
        if model_root is None:
            return found
        export_dir = Path(model_root) / "exported_models"
        if not export_dir.exists():
            return found
        for name in KNOWN_ANNOTATORS:
            candidate = export_dir / name
            if candidate.exists():
                found[name] = str(candidate)
        return found

    def _validate_audio_on_init(self):
        audio_dir = self.controler.audio_directory
        if audio_dir and Path(audio_dir).exists():
            self.dataset_input.value = str(audio_dir)
            self._load_external_dataset(None)
