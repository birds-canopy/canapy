import logging
import time
from pathlib import Path
import panel as pn
import pandas as pd
import soundfile as sf
from ..helpers import SubDash, SideBar
from canapy.corpus import Corpus

logger = logging.getLogger("canapy-dashboard")

pn.extension()

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")
KNOWN_ANNOTATORS = ["syn-esn", "nsyn-esn", "ensemble"]

# CSS Minimaliste : Juste la police, pas de couleurs de fond bizarres
MINIMAL_CSS = """
:host {
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background-color: white; 
}
.title-text {
    font-size: 24px;
    font-weight: bold;
    color: #333;
    margin-bottom: 20px;
}
.section-header {
    font-size: 18px;
    font-weight: 600;
    color: #555;
    margin-top: 20px;
    border-bottom: 2px solid #eee;
    padding-bottom: 5px;
    margin-bottom: 15px;
}
"""

class AnnotateDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)
        pn.config.raw_css.append(MINIMAL_CSS)
        
        self.sidebar = SideBar(self, "Annotation")
        self.parent = parent
        self.controler = self.parent.controler

        # --- Partie Gauche : Configuration ---
        self.config_panel = self._build_config_panel()

        # --- Partie Droite : Visualisation (Spinners) ---
        self.monitor_panel = self._build_monitor_panel()

        self.layout = pn.Row(
            self.sidebar,
            self.config_panel,
            pn.Spacer(width=20),
            self.monitor_panel,
            sizing_mode="stretch_both",
            background="#ffffff",
            margin=(20, 20, 20, 0),
        )
        
        self._external_corpus = None
        self._predictions = {}
        self._available_models = self._scan_models()
        self._validate_audio_on_init()

    def _build_config_panel(self):
        audio_dir = self.controler.audio_directory or Path.cwd()
        
        # Affichage simple du chemin (Texte brut, pas de fond gris)
        path_label = pn.pane.Markdown(f"**Source:** {audio_dir}")

        # Toggles pour sélectionner les modèles
        self.toggle_syn = pn.widgets.Toggle(name="Syn-ESN", value=False, button_type="default", sizing_mode="stretch_width")
        self.toggle_nsyn = pn.widgets.Toggle(name="NSyn-ESN", value=False, button_type="default", sizing_mode="stretch_width")
        self.toggle_ens = pn.widgets.Toggle(name="Ensemble", value=False, button_type="default", sizing_mode="stretch_width")
        
        # Callbacks pour changer la couleur des boutons quand actifs
        for t in [self.toggle_syn, self.toggle_nsyn, self.toggle_ens]:
            t.param.watch(self._on_toggle_change, 'value')

        self.btn_run = pn.widgets.Button(name="START ANNOTATION", button_type="primary", height=60, sizing_mode="stretch_width")
        self.btn_run.on_click(self._on_annotate)

        self.btn_export = pn.widgets.Button(name="Export Results", button_type="success", height=40, sizing_mode="stretch_width", disabled=True)
        self.btn_export.on_click(self._on_export)

        self.info_msg = pn.pane.HTML("", sizing_mode="stretch_width")

        return pn.Column(
            pn.pane.HTML("<div class='title-text'>Configuration</div>"),
            path_label,
            pn.pane.HTML("<div class='section-header'>Select Models</div>"),
            self.toggle_syn,
            self.toggle_nsyn,
            self.toggle_ens,
            pn.Spacer(height=30),
            self.btn_run,
            pn.Spacer(height=10),
            self.btn_export,
            pn.Spacer(height=20),
            self.info_msg,
            sizing_mode="stretch_width",
            min_width=220,
            max_width=380,
        )

    def _build_monitor_panel(self):
        # Création des spinners comme dans Train (Gros : 100x100)
        self.syn_indicator = pn.indicators.LoadingSpinner(value=False, width=100, height=100, color='primary')
        self.nsyn_indicator = pn.indicators.LoadingSpinner(value=False, width=100, height=100, color='primary')
        self.ens_indicator = pn.indicators.LoadingSpinner(value=False, width=100, height=100, color='primary')

        # Status HTML comme dans Train
        self.syn_status = pn.pane.HTML("<h2>Idle</h2>")
        self.nsyn_status = pn.pane.HTML("<h2>Idle</h2>")
        self.ens_status = pn.pane.HTML("<h2>Idle</h2>")

        # Layout en colonnes pour chaque modèle
        return pn.Column(
            pn.pane.HTML("<div class='title-text'>Process Monitor</div>"),
            pn.FlexBox(
                pn.Column(
                    pn.pane.HTML("<h3>Syn-ESN</h3>"),
                    self.syn_indicator,
                    self.syn_status,
                    sizing_mode="stretch_width",
                    align="center",
                    min_width=150,
                ),
                pn.Column(
                    pn.pane.HTML("<h3>NSyn-ESN</h3>"),
                    self.nsyn_indicator,
                    self.nsyn_status,
                    sizing_mode="stretch_width",
                    align="center",
                    min_width=150,
                ),
                pn.Column(
                    pn.pane.HTML("<h3>Ensemble</h3>"),
                    self.ens_indicator,
                    self.ens_status,
                    sizing_mode="stretch_width",
                    align="center",
                    min_width=150,
                ),
                flex_wrap="wrap",
                gap=20,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )

    def switch_status(self, obj, status, duration=0.0):
        # Logique identique à TrainDashboard
        if status == "annotating":
            obj.object = "<h2>Annotating...</h2>"
            obj.style = {"color": "blue"}
        elif status == "done":
            obj.object = f"<h2>Done !</h2><p>{round(duration, 2)} sec.</p>"
            obj.style = {"color": "green"}
        elif status == "skipped":
            obj.object = "<h2>Skipped</h2>"
            obj.style = {"color": "gray"}
        elif status == "idle":
            obj.object = "<h2>Idle</h2>"
            obj.style = {"color": "black"}

    def _on_toggle_change(self, event):
        if event.new:
            event.obj.button_type = "primary"
        else:
            event.obj.button_type = "default"

    def _on_annotate(self, event):
        if not self._external_corpus:
            self.info_msg.object = "<span style='color:red'>Error: No corpus loaded. Check folder.</span>"
            return
        
        # Récupération des choix
        chosen = []
        if self.toggle_syn.value: chosen.append("syn-esn")
        if self.toggle_nsyn.value: chosen.append("nsyn-esn")
        if self.toggle_ens.value: chosen.append("ensemble")

        if not chosen:
            self.info_msg.object = "<span style='color:orange'>Please select a model.</span>"
            return

        if "ensemble" in chosen:
            # Force activation for ensemble
            if not self.toggle_syn.value: self.toggle_syn.value = True
            if not self.toggle_nsyn.value: self.toggle_nsyn.value = True
            if "syn-esn" not in chosen: chosen.append("syn-esn")
            if "nsyn-esn" not in chosen: chosen.append("nsyn-esn")

        models_to_run = {k: v for k, v in self._available_models.items() if k in chosen}
        
        self.btn_run.disabled = True
        self.info_msg.object = "Running..."
        
        # Reset visual status
        for stat_obj in [self.syn_status, self.nsyn_status, self.ens_status]:
            self.switch_status(stat_obj, "idle")

        try:
            all_preds = {}

            # --- SYN ---
            if "syn-esn" in chosen:
                self.switch_status(self.syn_status, "annotating")
                self.syn_indicator.value = True
                tic = time.time()
                
                preds = self.controler.annotate_external(
                    self._external_corpus,
                    model_sources={"syn-esn": models_to_run["syn-esn"]},
                    use_in_memory=False
                )
                all_preds.update(preds)
                
                toc = time.time()
                self.switch_status(self.syn_status, "done", duration=toc-tic)
                self.syn_indicator.value = False
            else:
                self.switch_status(self.syn_status, "skipped")

            # --- NSYN ---
            if "nsyn-esn" in chosen:
                self.switch_status(self.nsyn_status, "annotating")
                self.nsyn_indicator.value = True
                tic = time.time()
                
                preds = self.controler.annotate_external(
                    self._external_corpus,
                    model_sources={"nsyn-esn": models_to_run["nsyn-esn"]},
                    use_in_memory=False
                )
                all_preds.update(preds)
                
                toc = time.time()
                self.switch_status(self.nsyn_status, "done", duration=toc-tic)
                self.nsyn_indicator.value = False
            else:
                self.switch_status(self.nsyn_status, "skipped")

            # --- ENSEMBLE ---
            if "ensemble" in chosen:
                self.switch_status(self.ens_status, "annotating")
                self.ens_indicator.value = True
                tic = time.time()
                
                ens_preds = self.controler.annotate_external(
                    self._external_corpus,
                    model_sources={"ensemble": models_to_run["ensemble"]},
                    use_in_memory=False
                )
                all_preds.update(ens_preds)
                
                toc = time.time()
                self.switch_status(self.ens_status, "done", duration=toc-tic)
                self.ens_indicator.value = False
            else:
                self.switch_status(self.ens_status, "skipped")

            self._predictions = all_preds
            self.btn_export.disabled = False
            self.info_msg.object = "<span style='color:green; font-weight:bold'>All tasks completed.</span>"
            logger.info("Annotation sequence finished.")

        except Exception as e:
            logger.exception("Annotation Error")
            self.info_msg.object = f"<span style='color:red'>Error: {str(e)}</span>"
            # Stop spinners in case of crash
            self.syn_indicator.value = False
            self.nsyn_indicator.value = False
            self.ens_indicator.value = False
        finally:
            self.btn_run.disabled = False

    def _on_export(self, event):
        if not self._predictions: return
        try:
            out_dir = Path(self.controler.output_directory) / "annotated_external"
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for name, pred in self._predictions.items():
                target = out_dir / ts / name
                self.controler.export_predictions(pred, target)
            
            self.info_msg.object = f"Exported to:<br>{out_dir.name}/{ts}"
        except Exception as e:
            self.info_msg.object = f"Export Error: {e}"

    def _scan_models(self):
        found = {}
        model_root = getattr(self.controler, "model_root", None)
        if model_root is None: return found
        model_root = Path(model_root)
        if not model_root.exists(): return found
        for name in KNOWN_ANNOTATORS:
            candidates = list(model_root.rglob(name))
            if candidates:
                found[name] = str(sorted(candidates, reverse=True)[0])
        return found

    def _validate_audio_on_init(self):
        folder_path = Path(self.controler.audio_directory or Path.cwd())
        
        if not folder_path.exists():
            self.info_msg.object = "Folder not found."
            return
            
        audio_files = sorted([f for f in folder_path.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS])
        if not audio_files:
            self.info_msg.object = "No audio files found."
            return
        
        durations = []
        valid_files = []
        
        # Fast load check
        try:
            for p in audio_files:
                try:
                    info = sf.info(str(p))
                    durations.append(info.duration)
                    valid_files.append(p)
                except Exception:
                    pass
        except Exception:
            # Fallback librosa
            import librosa
            for p in audio_files:
                try:
                    d = librosa.get_duration(path=str(p))
                    durations.append(d)
                    valid_files.append(p)
                except Exception:
                    pass

        if not valid_files:
            self.info_msg.object = "No readable audio."
            return

        data = pd.DataFrame({
            "label": ["UNLABELED"] * len(valid_files),
            "onset_s": [0.0] * len(valid_files),
            "offset_s": durations,
            "notated_path": [str(p.resolve()) for p in valid_files],
            "annot_path": [str(p.with_suffix('.csv').resolve()) for p in valid_files],
            "sequence": [0] * len(valid_files),
            "annotation": [p.stem for p in valid_files],
        })
        
        self._external_corpus = Corpus.from_df(
            df=data, annots_directory=None, config=self.controler.config, seq_ids=data['notated_path']
        )
        self._external_corpus.audio_directory = str(folder_path.resolve())
        self._external_corpus.spec_directory = str(folder_path.resolve())
        self._external_corpus.spec_ext = ".mfcc.npz"
        
        self.info_msg.object = f"Loaded {len(valid_files)} files ({sum(durations):.1f}s)"