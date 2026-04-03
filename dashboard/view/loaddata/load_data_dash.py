# Author: Axel Arnaud at 20/01/2026 <axel.arnaud<at>inria.fr>
# Licence: MIT License
# Copyright: Axel Arnaud
import logging
from pathlib import Path
import panel as pn
from ..helpers import SubDash, SideBar
from canapy.corpus import Corpus
from canapy.correction import Corrector
from canapy.transforms.commons.training import split_train_test

logger = logging.getLogger("canapy")


def _find_long_audio_files(audio_dir: Path, ext: str, max_duration_s: float) -> list:
    """Return paths of audio files whose duration exceeds max_duration_s.
    Uses soundfile.info() — reads only the file header, no audio data loaded.
    """
    try:
        import soundfile as sf
    except ImportError:
        return []
    long_files = []
    for p in sorted(audio_dir.rglob(f"*{ext}")):
        try:
            info = sf.info(str(p))
            if info.duration > max_duration_s:
                long_files.append(p)
        except Exception:
            continue
    return long_files


FORM_CSS = """
:host {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --primary-color: #4f46e5;
    --bg-color: #f3f4f6;
    --card-bg: #ffffff;
    --text-main: #111827;
    --text-muted: #6b7280;
    --border-color: #e5e7eb;
}
.main-dashboard-area {
    background-color: var(--bg-color);
    padding: 20px;
    height: 100%;
    width: 100%;
    box-sizing: border-box;
    overflow-y: auto;
}
.form-card {
    background-color: var(--card-bg);
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    border: 1px solid var(--border-color);
    padding: 25px 30px; 
    width: 100%;
    max-width: 850px;
    margin: 0 auto;
    box-sizing: border-box;
}
@media (max-width: 768px) {
    .form-card {
        padding: 15px;
    }
    .main-dashboard-area {
        padding: 10px;
    }
}
.page-title {
    font-size: 24px;
    font-weight: 800;
    color: var(--text-main);
    margin-bottom: 5px;
    letter-spacing: -0.025em;
}
.page-subtitle {
    font-size: 14px;
    color: var(--text-muted);
    margin-bottom: 15px;
}
.section-header {
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 2px solid var(--border-color);
    padding-bottom: 5px;
    margin-top: 10px;
    margin-bottom: 10px;
}
.input-label {
    font-size: 13px;
    font-weight: 600;
    color: #374151;
    margin-bottom: 4px;
    display: block;
}
.form-card .bk-input {
    border-radius: 6px !important;
    border: 1px solid var(--border-color) !important;
    font-size: 13px !important;
    color: var(--text-main) !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    transition: border-color 0.2s, box-shadow 0.2s;
    height: 38px !important; 
}
.form-card .bk-input:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important;
}
.form-card .bk-btn {
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    height: 38px !important; 
    line-height: 1.5 !important;
    margin: 0 !important; 
}
.form-card .bk-btn-default {
    background-color: #ffffff !important;
    color: #374151 !important;
    border: 1px solid #d1d5db !important;
}
.form-card .bk-btn-default:hover {
    background-color: #f9fafb !important;
    border-color: #9ca3af !important;
}
.form-card .action-button .bk-btn-primary {
    background-color: var(--primary-color) !important;
    border: none !important;
    font-size: 15px !important; 
    height: 45px !important; 
}
.form-card .action-button .bk-btn-primary:hover {
    background-color: #4338ca !important;
}
"""

class LoadDataDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)
        
        pn.config.raw_css.append(FORM_CSS)
        
        self.sidebar = SideBar(self, "Load Data")
        self.data_loaded = False

        self.header = pn.Column(
            pn.pane.Markdown("Load Data", css_classes=['page-title'], margin=0),
            pn.pane.Markdown("Configure your dataset sources and output targets.", css_classes=['page-subtitle'], margin=0),
            sizing_mode="stretch_width"
        )
        
        self.mode_label = pn.pane.HTML("<span class='input-label'>Loading Mode</span>", margin=0)
        self.mode_selector = pn.widgets.RadioBoxGroup(
            name="Loading Mode",
            options={"Combined Folder (Audio + Annots)": "combined", "Separate Folders": "separate"},
            value="combined",
            inline=True,
            sizing_mode="stretch_width",
            margin=(5, 0, 10, 0)
        )
        self.mode_selector.param.watch(self._update_mode_visibility, 'value')
        
        def create_input_block(label_text, placeholder, browse_callback, tooltip_text=None):
            lbl = pn.pane.HTML(f"<span class='input-label'>{label_text}</span>", margin=(0, 0, 2, 0))
            inp = pn.widgets.TextInput(placeholder=placeholder, sizing_mode="stretch_width", height=38, margin=0)
            btn = pn.widgets.Button(name="Browse", button_type="default", width=100, height=38, margin=0)
            btn.on_click(browse_callback)
            
            row_elements = [inp]
            if tooltip_text:
                tt = pn.widgets.TooltipIcon(
                    value=tooltip_text,
                    margin=(0, 10), align='center'
                )
                row_elements.append(tt)
            else:
                row_elements.append(pn.Spacer(width=10))

            row_elements.append(btn)
            
            row = pn.Row(*row_elements, sizing_mode="stretch_width", margin=0, align='center')
            container = pn.Column(lbl, row, sizing_mode="stretch_width", margin=(0, 0, 10, 0))
            return container, inp, btn

        self.combined_block, self.combined_input, self.combined_browse = \
            create_input_block(
                "Data Directory (-d)", 
                "/path/to/dataset", 
                self._browse_combined,
                "Select a folder containing BOTH audio and annotation files."
            )
        
        self.audio_block, self.audio_input, self.audio_browse = \
            create_input_block(
                "Audio Directory (-s)", 
                "/path/to/audio", 
                self._browse_audio,
                "Select the folder containing your raw audio files (.wav).\nSubdirectories are supported."
            )
        self.audio_block.visible = False 
        
        self.annots_block, self.annots_input, self.annots_browse = \
            create_input_block(
                "Annotations Directory (-a)", 
                "/path/to/annotations", 
                self._browse_annots,
                "Select the folder containing your annotation files (.csv or .txt).\nFilenames must match audio files."
            )
        self.annots_block.visible = False
        
        self.model_block, self.model_input, self.model_browse = \
            create_input_block(
                "Model Directory (-m)",
                "Optional — load pre-trained models",
                self._browse_model,
                "Select the directory where trained models and checkpoints will be loaded from."
            )
        
        self.output_block, self.output_input, self.output_browse = \
            create_input_block(
                "Output Directory (-o)", 
                "Defaults to ./output if empty", 
                self._browse_output,
                "Select the directory where results and logs will be saved."
            )
        
        def create_select_block(label, options, value):
            lbl = pn.pane.HTML(f"<span class='input-label'>{label}</span>", margin=(0, 0, 2, 0))
            sel = pn.widgets.Select(options=options, value=value, sizing_mode="stretch_width", height=38, margin=0)
            return pn.Column(lbl, sel, sizing_mode="stretch_width", min_width=180), sel

        self.audio_ext_block, self.audio_ext_input = create_select_block("Audio Extension", [".wav", ".npy"], ".wav")

        # SR detection section
        self._detected_sr = None
        self.sr_detected_display = pn.pane.HTML(
            "<span style='color:#6b7280;font-size:13px;'>Not detected — select a data directory above.</span>",
            sizing_mode="stretch_width",
            align="center",
        )
        self.downsample_toggle = pn.widgets.Toggle(
            name="Disabled",
            value=False,
            button_type="default",
            sizing_mode="stretch_width",
        )
        _target_sr_lbl = pn.pane.HTML("<span class='input-label'>Target SR (Hz)</span>", margin=(0, 0, 2, 0))
        self.target_sr_input = pn.widgets.IntInput(
            value=22050,
            start=1000,
            sizing_mode="stretch_width",
            height=38,
            margin=0,
        )
        self.target_sr_block = pn.Column(
            _target_sr_lbl,
            self.target_sr_input,
            sizing_mode="stretch_width",
            margin=(6, 0, 0, 0),
            visible=False,
        )

        def _on_downsample_toggle(event):
            self.downsample_toggle.name = "Enabled" if event.new else "Disabled"
            self.downsample_toggle.button_type = "success" if event.new else "default"
            self.target_sr_block.visible = event.new

        self.downsample_toggle.param.watch(_on_downsample_toggle, "value")
        self.combined_input.param.watch(lambda e: self._auto_detect_sr(), "value")
        self.audio_input.param.watch(lambda e: self._auto_detect_sr(), "value")

        self.global_status = pn.pane.Alert(
            "Please configure your paths below.", alert_type="info", sizing_mode="stretch_width", visible=False
        )
        
        self.btn_load = pn.widgets.Button(
            name="Initialize Dataset", button_type="primary", 
            sizing_mode="stretch_width", height=45, css_classes=['action-button']
        )
        self.btn_load.on_click(self._load_data)

        if getattr(self.controler, 'audio_directory', None):
            audio_dir = self.controler.audio_directory
            annots_dir = getattr(self.controler, 'annots_directory', None)
            
            if audio_dir == annots_dir:
                self.mode_selector.value = "combined"
                self.combined_input.value = str(audio_dir)
            else:
                self.mode_selector.value = "separate"
                self.audio_input.value = str(audio_dir)
                if annots_dir:
                    self.annots_input.value = str(annots_dir)
            
            self._update_mode_visibility(type('Event', (object,), {'new': self.mode_selector.value})())

        if getattr(self.controler, 'model_root', None):
            self.model_input.value = str(self.controler.model_root)

        if getattr(self.controler, 'output_directory', None):
            self.output_input.value = str(self.controler.output_directory)

        if getattr(self.controler, 'audio_ext', None):
            self.audio_ext_input.value = self.controler.audio_ext
        
        card_content = pn.Column(
            self.header,

            pn.pane.HTML("<div class='section-header'>1. Source Selection</div>"),
            self.mode_label,
            self.mode_selector,
            
            self.combined_block,
            self.audio_block,
            self.annots_block,

            pn.pane.HTML("<div class='section-header'>2. Sampling Rate</div>"),
            pn.Row(
                pn.pane.HTML("<b>Detected SR</b>", width=120, align="center"),
                self.sr_detected_display,
                pn.widgets.TooltipIcon(
                    value="Sample rate detected from the first audio file found in the selected directory.",
                    margin=(0, 10), align="center",
                ),
                sizing_mode="stretch_width",
                margin=(4, 0),
            ),
            pn.Row(
                pn.pane.HTML("<b>Downsample</b>", width=120, align="center"),
                self.downsample_toggle,
                pn.widgets.TooltipIcon(
                    value="Enable to resample audio to a lower target sample rate during processing.",
                    margin=(0, 10), align="center",
                ),
                sizing_mode="stretch_width",
                margin=(4, 0),
            ),
            self.target_sr_block,

            pn.pane.HTML("<div class='section-header'>3. Configuration</div>"),
            self.model_block,
            self.output_block,
            
            pn.Spacer(height=5),
            pn.FlexBox(
                self.audio_ext_block,
                flex_wrap="wrap",
                gap=20,
                sizing_mode="stretch_width"
            ),
            
            pn.Spacer(height=15),
            self.global_status,
            pn.Spacer(height=5),
            
            self.btn_load,
            
            css_classes=['form-card'],
            sizing_mode="stretch_width",
            max_width=850,
        )

        self.layout = pn.Row(
            self.sidebar,
            pn.Column(
                pn.Spacer(height=20),
                card_content,
                pn.Spacer(height=20),
                css_classes=['main-dashboard-area'],
                sizing_mode="stretch_both",
            ),
            sizing_mode="stretch_both",
            margin=0
        )
    
    def _update_mode_visibility(self, event):
        is_separate = event.new == "separate"
        self.combined_block.visible = not is_separate
        self.annots_block.visible = is_separate
        self.audio_block.visible = is_separate
    
    def _browse_combined(self, event):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        directory = filedialog.askdirectory(title="Select Data Directory")
        if directory:
            self.combined_input.value = directory
            logger.info(f"Selected combined directory: {directory}")
        root.destroy()
    
    def _browse_annots(self, event):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        directory = filedialog.askdirectory(title="Select Annotations Directory")
        if directory:
            self.annots_input.value = directory
            logger.info(f"Selected annotations directory: {directory}")
        root.destroy()
    
    def _browse_audio(self, event):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        directory = filedialog.askdirectory(title="Select Audio Directory")
        if directory:
            self.audio_input.value = directory
            logger.info(f"Selected audio directory: {directory}")
        root.destroy()
    
    def _browse_model(self, event):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        directory = filedialog.askdirectory(title="Select Model Directory (-m)")
        if directory:
            self.model_input.value = directory
            logger.info(f"Selected model directory: {directory}")
        root.destroy()
    
    def _browse_output(self, event):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_input.value = directory
            logger.info(f"Selected output directory: {directory}")
        root.destroy()

    def _validate_path(self, path_str, must_exist=True):
        if not path_str or path_str.strip() == "":
            return False, "Path is empty"
        path = Path(path_str)
        if must_exist and not path.exists():
            return False, f"Path does not exist: {path}"
        return True, None
    
    def _load_data(self, event):
        try:
            self.global_status.object = "Loading data..."
            self.global_status.alert_type = "info"
            self.global_status.visible = True
            
            if not self.output_input.value or self.output_input.value.strip() == "":
                default_out = Path.cwd() / "output"
                try:
                    default_out.mkdir(parents=True, exist_ok=True)
                    self.output_input.value = str(default_out)
                except Exception as e:
                    self.global_status.object = f"Error creating default output: {e}"
                    self.global_status.alert_type = "danger"
                    return

            is_combined = self.mode_selector.value == "combined"
            
            if is_combined:
                valid, err = self._validate_path(self.combined_input.value)
                if not valid:
                    self.global_status.object = f"Error: {err}"
                    self.global_status.alert_type = "danger"
                    return
                audio_dir = Path(self.combined_input.value)
                annots_dir = Path(self.combined_input.value)
            else:
                valid_audio, err_audio = self._validate_path(self.audio_input.value)
                valid_annots, err_annots = self._validate_path(self.annots_input.value)
                if not valid_audio:
                    self.global_status.object = f"Error in audio path: {err_audio}"
                    self.global_status.alert_type = "danger"
                    return
                if not valid_annots:
                    self.global_status.object = f"Error in annotations path: {err_annots}"
                    self.global_status.alert_type = "danger"
                    return
                audio_dir = Path(self.audio_input.value)
                annots_dir = Path(self.annots_input.value)
            
            output_dir = Path(self.output_input.value)
            
            model_root = None
            has_model = False

            if self.model_input.value and self.model_input.value.strip() != "":
                m_path = Path(self.model_input.value)
                if m_path.is_dir():
                    model_root = m_path
                    has_model = True
                else:
                    logger.warning("Model path provided but does not exist or is not a directory.")

            self.controler.audio_directory = audio_dir
            self.controler.annots_directory = annots_dir
            self.controler.output_directory = output_dir
            self.controler.config_path = None
            self.controler.model_root = model_root
            config_path = None
            self.controler.annot_format = "marron1csv"
            self.controler.audio_ext = self.audio_ext_input.value
            
            logger.info("Creating corpus...")
            self.controler.corpus = Corpus.from_directory(
                audio_directory=audio_dir,
                annots_directory=annots_dir,
                config_path=config_path if config_path else None,
                annot_format=self.controler.annot_format,
                audio_ext=self.controler.audio_ext,
            )
            self.controler.config = self.controler.corpus.config

            # Apply sampling rate from load-data section
            if self.downsample_toggle.value:
                target_sr = self.target_sr_input.value
            elif self._detected_sr:
                target_sr = self._detected_sr
            else:
                target_sr = None
            if target_sr:
                self.controler.config.data["transforms"]["audio"]["sampling_rate"] = target_sr

            self.controler.corrector = Corrector(
                output_dir / "checkpoints",
                [{"class": dict(), "annot": dict()}],
            )
            self.controler.initialize_output()
            self.controler.initialize_models()
            self.controler.initialize_annots()
            self.controler.corpus = split_train_test(self.controler.corpus, redo=True)
            self.controler.compute_classes()
            
            self.controler._is_ready = True
            self.controler._step = "home"

            # Check for audio files that are too long to be processed safely.
            # Loading a 3h file at 96 kHz requires ~4 GB of RAM peak (raw audio
            # + resampling buffer), which will crash the training pipeline.
            _MAX_DURATION_S = 1800  # 30 minutes
            _long_files = _find_long_audio_files(audio_dir, self.audio_ext_input.value, _MAX_DURATION_S)

            if has_model:
                msg = "Data & Models loaded! (Annotation enabled)"
            else:
                msg = "Data loaded successfully! (Training pipeline enabled)"

            if _long_files:
                _names = ", ".join(f"<code>{p.name}</code>" for p in _long_files[:3])
                if len(_long_files) > 3:
                    _names += f" and {len(_long_files) - 3} more"
                self.global_status.object = (
                    f"{msg}<br><br>"
                    f"⚠️ <b>Warning — audio files too long to train on:</b> {_names}. "
                    f"Files longer than {_MAX_DURATION_S // 60} minutes will likely crash "
                    f"the training pipeline due to memory constraints. "
                    f"Please segment them into shorter clips before training."
                )
                self.global_status.alert_type = "warning"
            else:
                self.global_status.object = msg
                self.global_status.alert_type = "success"
            
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            self.global_status.object = f"Error: {str(e)}"
            self.global_status.alert_type = "danger"

    def _detect_sr_from_dir(self, directory: Path):
        """Return the sample rate of the first audio file found in directory, or None."""
        try:
            import soundfile as sf
            ext = self.audio_ext_input.value if hasattr(self, "audio_ext_input") else ".wav"
            for p in directory.rglob(f"*{ext}"):
                try:
                    return sf.info(str(p)).samplerate
                except Exception:
                    continue
        except ImportError:
            pass
        return None

    def _auto_detect_sr(self):
        """Auto-detect SR from the currently selected audio directory."""
        if self.mode_selector.value == "combined":
            dir_str = self.combined_input.value
        else:
            dir_str = self.audio_input.value

        if not dir_str or not dir_str.strip():
            return

        p = Path(dir_str)
        if not p.exists():
            return

        sr = self._detect_sr_from_dir(p)
        if sr:
            self._detected_sr = sr
            self.sr_detected_display.object = (
                f"<span style='color:#059669;font-weight:600;font-size:13px;'>{sr} Hz</span>"
            )
            if not self.downsample_toggle.value:
                self.target_sr_input.value = sr
        else:
            self._detected_sr = None
            self.sr_detected_display.object = (
                "<span style='color:#dc2626;font-size:13px;'>Could not detect — no audio files found.</span>"
            )