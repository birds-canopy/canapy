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
        
        def create_input_block(label_text, placeholder, browse_callback):
            lbl = pn.pane.HTML(f"<span class='input-label'>{label_text}</span>", margin=(0, 0, 2, 0))
            inp = pn.widgets.TextInput(placeholder=placeholder, sizing_mode="stretch_width", height=38, margin=0)
            btn = pn.widgets.Button(name="Browse", button_type="default", width=100, height=38, margin=0)
            btn.on_click(browse_callback)
            row = pn.Row(inp, pn.Spacer(width=10), btn, sizing_mode="stretch_width", margin=0, align='center')
            container = pn.Column(lbl, row, sizing_mode="stretch_width", margin=(0, 0, 10, 0))
            return container, inp, btn

        self.combined_block, self.combined_input, self.combined_browse = \
            create_input_block("Data Directory (-d)", "/path/to/dataset", self._browse_combined)
        
        self.audio_block, self.audio_input, self.audio_browse = \
            create_input_block("Audio Directory", "/path/to/audio", self._browse_audio)
        self.audio_block.visible = False 
        
        self.annots_block, self.annots_input, self.annots_browse = \
            create_input_block("Annotations Directory", "/path/to/annotations", self._browse_annots)
        self.annots_block.visible = False
        
        self.config_block, self.config_input, self.config_browse = \
            create_input_block("Config File / Model Directory (-c)", "Required for Annotation...", self._browse_config)
        
        self.output_block, self.output_input, self.output_browse = \
            create_input_block("Output Directory", "Defaults to ./output if empty", self._browse_output)
        
        def create_select_block(label, options, value):
            lbl = pn.pane.HTML(f"<span class='input-label'>{label}</span>", margin=(0, 0, 2, 0))
            sel = pn.widgets.Select(options=options, value=value, sizing_mode="stretch_width", height=38, margin=0)
            return pn.Column(lbl, sel, sizing_mode="stretch_width", min_width=180), sel

        self.annot_fmt_block, self.annot_format_input = create_select_block("Annotation Format", ["marron1csv", "raven", "audacity"], "marron1csv")
        self.audio_ext_block, self.audio_ext_input = create_select_block("Audio Extension", [".wav", ".npy"], ".wav")
        
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
            self.config_input.value = str(self.controler.model_root)
        elif getattr(self.controler, 'config_path', None):
            self.config_input.value = str(self.controler.config_path)

        if getattr(self.controler, 'output_directory', None):
            self.output_input.value = str(self.controler.output_directory)

        if getattr(self.controler, 'annot_format', None):
            self.annot_format_input.value = self.controler.annot_format
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
            
            pn.pane.HTML("<div class='section-header'>2. Configuration</div>"),
            self.config_block,
            self.output_block,
            
            pn.Spacer(height=5),
            pn.FlexBox(
                self.annot_fmt_block, 
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
            sizing_mode="stretch_width"
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
    
    def _browse_config(self, event):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        answer = filedialog.askdirectory(title="Select Model Directory (or Cancel for config file)")
        if answer:
            self.config_input.value = answer
            logger.info(f"Selected model directory: {answer}")
        else:
            answer = filedialog.askopenfilename(title="Select Config File", filetypes=[("TOML files", "*.toml"), ("All files", "*.*")])
            if answer:
                self.config_input.value = answer
                logger.info(f"Selected config file: {answer}")
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
            
            config_path = None
            model_root = None
            has_conf = False
            
            if self.config_input.value and self.config_input.value.strip() != "":
                c_path = Path(self.config_input.value)
                if c_path.exists():
                    has_conf = True
                    if c_path.is_dir():
                        model_root = c_path
                    elif c_path.is_file():
                        config_path = c_path
                else:
                    logger.warning("Config path provided but does not exist.")
            
            self.controler.audio_directory = audio_dir
            self.controler.annots_directory = annots_dir
            self.controler.output_directory = output_dir
            self.controler.config_path = config_path
            self.controler.model_root = model_root
            self.controler.annot_format = self.annot_format_input.value
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
            
            if has_conf:
                msg = "Data & Configuration loaded! (Training & Annotation enabled)"
            else:
                msg = "Data loaded successfully! (Training pipeline enabled)"
            
            self.global_status.object = msg
            self.global_status.alert_type = "success"
            
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            self.global_status.object = f"Error: {str(e)}"
            self.global_status.alert_type = "danger"