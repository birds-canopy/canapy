# Author: Axel Arnaud
# Licence: MIT License
# Copyright: Axel Arnaud
import logging
import math
from pathlib import Path
import panel as pn
from ..helpers import SubDash, SideBar

_PRESETS_DIR = Path(__file__).parents[3] / "config" / "presets"

_PRESET_PALETTE = [
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
    "#c5b0d5", "#c49c94", "#f7b6d2", "#dbdb8d",
    "#9edae5", "#c7c7c7",
]


def _deep_update(base, override):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _preset_btn_stylesheet(color, selected=False):
    border = "#374151" if selected else "transparent"
    return f"""
        button.bk-btn {{
            background-color: {color} !important;
            border: 2px solid {border} !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            color: #374151 !important;
            box-shadow: none !important;
            white-space: normal !important;
            line-height: 1.3 !important;
        }}
        button.bk-btn:hover {{
            opacity: 0.82 !important;
            border-color: #374151 !important;
        }}
    """

logger = logging.getLogger("canapy")

SETTINGS_CSS = """
:host {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.settings-section-header {
    font-size: 15px;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 8px;
    margin-bottom: 16px;
}
.settings-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 20px;
}
.settings-subsection-header {
    font-size: 12px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 5px;
    margin-top: 18px;
    margin-bottom: 10px;
}
"""

PARAM_HELP = {
    "sampling_rate": "Sampling rate (Hz) of your audio files. librosa will resample to this value. Common values: 22050, 44100, 48000.",
    "fmin": "Minimum frequency (Hz) for MFCC computation. Frequencies below this value are ignored.",
    "fmax": "Maximum frequency (Hz) for MFCC computation. Frequencies above this value are ignored.",
    "win_length": (
        "Window length (seconds) for spectrogram frames."
    ),
    "hop_length": (
        "Hop length (seconds) between successive frames. "
        "Auto-filled to win_length / 2 when win_length changes, but can be overridden."
    ),
    "n_fft": (
        "FFT window size (samples). "
        "Auto-filled to the next power of 2 ≥ win_length × SR when win_length changes, "
        "but can be overridden. Must be ≥ win_length × sampling_rate."
    ),
    "sr": "Spectral radius of the reservoir weight matrix. Controls the echo state property (typical: 0.1–1.5).",
    "leak": "Leak rate of the reservoir neurons. Controls the temporal memory of the reservoir (0 < leak ≤ 1).",
    "iss": "Input scaling for MFCC features. Controls how strongly raw MFCCs drive the reservoir.",
    "isd": "Input scaling for delta (first-order derivative) features.",
    "isd2": "Input scaling for delta-delta (second-order derivative) features.",
    "ridge": "Ridge regression regularization coefficient. Higher values reduce overfitting.",
    "opt_parallel": (
        "Normal (TPE): Bayesian sequential search, more sample-efficient. "
        "Parallel (Random): evaluates multiple configurations simultaneously — "
        "faster on multi-core machines but requires more total trials."
    ),
    "opt_max_percentage": (
        "Fraction of training sequences used during each optimization trial. "
        "Lower values speed up the search but may reduce accuracy. "
        "Recommended: 0.3–0.5 for large datasets, 1.0 for small ones."
    ),
    "opt_n_jobs": (
        "Number of parallel workers for the HP search (Parallel mode only). "
        "Each worker is a separate process — higher values speed up the search "
        "but use more CPU and RAM. Recommended: 4–8. "
        "Has no effect in Normal (TPE) mode."
    ),
    "opt_max_evals": (
        "Maximum number of hyperparameter configurations to evaluate. "
        "Higher values improve the chance of finding the best configuration "
        "but increase total search time. Recommended: 50–200."
    ),
    "merge_consecutive_labels": (
        "If enabled, consecutive annotations with the same label separated by a silence "
        "shorter than min_silence_gap are merged into a single annotation. "
        "Disable if your protocol distinguishes repeated identical labels."
    ),
}


def _make_param_row(label: str, widget: pn.widgets.Widget, param_name: str):
    tt = pn.widgets.TooltipIcon(
        value=PARAM_HELP[param_name],
        margin=(0, 10), align='center'
    )
    row = pn.Row(
        pn.pane.HTML(f"<b>{label}</b>", width=180, align="center"),
        widget,
        tt,
        sizing_mode="stretch_width",
        margin=(4, 0),
    )
    return pn.Column(row, sizing_mode="stretch_width", margin=(0, 0, 8, 0))


class SettingsDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        pn.config.raw_css.append(SETTINGS_CSS)

        self.sidebar = SideBar(self, "Settings")

        self.status = pn.pane.Alert(
            "Modify parameters below and click Apply.",
            alert_type="info",
            sizing_mode="stretch_width",
        )

        cfg = self.controler.config

        self.fmin_input = pn.widgets.IntInput(
            value=int(cfg.transforms.audio.data["fmin"]),
            start=0,
            end=22050,
            step=100,
            sizing_mode="stretch_width",
        )
        self.fmax_input = pn.widgets.IntInput(
            value=int(cfg.transforms.audio.data["fmax"]),
            start=0,
            end=22050,
            step=100,
            sizing_mode="stretch_width",
        )
        self.win_length_input = pn.widgets.FloatInput(
            value=float(cfg.transforms.audio.data["win_length"]),
            start=0.001,
            end=1.0,
            step=0.005,
            sizing_mode="stretch_width",
        )
        _win = float(cfg.transforms.audio.data["win_length"])
        _sr = cfg.transforms.audio.data.get("sampling_rate", 44100)
        self.hop_length_input = pn.widgets.FloatInput(
            value=round(_win / 2, 6),
            start=0.0001,
            step=0.001,
            sizing_mode="stretch_width",
        )
        self.n_fft_input = pn.widgets.IntInput(
            value=2 ** math.ceil(math.log2(max(_win * int(_sr), 1))),
            start=64,
            sizing_mode="stretch_width",
        )

        self.win_length_input.param.watch(self._on_win_length_change, "value")

        self.sr_input = pn.widgets.FloatInput(
            value=float(cfg.model.syn.data["sr"]),
            start=0.0,
            end=5.0,
            step=0.05,
            sizing_mode="stretch_width",
        )
        self.leak_input = pn.widgets.FloatInput(
            value=float(cfg.model.syn.data["leak"]),
            start=0.0001,
            end=1.0,
            step=0.01,
            sizing_mode="stretch_width",
        )
        self.iss_input = pn.widgets.FloatInput(
            value=float(cfg.model.syn.data["iss"]),
            start=1e-6,
            end=1.0,
            step=0.0001,
            sizing_mode="stretch_width",
        )
        self.isd_input = pn.widgets.FloatInput(
            value=float(cfg.model.syn.data["isd"]),
            start=1e-6,
            end=1.0,
            step=0.001,
            sizing_mode="stretch_width",
        )
        self.isd2_input = pn.widgets.FloatInput(
            value=float(cfg.model.syn.data["isd2"]),
            start=1e-6,
            end=1.0,
            step=0.001,
            sizing_mode="stretch_width",
        )
        self.ridge_input = pn.widgets.FloatInput(
            value=float(cfg.model.syn.data["ridge"]),
            start=1e-12,
            end=1.0,
            step=1e-9,
            sizing_mode="stretch_width",
        )

        _merge_val = bool(cfg.transforms.annots.data.get("merge_consecutive_labels", True))
        self.merge_labels_input = pn.widgets.Toggle(
            name="Enabled" if _merge_val else "Disabled",
            value=_merge_val,
            button_type="success" if _merge_val else "default",
            sizing_mode="stretch_width",
        )

        def _on_merge_toggle(event):
            self.merge_labels_input.name = "Enabled" if event.new else "Disabled"
            self.merge_labels_input.button_type = "success" if event.new else "default"

        self.merge_labels_input.param.watch(_on_merge_toggle, "value")

        self.opt_parallel_input = pn.widgets.Select(
            options={"Normal (TPE)": False, "Parallel (Random)": True},
            value=self.controler.opt_parallel,
            sizing_mode="stretch_width",
        )
        self.opt_percentage_input = pn.widgets.FloatSlider(
            value=self.controler.opt_max_percentage,
            start=0.1,
            end=1.0,
            step=0.05,
            sizing_mode="stretch_width",
        )
        import os as _os
        self.opt_n_jobs_input = pn.widgets.IntSlider(
            value=self.controler.opt_n_jobs,
            start=1,
            end=max(1, _os.cpu_count() or 8),
            step=1,
            sizing_mode="stretch_width",
        )
        self.opt_max_evals_input = pn.widgets.IntInput(
            value=self.controler.opt_max_evals,
            start=10,
            step=10,
            sizing_mode="stretch_width",
        )

        self.advanced_toggle = pn.widgets.Toggle(
            name="Show advanced parameters",
            value=False,
            button_type="default",
            sizing_mode="stretch_width",
        )

        # --- Preset state ---
        self._selected_preset_path = None
        self._preset_btns = []
        self._preset_paths = []
        self._preset_colors = []

        # --- Load Config widgets ---
        self.config_path_input = pn.widgets.TextInput(
            placeholder="Path to .toml config file...",
            sizing_mode="stretch_width",
            height=36,
            margin=0,
        )
        self.config_browse_btn = pn.widgets.Button(
            name="Browse", button_type="default", width=80, height=36, margin=0,
        )
        self.config_browse_btn.on_click(self._browse_config)
        self.config_load_btn = pn.widgets.Button(
            name="Load Config", button_type="success",
            sizing_mode="stretch_width", height=36,
        )
        self.config_load_btn.on_click(self._load_config)
        self.config_load_status = pn.pane.HTML("", margin=(4, 0, 0, 0))

        self.btn_apply = pn.widgets.Button(
            name="Apply",
            button_type="primary",
            sizing_mode="stretch_width",
            height=45,
        )
        self.btn_apply.on_click(self._apply)

        self.advanced_audio_block = pn.Column(
            _make_param_row("hop_length (s)", self.hop_length_input, "hop_length"),
            _make_param_row("n_fft (samples)", self.n_fft_input, "n_fft"),
            visible=False,
            sizing_mode="stretch_width",
        )

        self.reservoir_block = pn.Column(
            pn.pane.HTML("<div class='settings-subsection-header'>Reservoir (syn & nsyn)</div>"),
            _make_param_row("Spectral radius (sr)", self.sr_input, "sr"),
            _make_param_row("Leak rate (leak)", self.leak_input, "leak"),
            _make_param_row("Input scaling MFCC (iss)", self.iss_input, "iss"),
            _make_param_row("Input scaling delta (isd)", self.isd_input, "isd"),
            _make_param_row("Input scaling delta2 (isd2)", self.isd2_input, "isd2"),
            _make_param_row("Ridge coefficient", self.ridge_input, "ridge"),
            visible=False,
            sizing_mode="stretch_width",
        )

        def _on_advanced_toggle(event):
            self.advanced_toggle.button_type = "primary" if event.new else "default"
            self.advanced_audio_block.visible = event.new
            self.reservoir_block.visible = event.new

        self.advanced_toggle.param.watch(_on_advanced_toggle, "value")

        species_params_card = pn.Column(
            pn.pane.HTML("<div class='settings-section-header'>Species Parameters</div>"),
            pn.Row(
                pn.Spacer(),
                self.advanced_toggle,
                width=300,
                margin=(0, 0, 10, 0),
            ),

            pn.pane.HTML("<div class='settings-subsection-header'>Audio</div>"),
            _make_param_row("fmin (Hz)", self.fmin_input, "fmin"),
            _make_param_row("fmax (Hz)", self.fmax_input, "fmax"),
            _make_param_row("win_length (s)", self.win_length_input, "win_length"),
            self.advanced_audio_block,

            pn.pane.HTML("<div class='settings-subsection-header'>Annotations</div>"),
            _make_param_row("Merge consecutive labels", self.merge_labels_input, "merge_consecutive_labels"),

            self.reservoir_block,

            css_classes=["settings-card"],
            sizing_mode="stretch_width",
        )

        hp_card = pn.Column(
            pn.pane.HTML("<div class='settings-section-header'>Hyperparameter Search</div>"),
            _make_param_row("Search mode", self.opt_parallel_input, "opt_parallel"),
            _make_param_row("Max evaluations", self.opt_max_evals_input, "opt_max_evals"),
            _make_param_row("Data fraction", self.opt_percentage_input, "opt_max_percentage"),
            _make_param_row("Parallel workers (n_jobs)", self.opt_n_jobs_input, "opt_n_jobs"),
            css_classes=["settings-card"],
            sizing_mode="stretch_width",
        )

        preset_widget, self._preset_status = self._build_preset_selector()
        self.preset_apply_btn = pn.widgets.Button(
            name="Apply Preset",
            button_type="primary",
            sizing_mode="stretch_width",
            height=38,
        )
        self.preset_apply_btn.on_click(self._apply_preset)

        species_card = pn.Column(
            pn.pane.HTML("<div class='settings-section-header'>Species</div>"),
            pn.pane.HTML(
                "<span style='font-size:12px;color:#6b7280;'>"
                "Select a preset to pre-fill species parameters."
                "</span>",
                margin=(0, 0, 10, 0),
            ),
            preset_widget,
            self._preset_status,
            pn.Spacer(height=8),
            self.preset_apply_btn,

            pn.pane.HTML("<div class='settings-subsection-header'>Load Config</div>"),
            pn.pane.HTML(
                "<span style='font-size:12px;color:#6b7280;'>"
                "Override settings by loading a .toml config file."
                "</span>",
                margin=(0, 0, 6, 0),
            ),
            pn.Row(
                self.config_path_input,
                self.config_browse_btn,
                sizing_mode="stretch_width",
                margin=0,
                align="center",
            ),
            pn.Spacer(height=6),
            self.config_load_btn,
            self.config_load_status,

            css_classes=["settings-card"],
            sizing_mode="stretch_width",
        )

        params_column = pn.Column(
            species_params_card,
            hp_card,
            self.btn_apply,
            sizing_mode="stretch_width",
        )

        main_content = pn.Column(
            pn.pane.HTML("<h1 style='color:#1e293b;margin:0 0 6px 0;'>Settings</h1>"),
            self.status,
            pn.Spacer(height=20),
            pn.Row(
                species_card,
                pn.Spacer(width=24),
                params_column,
                sizing_mode="stretch_width",
                align="start",
            ),
            pn.Spacer(height=40),
            sizing_mode="stretch_width",
        )

        self.layout = pn.Row(
            self.sidebar,
            pn.Column(
                pn.Spacer(height=30),
                main_content,
                sizing_mode="stretch_width",
                styles={"padding": "0 40px", "overflow-y": "auto"},
            ),
            sizing_mode="stretch_both",
            background="#ffffff",
        )

    def _build_preset_selector(self):
        status = pn.pane.HTML("", margin=(4, 0, 0, 0))

        if not _PRESETS_DIR.exists():
            return pn.pane.HTML("<i style='color:#9ca3af;font-size:12px;'>No presets directory found.</i>"), status

        preset_files = sorted(
            p for p in _PRESETS_DIR.iterdir()
            if p.suffix in (".toml", ".yml", ".yaml") and p.is_file()
        )

        if not preset_files:
            return pn.pane.HTML("<i style='color:#9ca3af;font-size:12px;'>No presets found.</i>"), status

        for i, path in enumerate(preset_files):
            color = _PRESET_PALETTE[i % len(_PRESET_PALETTE)]
            name = path.stem.replace("_", " ").title()
            btn = pn.widgets.Button(
                name=name,
                stylesheets=[_preset_btn_stylesheet(color, selected=False)],
                width=120,
                height=48,
            )
            btn.on_click(lambda e, idx=i: self._on_preset_click(idx))
            self._preset_btns.append(btn)
            self._preset_paths.append(path)
            self._preset_colors.append(color)

        flex = pn.FlexBox(*self._preset_btns, flex_wrap="wrap", gap=8, sizing_mode="stretch_width")
        return flex, status

    def _on_preset_click(self, idx):
        path = self._preset_paths[idx]
        if self._selected_preset_path == path:
            self._selected_preset_path = None
            for i, (btn, color) in enumerate(zip(self._preset_btns, self._preset_colors)):
                btn.stylesheets = [_preset_btn_stylesheet(color, selected=False)]
            self._preset_status.object = ""
            return
        self._selected_preset_path = path
        for i, (btn, color) in enumerate(zip(self._preset_btns, self._preset_colors)):
            btn.stylesheets = [_preset_btn_stylesheet(color, selected=(i == idx))]
        name = path.stem.replace("_", " ").title()
        self._preset_status.object = (
            f"<span style='font-size:12px;color:#059669;'><b>{name}</b> selected.</span>"
        )

    def _apply_preset(self, _):
        if self._selected_preset_path is None:
            self._preset_status.object = (
                "<span style='font-size:12px;color:#dc2626;'>No preset selected.</span>"
            )
            return
        try:
            from config.config import Config as _Config
            preset = _Config.from_file(self._selected_preset_path)
            _deep_update(self.controler.config.data, preset.data)
            self._refresh_widgets_from_config()
            name = self._selected_preset_path.stem.replace("_", " ").title()
            self._preset_status.object = (
                f"<span style='font-size:12px;color:#059669;'>Preset <b>{name}</b> applied.</span>"
            )
            self.status.object = f"Preset '{name}' applied — click Apply to confirm."
            self.status.alert_type = "warning"
            logger.info(f"Applied preset: {self._selected_preset_path.stem}")
        except Exception as e:
            self._preset_status.object = (
                f"<span style='font-size:12px;color:#dc2626;'>Error: {e}</span>"
            )

    def _browse_config(self, _):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select Config File",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        root.destroy()
        if path:
            self.config_path_input.value = path

    def _load_config(self, _):
        path_str = self.config_path_input.value.strip()
        if not path_str:
            self.config_load_status.object = (
                "<span style='font-size:12px;color:#dc2626;'>Please select a config file first.</span>"
            )
            return
        path = Path(path_str)
        if not path.exists():
            self.config_load_status.object = (
                "<span style='font-size:12px;color:#dc2626;'>File not found.</span>"
            )
            return
        try:
            from config.config import Config as _Config
            loaded = _Config.from_file(path)
            _deep_update(self.controler.config.data, loaded.data)
            self._refresh_widgets_from_config()
            self.config_load_status.object = (
                "<span style='font-size:12px;color:#059669;'>Config loaded.</span>"
            )
            self.status.object = f"Config '{path.name}' loaded — click Apply to confirm."
            self.status.alert_type = "warning"
            logger.info(f"Loaded config: {path}")
        except Exception as e:
            self.config_load_status.object = (
                f"<span style='font-size:12px;color:#dc2626;'>Error: {e}</span>"
            )

    def _refresh_widgets_from_config(self):
        cfg = self.controler.config
        audio = cfg.transforms.audio.data
        self.fmin_input.value = int(audio["fmin"])
        self.fmax_input.value = int(audio["fmax"])
        win = float(audio["win_length"])
        sr = audio.get("sampling_rate", 44100)
        self.win_length_input.value = win
        self.hop_length_input.value = float(audio.get("hop_length", round(win / 2, 6)))
        self.n_fft_input.value = int(
            audio.get("n_fft", 2 ** math.ceil(math.log2(max(win * int(sr), 1))))
        )
        syn = cfg.model.syn.data
        self.sr_input.value = float(syn["sr"])
        self.leak_input.value = float(syn["leak"])
        self.iss_input.value = float(syn["iss"])
        self.isd_input.value = float(syn["isd"])
        self.isd2_input.value = float(syn["isd2"])
        self.ridge_input.value = float(syn["ridge"])
        merge = bool(cfg.transforms.annots.data.get("merge_consecutive_labels", True))
        self.merge_labels_input.value = merge
        self.merge_labels_input.name = "Enabled" if merge else "Disabled"
        self.merge_labels_input.button_type = "success" if merge else "default"

    def _on_win_length_change(self, event):
        win = event.new
        self.hop_length_input.value = round(win / 2, 6)
        sr = self.controler.config.data["transforms"]["audio"].get("sampling_rate", 44100)
        win_samples = win * int(sr)
        self.n_fft_input.value = 2 ** math.ceil(math.log2(max(win_samples, 1)))

    def _apply(self, _):
        try:
            self._validate()
        except ValueError as e:
            self.status.object = str(e)
            self.status.alert_type = "danger"
            return

        cfg = self.controler.config

        audio_data = cfg.data["transforms"]["audio"]
        audio_data["fmin"] = self.fmin_input.value
        audio_data["fmax"] = self.fmax_input.value
        audio_data["win_length"] = self.win_length_input.value
        audio_data["hop_length"] = self.hop_length_input.value
        audio_data["n_fft"] = self.n_fft_input.value

        for section in ("syn", "nsyn"):
            model_data = cfg.data["model"][section]
            model_data["sr"] = self.sr_input.value
            model_data["leak"] = self.leak_input.value
            model_data["iss"] = self.iss_input.value
            model_data["isd"] = self.isd_input.value
            model_data["isd2"] = self.isd2_input.value
            model_data["ridge"] = self.ridge_input.value

        cfg.data["transforms"]["annots"]["merge_consecutive_labels"] = self.merge_labels_input.value

        self.controler.opt_parallel = self.opt_parallel_input.value
        self.controler.opt_max_percentage = self.opt_percentage_input.value
        self.controler.opt_n_jobs = self.opt_n_jobs_input.value
        self.controler.opt_max_evals = self.opt_max_evals_input.value

        logger.info("Settings applied to config.")
        sr = cfg.data["transforms"]["audio"].get("sampling_rate", 0)
        nyquist = sr / 2 if sr else 0
        if nyquist and self.fmax_input.value >= nyquist:
            self.status.object = (
                f"Settings applied — Warning: fmax ({self.fmax_input.value} Hz) "
                f"≥ Nyquist frequency ({nyquist:.0f} Hz = sr/2). "
                f"librosa will cap fmax at {nyquist:.0f} Hz."
            )
            self.status.alert_type = "warning"
        else:
            self.status.object = "Settings applied successfully."
            self.status.alert_type = "success"

    def _validate(self):
        if self.fmin_input.value >= self.fmax_input.value:
            raise ValueError("fmin must be strictly less than fmax.")
        if self.win_length_input.value <= 0:
            raise ValueError("win_length must be strictly positive.")
        if self.hop_length_input.value <= 0:
            raise ValueError("hop_length must be strictly positive.")
        sr = self.controler.config.data["transforms"]["audio"].get("sampling_rate", 44100)
        win_samples = int(self.win_length_input.value * int(sr))
        if self.n_fft_input.value < win_samples:
            next_pow2 = 2 ** math.ceil(math.log2(max(win_samples, 1)))
            raise ValueError(
                f"n_fft ({self.n_fft_input.value}) must be ≥ win_length × SR "
                f"({win_samples} samples). Suggested: {next_pow2}."
            )
        if self.sr_input.value <= 0:
            raise ValueError("Spectral radius must be strictly positive.")
        if not (0 < self.leak_input.value <= 1):
            raise ValueError("Leak rate must be in (0, 1].")