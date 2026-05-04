# Author: Axel Arnaud
# Licence: MIT License
# Copyright: Axel Arnaud
import logging
import math
from pathlib import Path
import panel as pn
from ..helpers import SubDash, SideBar, pick_file, custom_tooltip

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
    "fmin": "Minimum frequency (Hz) for MFCC computation. Frequencies below this value are ignored.",
    "fmax": "Maximum frequency (Hz) for MFCC computation. Frequencies above this value are ignored. Capped at sr/2 (Nyquist frequency).",
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
    "opt_seed": (
        "Random seed for the hyperparameter search. "
        "Fixing this value makes the search reproducible: two runs with the same seed "
        "and parameters will explore the same configurations."
    ),
    "merge_consecutive_labels": (
        "If enabled, consecutive annotations with the same label separated by a silence "
        "shorter than min_silence_gap are merged into a single annotation. "
        "Disable if your protocol distinguishes repeated identical labels."
    ),
    "lonely_labels": (
        "Labels that are never merged, even when merge is enabled and two identical "
        "annotations are very close in time. Useful for isolated vocalisations (calls, "
        "noise) that should remain distinct occurrences. "
        "Enter labels separated by commas, e.g.: cri, TRASH, call. "
        "Has no effect when 'Merge consecutive labels' is disabled."
    ),
    "silence_tag": (
        "Label used to represent silence intervals between annotations. "
        "Silence segments are inserted automatically during pre-processing and are "
        "excluded from training metrics. Must match the tag used in your annotation files "
        "if you already have silence labels."
    ),
    "iterative_fit": (
        "Normal fit: standard mode — accumulates all reservoir state matrices in RAM "
        "before solving Ridge. Slightly faster but RAM usage grows with dataset size "
        "(O(n_frames × n_units)). "
        "Iterative fit: processes one sequence at a time, keeping only one state matrix "
        "in RAM at once. Mathematically equivalent result, constant peak RAM. "
        "Recommended for large datasets."
    ),
    "n_mfcc": "Number of MFCC coefficients extracted per frame. Changing this requires recomputing spectrograms.",
    "lifter": "MFCC liftering coefficient. Applies a sinusoidal lift to de-emphasise low-order coefficients. 0 = no liftering.",
    "audio_features": "Feature types extracted from audio. mfcc = raw coefficients, delta = 1st derivative, delta2 = 2nd derivative. Changing this requires recomputing spectrograms.",
    "delta_padding": "Padding mode used to compute 1st-order MFCC derivatives at sequence boundaries.",
    "delta2_padding": "Padding mode used to compute 2nd-order MFCC derivatives at sequence boundaries.",
    "time_precision": "Minimum time resolution (seconds) for annotation onsets/offsets. Values are rounded to this precision.",
    "min_label_duration": "Annotations shorter than this (seconds) are discarded during preprocessing.",
    "min_silence_gap": "Silence gaps shorter than this (seconds) between two annotations are absorbed by the surrounding labels.",
    "test_ratio": (
        "Fraction of audio files held out for evaluation. Applied when moving from preprocessing to training. "
        "Changing this invalidates the current train/test split and resets model training."
    ),
    "max_sequences": (
        "Maximum number of audio files used for training. -1 = all files. "
        "Changing this invalidates the current train/test split and resets model training."
    ),
    "min_silence_duration": (
        "Silence segments shorter than this (seconds) are discarded from the non-syntactic (NSyn) training set. "
        "Changing this resets model training."
    ),
    "noise_std": (
        "Standard deviation of Gaussian noise added to augmented NSyn training samples. "
        "Changing this resets model training."
    ),
    "units": "Number of neurons in the ESN reservoir. Larger reservoirs capture more complex temporal patterns but use more RAM and are slower to train.",
    "workers": "Number of parallel workers for ESN training and inference. -1 = all available CPUs.",
    "backend": "Joblib parallelisation backend. multiprocessing = separate processes (safe, high overhead); threading = shared memory (faster but GIL-limited); loky = robust multiprocessing.",
    "seed": "Global random seed for reproducibility. Affects reservoir initialisation and train/test split.",
    "min_segment_proportion_agreement": (
        "Minimum fraction of time during which all annotators must agree on the same label "
        "for a segment to be considered correctly annotated. Used in the evaluation view. "
        "Range 0–1."
    ),
}


def _make_param_row(label: str, widget: pn.widgets.Widget, param_name: str):
    tt = custom_tooltip(PARAM_HELP[param_name], direction="left", margin=(0, 10))
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

        _sr = cfg.transforms.audio.data.get("sampling_rate", 44100)
        self.fmin_input = pn.widgets.IntInput(
            value=int(cfg.transforms.audio.data["fmin"]),
            start=0,
            step=100,
            sizing_mode="stretch_width",
        )
        self.fmax_input = pn.widgets.IntInput(
            value=int(cfg.transforms.audio.data["fmax"]),
            start=0,
            end=int(_sr) // 2,
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
            self.lonely_labels_input.disabled = not event.new

        self.merge_labels_input.param.watch(_on_merge_toggle, "value")

        _lonely_raw = cfg.transforms.annots.data.get("lonely_labels", [])
        _lonely_str = ", ".join(_lonely_raw) if _lonely_raw else ""
        self.lonely_labels_input = pn.widgets.TextInput(
            value=_lonely_str,
            placeholder="e.g. cri, TRASH, call",
            disabled=not _merge_val,
            sizing_mode="stretch_width",
        )

        _silence_tag = cfg.transforms.annots.data.get("silence_tag", "SIL")
        self.silence_tag_input = pn.widgets.TextInput(
            value=str(_silence_tag),
            placeholder="e.g. SIL",
            sizing_mode="stretch_width",
        )

        self.opt_parallel_input = pn.widgets.Select(
            options={"Bayesian sequential optimization": False, "Fast random parallel optimization": True},
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
        self.opt_seed_input = pn.widgets.IntInput(
            value=self.controler.opt_seed,
            start=0,
            step=1,
            sizing_mode="stretch_width",
        )

        # --- Advanced audio ---
        _adv_audio = cfg.transforms.audio.data
        self.n_mfcc_input = pn.widgets.IntInput(
            value=int(_adv_audio.get("n_mfcc", 13)),
            start=1, end=128, sizing_mode="stretch_width",
        )
        self.lifter_input = pn.widgets.IntInput(
            value=int(_adv_audio.get("lifter", 40)),
            start=0, end=200, sizing_mode="stretch_width",
        )
        _all_feats = ["mfcc", "delta", "delta2"]
        _cur_feats = list(_adv_audio.get("audio_features", _all_feats))
        self.audio_features_input = pn.widgets.CheckBoxGroup(
            options=_all_feats, value=_cur_feats,
            inline=True, sizing_mode="stretch_width",
        )
        _PADDING_OPTS = ["wrap", "interp", "nearest", "mirror", "constant"]
        self.delta_padding_input = pn.widgets.Select(
            options=_PADDING_OPTS,
            value=_adv_audio.get("delta", {}).get("padding", "wrap"),
            sizing_mode="stretch_width",
        )
        self.delta2_padding_input = pn.widgets.Select(
            options=_PADDING_OPTS,
            value=_adv_audio.get("delta2", {}).get("padding", "wrap"),
            sizing_mode="stretch_width",
        )

        # --- Advanced annotations ---
        _adv_annots = cfg.transforms.annots.data
        self.time_precision_input = pn.widgets.FloatInput(
            value=float(_adv_annots.get("time_precision", 0.001)),
            start=1e-6, step=0.001, sizing_mode="stretch_width",
        )
        self.min_label_dur_input = pn.widgets.FloatInput(
            value=float(_adv_annots.get("min_label_duration", 0.02)),
            start=0.0, step=0.005, sizing_mode="stretch_width",
        )
        self.min_silence_gap_input = pn.widgets.FloatInput(
            value=float(_adv_annots.get("min_silence_gap", 0.001)),
            start=0.0, step=0.001, sizing_mode="stretch_width",
        )

        # --- Advanced training ---
        _adv_training = cfg.data.get("transforms", {}).get("training", {})
        self.test_ratio_input = pn.widgets.FloatSlider(
            value=float(_adv_training.get("test_ratio", 0.2)),
            start=0.05, end=0.5, step=0.05, sizing_mode="stretch_width",
        )
        self.max_sequences_input = pn.widgets.IntInput(
            value=int(_adv_training.get("max_sequences", -1)),
            start=-1, step=1, sizing_mode="stretch_width",
        )
        _adv_balance = _adv_training.get("balance", {})
        self.min_silence_dur_input = pn.widgets.FloatInput(
            value=float(_adv_balance.get("min_silence_duration", 0.2)),
            start=0.0, step=0.05, sizing_mode="stretch_width",
        )
        _adv_aug = _adv_balance.get("data_augmentation", {})
        self.noise_std_input = pn.widgets.FloatInput(
            value=float(_adv_aug.get("noise_std", 0.01)),
            start=0.0, step=0.001, sizing_mode="stretch_width",
        )

        # --- Advanced model architecture ---
        self.units_input = pn.widgets.IntInput(
            value=int(cfg.model.syn.data.get("units", 1000)),
            start=100, step=100, sizing_mode="stretch_width",
        )
        _BACKEND_OPTS = ["multiprocessing", "threading", "loky", "sequence"]
        self.backend_input = pn.widgets.Select(
            options=_BACKEND_OPTS,
            value=cfg.model.syn.data.get("backend", "multiprocessing"),
            sizing_mode="stretch_width",
        )
        self.workers_input = pn.widgets.IntInput(
            value=int(cfg.model.syn.data.get("workers", -1)),
            start=-1, step=1, sizing_mode="stretch_width",
        )

        # --- Misc & Correction ---
        self.seed_input = pn.widgets.IntInput(
            value=int(cfg.data.get("misc", {}).get("seed", 42)),
            start=0, step=1, sizing_mode="stretch_width",
        )
        self.min_agreement_input = pn.widgets.FloatSlider(
            value=float(cfg.data.get("correction", {}).get("min_segment_proportion_agreement", 0.66)),
            start=0.0, end=1.0, step=0.01, sizing_mode="stretch_width",
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
        self.apply_status = pn.pane.HTML("", margin=(4, 0, 0, 0))

        _iterative_fit_val = bool(
            cfg.data.get("transforms", {}).get("training", {}).get("iterative_fit", False)
        )
        self.fit_mode_input = pn.widgets.Select(
            options={
                "Normal fit (fast, high RAM on large datasets)": False,
                "Iterative fit (memory-efficient, slightly slower)": True,
            },
            value=_iterative_fit_val,
            sizing_mode="stretch_width",
        )

        self.advanced_audio_block = pn.Column(
            _make_param_row("hop_length (s)", self.hop_length_input, "hop_length"),
            _make_param_row("n_fft (samples)", self.n_fft_input, "n_fft"),
            _make_param_row("n_mfcc", self.n_mfcc_input, "n_mfcc"),
            _make_param_row("Lifter", self.lifter_input, "lifter"),
            _make_param_row("Audio features", self.audio_features_input, "audio_features"),
            _make_param_row("Delta padding", self.delta_padding_input, "delta_padding"),
            _make_param_row("Delta2 padding", self.delta2_padding_input, "delta2_padding"),
            visible=False,
            sizing_mode="stretch_width",
        )

        self.advanced_annots_block = pn.Column(
            _make_param_row("Time precision (s)", self.time_precision_input, "time_precision"),
            _make_param_row("Min label duration (s)", self.min_label_dur_input, "min_label_duration"),
            _make_param_row("Min silence gap (s)", self.min_silence_gap_input, "min_silence_gap"),
            _make_param_row("Lonely labels", self.lonely_labels_input, "lonely_labels"),
            _make_param_row("Silence tag", self.silence_tag_input, "silence_tag"),
            visible=False,
            sizing_mode="stretch_width",
        )

        self.advanced_training_block = pn.Column(
            pn.pane.HTML("<div class='settings-subsection-header'>Training</div>"),
            pn.pane.HTML(
                "<p style='font-size:12px;color:#e67e22;margin:0 0 8px 0;'>"
                "⚠ Changing these parameters invalidates the current train/test split "
                "and resets model training.</p>"
            ),
            _make_param_row("Test ratio", self.test_ratio_input, "test_ratio"),
            _make_param_row("Max sequences (-1 = all)", self.max_sequences_input, "max_sequences"),
            _make_param_row("Min silence duration (s)", self.min_silence_dur_input, "min_silence_duration"),
            _make_param_row("Noise std (augmentation)", self.noise_std_input, "noise_std"),
            visible=False,
            sizing_mode="stretch_width",
        )

        self.reservoir_block = pn.Column(
            pn.pane.HTML("<div class='settings-subsection-header'>Reservoir (syn & nsyn)</div>"),
            pn.pane.HTML("<p style='font-size:12px;color:#64748b;margin:0 0 10px 0;'>These parameters can be automatically optimised by the hyperparameter search run just before model training.</p>"),
            _make_param_row("Spectral radius (sr)", self.sr_input, "sr"),
            _make_param_row("Leak rate (leak)", self.leak_input, "leak"),
            _make_param_row("Input scaling MFCC (iss)", self.iss_input, "iss"),
            _make_param_row("Input scaling delta (isd)", self.isd_input, "isd"),
            _make_param_row("Input scaling delta2 (isd2)", self.isd2_input, "isd2"),
            _make_param_row("Ridge coefficient", self.ridge_input, "ridge"),
            _make_param_row("Fit mode", self.fit_mode_input, "iterative_fit"),
            pn.pane.HTML("<div class='settings-subsection-header'>Model architecture</div>"),
            _make_param_row("N units", self.units_input, "units"),
            _make_param_row("Workers (-1 = all)", self.workers_input, "workers"),
            _make_param_row("Backend", self.backend_input, "backend"),
            pn.pane.HTML("<div class='settings-subsection-header'>Misc & Correction</div>"),
            _make_param_row("Random seed", self.seed_input, "seed"),
            _make_param_row("Min agreement (correction)", self.min_agreement_input, "min_segment_proportion_agreement"),
            visible=False,
            sizing_mode="stretch_width",
        )

        def _on_advanced_toggle(event):
            self.advanced_toggle.button_type = "primary" if event.new else "default"
            self.advanced_audio_block.visible = event.new
            self.advanced_annots_block.visible = event.new
            self.advanced_training_block.visible = event.new
            self.reservoir_block.visible = event.new

        self.advanced_toggle.param.watch(_on_advanced_toggle, "value")

        species_params_card = pn.Column(
            pn.pane.HTML("<div class='settings-section-header'>Species Parameters</div>"),
            pn.Row(
                pn.Spacer(),
                self.advanced_toggle,
                sizing_mode="stretch_width",
                margin=(0, 0, 10, 0),
            ),

            pn.pane.HTML("<div class='settings-subsection-header'>Audio</div>"),
            _make_param_row("fmin (Hz)", self.fmin_input, "fmin"),
            _make_param_row("fmax (Hz)", self.fmax_input, "fmax"),
            _make_param_row("win_length (s)", self.win_length_input, "win_length"),
            self.advanced_audio_block,

            pn.pane.HTML("<div class='settings-subsection-header'>Annotations</div>"),
            _make_param_row("Merge consecutive labels", self.merge_labels_input, "merge_consecutive_labels"),
            self.advanced_annots_block,

            self.advanced_training_block,
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
            _make_param_row("Search seed", self.opt_seed_input, "opt_seed"),
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
                pn.Spacer(width=8),
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
            min_width=260,
        )

        params_column = pn.Column(
            species_params_card,
            hp_card,
            self.btn_apply,
            self.apply_status,
            sizing_mode="stretch_width",
            min_width=320,
        )

        main_content = pn.Column(
            pn.pane.HTML("<h1 style='color:#1e293b;margin:0 0 6px 0;'>Settings</h1>"),
            pn.Spacer(height=20),
            pn.Row(
                species_card,
                pn.Spacer(width=24),
                params_column,
                sizing_mode="stretch_width",
            ),
            pn.Spacer(height=40),
            sizing_mode="stretch_width",
            max_width=1400,
        )

        self.layout = pn.Row(
            self.sidebar,
            pn.Column(
                pn.Spacer(height=30),
                main_content,
                sizing_mode="stretch_width",
                styles={"padding": "0 24px", "overflow-y": "auto"},
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
        # name = path.stem.replace("_", " ").title()


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
            self.controler._config_display_name = self._selected_preset_path.stem
            self._refresh_widgets_from_config()
            self._apply(None)
            self.controler._settings_dirty = False
            name = self._selected_preset_path.stem.replace("_", " ").title()
            self._preset_status.object = (
                f"<span style='font-size:12px;color:#059669;'>Preset <b>{name}</b> applied.</span>"
            )
            self.status.object = f"Preset '{name}' applied successfully."
            self.status.alert_type = "success"
            logger.info(f"Applied preset: {self._selected_preset_path.stem}")
        except Exception as e:
            self._preset_status.object = (
                f"<span style='font-size:12px;color:#dc2626;'>Error: {e}</span>"
            )

    def _browse_config(self, _):
        config_dir = Path(__file__).parents[3] / "config"
        initialdir = str(config_dir) if config_dir.exists() else None
        path = pick_file(
            title="Select Config File",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
            initialdir=initialdir,
        )
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
            self.controler.config_path = path
            self.controler._config_display_name = path.stem
            self._refresh_widgets_from_config()
            self._apply(None)
            self.controler._settings_dirty = False
            self.config_load_status.object = (
                "<span style='font-size:12px;color:#059669;'>Config loaded and applied.</span>"
            )
            self.status.object = f"Config '{path.name}' applied successfully."
            self.status.alert_type = "success"
            logger.info(f"Loaded config: {path}")
        except Exception as e:
            self.config_load_status.object = (
                f"<span style='font-size:12px;color:#dc2626;'>Error: {e}</span>"
            )

    def _refresh_widgets_from_config(self):
        cfg = self.controler.config
        audio = cfg.transforms.audio.data

        # Basic audio
        self.fmin_input.value = int(audio["fmin"])
        sr = audio.get("sampling_rate", 44100)
        self.fmax_input.end = int(sr) // 2
        self.fmax_input.value = min(int(audio["fmax"]), int(sr) // 2)
        win = float(audio["win_length"])
        self.win_length_input.value = win
        self.hop_length_input.value = float(audio.get("hop_length", round(win / 2, 6)))
        self.n_fft_input.value = int(
            audio.get("n_fft", 2 ** math.ceil(math.log2(max(win * int(sr), 1))))
        )

        # Advanced audio
        self.n_mfcc_input.value = int(audio.get("n_mfcc", 13))
        self.lifter_input.value = int(audio.get("lifter", 40))
        _all_feats = ["mfcc", "delta", "delta2"]
        _cur_feats = [f for f in list(audio.get("audio_features", _all_feats)) if f in _all_feats]
        self.audio_features_input.value = _cur_feats
        self.delta_padding_input.value = audio.get("delta", {}).get("padding", "wrap")
        self.delta2_padding_input.value = audio.get("delta2", {}).get("padding", "wrap")

        # Model reservoir
        syn = cfg.model.syn.data
        self.sr_input.value = float(syn["sr"])
        self.leak_input.value = float(syn["leak"])
        self.iss_input.value = float(syn["iss"])
        self.isd_input.value = float(syn["isd"])
        self.isd2_input.value = float(syn["isd2"])
        self.ridge_input.value = float(syn["ridge"])

        # Model architecture
        self.units_input.value = int(syn.get("units", 1000))
        self.workers_input.value = int(syn.get("workers", -1))
        _bk = syn.get("backend", "multiprocessing")
        if _bk in self.backend_input.options:
            self.backend_input.value = _bk

        # Annotation params
        annots = cfg.transforms.annots.data
        merge = bool(annots.get("merge_consecutive_labels", True))
        self.merge_labels_input.value = merge
        self.merge_labels_input.name = "Enabled" if merge else "Disabled"
        self.merge_labels_input.button_type = "success" if merge else "default"
        lonely_raw = annots.get("lonely_labels", [])
        self.lonely_labels_input.value = ", ".join(lonely_raw) if lonely_raw else ""
        self.lonely_labels_input.disabled = not merge
        self.silence_tag_input.value = str(annots.get("silence_tag", "SIL"))
        self.time_precision_input.value = float(annots.get("time_precision", 0.001))
        self.min_label_dur_input.value = float(annots.get("min_label_duration", 0.02))
        self.min_silence_gap_input.value = float(annots.get("min_silence_gap", 0.001))

        # Training params
        training = cfg.data.get("transforms", {}).get("training", {})
        self.test_ratio_input.value = float(training.get("test_ratio", 0.2))
        self.max_sequences_input.value = int(training.get("max_sequences", -1))
        balance = training.get("balance", {})
        self.min_silence_dur_input.value = float(balance.get("min_silence_duration", 0.2))
        aug = balance.get("data_augmentation", {})
        self.noise_std_input.value = float(aug.get("noise_std", 0.01))
        self.fit_mode_input.value = bool(training.get("iterative_fit", False))

        # Misc & Correction
        self.seed_input.value = int(cfg.data.get("misc", {}).get("seed", 42))
        self.min_agreement_input.value = float(
            cfg.data.get("correction", {}).get("min_segment_proportion_agreement", 0.66)
        )

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

        # --- Snapshot old audio params for dirty detection ---
        _audio_keys_basic = ("fmin", "fmax", "win_length", "hop_length", "n_fft")
        _old_audio = {k: audio_data.get(k) for k in _audio_keys_basic}
        _old_n_mfcc = audio_data.get("n_mfcc")
        _old_lifter = audio_data.get("lifter")
        _old_audio_features = sorted(audio_data.get("audio_features", []))
        _old_delta_padding = audio_data.get("delta", {}).get("padding")
        _old_delta2_padding = audio_data.get("delta2", {}).get("padding")

        # --- Apply basic audio params ---
        audio_data["fmin"] = self.fmin_input.value
        audio_data["fmax"] = self.fmax_input.value
        audio_data["win_length"] = self.win_length_input.value
        audio_data["hop_length"] = self.hop_length_input.value
        audio_data["n_fft"] = self.n_fft_input.value

        # --- Apply advanced audio params ---
        audio_data["n_mfcc"] = self.n_mfcc_input.value
        audio_data["lifter"] = self.lifter_input.value
        audio_data["audio_features"] = list(self.audio_features_input.value)
        audio_data.setdefault("delta", {})["padding"] = self.delta_padding_input.value
        audio_data.setdefault("delta2", {})["padding"] = self.delta2_padding_input.value

        # --- Detect audio changes → invalidate MFCC cache ---
        _audio_changed = (
            any(audio_data.get(k) != _old_audio[k] for k in _audio_keys_basic)
            or audio_data.get("n_mfcc") != _old_n_mfcc
            or audio_data.get("lifter") != _old_lifter
            or sorted(audio_data.get("audio_features", [])) != _old_audio_features
            or audio_data.get("delta", {}).get("padding") != _old_delta_padding
            or audio_data.get("delta2", {}).get("padding") != _old_delta2_padding
        )
        if _audio_changed:
            self.controler._audio_params_dirty = True
        self.controler._settings_dirty = True

        # --- Model reservoir + architecture params (syn & nsyn) ---
        for section in ("syn", "nsyn"):
            model_data = cfg.data["model"][section]
            model_data["sr"] = self.sr_input.value
            model_data["leak"] = self.leak_input.value
            model_data["iss"] = self.iss_input.value
            model_data["isd"] = self.isd_input.value
            model_data["isd2"] = self.isd2_input.value
            model_data["ridge"] = self.ridge_input.value
            model_data["units"] = self.units_input.value
            model_data["workers"] = self.workers_input.value
            model_data["backend"] = self.backend_input.value

        # --- Annotation params ---
        annots_data = cfg.data["transforms"]["annots"]
        annots_data["merge_consecutive_labels"] = self.merge_labels_input.value
        annots_data["lonely_labels"] = [
            x.strip() for x in self.lonely_labels_input.value.split(",") if x.strip()
        ]
        annots_data["silence_tag"] = self.silence_tag_input.value.strip() or "SIL"
        annots_data["time_precision"] = self.time_precision_input.value
        annots_data["min_label_duration"] = self.min_label_dur_input.value
        annots_data["min_silence_gap"] = self.min_silence_gap_input.value

        # --- Training params (with dirty detection) ---
        training_data = cfg.data["transforms"]["training"]
        _old_test_ratio = training_data.get("test_ratio")
        _old_max_sequences = training_data.get("max_sequences")
        _old_balance = training_data.get("balance", {})
        _old_min_sil_dur = _old_balance.get("min_silence_duration")
        _old_noise_std = _old_balance.get("data_augmentation", {}).get("noise_std")

        training_data["iterative_fit"] = self.fit_mode_input.value
        training_data["test_ratio"] = self.test_ratio_input.value
        training_data["max_sequences"] = self.max_sequences_input.value
        training_data.setdefault("balance", {})["min_silence_duration"] = self.min_silence_dur_input.value
        training_data["balance"].setdefault("data_augmentation", {})["noise_std"] = self.noise_std_input.value

        _training_changed = (
            training_data.get("test_ratio") != _old_test_ratio
            or training_data.get("max_sequences") != _old_max_sequences
            or training_data["balance"].get("min_silence_duration") != _old_min_sil_dur
            or training_data["balance"]["data_augmentation"].get("noise_std") != _old_noise_std
        )
        if _training_changed:
            self.controler._training_params_dirty = True

        # --- Misc & Correction ---
        cfg.data["misc"]["seed"] = self.seed_input.value
        cfg.data["correction"]["min_segment_proportion_agreement"] = self.min_agreement_input.value

        # --- HP Search ---
        self.controler.opt_parallel = self.opt_parallel_input.value
        self.controler.opt_max_percentage = self.opt_percentage_input.value
        self.controler.opt_n_jobs = self.opt_n_jobs_input.value
        self.controler.opt_max_evals = self.opt_max_evals_input.value
        self.controler.opt_seed = self.opt_seed_input.value

        logger.info("Settings applied to config.")
        self.status.object = "Settings applied successfully."
        self.status.alert_type = "success"
        self.apply_status.object = (
            "<span style='font-size:12px;color:#059669;'>✓ Settings applied successfully.</span>"
        )

    def _validate(self):
        sr = self.controler.config.data["transforms"]["audio"].get("sampling_rate", 44100)
        nyquist = int(sr) // 2
        if self.fmax_input.value > nyquist:
            raise ValueError(
                f"fmax ({self.fmax_input.value} Hz) exceeds the Nyquist frequency "
                f"({nyquist} Hz = sr/2 with sr={sr} Hz). "
                f"Set fmax ≤ {nyquist} Hz."
            )
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