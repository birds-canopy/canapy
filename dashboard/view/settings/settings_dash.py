# Author: Axel Arnaud
# Licence: MIT License
# Copyright: Axel Arnaud
import logging
import math
import panel as pn
from ..helpers import SubDash, SideBar

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
"""

PARAM_HELP = {
    "sampling_rate": "Sampling rate (Hz) of your audio files. librosa will resample to this value. Common values: 22050, 44100, 48000.",
    "fmin": "Minimum frequency (Hz) for MFCC computation. Frequencies below this value are ignored.",
    "fmax": "Maximum frequency (Hz) for MFCC computation. Frequencies above this value are ignored.",
    "win_length": (
        "Window length (seconds) for spectrogram frames. "
        "hop_length is automatically set to win_length / 2."
    ),
    "n_fft": (
        "FFT window size (samples). Must be ≥ win_length × sampling_rate. "
        "Should be a power of 2 (512, 1024, 2048, 4096, 8192…). "
        "At high sampling rates (e.g. 192 kHz), increase this value accordingly."
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

        # --- Audio widgets ---
        self.sampling_rate_input = pn.widgets.IntInput(
            value=int(cfg.transforms.audio.data["sampling_rate"]),
            start=1000,
            sizing_mode="stretch_width",
        )
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
        self.n_fft_input = pn.widgets.IntInput(
            value=int(cfg.transforms.audio.data["n_fft"]),
            start=64,
            sizing_mode="stretch_width",
        )
        self.hop_length_display = pn.pane.HTML(
            self._hop_label(float(cfg.transforms.audio.data["win_length"])),
            sizing_mode="stretch_width",
            align="center",
        )
        self.win_length_input.param.watch(self._update_hop_display, "value")

        # --- Reservoir widgets ---
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

        # --- Optimization widgets ---
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

        # --- Apply button ---
        self.btn_apply = pn.widgets.Button(
            name="Apply",
            button_type="primary",
            sizing_mode="stretch_width",
            height=45,
        )
        self.btn_apply.on_click(self._apply)

        # --- Layout ---
        audio_card = pn.Column(
            pn.pane.HTML("<div class='settings-section-header'>Audio</div>"),
            _make_param_row("Sampling rate (Hz)", self.sampling_rate_input, "sampling_rate"),
            _make_param_row("fmin (Hz)", self.fmin_input, "fmin"),
            _make_param_row("fmax (Hz)", self.fmax_input, "fmax"),
            _make_param_row("win_length (s)", self.win_length_input, "win_length"),
            pn.Row(
                pn.pane.HTML("<b>hop_length (s)</b>", width=180, align="center"),
                self.hop_length_display,
                sizing_mode="stretch_width",
            ),
            _make_param_row("n_fft (samples)", self.n_fft_input, "n_fft"),
            css_classes=["settings-card"],
            sizing_mode="stretch_width",
        )

        reservoir_card = pn.Column(
            pn.pane.HTML("<div class='settings-section-header'>Reservoir (syn & nsyn)</div>"),
            _make_param_row("Spectral radius (sr)", self.sr_input, "sr"),
            _make_param_row("Leak rate (leak)", self.leak_input, "leak"),
            _make_param_row("Input scaling MFCC (iss)", self.iss_input, "iss"),
            _make_param_row("Input scaling delta (isd)", self.isd_input, "isd"),
            _make_param_row("Input scaling delta2 (isd2)", self.isd2_input, "isd2"),
            _make_param_row("Ridge coefficient", self.ridge_input, "ridge"),
            css_classes=["settings-card"],
            sizing_mode="stretch_width",
        )

        optimization_card = pn.Column(
            pn.pane.HTML("<div class='settings-section-header'>Hyperparameter Search</div>"),
            _make_param_row("Search mode", self.opt_parallel_input, "opt_parallel"),
            _make_param_row("Data fraction", self.opt_percentage_input, "opt_max_percentage"),
            css_classes=["settings-card"],
            sizing_mode="stretch_width",
        )

        main_content = pn.Column(
            pn.pane.HTML("<h2 style='color:#1e293b;margin-bottom:4px;'>Settings</h2>"),
            self.status,
            pn.Spacer(height=20),
            audio_card,
            reservoir_card,
            optimization_card,
            self.btn_apply,
            pn.Spacer(height=40),
            sizing_mode="stretch_width",
            max_width=800,
        )

        self.layout = pn.Row(
            self.sidebar,
            pn.Spacer(width=50),
            pn.Column(
                pn.Spacer(height=40),
                main_content,
                pn.Spacer(height=40),
                sizing_mode="stretch_width",
            ),
            pn.Spacer(width=50),
            sizing_mode="stretch_both",
            background="#ffffff",
        )

    # ------------------------------------------------------------------
    def _hop_label(self, win_length: float) -> str:
        hop = round(win_length / 2, 6)
        return f"<span style='color:#64748b;font-size:13px;'>{hop} s (win_length / 2, read-only)</span>"

    def _update_hop_display(self, event):
        self.hop_length_display.object = self._hop_label(event.new)

    def _apply(self, _):
        try:
            self._validate()
        except ValueError as e:
            self.status.object = str(e)
            self.status.alert_type = "danger"
            return

        cfg = self.controler.config

        audio_data = cfg.data["transforms"]["audio"]
        audio_data["sampling_rate"] = self.sampling_rate_input.value
        audio_data["fmin"] = self.fmin_input.value
        audio_data["fmax"] = self.fmax_input.value
        audio_data["win_length"] = self.win_length_input.value
        audio_data["hop_length"] = round(self.win_length_input.value / 2, 6)
        audio_data["n_fft"] = self.n_fft_input.value

        for section in ("syn", "nsyn"):
            model_data = cfg.data["model"][section]
            model_data["sr"] = self.sr_input.value
            model_data["leak"] = self.leak_input.value
            model_data["iss"] = self.iss_input.value
            model_data["isd"] = self.isd_input.value
            model_data["isd2"] = self.isd2_input.value
            model_data["ridge"] = self.ridge_input.value

        self.controler.opt_parallel = self.opt_parallel_input.value
        self.controler.opt_max_percentage = self.opt_percentage_input.value

        logger.info("Settings applied to config.")
        nyquist = self.sampling_rate_input.value / 2
        if self.fmax_input.value >= nyquist:
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
        if self.sampling_rate_input.value < 1000:
            raise ValueError("Sampling rate must be at least 1000 Hz.")
        if self.fmin_input.value >= self.fmax_input.value:
            raise ValueError("fmin must be strictly less than fmax.")
        win_samples = int(self.win_length_input.value * self.sampling_rate_input.value)
        if self.n_fft_input.value < win_samples:
            next_pow2 = 2 ** math.ceil(math.log2(win_samples))
            raise ValueError(
                f"n_fft ({self.n_fft_input.value}) must be ≥ win_length × sampling_rate "
                f"({win_samples} samples). Suggested value: {next_pow2}."
            )
        if self.win_length_input.value <= 0:
            raise ValueError("win_length must be strictly positive.")
        if self.sr_input.value <= 0:
            raise ValueError("Spectral radius must be strictly positive.")
        if not (0 < self.leak_input.value <= 1):
            raise ValueError("Leak rate must be in (0, 1].")