# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Syllable segmentation panel of the Annotate page.

Segmentation runs on model predictions, so the panel has nothing to show until
the models have run. It holds three states:

``waiting``      no predictions yet, the panel explains what to do first;
``ready``        predictions are in, settings and preview are live;
``unavailable``  the models predict syllables already, splitting again would
                 misuse Otsu.

The settings write straight into the session config, which is what
``to_syllable_level`` reads at export time. A setting that only moved the
preview would let the picture and the exported file disagree.
"""
import logging

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import soundfile as sf
from scipy.signal import stft

from canapy.segmentation import segment_signal, segmentation_params, pools_period
from canapy.segmentation.corpus import untouched_labels
from canapy.segmentation.features import contours
from canapy.segmentation.grouping import estimate, pool_by_label

logger = logging.getLogger("canapy")

PREVIEW_MIN_SYLLABLES = 3
PHRASES_TRIED = 12
# how many phrases of a label the preview pools a period over
POOLED_PREVIEW = 30
# how many phrases are laid out at once when a whole label is inspected
OVERVIEW_PHRASES = 24
OVERVIEW_ROW_IN = 0.85
PHRASE_COLOUR = "#60a5fa"
SYLLABLE_COLOUR = "#4ade80"
# extremes make poor examples: too short to split, too long to read
DURATION_QUANTILES = (0.10, 0.90)
SPECTROGRAM_NPERSEG = 512
SPECTROGRAM_HOP_S = 0.001
SPECTROGRAM_FMAX = 11000.0
SPECTROGRAM_FLOOR_DB = -70.0

SEGMENTATION_CSS = """
.seg-card {
    background-color: #ffffff;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    padding: 18px 20px;
    border: 1px solid #e5e7eb;
    box-sizing: border-box;
}
.seg-section {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #6b7280;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 6px;
    margin: 4px 0 12px 0;
}
.seg-lead {
    font-size: 13px;
    color: #374151;
    line-height: 1.5;
}
.seg-hint {
    font-size: 12px;
    color: #9ca3af;
}
.seg-details .bk-panel-models-layout-Card {
    border: none;
}
"""

pn.config.raw_css.append(SEGMENTATION_CSS)

LEAD_TEXT = (
    "Splits each predicted phrase into the syllables it contains, keeping the "
    "phrase label on each."
)

DETAILS_TEXT = (
    "This is built for species whose vocalisations are made of repetitions, "
    "such as canaries, zebra finches and humpback whales, where a phrase is "
    "one syllable type repeated, or a motif repeated. A phrase that holds no "
    "repetition is returned untouched.<br><br>"
    "An <b>acoustic element</b> is every sound set apart by a silence. "
    "A <b>repeated unit</b> is one full cycle of the repetition, elements "
    "included. On a hierarchical phrase the first finds two to three times as "
    "many syllables as the second.<br><br>"
    "<b>Export Segmented Annotations</b> then writes both levels: the phrases "
    "the models predicted, and their segmented version. Re-importing the "
    "segmented output as a corpus requires lowering "
    "<code>min_label_duration</code>, otherwise postprocessing merges the "
    "syllables back into phrases."
)

WAITING_TEXT = (
    "<b>Segmentation applies to model predictions.</b> Run the annotation "
    "first: the models predict phrases, which this module then splits into "
    "syllables."
)

UNAVAILABLE_TEXT = (
    "<b>These predictions are already at syllable level.</b> Splitting them "
    "again would misuse Otsu, which assumes an alternation of song and "
    "silence that a phrase guarantees and an isolated syllable does not."
)

TARGET_HELP = (
    "An acoustic element is every sound set apart by a silence. A repeated "
    "unit is one full cycle of the repetition, elements included."
)

POOL_HELP = (
    "Estimates the period across every phrase carrying the label. Unchecked, "
    "each phrase is measured on its own."
)

DURATION_HELP = "Shorter sounds are discarded as clicks rather than syllables."

MODEL_HELP = "The model whose predictions are segmented, here and at export."

EXPORT_HINT = (
    "Export Segmented Annotations writes both levels. See What this does."
)


def _toggle_btn_css(opened=False):
    angle = "90deg" if opened else "0deg"
    return f"""
    button.bk-btn::before {{
        content: "▶";
        display: inline-block;
        font-size: 11px;
        margin-right: 7px;
        transform: rotate({angle});
        transition: transform 0.15s ease;
        color: inherit;
        vertical-align: middle;
    }}
    """


def _note(text, colour="#6b7280"):
    return f"<span style='color:{colour};font-size:12px;'>{text}</span>"


def _section(title):
    """Small uppercase rule, the same one the cards above the panel use."""
    return pn.pane.HTML(f"<div class='seg-section'>{title}</div>",
                        margin=(0, 0, 0, 0), sizing_mode="stretch_width")


def _hint(help_text, margin=(0, 0, 0, -4)):
    """The tooltip mark Panel draws for `description`, for widgets without it."""
    return pn.widgets.TooltipIcon(value=help_text, margin=margin)


def _field(label, widget, help_text=""):
    """A widget with the label Panel does not draw for it, plus its tooltip."""
    header = pn.pane.HTML(f"<span style='font-size:11px;color:#6b7280;"
                          f"font-weight:600;text-transform:uppercase;"
                          f"letter-spacing:0.6px;'>{label}</span>",
                          margin=(0, 0, 0, 0))
    return pn.Column(
        pn.Row(header, _hint(help_text, margin=(-6, 0, 0, -4)),
               margin=(0, 0, 2, 0)) if help_text else header,
        widget, margin=0,
    )


def _with_hint(widget, help_text):
    """A checkbox and its tooltip, side by side."""
    return pn.Row(widget, _hint(help_text), margin=0)


def candidate_phrases(corpus, untouched):
    """Phrases worth showing: segmentable, and away from the duration extremes."""
    df = corpus.dataset
    df = df[~df["label"].astype(str).isin(untouched)].copy()
    if df.empty:
        return df

    df["duration"] = df["offset_s"] - df["onset_s"]
    low, high = df["duration"].quantile(DURATION_QUANTILES)
    trimmed = df[(df["duration"] >= low) & (df["duration"] <= high)]
    return trimmed if len(trimmed) else df


def label_options(candidates):
    """Labels of the corpus, in the order the repertoire lists them."""
    return sorted(candidates["label"].astype(str).unique())


def opening_label(candidates):
    return str(candidates["label"].value_counts().idxmax())


def load_phrase(row):
    """Audio of one phrase, or None if the file cannot be read."""
    onset, offset = float(row["onset_s"]), float(row["offset_s"])
    try:
        signal, sr = sf.read(str(row["notated_path"]), dtype="float64")
    except Exception:
        return None
    if signal.ndim > 1:
        signal = signal[:, 0]
    return dict(label=str(row["label"]), sr=sr, duration=offset - onset,
                signal=signal[int(onset * sr): int(offset * sr)])


def pooled_for_label(candidates, label, params, limit=POOLED_PREVIEW):
    """Period of a label, pooled as the export will pool it.

    Capped at `limit` phrases to keep the preview responsive.
    """
    if not params.get("group_syllables"):
        return {}

    group = candidates[candidates["label"].astype(str) == str(label)]
    estimates = []
    for _, row in group.head(limit).iterrows():
        example = load_phrase(row)
        if example is None:
            continue
        segments, _ = segment_signal(example["signal"], example["sr"],
                                     **{**params, "group_syllables": False})
        if len(segments) < 2:
            continue
        times, energy, _, _ = contours(example["signal"], example["sr"],
                                       band=params["band"], hop=params["hop"])
        measured = estimate(segments, times, energy)
        if measured:
            estimates.append(measured)

    return pool_by_label({str(label): estimates}).get(str(label), {})


def phrase_for_label(candidates, label, params, pooled):
    """One phrase standing for a label: median duration, and it splits.

    Deterministic, so the same label always opens on the same picture.

    The phrases nearest the median are tried first, then the rest of the label
    in order of duration.
    """
    group = candidates[candidates["label"].astype(str) == str(label)]
    if not len(group):
        return None

    # ordered by position, not by index label: a corpus assembled file by file
    # carries duplicate labels, on which `reindex` refuses to work at all
    median = group["duration"].median()
    order = (group["duration"] - median).abs().to_numpy().argsort(kind="stable")
    ordered = group.iloc[order]

    fallback = None
    for rank, (_, row) in enumerate(ordered.iterrows()):
        # a readable phrase is in hand and the likely ones are exhausted
        if fallback is not None and rank >= PHRASES_TRIED:
            break
        example = load_phrase(row)
        if example is None or not len(example["signal"]):
            continue
        segments, _ = segment_signal(example["signal"], example["sr"], **params,
                                     period=pooled.get("period"))
        example = {**example, "segments": segments}
        if len(segments) >= PREVIEW_MIN_SYLLABLES:
            return example
        fallback = fallback or example
    return fallback


def phrases_of_label(candidates, label, params, pooled, limit=OVERVIEW_PHRASES):
    """Load and segment up to `limit` phrases of a label, longest first."""
    group = candidates[candidates["label"].astype(str) == str(label)]
    group = group.sort_values("duration", ascending=False).head(limit)

    examples = []
    for _, row in group.iterrows():
        example = load_phrase(row)
        if example is None:
            continue
        segments, _ = segment_signal(example["signal"], example["sr"], **params,
                                     period=pooled.get("period"))
        examples.append({**example, "segments": segments})
    return examples


def _spectrogram(signal, sr):
    hop = max(1, int(SPECTROGRAM_HOP_S * sr))
    # stft raises on nperseg = 0 rather than saying so
    nperseg = max(8, min(SPECTROGRAM_NPERSEG, len(signal)))
    freqs, times, spectrum = stft(
        signal, fs=sr, nperseg=nperseg, noverlap=max(0, nperseg - hop)
    )
    power = np.abs(spectrum) ** 2
    db = 10 * np.log10(power + 1e-20)
    keep = freqs <= SPECTROGRAM_FMAX
    return freqs[keep], times, db[keep] - db.max()


def draw_example(example, note=""):
    """Before / after strips: one phrase annotation, then its syllables.

    ``note`` names the settings that produced the lower strip, so that two
    pictures taken with different ones can be told apart.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 3.8), sharex=True)
    freqs, times, db = _spectrogram(example["signal"], example["sr"])

    for ax in axes:
        ax.pcolormesh(times, freqs / 1000, db, cmap="magma",
                      vmin=SPECTROGRAM_FLOOR_DB, vmax=0.0, shading="auto",
                      rasterized=True)
        ax.set_xlim(0, example["duration"])
        ax.set_ylabel("kHz", fontsize=8)
        ax.tick_params(labelsize=7)

    def annotate(ax, spans, colour, fontsize):
        for start, end in spans:
            ax.axvspan(start, end, color=colour, alpha=0.20, lw=0)
            ax.axvline(start, color=colour, lw=0.8, alpha=0.9)
            ax.axvline(end, color=colour, lw=0.8, alpha=0.9)
            # inside the span, where the tick labels do not already sit
            ax.text((start + end) / 2, 0.93, example["label"],
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=fontsize, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=colour,
                              edgecolor="none", alpha=0.85))

    annotate(axes[0], [(0.0, example["duration"])], PHRASE_COLOUR, 9)
    axes[0].set_title(f"predicted, 1 annotation of {example['duration'] * 1000:.0f} ms",
                      fontsize=9, loc="left", color="#374151")

    annotate(axes[1], example["segments"], SYLLABLE_COLOUR, 7)
    axes[1].set_title(
        f"segmented, {len(example['segments'])} annotations, same label"
        + (f"  ({note})" if note else ""),
        fontsize=9, loc="left", color="#374151")
    axes[1].set_xlabel("seconds", fontsize=8)

    fig.tight_layout()
    return fig


def draw_overview(examples, label, pooled):
    """Every phrase of a label, stacked, with its segmentation."""
    rows = len(examples)
    fig, axes = plt.subplots(rows, 1, figsize=(9, OVERVIEW_ROW_IN * rows + 0.8),
                             squeeze=False)
    axes = axes[:, 0]

    for ax, example in zip(axes, examples):
        freqs, times, db = _spectrogram(example["signal"], example["sr"])
        ax.pcolormesh(times, freqs / 1000, db, cmap="magma",
                      vmin=SPECTROGRAM_FLOOR_DB, vmax=0.0, shading="auto",
                      rasterized=True)
        for start, end in example["segments"]:
            ax.axvspan(start, end, color=SYLLABLE_COLOUR, alpha=0.22, lw=0)
            ax.axvline(start, color=SYLLABLE_COLOUR, lw=0.8)
            ax.axvline(end, color=SYLLABLE_COLOUR, lw=0.8)
        ax.set_xlim(0, example["duration"])
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
        if ax is not axes[-1]:
            ax.set_xticklabels([])
        ax.text(1.004, 0.5, f"{len(example['segments'])}", transform=ax.transAxes,
                va="center", fontsize=7, fontweight="bold", color="#374151")

    title = f"{label}, {rows} phrases. The count on the right is the syllables."
    period = pooled.get("period")
    if period:
        title += f" Pooled period {period * 1000:.0f} ms."
    else:
        title += " Period measured on each phrase."
    axes[0].set_title(title, fontsize=9, loc="left", color="#374151")
    axes[-1].set_xlabel("seconds", fontsize=8)
    fig.tight_layout()
    return fig


class SyllableSegmentationPanel:
    """Collapsible module of the Annotate page.

    The owner feeds it predictions with `set_predictions` once the models have
    run, and reads `enabled` at export time to know whether to write the
    segmented level as well.
    """

    def __init__(self, owner):
        self.owner = owner
        self.controler = owner.controler
        self._predictions = {}
        self._candidates = None
        self._example = None
        self._figure = None
        self._drawn = False
        self._loading_options = False
        self._state = "waiting"

        self.toggle_btn = pn.widgets.Button(
            name="Syllable Segmentation", button_type="primary", width=200
        )
        self.toggle_btn.stylesheets = [_toggle_btn_css(False)]
        self.toggle_btn.on_click(self._on_toggle)

        self.summary = pn.pane.Markdown(
            "", styles={"font-size": "13px", "color": "#6b7280"},
            align="center", margin=(0, 0, 0, 12),
        )
        self.collapsed = pn.Row(
            self.toggle_btn, self.summary, align="center",
            sizing_mode="stretch_width",
        )

        self.intro = pn.pane.Alert(WAITING_TEXT, alert_type="light",
                                   margin=(0, 0, 8, 0))
        self.lead = pn.pane.HTML(f"<div class='seg-lead'>{LEAD_TEXT}</div>",
                                 margin=(0, 0, 2, 0),
                                 sizing_mode="stretch_width")
        self.details = pn.Card(
            pn.pane.HTML(f"<div class='seg-lead'>{DETAILS_TEXT}</div>",
                         sizing_mode="stretch_width"),
            title="What this does", collapsed=True, collapsible=True,
            header_background="#f9fafb", css_classes=["seg-details"],
            margin=(0, 0, 12, 0), sizing_mode="stretch_width",
        )

        self.model_select = pn.widgets.Select(name="Predictions of", options=[],
                                              width=170, description=MODEL_HELP)
        self.model_select.param.watch(self._on_model_change, "value")

        self.target_select = pn.widgets.RadioButtonGroup(
            name="A syllable is", options=["acoustic element", "repeated unit"],
            button_type="default", button_style="outline", width=300,
        )
        self.target_select.param.watch(self._on_target_change, "value")

        self.pool_toggle = pn.widgets.Checkbox(
            name="Estimate the period per label", value=True, width=230,
        )
        self.pool_toggle.param.watch(self._on_setting_change, "value")

        self.duration_input = pn.widgets.FloatInput(
            name="Minimum syllable duration (s)",
            start=0.001, end=0.100, step=0.001, width=200,
            description=DURATION_HELP,
        )
        self.duration_input.param.watch(self._on_setting_change, "value")

        self.pool_row = pn.Column(_with_hint(self.pool_toggle, POOL_HELP),
                                  align="end", margin=(20, 0, 0, 0))

        self.label_select = pn.widgets.Select(name="Example label", options=[],
                                              width=170)
        self.label_select.param.watch(self._on_label_change, "value")

        self.overview_toggle = pn.widgets.Checkbox(
            name="Show every phrase of this label", value=False, width=240,
        )
        self.overview_toggle.param.watch(self._on_overview_change, "value")

        # a container rather than a pane: pn.pane.Matplotlib raises when its
        # model is built while it still holds no figure
        self.preview = pn.Column(sizing_mode="stretch_width")
        self.status = pn.pane.HTML("", margin=(4, 0, 0, 0))
        self.footnote = pn.pane.HTML(
            f"<div class='seg-hint'>{EXPORT_HINT}</div>",
            margin=(10, 0, 0, 2), sizing_mode="stretch_width",
        )

        self.settings_row = pn.Row(
            self.model_select, pn.Spacer(width=20),
            _field("A syllable is", self.target_select, TARGET_HELP),
            pn.Spacer(width=20), self.duration_input,
            pn.Spacer(width=20),
            self.pool_row,
            sizing_mode="stretch_width",
        )
        self.example_row = pn.Row(
            self.label_select, pn.Spacer(width=20),
            pn.Column(self.overview_toggle, align="end", margin=(20, 0, 0, 0)),
            sizing_mode="stretch_width",
        )
        self.controls = pn.Column(
            self.lead,
            self.details,
            _section("Settings"),
            self.settings_row,
            _section("Preview"),
            self.example_row,
            self.preview,
            self.footnote,
            sizing_mode="stretch_width",
        )

        self.expanded = pn.Column(
            self.intro, self.controls, self.status,
            visible=False, sizing_mode="stretch_width",
        )
        self.layout = pn.Column(
            self.collapsed, self.expanded,
            css_classes=["seg-card"], sizing_mode="stretch_width",
            styles={"margin-bottom": "15px"},
        )

        self._loading_options = True
        try:
            self._read_config()
        finally:
            self._loading_options = False
        self._apply_state()

    # -- public API ---------------------------------------------------------
    @property
    def enabled(self):
        """Whether there is anything the export could segment."""
        return self._state == "ready"

    def set_predictions(self, predictions):
        """Hand the panel what the models produced, one corpus per model."""
        self._predictions = dict(predictions or {})
        self._candidates = None
        self._example = None
        self._drawn = False

        if not self._predictions:
            self._state = "waiting"
        else:
            self._state = "ready"
            self._set_options(self.model_select, list(self._predictions),
                              next(iter(self._predictions)))
        self._apply_state()
        if self._state == "ready" and self.expanded.visible:
            self._load_options()

    def mark_unavailable(self):
        """The models predict syllables already; there is nothing to split."""
        self._state = "unavailable"
        self._apply_state()

    def reset(self):
        """Back to the waiting state, when a new run starts."""
        self.set_predictions({})

    # -- state --------------------------------------------------------------
    def _apply_state(self):
        ready = self._state == "ready"
        self.intro.visible = not ready
        self.intro.object = (UNAVAILABLE_TEXT if self._state == "unavailable"
                             else WAITING_TEXT)
        self.intro.alert_type = "warning" if self._state == "unavailable" else "light"
        self.controls.visible = ready
        self.summary.object = {
            "waiting": "*run the annotation first*",
            "ready": f"**{len(self._predictions)}** model(s) ready to segment",
            "unavailable": "*not applicable*",
        }[self._state]
        if self._state != "ready":
            self._clear_preview("")

    def _read_config(self):
        """Open on what the config says, since that is what the export reads."""
        params = segmentation_params(self.controler.config)
        self.target_select.value = ("repeated unit" if params["group_syllables"]
                                    else "acoustic element")
        self.duration_input.value = float(params["min_syllable_duration"])
        self.pool_toggle.value = pools_period(self.controler.config)
        self._sync_pool_visibility()

    def _write_config(self):
        section = self.controler.config.data.setdefault("segmentation", {})
        section["group_syllables"] = self.target_select.value == "repeated unit"
        section["min_syllable_duration"] = float(self.duration_input.value)
        section["pool_period"] = bool(self.pool_toggle.value)

    def _sync_pool_visibility(self):
        grouped = self.target_select.value == "repeated unit"
        self.pool_toggle.visible = grouped
        self.pool_row.visible = grouped

    def _params(self):
        return segmentation_params(self.controler.config)

    def _preview_note(self, params):
        """What the picture was drawn with, when the setting is not visible on it."""
        if not params.get("group_syllables"):
            return ""
        return ("period per label" if pools_period(self.controler.config)
                else "period per phrase")

    # -- events -------------------------------------------------------------
    def _on_toggle(self, event):
        self.expanded.visible = not self.expanded.visible
        self.toggle_btn.stylesheets = [_toggle_btn_css(self.expanded.visible)]
        if not self.expanded.visible:
            return
        # the Settings page edits the same keys, so re-read on every opening
        self._loading_options = True
        try:
            self._read_config()
        finally:
            self._loading_options = False
        if self._state == "ready" and not self._drawn:
            self._load_options()

    def _on_target_change(self, event):
        if self._loading_options:
            return
        self._write_config()
        self._sync_pool_visibility()
        # the phrase stays put: the point is to see this setting act on it
        if self.expanded.visible and self._state == "ready":
            self._draw_selected(keep_phrase=True)

    def _on_setting_change(self, event):
        if self._loading_options:
            return
        self._write_config()
        if self.expanded.visible and self._state == "ready":
            self._draw_selected(keep_phrase=True)

    def _on_model_change(self, event):
        if self._loading_options or self._state != "ready":
            return
        self._candidates = None
        self._example = None
        self._load_options()

    def _on_label_change(self, event):
        if self._loading_options:
            return
        self._draw_selected()

    def _on_overview_change(self, event):
        if self.expanded.visible and self._state == "ready":
            self._draw_selected(keep_phrase=True)

    # -- preview ------------------------------------------------------------
    def _preview_corpus(self):
        return self._predictions.get(self.model_select.value)

    def _clear_preview(self, message, colour="#6b7280"):
        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None
        self.preview[:] = []
        self.status.object = _note(message, colour) if message else ""

    def _set_options(self, widget, options, value):
        # assigning options fires the value watcher on its own; the guard keeps
        # it from drawing something the menu is about to replace
        self._loading_options = True
        try:
            widget.options = list(options)
            if value is not None:
                widget.value = value
        finally:
            self._loading_options = False

    def _load_options(self):
        """Fill the label menu from the selected model's predictions."""
        self._drawn = True
        corpus = self._preview_corpus()

        candidates = None
        if corpus is not None and not corpus.dataset.empty:
            candidates = candidate_phrases(
                corpus, untouched_labels(self.controler.config))

        if candidates is None or not len(candidates):
            self._candidates = None
            self._set_options(self.label_select, [], None)
            self._example = None
            self._clear_preview("No labelled phrase to show for this model.",
                                "#d97706")
            return

        self._candidates = candidates
        self._set_options(self.label_select, label_options(candidates),
                          opening_label(candidates))
        self._draw_selected()

    def _draw_selected(self, keep_phrase=False):
        if self._candidates is None:
            return

        label = self.label_select.value
        params = self._params()
        failure = None
        try:
            pooled = (pooled_for_label(self._candidates, label, params)
                      if pools_period(self.controler.config) else {})
            if self.overview_toggle.value:
                figure = self._overview_figure(label, params, pooled)
            else:
                figure = self._single_figure(label, params, pooled, keep_phrase)
        except Exception as error:
            logger.exception("Segmentation preview failed")
            failure = f"Preview failed: {error}"
            figure = None

        if figure is None:
            self._clear_preview(
                failure or f"No phrase of {label} could be drawn.", "#d97706")
            return

        if self._figure is not None:
            plt.close(self._figure)
        self._figure = figure
        self.preview[:] = [
            pn.pane.Matplotlib(self._figure, sizing_mode="stretch_width", tight=True)
        ]
        self.status.object = ""

    def _overview_figure(self, label, params, pooled):
        examples = phrases_of_label(self._candidates, label, params, pooled)
        if not examples:
            self._example = None
            return None
        self._example = examples[0]
        return draw_overview(examples, label, pooled)

    def _single_figure(self, label, params, pooled, keep_phrase):
        if keep_phrase and self._example is not None:
            segments, _ = segment_signal(
                self._example["signal"], self._example["sr"], **params,
                period=pooled.get("period"),
            )
            self._example = {**self._example, "segments": segments}
        else:
            self._example = phrase_for_label(self._candidates, label, params, pooled)
        if not self._example:
            return None
        return draw_example(self._example, self._preview_note(params))
