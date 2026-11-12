# Licence: BSD-3-Clause
"""Section of the Outliers tab for the files whose annotations seem shifted in time
(canapy.outliers.find_shifted_files): the spectrogram of a whole file with its annotations
and the shifted ones, its audio, and the shift applied to its annotations."""
import logging
import pathlib

import librosa
import numpy as np
import pandas as pd
import panel as pn
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ..helpers import SubDash

logger = logging.getLogger("canapy")

N_FFT, HOP = 1024, 256  # spectrogram of the whole file


def _text_color(color):
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return "#111827" if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


class ShiftTool(SubDash):
    """The suspicious files (table), and for the chosen file (a click on its row, or any
    file of the list): its spectrogram, its annotations (top band) and the shifted ones
    (lower band), its audio, and a button applying the shift to its annotations."""

    def __init__(self, parent, params):
        super().__init__(parent)
        self.params = params
        self.colors = {}
        self.shifted = None
        self._audio = {}  # notated_path: (audio, its spectrogram in dB)

        self.table = pn.widgets.Tabulator(
            pd.DataFrame(), show_index=False, disabled=True, selectable=1, height=170,
            sizing_mode="stretch_width", layout="fit_data_stretch",
        )
        self.table.param.watch(self._on_select, "selection")
        self.none_found = pn.pane.Alert(
            "No file with shifted annotations detected; any file can still be checked below.",
            alert_type="success", visible=False,
        )
        self.file_select = pn.widgets.Select(name="File", options={}, width=340)
        self.file_select.param.watch(lambda event: self._on_file(), "value")
        max_shift = params["max_shift"]
        self.shift_slider = pn.widgets.FloatSlider(
            name="Shift of the annotations (s)", start=-max_shift, end=max_shift,
            step=params["shift_step"], value=0.0, width=340,
        )
        self.shift_slider.param.watch(lambda event: self._draw(), "value_throttled")
        self.apply_btn = pn.widgets.Button(
            name="Apply the shift to this file", button_type="warning", width=220, align="end"
        )
        self.apply_btn.on_click(self.on_click_apply)
        self.msg = pn.pane.Markdown("", styles={"font-size": "12px", "color": "#16a34a"})
        self.figure = pn.Column(sizing_mode="stretch_width")  # (the spectrogram of the file)
        self.player = pn.Column(sizing_mode="stretch_width")

        self.layout = pn.Column(
            pn.pane.Markdown("### 3. Shifted annotations", margin=(10, 0, 6, 0)),
            pn.pane.Markdown(
                "Files whose segments rarely sound like their label: their annotations may be "
                "shifted in time (best shift: the shift of the annotations agreeing best with "
                "the other files). Click a file (or choose any file), adjust the shift while "
                "looking at the bands (top: the annotations, below: shifted) and listening, then "
                "apply it.",
                styles={"font-size": "12px", "color": "#6b7280"},
            ),
            self.none_found,
            self.table,
            pn.Row(self.file_select, self.shift_slider, self.apply_btn, align="end"),
            self.msg,
            self.figure,
            self.player,
            sizing_mode="stretch_width",
            visible=False,
        )

    def set_results(self, shifted, colors):
        """The suspicious files of a detection (find_shifted_files) and the label colors."""
        self.colors = colors
        self.shifted = shifted
        if len(shifted):
            table = shifted.assign(file=shifted["notated_path"].map(lambda p: pathlib.Path(p).name))
            table = table[["file", "segments", "agreement", "outliers", "knn_agreement",
                           "best_shift", "knn_agreement_after_shift"]]
            for column in ("agreement", "knn_agreement", "knn_agreement_after_shift"):
                table[column] = table[column].map(lambda v: f"{v:.0%}")
            table["best_shift"] = table["best_shift"].map(lambda v: f"{v:+.2f} s")
            self.table.value = table.rename(columns={
                "agreement": "sound like their label", "knn_agreement": "kNN agreement",
                "best_shift": "best shift", "knn_agreement_after_shift": "kNN agreement after it"})
        self.table.visible = len(shifted) > 0
        self.none_found.visible = len(shifted) == 0

        # all the files, the suspicious ones first
        suspicious = list(shifted["notated_path"]) if len(shifted) else []
        files = sorted(self.controler.corpus.dataset["notated_path"].astype(str).unique())
        ordered = suspicious + [f for f in files if f not in suspicious]
        self.file_select.options = {
            pathlib.Path(f).name + (" (suspicious)" if f in suspicious else ""): f for f in ordered
        }
        self.msg.object = ""
        self.layout.visible = True
        if suspicious:
            self.table.selection = [0]  # (shows the first one at its best shift)
        else:
            self._on_file()  # (the first file of the list)

    def _best_shift(self, path):
        if self.shifted is None or not len(self.shifted):
            return 0.0
        rows = self.shifted[self.shifted["notated_path"] == path]
        return float(rows["best_shift"].iloc[0]) if len(rows) else 0.0

    def _on_select(self, event):
        if not event.new:
            return
        path = self.shifted["notated_path"].iloc[event.new[0]]
        if self.file_select.value == path:
            self._on_file()
        else:
            self.file_select.value = path  # (-> _on_file)

    def _on_file(self):
        path = self.file_select.value
        if path is None:
            return
        self.shift_slider.value = self._best_shift(path)
        self.msg.object = ""
        self._draw()
        audio, _ = self._load(path)
        rate = round(self.controler.config.transforms.audio.sampling_rate)
        self.player.objects = [pn.pane.Audio(
            np.int16(audio / max(np.abs(audio).max(), 1e-12) * 0.9 * np.iinfo(np.int16).max),
            sample_rate=rate, sizing_mode="stretch_width", height=40)]

    def _load(self, path):
        if path not in self._audio:
            rate = self.controler.config.transforms.audio.sampling_rate
            audio, _ = librosa.load(path, sr=rate)
            spec = librosa.amplitude_to_db(
                np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP)), ref=np.max)
            self._audio[path] = (audio, spec)
        return self._audio[path]

    def _annotations(self, path):
        df = self.controler.corpus.dataset
        silence = self.controler.config.transforms.annots.silence_tag
        rows = df[(df["notated_path"].astype(str) == path) & (df["label"] != silence)]
        return rows[["onset_s", "offset_s", "label"]].to_numpy()

    def _draw(self):
        path = self.file_select.value
        if path is None:
            return
        audio, spec = self._load(path)
        audio_config = self.controler.config.transforms.audio
        rate = audio_config.sampling_rate
        duration = len(audio) / rate
        shift = self.shift_slider.value

        fig = Figure(figsize=(16, 3.2))
        ax = fig.subplots()
        fmax = min(getattr(audio_config, "fmax", rate / 2) or rate / 2, rate / 2)
        ax.imshow(spec, origin="lower", aspect="auto", cmap="magma",
                  extent=(0, duration, 0, rate / 2))
        ax.set_ylim(0, fmax)
        ax.set_xlim(0, duration)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Hz")
        # the bands: the annotations (top) and the shifted ones (below)
        for (low, high), offset in (((0.88, 1.0), 0.0), ((0.76, 0.88), shift)):
            for onset, end, label in self._annotations(path):
                start, stop = max(onset + offset, 0.0), min(end + offset, duration)
                if stop <= start:
                    continue
                color = self.colors.get(label, "#9ca3af")
                ax.add_patch(Rectangle((start, low), stop - start, high - low, color=color,
                                       transform=ax.get_xaxis_transform(), alpha=0.9, lw=0))
                ax.text((start + stop) / 2, (low + high) / 2, label, ha="center", va="center",
                        fontsize=7, color=_text_color(color), transform=ax.get_xaxis_transform(),
                        clip_on=True)
        ax.text(0.003, 0.94, "annotations", transform=ax.transAxes, fontsize=7, color="white",
                va="center", bbox=dict(fc="black", alpha=0.6, lw=0))
        ax.text(0.003, 0.82, f"shifted {shift:+.2f} s", transform=ax.transAxes, fontsize=7,
                color="white", va="center", bbox=dict(fc="black", alpha=0.6, lw=0))
        fig.tight_layout()
        self.figure.objects = [pn.pane.Matplotlib(fig, dpi=100, tight=True,
                                                  sizing_mode="scale_width")]

    def on_click_apply(self, event):
        path, shift = self.file_select.value, self.shift_slider.value
        if path is None or shift == 0:
            return
        n_left = self.controler.shift_file_annotations(path, shift)
        self.msg.object = (
            f"Annotations of {pathlib.Path(path).name} shifted by {shift:+.2f} s ({n_left} "
            "left in the audio). Run the detection again to update the outliers."
        )
        self.shift_slider.value = 0.0
        self._draw()
