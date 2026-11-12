# Licence: BSD-3-Clause
"""Tab of the Preprocess and Eval pages listing the segments whose label does not match their
neighbourhood (canapy.outliers), to listen to them and correct their label."""
import base64
import io
import logging
import pathlib
import threading

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
import soundfile as sf
from bokeh.palettes import Category10, Category20, turbo

from canapy.correction import match_annotation
from canapy.outliers import find_outliers, find_shifted_files, outliers_params

from ..helpers import Registry, SubDash, custom_tooltip
from .shifts import ShiftTool

logger = logging.getLogger("canapy")

# columns of the table of the outliers (the last one editable)
COLUMNS = ["#", "file", "onset (s)", "duration (s)", "label", "sounds like", "shifted file",
           "correction"]


# scrolls the page to the sample of the clicked row (css class "outlier-detail", searched
# through the shadow roots of the Panel components) once it is rendered, only if it is not
# entirely in sight, and as little as needed
SCROLL_TO_DETAIL = """
setTimeout(() => {
  const find = (root) => {
    for (const el of root.querySelectorAll('*')) {
      if (el.classList && el.classList.contains('outlier-detail')) return el;
      if (el.shadowRoot) { const found = find(el.shadowRoot); if (found) return found; }
    }
    return null;
  };
  const detail = find(document);
  if (!detail) return;
  const box = detail.getBoundingClientRect();
  if (box.top < 0 || box.bottom > window.innerHeight) {
    detail.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  }
}, 300);
"""


CONTEXT_S = 0.5  # s of audio before and after the segment, in the second player
TYPICAL = 10  # typical segments of a label, shown in turn next to an outlier sounding like it


def _wav(path, start_s, end_s):
    """(the audio of the file between the two times as a base64 WAV, its duration in s)."""
    info = sf.info(path)
    start = max(0, int(round(start_s * info.samplerate)))
    stop = min(info.frames, int(round(end_s * info.samplerate)))
    audio, rate = sf.read(path, start=start, stop=max(stop, start + 1), always_2d=True)
    audio = audio.mean(axis=1)
    audio = 0.9 * audio / max(abs(audio).max(), 1e-12)  # (each extract at the same loudness)
    buffer = io.BytesIO()
    sf.write(buffer, audio, rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buffer.getvalue()).decode(), len(audio) / rate


def _player(wav, duration, autoplay=False, nonce=0):
    """HTML of a compact player: a button playing the audio from its start, a progress bar,
    and the time in tenths of a second (hundredths below one second)."""
    digits = 1 if duration >= 1 else 2
    total = f"{duration:.{digits}f}"
    tick = (
        "const a=this,p=a.parentNode,t=p.querySelector('.t'),g=p.querySelector('progress');"
        "const f=()=>{const end=a.ended;"
        f"t.textContent=(end?'{total}':a.currentTime.toFixed({digits}))+' / {total} s';"
        "g.value=end?a.duration:a.currentTime;"
        "if(!a.paused&&!end)requestAnimationFrame(f);};f();"
    )
    return (
        f'<div style="display:flex;align-items:center;gap:8px;font:12px sans-serif" data-n="{nonce}">'
        '<button title="play from the start" style="width:30px;height:26px;border:1px solid #d1d5db;'
        'border-radius:4px;background:white;cursor:pointer" '
        "onclick=\"const a=this.parentNode.querySelector('audio');a.currentTime=0;a.play();\">"
        '&#9654;</button>'
        f'<progress max="{duration:.4f}" value="0" style="width:90px;height:8px"></progress>'
        f'<span class="t" style="font-variant-numeric:tabular-nums;color:#374151">'
        f'{0:.{digits}f} / {total} s</span>'
        f'<audio preload="auto"{" autoplay" if autoplay else ""} onplay="{tick}" '
        f'src="data:audio/wav;base64,{wav}"></audio></div>'
    )


def _palette(n):
    if n <= 10:
        return list(Category10[10][:n])
    if n <= 20:
        return list(Category20[20][:n])
    return list(turbo(n))


def _text_color(color):
    """Black or white, readable on the color (hex)."""
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return "#111827" if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


class OutlierTool(SubDash):
    """Detection of the outliers (button), their 3D map and the table of their samples: a
    click on a row (or on an outlier of the map) shows and plays it, its correction is typed
    in the table."""

    def __init__(self, parent):
        super().__init__(parent)
        self.result = None
        self.outliers = None
        self.colors = {}
        self.registry = Registry()  # (the view of each shown outlier, by key)
        self._typical_keys = {}  # label: keys of its typical segments
        self._plays = 0  # (a new player at each play: autoplayed)

        params = outliers_params(self.controler.config)
        self.params = params
        self.neighbors_input = pn.widgets.IntInput(
            name="Neighbors", value=params["neighbors"], start=2, end=100, width=110
        )
        self.min_other_input = pn.widgets.IntInput(
            name="Min. other label", value=params["min_other"], start=1, end=100, width=110
        )
        self.percentile_input = pn.widgets.FloatInput(
            name="Percentile", value=params["percentile"], start=50.0, end=100.0, step=0.5, width=110
        )
        self.dimension_input = pn.widgets.RadioButtonGroup(
            name="UMAP", options={"2D": 2, "3D": 3},
            value=3 if params["n_components"] >= 3 else 2, button_type="default",
            button_style="outline", width=110, align="end",
        )
        params_tooltip = custom_tooltip(
            "Each segment is described by the mean and deviation of its MFCC and their\n"
            "deltas, and its duration, embedded in 2D or 3D by UMAP\n"
            "(the two give different outliers: a 2D UMAP is not a projection of the 3D one).\n"
            "A segment is an outlier when:\n"
            "- none of its 'Neighbors' nearest segments has its label,\n"
            "- the nearest segment of its label is further than for 'Percentile' %\n"
            "  of the segments,\n"
            "- at least 'Min. other label' of its neighbors share another label\n"
            "  (the label it sounds like).",
            direction="right",
        )
        self.detect_btn = pn.widgets.Button(
            name="Detect outliers", button_type="primary", width=160, height=40, align="end"
        )
        self.detect_btn.on_click(self.run_detection)
        self.status = pn.pane.Markdown("", styles={"font-size": "12px", "color": "#6b7280"})

        self.plot_area = pn.Column(sizing_mode="stretch_width")  # (the map, once detected)
        self.shift_tool = ShiftTool(self, params)  # (files whose annotations seem shifted)

        self.save_btn = pn.widgets.Button(
            name="Save corrections", button_type="primary", width=200, disabled=True
        )
        self.save_btn.on_click(self.on_click_save)
        self.save_msg = pn.pane.HTML(styles=dict(color="green", background="white"), visible=False)
        self.table = pn.widgets.Tabulator(
            pd.DataFrame(columns=COLUMNS), show_index=False, selectable=1, height=360,
            sizing_mode="stretch_width", layout="fit_data_stretch", visible=False,
            editors={column: None for column in COLUMNS[:-1]},
            widths={"correction": 140},
        )
        self.table.style.apply(self._label_style, subset=["label", "sounds like"])
        self.table.param.watch(self._on_select, "selection")
        self.table.on_edit(self._on_edit)
        self.table_help = pn.pane.Markdown(
            "Click a row (or an outlier of the map) to see and hear it; type its new label in "
            "the *correction* column (empty: unchanged), then save.",
            styles={"font-size": "12px", "color": "#6b7280"}, visible=False,
        )
        self.detail = pn.Column(sizing_mode="stretch_width")  # (the sample of the clicked row)
        # (its value is incremented at each click on a row: the page scrolls to the sample)
        self._scroll = pn.widgets.IntInput(value=0, visible=False)
        self._scroll.jscallback(value=SCROLL_TO_DETAIL)

        self.layout = pn.Column(
            pn.pane.Markdown("### 1. Detect outliers", margin=(0, 0, 6, 0)),
            pn.Row(
                self.neighbors_input,
                self.min_other_input,
                self.percentile_input,
                pn.Column(pn.pane.Markdown("UMAP", margin=(0, 5, 2, 5),
                                           styles={"font-size": "13px"}),
                          self.dimension_input),
                params_tooltip,
                pn.Spacer(width=10),
                self.detect_btn,
                align="end",
            ),
            self.status,
            self.plot_area,
            pn.pane.Markdown("### 2. Correct outliers", margin=(10, 0, 6, 0)),
            self.table_help,
            self.table,
            self.detail,
            self._scroll,
            pn.Row(self.save_btn, self.save_msg, sizing_mode="stretch_width"),
            self.shift_tool.layout,
            sizing_mode="stretch_width",
        )

    def run_detection(self, event):
        self.detect_btn.disabled = True
        self.detect_btn.loading = True
        self.save_msg.visible = False
        n_segments = int((self.controler.corpus.dataset["label"]
                          != self.controler.config.transforms.annots.silence_tag).sum())
        self.status.object = (
            f"Embedding the MFCC of {n_segments} segments "
            f"(UMAP in {self.dimension_input.value}D), "
            "this can take a few minutes..."
        )
        params = {
            **self.params,
            "neighbors": self.neighbors_input.value,
            "min_other": self.min_other_input.value,
            "percentile": self.percentile_input.value,
            "n_components": self.dimension_input.value,
        }

        def run():
            try:
                result = find_outliers(self.controler.corpus, params)
                shifted = find_shifted_files(self.controler.corpus, result, params)
                with pn.io.unlocked():
                    self._show_results(result, shifted)
            except Exception as e:
                logger.error(f"Outlier detection failed: {e}")
                self.status.object = f"Error: {e}"
            finally:
                self.detect_btn.loading = False
                self.detect_btn.disabled = False

        threading.Thread(target=run, daemon=True).start()

    def _show_results(self, result, shifted):
        self.result = result
        outliers = result[result["outlier"]]
        per_file = (result["sounds_like"] == result["label"]).groupby(result["notated_path"]).mean()
        self.status.object = (
            f"**{len(outliers)} outliers** among {len(result)} segments"
            + f". Segments sounding like their label: {per_file.mean():.0%} per file on average"
            + (f", less than {self.params['min_file_agreement']:.0%} in {len(shifted)} files." if len(shifted) else ".")
        )

        self._update_plot()
        self.shift_tool.set_results(shifted, self.colors)

        # the outliers, those of the suspicious files last
        suspicious = set(shifted["notated_path"]) if len(shifted) else set()
        order = sorted(range(len(outliers)), key=lambda i: (
            outliers["notated_path"].iloc[i] in suspicious, outliers["label"].iloc[i],
            outliers["notated_path"].iloc[i], outliers["onset_s"].iloc[i]))
        self.outliers = outliers.iloc[order].reset_index(drop=True)
        self.registry.clean()
        self._typical_keys = {}
        self.detail.objects = []
        self.save_btn.disabled = True
        self.save_btn.name = "Save corrections"

        o = self.outliers
        table = pd.DataFrame({
            "#": range(1, len(o) + 1),
            "file": o["notated_path"].map(lambda p: pathlib.Path(p).name),
            "onset (s)": o["onset_s"].round(2),
            "duration (s)": (o["offset_s"] - o["onset_s"]).round(2),
            "label": o["label"],
            "sounds like": o["sounds_like"],
            "shifted file": o["notated_path"].map(lambda p: "yes" if p in suspicious else ""),
            "correction": "",
        })
        self.table.editors = {
            **{column: None for column in COLUMNS[:-1]},
            "correction": {"type": "list", "values": sorted(self.controler.classes or []),
                           "autocomplete": True, "freetext": True, "listOnEmpty": True,
                           "clearable": True},
        }
        self.table.value = table
        self.table.visible = self.table_help.visible = len(o) > 0
        if len(o) == 0:
            self.detail.objects = [pn.pane.Alert("No outlier found.", alert_type="success")]

    def _label_style(self, column):
        """Each label cell on the color of its label (as in the map)."""
        styles = []
        for label in column:
            color = self.colors.get(label)
            styles.append(f"background-color: {color}; color: {_text_color(color)}; font-weight: bold"
                          if color else "")
        return styles

    def _update_plot(self):
        """The UMAP of the segments (2D or 3D, that of the detection), the outliers as
        diamonds (a click shows them)."""
        result = self.result
        labels = sorted(result["label"].unique())
        self.colors = dict(zip(labels, _palette(len(labels))))
        files = result["notated_path"].map(lambda p: pathlib.Path(p).name)
        text = [f"{label} (sounds like {like})<br>{file}, {onset:.2f} s"
                for label, like, file, onset in zip(result["label"], result["sounds_like"], files,
                                                    result["onset_s"])]
        result = result.assign(text=text)

        three_d = "umap_2" in result
        trace = go.Scatter3d if three_d else go.Scattergl
        size, outlier_size = (2, 5) if three_d else (4, 10)

        def coordinates(points):
            xyz = dict(x=points["umap_0"], y=points["umap_1"])
            return {**xyz, "z": points["umap_2"]} if three_d else xyz

        fig = go.Figure()
        for label in labels:
            points = result[(result["label"] == label) & ~result["outlier"]]
            fig.add_trace(trace(
                **coordinates(points), mode="markers",
                name=label, legendgroup=label, text=points["text"], hoverinfo="text",
                marker=dict(size=size, color=self.colors[label], opacity=0.4),
            ))
        outliers = result[result["outlier"]]
        fig.add_trace(trace(
            **coordinates(outliers), mode="markers",
            name="outliers", text=outliers["text"], hoverinfo="text",
            customdata=outliers["key"],
            marker=dict(size=outlier_size, symbol="diamond",
                        color=outliers["label"].map(self.colors),
                        line=dict(color="black", width=2), opacity=1.0),
        ))
        hidden = dict(visible=False)
        fig.update_layout(
            height=520, margin=dict(l=0, r=0, t=30, b=0),
            title=dict(text=f"UMAP of the segments in {'3D' if three_d else '2D'} "
                            "(diamonds: outliers, click one to see it)",
                       font=dict(size=13)),
            legend=dict(itemsizing="constant"),
            plot_bgcolor="white",
        )
        if three_d:
            fig.update_layout(scene=dict(xaxis=hidden, yaxis=hidden, zaxis=hidden))
        else:
            fig.update_layout(xaxis=hidden, yaxis=hidden)
        plot = pn.pane.Plotly(fig, height=520, sizing_mode="stretch_width")
        plot.param.watch(self._on_plot_click, "click_data")
        self.plot_area.objects = [plot]

    def _on_plot_click(self, event):
        """A click on an outlier of the map selects its row."""
        points = (event.new or {}).get("points", [])
        key = points[0].get("customdata") if points else None
        if key is None or self.outliers is None:
            return
        rows = self.outliers.index[self.outliers["key"] == key]
        if len(rows):
            self.table.selection = [int(rows[0])]

    def _badge(self, label, size=14):
        """HTML of a label on the color of its class (as in the map and the table)."""
        color = self.colors.get(label, "#9ca3af")
        return (f'<span style="background:{color};color:{_text_color(color)};padding:2px 10px;'
                f'border-radius:4px;font-weight:600;font-size:{size}px">{label}</span>')

    def _dataset_row(self, key):
        """The current row of the dataset of a segment (its label corrected since the
        detection included), or None."""
        dataset = self.controler.corpus.dataset
        time_precision = self.controler.config.transforms.annots.time_precision
        matches = match_annotation(dataset, key, time_precision)
        return dataset.loc[[matches[0]]] if len(matches) else None

    def _typical(self, label):
        """Keys of the TYPICAL segments of the label nearest to the center of its segments in
        the UMAP (outliers excepted)."""
        if label not in self._typical_keys:
            result = self.result
            rows = result[(result["label"] == label) & ~result["outlier"]]
            coordinates = rows[[c for c in rows.columns if c.startswith("umap_")]].to_numpy()
            order = np.argsort(np.linalg.norm(coordinates - coordinates.mean(axis=0), axis=1))
            self._typical_keys[label] = list(rows["key"].iloc[order[:TYPICAL]])
        return self._typical_keys[label]

    def _thumbnail(self, sample):
        spec = self.controler.load_repertoire(sample)[0]
        return pn.pane.Matplotlib(spec[0], format="png", tight=True, height=80, width=160,
                                  margin=(0, 10, 0, 0), align="center")

    def _on_select(self, event):
        """The selected outlier (its spectrogram and its players, on the color of its label)
        next to a typical segment of the label it sounds like."""
        if not event.new or self.outliers is None:
            return
        row = event.new[0]
        outlier = self.outliers.iloc[row]
        if self.registry.get(outlier.key) is None:
            sample = self._dataset_row(outlier.key)
            if sample is None:
                self.detail.objects = [pn.pane.Alert("This segment is no longer in the dataset.",
                                                     alert_type="warning")]
                return
            segment = sample.iloc[0]
            path = str(segment.notated_path)
            wav = _wav(path, segment.onset_s, segment.offset_s)
            context = _wav(path, segment.onset_s - CONTEXT_S, segment.offset_s + CONTEXT_S)
            segment_player = pn.pane.HTML(_player(*wav), margin=(2, 0))
            label_style = {"font-size": "12px", "color": "#6b7280"}
            current = pn.Column(
                pn.pane.HTML(
                    f'{self._badge(segment.label)} <b style="margin-left:6px">#{row + 1} this '
                    f'segment</b> <span style="color:#6b7280;font-size:12px;margin-left:6px">'
                    f'{pathlib.Path(path).name}, {segment.onset_s:.2f} s, '
                    f'{segment.offset_s - segment.onset_s:.2f} s long</span>'),
                pn.Row(
                    self._thumbnail(sample),
                    pn.Column(
                        pn.Row(pn.pane.Markdown("**Segment**", styles=label_style, width=70,
                                                margin=(0, 5), align="center"), segment_player),
                        pn.Row(pn.pane.Markdown(f"± {CONTEXT_S:g} s",
                                                styles=label_style, width=70, margin=(0, 5),
                                                align="center"),
                               pn.pane.HTML(_player(*context), margin=(2, 0))),
                    ),
                ),
                styles={"border-left": f"6px solid {self.colors.get(segment.label, '#9ca3af')}",
                        "padding-left": "10px"},
                sizing_mode="stretch_width",
            )
            # (side by side, one below the other on a narrow screen)
            current.sizing_mode = None
            current.width = 520
            proposed = self._proposed(row, outlier.sounds_like)
            proposed.sizing_mode = None
            proposed.width = 480
            view = pn.FlexBox(
                current, proposed, flex_wrap="wrap", gap="12px",
                sizing_mode="stretch_width",
                css_classes=["outlier-detail"],  # (found by SCROLL_TO_DETAIL)
                styles={"border": "1px solid #e5e7eb", "border-radius": "8px", "padding": "8px"},
            )
            self.registry[outlier.key] = (view, segment_player, wav)
        view, segment_player, wav = self.registry[outlier.key]
        self.detail.objects = [view]
        self._scroll.value += 1  # (the page scrolls to it, if it is not in sight)
        # the segment is played at each click on its row (a new player, autoplayed)
        self._plays += 1
        segment_player.object = _player(*wav, autoplay=True, nonce=self._plays)

    def _proposed(self, row, label):
        """A typical segment of the label the outlier sounds like (the next one at each click
        on "Another example"), and a button relabeling the outlier with it."""
        keys = self._typical(label)
        header = pn.pane.HTML("")
        body = pn.Row()
        state = {"index": -1}

        def show(event=None):
            if not keys:
                header.object = f"{self._badge(label)} <b>sounds like</b> (no typical segment)"
                return
            state["index"] = (state["index"] + 1) % len(keys)
            sample = self._dataset_row(keys[state["index"]])
            if sample is None:
                return
            segment = sample.iloc[0]
            header.object = (
                f'{self._badge(label)} <b style="margin-left:6px">sounds like {label}</b> '
                f'<span style="color:#6b7280;font-size:12px;margin-left:6px">typical {label} '
                f'{state["index"] + 1}/{len(keys)}, {segment.offset_s - segment.onset_s:.2f} s long'
                '</span>')
            self._plays += 1
            body.objects = [
                self._thumbnail(sample),
                pn.pane.HTML(_player(*_wav(str(segment.notated_path), segment.onset_s,
                                           segment.offset_s),
                                     autoplay=event is not None, nonce=self._plays),
                             margin=(2, 0), align="center"),
            ]

        another = pn.widgets.Button(name="Another example", button_type="default", width=150)
        another.on_click(show)
        relabel = pn.widgets.Button(name=f"Relabel as {label}", button_type="primary", width=150)
        relabel.on_click(lambda event: self._relabel(row, label))
        show()
        color = self.colors.get(label, "#9ca3af")
        return pn.Column(
            header, body, pn.Row(another, relabel),
            styles={"border-left": f"6px solid {color}", "padding-left": "10px"},
            sizing_mode="stretch_width",
        )

    def _relabel(self, row, label):
        """The proposed label typed in the correction column of the row."""
        self.table.patch({"correction": [(self.table.value.index[row], label)]})
        self._on_edit(None)

    def _corrections(self):
        """{key: new label} of the rows whose correction is filled in."""
        table = self.table.value
        filled = table["correction"].fillna("").astype(str).str.strip()
        return {self.outliers["key"].iloc[i]: filled.iloc[i]
                for i in range(len(table)) if filled.iloc[i] != ""}

    def _on_edit(self, event):
        n = len(self._corrections())
        self.save_btn.disabled = n == 0
        self.save_btn.name = f"Save {n} correction{'s' if n > 1 else ''}" if n else "Save corrections"
        self.save_msg.visible = False

    def on_click_save(self, events):
        corrections = self._corrections()
        if not corrections:
            return
        self.controler.apply_live_corrections(corrections, "annot")
        # the table shows the new labels
        table = self.table.value.copy()
        for i, key in enumerate(self.outliers["key"]):
            if key in corrections:
                table.loc[table.index[i], "label"] = corrections[key]
                self.outliers.loc[i, "label"] = corrections[key]
        table["correction"] = ""
        self.table.value = table
        self.registry.clean()
        self._typical_keys = {}
        self.detail.objects = []
        self.save_btn.disabled = True
        self.save_btn.name = "Save corrections"
        self.save_msg.object = f"Saved {len(corrections)} corrections."
        self.save_msg.visible = True
        # the other tools of the page (Preprocess or Eval) show the corrected labels
        self.parent.refresh_after_outlier_save()
