# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Dragging the onset and offset of one annotation on its spectrogram.

Shown as a modal over the page: a sample card is too short to work in, and the
boundaries need room to be placed precisely.

The boundaries are moved by three handles: one per boundary, and one in the
middle that shifts the whole segment. The drag is driven by pan events on the
figure rather than by one of Bokeh's editing tools.

Neither tool fits. ``BoxEditTool`` grabs on the box outline, a couple of pixels
wide, and moves the box freely in two dimensions when only time means anything.
``PointDrawTool`` can be pinned to one height, but it dims every handle it did
not select and only writes ``source.data`` back when the mouse is released, so
the segment does not follow the cursor. Pan events give the live redraw and
never touch a vertical coordinate, since nothing here reads one.
"""
import logging

import panel as pn
from bokeh.events import Pan, PanEnd, PanStart
from bokeh.models import ColumnDataSource, CustomJS, Div, LinearColorMapper, Range1d
from bokeh.plotting import figure

from canapy.plots import segment_melspectrogram_data

logger = logging.getLogger("canapy-dashboard")

MARGIN_S = 0.5
# how far the view can be widened when an annotation sits well off the sound it
# should cover, and how far it can be narrowed again
MARGIN_MIN_S = 0.25
MARGIN_MAX_S = 4.0
MARGIN_STEP = 2.0
PLOT_HEIGHT = 420
HANDLE_SIZE = 20
BOX_COLOUR = "#4ade80"
EDGE_COLOUR = "#16a34a"
MOVE_COLOUR = "#2563eb"
NEIGHBOUR_COLOUR = "#f97316"

MODAL_CSS = """
.bounds-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(15, 23, 42, 0.55);
    z-index: 9998;
}
.bounds-modal {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 92vw;
    max-width: 1200px;
    max-height: 90vh;
    overflow: auto;
    background: #ffffff;
    border-radius: 10px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
    padding: 18px 22px 14px 22px;
    z-index: 9999;
    box-sizing: border-box;
}
"""

# Grabs whichever handle is nearest and redraws everything as the mouse moves,
# so the segment follows the cursor instead of jumping on release.
_PAN_START_JS = """
const x = cb_obj.x;
const d = state.data;
const near = [
    {kind: 'onset', at: d.onset[0]},
    {kind: 'move', at: (d.onset[0] + d.offset[0]) / 2},
    {kind: 'offset', at: d.offset[0]},
];
let best = near[0];
for (const candidate of near) {
    if (Math.abs(candidate.at - x) < Math.abs(best.at - x)) { best = candidate; }
}
// beyond this the click is not aimed at any handle, and the drag is ignored
const reach = (window_src.data.t1[0] - window_src.data.t0[0]) * 0.06;
state.data = {
    onset: d.onset, offset: d.offset,
    grabbed: [Math.abs(best.at - x) <= reach ? best.kind : 'none'],
    from_x: [x], from_onset: [d.onset[0]], from_offset: [d.offset[0]],
};
"""

_PAN_JS = """
const d = state.data;
const grabbed = d.grabbed[0];
if (grabbed === 'none') { return; }

const shift = cb_obj.x - d.from_x[0];
const t0 = window_src.data.t0[0], t1 = window_src.data.t1[0];
let onset = d.from_onset[0], offset = d.from_offset[0];

if (grabbed === 'move') {
    const span = offset - onset;
    onset += shift; offset += shift;
    if (onset < t0) { onset = t0; offset = t0 + span; }
    if (offset > t1) { offset = t1; onset = t1 - span; }
} else if (grabbed === 'onset') {
    onset = Math.min(Math.max(onset + shift, t0), offset);
} else {
    offset = Math.max(Math.min(offset + shift, t1), onset);
}

state.data = {
    onset: [onset], offset: [offset], grabbed: d.grabbed,
    from_x: d.from_x, from_onset: d.from_onset, from_offset: d.from_offset,
};
box.data = {x: [(onset + offset) / 2], y: [y], width: [offset - onset], height: [height]};
handles.data = {
    x: [onset, (onset + offset) / 2, offset], y: [y, y, y],
    colour: handles.data.colour, marker: handles.data.marker,
};
readout.text = (
    "<span style='font-family:monospace;font-size:14px;'>onset <b>" +
    onset.toFixed(3) + "</b> s &nbsp;&nbsp; offset <b>" + offset.toFixed(3) +
    "</b> s &nbsp;&nbsp; <span style='color:#6b7280;'>(" +
    (offset - onset).toFixed(3) + " s)</span></span>"
);
"""

# only on release does Python hear about it, so the overlap notice is computed
# once per drag rather than on every mouse move
_PAN_END_JS = """
const d = state.data;
state.data = {
    onset: d.onset, offset: d.offset, grabbed: ['none'],
    from_x: d.from_x, from_onset: d.from_onset, from_offset: d.from_offset,
};
committed.data = {onset: d.onset, offset: d.offset};
"""


class BoundsEditor:
    """Spectrogram, three draggable handles, and Validate / Cancel."""

    def __init__(self, row_data, key, on_validate, on_cancel, config, neighbours=None):
        self.key = key
        self.row_data = row_data
        self._on_validate = on_validate
        self._on_cancel = on_cancel
        self.neighbours = list(neighbours) if neighbours is not None else []

        self.audio = config.transforms.audio
        self.margin_s = MARGIN_S
        onset, offset = float(row_data.onset_s), float(row_data.offset_s)
        spec, (t0, t1) = self._spectrogram(self.margin_s)
        self.window = (t0, t1)
        height = float(spec.shape[0])
        mid_y = height / 2
        self._height = height

        self.figure = figure(
            height=PLOT_HEIGHT, sizing_mode="stretch_width",
            x_range=Range1d(t0, t1), y_range=Range1d(0, height),
            # no pan tool: pan events drive the handles instead
            tools="xwheel_zoom,reset", toolbar_location="right",
            x_axis_label="seconds",
        )
        self.figure.yaxis.visible = False
        self.figure.grid.visible = False
        self.spectrogram = ColumnDataSource(data=dict(
            image=[spec], x=[t0], y=[0.0], dw=[t1 - t0], dh=[height],
        ))
        self.figure.image(image="image", x="x", y="y", dw="dw", dh="dh",
                          source=self.spectrogram,
                          color_mapper=LinearColorMapper(palette="Magma256"),
                          level="image")
        # read by the drag callbacks, so widening the view is picked up without
        # rebuilding them
        self.window_src = ColumnDataSource(data=dict(t0=[t0], t1=[t1]))
        self._draw_neighbours(height)

        self.box = ColumnDataSource(data=dict(
            x=[(onset + offset) / 2], y=[mid_y],
            width=[offset - onset], height=[height],
        ))
        self.figure.rect("x", "y", "width", "height", source=self.box,
                         fill_alpha=0.16, fill_color=BOX_COLOUR,
                         line_color=BOX_COLOUR, line_width=2)

        self.handles = ColumnDataSource(data=dict(
            x=[onset, (onset + offset) / 2, offset],
            y=[mid_y] * 3,
            colour=[EDGE_COLOUR, MOVE_COLOUR, EDGE_COLOUR],
            marker=["square", "circle", "square"],
        ))
        # full-height marks so a boundary is read exactly, while the handle
        # below is what the mouse actually has to hit
        self.figure.segment(x0="x", x1="x", y0=0, y1=height, source=self.handles,
                            line_color="colour", line_width=2)
        self.figure.scatter(
            "x", "y", source=self.handles, size=HANDLE_SIZE, marker="marker",
            fill_color="colour", line_color="white", line_width=2,
        )

        # what the drag is doing, and what it has settled on
        self._state = ColumnDataSource(data=dict(
            onset=[onset], offset=[offset], grabbed=["none"],
            from_x=[0.0], from_onset=[onset], from_offset=[offset],
        ))
        self.committed = ColumnDataSource(data=dict(onset=[onset], offset=[offset]))

        self.readout = Div(sizing_mode="stretch_width")
        self.warning = pn.pane.HTML(sizing_mode="stretch_width", margin=(4, 0, 0, 0))
        self._refresh_readout(onset, offset)

        args = dict(state=self._state, box=self.box, handles=self.handles,
                    committed=self.committed, readout=self.readout,
                    window_src=self.window_src, y=mid_y, height=height)
        self.figure.js_on_event(PanStart, CustomJS(args=args, code=_PAN_START_JS))
        self.figure.js_on_event(Pan, CustomJS(args=args, code=_PAN_JS))
        self.figure.js_on_event(PanEnd, CustomJS(args=args, code=_PAN_END_JS))
        self.committed.on_change("data", self._on_committed)

        self.layout = self._build_layout(row_data)

    def _spectrogram(self, margin_s):
        return segment_melspectrogram_data(
            self.row_data.notated_path,
            float(self.row_data.onset_s), float(self.row_data.offset_s),
            sampling_rate=self.audio.sampling_rate, hop_length=self.audio.hop_length,
            n_fft=self.audio.n_fft, win_length=self.audio.win_length,
            fmin=self.audio.fmin, fmax=self.audio.fmax, margin_s=margin_s,
        )[:2]

    def _set_margin(self, margin_s):
        """Redraw on a wider or narrower window, leaving the boundaries alone.

        Widened on both sides at once: an annotation that sits off the sound it
        should cover can be off in either direction, and one button that always
        helps beats two that are right half the time.
        """
        margin_s = min(max(margin_s, MARGIN_MIN_S), MARGIN_MAX_S)
        if margin_s == self.margin_s:
            return
        try:
            spec, (t0, t1) = self._spectrogram(margin_s)
        except Exception:
            logger.exception("Could not widen the bounds editor view")
            return

        self.margin_s = margin_s
        self.window = (t0, t1)
        self.spectrogram.data = dict(
            image=[spec], x=[t0], y=[0.0], dw=[t1 - t0], dh=[self._height],
        )
        self.window_src.data = dict(t0=[t0], t1=[t1])
        self.figure.x_range.start, self.figure.x_range.end = t0, t1
        self._refresh_view_label()

    def _refresh_view_label(self):
        self.view_label.object = (
            f"<span style='font-size:12px;color:#6b7280;'>"
            f"± {self.margin_s:.2f} s around the segment</span>"
        )
        self.wider_btn.disabled = self.margin_s >= MARGIN_MAX_S
        self.narrower_btn.disabled = self.margin_s <= MARGIN_MIN_S

    def _build_layout(self, row_data):
        validate = pn.widgets.Button(name="Validate", button_type="success", width=130, height=36)
        cancel = pn.widgets.Button(name="Cancel", button_type="default", width=110, height=36)
        validate.on_click(self._validate)
        cancel.on_click(lambda event: self._on_cancel())

        self.wider_btn = pn.widgets.Button(
            name="− Wider view", button_type="default", width=120, height=30,
            description="Show more before and after the segment",
        )
        self.narrower_btn = pn.widgets.Button(
            name="+ Closer view", button_type="default", width=120, height=30,
            description="Show less around the segment",
        )
        self.wider_btn.on_click(lambda e: self._set_margin(self.margin_s * MARGIN_STEP))
        self.narrower_btn.on_click(lambda e: self._set_margin(self.margin_s / MARGIN_STEP))
        self.view_label = pn.pane.HTML(margin=(6, 0, 0, 6))
        self._refresh_view_label()

        header = pn.pane.HTML(
            f"<div style='font-size:15px;font-weight:700;color:#111827;'>"
            f"Edit boundaries — <span style='color:#2563eb;'>{row_data.label}</span></div>"
            f"<div style='font-size:12px;color:#6b7280;margin-top:2px;'>"
            f"Drag a green handle to move one boundary, the blue one to shift "
            f"the whole segment.</div>",
            sizing_mode="stretch_width",
        )
        return pn.Column(
            header,
            pn.Row(self.wider_btn, self.narrower_btn, self.view_label,
                   align="center", sizing_mode="stretch_width", margin=(4, 0)),
            pn.pane.Bokeh(self.figure, sizing_mode="stretch_width"),
            pn.Row(pn.pane.Bokeh(self.readout, sizing_mode="stretch_width"),
                   pn.Spacer(width=20), cancel, validate,
                   align="center", sizing_mode="stretch_width"),
            self.warning,
            sizing_mode="stretch_width",
            css_classes=["bounds-modal"],
        )

    def _draw_neighbours(self, height):
        """The annotations on either side, so a boundary is placed in context.

        One renderer for all of them, drawn wherever they are: widening the
        view brings the further ones into sight without redrawing anything.
        """
        if not self.neighbours:
            return
        self.figure.rect(
            x=[(o.onset_s + o.offset_s) / 2 for o in self.neighbours],
            y=[height / 2] * len(self.neighbours),
            width=[o.offset_s - o.onset_s for o in self.neighbours],
            height=[height] * len(self.neighbours),
            fill_alpha=0.06, fill_color=NEIGHBOUR_COLOUR,
            line_color=NEIGHBOUR_COLOUR, line_width=1, line_dash="dashed",
            level="underlay",
        )

    def bounds(self):
        """The boundaries the last drag settled on, clamped to the window."""
        data = self.committed.data
        if not len(data.get("onset", [])):
            return float(self.row_data.onset_s), float(self.row_data.offset_s)
        t0, t1 = self.window
        onset, offset = sorted((float(data["onset"][0]), float(data["offset"][0])))
        return max(t0, onset), min(t1, offset)

    def _on_committed(self, attr, old, new):
        self._refresh_readout(*self.bounds())

    def _refresh_readout(self, onset, offset):
        self.readout.text = (
            f"<span style='font-family:monospace;font-size:14px;'>"
            f"onset <b>{onset:.3f}</b> s &nbsp;&nbsp; offset <b>{offset:.3f}</b> s "
            f"&nbsp;&nbsp; <span style='color:#6b7280;'>({offset - onset:.3f} s)</span></span>"
        )
        self.warning.object = self._overlap_warning(onset, offset)

    def _overlap_warning(self, onset, offset):
        """Name the neighbours the segment now covers, without refusing it."""
        crossed = [
            other for other in self.neighbours
            if onset < other.offset_s and offset > other.onset_s
        ]
        if not crossed:
            return ""
        names = ", ".join(sorted({str(other.label) for other in crossed}))
        return (
            f"<span style='color:#d97706;font-size:12px;'>Overlaps {names}. "
            f"This segment takes precedence: {names} will be trimmed back to "
            f"its edge, or dropped if it is covered entirely.</span>"
        )

    def _validate(self, event):
        onset, offset = self.bounds()
        if offset - onset <= 0:
            self.warning.object = (
                "<span style='color:#dc2626;font-size:12px;'>"
                "A segment cannot end before it starts.</span>"
            )
            return
        self._on_validate(self.key, (onset, offset))
