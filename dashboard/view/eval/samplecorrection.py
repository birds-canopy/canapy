# Author: Nathan Trouvain at 18/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: BSD-3-Clause
# Copyright: Nathan Trouvain
import logging
import pathlib

import panel as pn

from canapy.plots import plot_bokeh_label_count
from canapy.correction import make_annot_key

from .classmerge import RepertoireView
from ..helpers import SubDash, Registry, SideBar, custom_tooltip
from ..ordering import anchor_of, anchors_of, restore_page
from ..bounds_editor import MARGIN_MAX_S, MODAL_CSS, BoundsEditor

logger = logging.getLogger("canapy-dashboard")

SAMPLE_CSS = """
.sample-card {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    margin-bottom: 15px;
}
.selector-card {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 5px;
    transition: all 0.2s;
}
.selector-card:hover {
    border-color: #3b82f6;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.sample-row-card {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
"""

class SampleCorrectionDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        pn.config.raw_css.append(SAMPLE_CSS)
        pn.config.raw_css.append(MODAL_CSS)

        # mounted at the root rather than inside a card: a fixed-position
        # overlay must not sit under an ancestor that could clip it
        self.modal_area = pn.Column(visible=False, height=0, width=0, margin=0)

        self.registry = Registry()
        self.registry["sample"] = dict()
        self.registry["class"] = dict()

        self.total_corrected = 0
        # class being corrected, and where each class was left, so a save does
        # not send the user back to "select a class above"
        self._active_label = None
        self._places = {}

        self.class_selectors = pn.FlexBox(justify_content='start', gap=10, align_items='center')
        self.build_class_selectors()

        self.save_btn = pn.widgets.Button(
            name="Save all", 
            sizing_mode="stretch_width", 
            button_type="primary",
            disabled=True 
        )
        self.save_btn.on_click(self.on_click_save)
        self.save_msg = pn.pane.HTML(styles=dict(color="green", background="white"), visible=False)

        self.repertoire_view = RepertoireView(
            self, num_panel=1, orientation="column", num_samples=100, page_size=5,
        )

        fig, self.counts = plot_bokeh_label_count(self.controler.misclassified_segments)
        fig.min_border = 0
        self.bokeh_pane = pn.pane.Bokeh(fig, sizing_mode="stretch_width", height=280)

        self.sample_container = pn.Column(
            pn.pane.Alert("Select a class above to start correcting.", alert_type="light"),
            sizing_mode="stretch_width",
            scroll=True,
            min_height=300,
        )

        self.layout = pn.Column(
            self.modal_area,
            self.bokeh_pane,
            pn.Spacer(height=20),
            pn.Column(
                pn.pane.Markdown("### 1. Select a class to correct", margin=(10, 0, 10, 0)),
                pn.Column(
                    self.class_selectors,
                    scroll=True,
                    min_height=200,
                    styles={
                        "background": "#f8f9fa",
                        "padding": "15px",
                        "border-radius": "8px",
                        "border": "1px solid #dee2e6",
                    },
                    sizing_mode="stretch_width",
                ),
                pn.pane.Markdown("### 2. Correct samples", margin=(20, 0, 10, 0)),
                pn.Row(
                    self.save_msg,
                    self.save_btn,
                    sizing_mode="stretch_width",
                    align="center",
                    margin=(0, 0, 8, 0),
                ),
                sizing_mode="stretch_width",
            ),
            pn.Spacer(height=10),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("**Samples to correct**", margin=(0, 0, 6, 0)),
                    pn.pane.Markdown("Browse samples that have been missclassified by the models and correct their label if it's wrong."),
                    self.sample_container,
                    sizing_mode="stretch_width",
                ),
                pn.Spacer(width=20),
                pn.Column(
                    pn.pane.Markdown("**Audio repertoire**", margin=(0, 0, 6, 0)),
                    self.repertoire_view.layout,
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )

    def open_modal(self, content):
        """Show ``content`` over the page, behind a backdrop that blocks clicks."""
        backdrop = pn.pane.HTML(
            "<div class='bounds-backdrop'></div>", width=0, height=0, margin=0
        )
        self.modal_area.objects = [backdrop, content]
        self.modal_area.visible = True

    def close_modal(self):
        self.modal_area.visible = False
        self.modal_area.objects = []

    def build_class_selectors(self):
        self.class_selectors.objects = []
        self.registry["class"] = {}
        misclass = self.controler.misclassified_segments
        for lbl in misclass.label.unique():
            n_error = len(misclass.query("label==@lbl"))
            display = ClassSelectionView(lbl, n_error, self)
            self.registry["class"][lbl] = display
            self.class_selectors.append(display.layout)

    def get_sample_corrector(self, label):
        if self.registry["sample"].get(label) is not None:
            return self.registry["sample"][label].layout
        anchor, previous = self._places.pop(label, (None, []))
        sample_corrector = SampleCorrectorView(
            self, label, anchor=anchor, previous_anchors=previous
        )
        self.registry["sample"][label] = sample_corrector
        return sample_corrector.layout

    def highlight_selection(self, active_label):
        for lbl, view in self.registry["class"].items():
            if lbl == active_label:
                view.select_btn.button_type = "primary"
            else:
                view.select_btn.button_type = "default"

    def listen_class_selection(self, label):
        self._active_label = label
        self.highlight_selection(label)
        self.save_msg.visible = False
        sample_corrector_layout = self.get_sample_corrector(label)
        self.sample_container.objects = [sample_corrector_layout]

    def listen_correction(self, label, increment):
        self.registry["class"][label].receive_correction(increment)
        self.total_corrected += increment
        self.save_btn.disabled = self.total_corrected == 0

    def check_correction(self, label):
        return label in self.controler.classes if label != "" else True

    def _refresh_after_annot_save(self):
        saved_counts = {
            lbl: view.num_corrected
            for lbl, view in self.registry["class"].items()
        }
        # the corpus changed: remember which sample each class was showing, so
        # reselecting it reopens there rather than on page 1
        self._places = {
            label: (view.current_anchor(), view.anchors())
            for label, view in self.registry["sample"].items()
        }
        self.registry["sample"] = {}
        self.build_class_selectors()
        fig, self.counts = plot_bokeh_label_count(self.controler.misclassified_segments)
        fig.min_border = 0
        self.bokeh_pane.object = fig
        for lbl, count in saved_counts.items():
            if lbl in self.registry["class"] and count > 0:
                view = self.registry["class"][lbl]
                view.num_corrected = count
                view.count_btn.name = f"{count} / {view.num_error}"
                view.count_btn.button_type = "success"
        # reopen the class that was being corrected, on the sample it was on
        if self._active_label in self.registry["class"]:
            self.listen_class_selection(self._active_label)
        else:
            self.sample_container.objects = [
                pn.pane.Alert("Select a class above to start correcting.", alert_type="light")
            ]
        self.repertoire_view.update_classes()
        self.parent.merge_dashboard.corrector.rebuild_grid()

    def on_click_save(self, events):
        new_corrections, new_bounds = {}, {}
        for sample_corrector in self.registry["sample"].values():
            new_corrections.update(sample_corrector.corrections)
            new_bounds.update(sample_corrector.bounds_corrections)
        if not new_corrections and not new_bounds:
            return
        if new_corrections:
            self.controler.apply_live_corrections(new_corrections, "annot")
        # after the labels: moving an onset changes the key a label correction
        # on the same annotation was captured under
        if new_bounds:
            self.controler.apply_live_corrections(new_bounds, "bounds")
        self._refresh_after_annot_save()
        self.save_msg.object = "Saved!"
        self.save_msg.visible = True


class ClassSelectionView(SubDash):
    def __init__(self, label, num_error, parent):
        super().__init__(parent)

        self.label = label
        self.num_error = num_error
        self.num_corrected = 0

        self.select_btn = pn.widgets.Button(
            name=label,
            button_type="default",
            height=35,
            sizing_mode="stretch_width",
        )
        self.select_btn.on_click(self.on_click_notify_display)

        self.count_btn = pn.widgets.Button(
            name=f"0 / {num_error}",
            button_type="light",
            sizing_mode="stretch_width",
            height=25,
        )
        self.count_btn.on_click(self.on_click_notify_display)

        self.layout = pn.Column(
            self.select_btn,
            self.count_btn,
            width=110,
            css_classes=['selector-card'],
        )

    def on_click_notify_display(self, events):
        self.parent.listen_class_selection(self.label)

    def receive_correction(self, increment):
        self.num_corrected += increment
        self.count_btn.name = f"{self.num_corrected} / {self.num_error}"
        self.count_btn.button_type = "success" if self.num_corrected > 0 else "light"


class SampleCorrectorView(SubDash):
    PAGE_SIZE = 25

    def __init__(self, parent, label, anchor=None, previous_anchors=None):
        super().__init__(parent)

        misclass = self.controler.misclassified_segments.query("label==@label")
        self.misclassified_segments = (
            misclass if not misclass.empty
            else self.controler.corpus.dataset.query("label==@label")
        )
        self.label = label
        self.registry = Registry()
        self._syncing = False

        n = len(self.misclassified_segments)
        self._n_pages = max(1, -(-n // self.PAGE_SIZE))
        self._container = pn.Column(sizing_mode="stretch_width")

        # a correction rebuilt the corpus: reopen on the sample that was on
        # screen rather than on page 1 (see dashboard/view/ordering.py)
        self._current_page = 0
        if anchor is not None:
            page, _ = restore_page(anchor, previous_anchors or [],
                                   self.misclassified_segments, self.PAGE_SIZE)
            self._current_page = min(page, self._n_pages - 1)

        pager_kwargs = dict(
            options=list(range(1, self._n_pages + 1)),
            value=self._current_page + 1,
            button_style="outline",
            button_type="default",
            visible=self._n_pages > 1,
        )
        self._pager_top = pn.widgets.RadioButtonGroup(**pager_kwargs)
        self._pager_bot = pn.widgets.RadioButtonGroup(**pager_kwargs)
        self._pager_top.param.watch(self._on_page_change_top, "value")
        self._pager_bot.param.watch(self._on_page_change_bot, "value")

        self.layout = pn.Column(
            pn.Row(pn.Spacer(), self._pager_top, pn.Spacer(), sizing_mode="stretch_width"),
            self._container,
            pn.Row(pn.Spacer(), self._pager_bot, pn.Spacer(), sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )
        if self.misclassified_segments.empty:
            self._container.objects = [pn.pane.Markdown("_No samples found._")]
        else:
            self._render_page(self._current_page)

    @property
    def corrections(self):
        return {
            i: display.correction
            for i, display in self.registry.items()
            if display.correction != ""
        }

    @property
    def bounds_corrections(self):
        return {
            i: display.bounds_correction
            for i, display in self.registry.items()
            if display.bounds_correction is not None
        }

    def current_anchor(self):
        """The sample opening the current page, to come back to it later."""
        start = self._current_page * self.PAGE_SIZE
        if start >= len(self.misclassified_segments):
            return None
        return anchor_of(self.misclassified_segments.iloc[start])

    def anchors(self):
        return anchors_of(self.misclassified_segments)

    def _render_page(self, page_index):
        self._current_page = page_index
        start = page_index * self.PAGE_SIZE
        batch = self.misclassified_segments.iloc[start:start + self.PAGE_SIZE]
        if batch.empty:
            return
        segments = self.controler.load_repertoire(batch)
        rows = []
        for i, (segment, (idx, annots)) in enumerate(zip(segments, batch.iterrows())):
            display_num = start + i + 1
            # Key corrections by a stable identifier (notated_path, onset_s)
            # instead of the volatile dataset index, so they survive the
            # re-indexing caused by class merges / preprocessing.
            key = make_annot_key(annots.notated_path, annots.onset_s)
            if self.registry.get(key) is None:
                self.registry[key] = SingleSampleCorrectorView(
                    annots, segment, self.parent, display_num
                )
            rows.append(self.registry[key].layout)
        self._container.objects = rows

    def _on_page_change_top(self, event):
        if self._syncing:
            return
        self._syncing = True
        self._pager_bot.value = event.new
        self._syncing = False
        self._render_page(event.new - 1)

    def _on_page_change_bot(self, event):
        if self._syncing:
            return
        self._syncing = True
        self._pager_top.value = event.new
        self._syncing = False
        self._render_page(event.new - 1)


class SingleSampleCorrectorView(SubDash):
    def __init__(self, repertoire_entry, spec, parent, display_num):
        super().__init__(parent)

        self.label = repertoire_entry.label
        self.repertoire_entry = repertoire_entry
        # the same key the registry and the correction store use
        self.idx = make_annot_key(repertoire_entry.notated_path, repertoire_entry.onset_s)
        self.bounds_corrected = False
        self.bounds_correction = None

        self.predictions = self.repertoire_entry.filter(regex="pred_.*")
        self.predictions.index = [p.split("_")[1] for p in self.predictions.index]

        self.text_input = pn.widgets.TextInput(placeholder="Correction", width=130, margin=(5, 0))
        self.corrected = False
        self.text_input.param.watch(self.on_correction_notify, "value")

        sampling_rate = round(self.controler.config.transforms.audio.sampling_rate)

        self.number_pane = pn.pane.Markdown(
            f"**#{display_num}**",
            styles={'font-size': '14px', 'color': '#9ca3af', 'font-weight': '600'},
            width=40, 
            align='center'
        )

        self.img = pn.pane.Matplotlib(
            spec[0],
            format="png",
            dpi=200,
            tight=True,
            height=80,
            sizing_mode="stretch_width",
            min_width=100,
            max_width=200,
        )

        self.audio_col = pn.Column(
            pn.pane.Audio(spec[1], sample_rate=sampling_rate, height=30, sizing_mode="stretch_width"),
            pn.Spacer(height=5),
            pn.pane.Audio(spec[2], sample_rate=sampling_rate, height=30, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
            min_width=220,
            max_width=340,
            styles={"overflow": "hidden"},
        )

        models_preds = ", ".join([f"{idx}: {p}" for idx, p in self.predictions.items()])
        notated_file = pathlib.Path(self.repertoire_entry.notated_path).stem

        file_tooltip = custom_tooltip(
            f"File: {notated_file}\n"
            f"Time: {self.repertoire_entry.onset_s:.2f}s - {self.repertoire_entry.offset_s:.2f}s\n"
            f"Predictions: {models_preds}",
            direction="right",
        )

        self.label_status = pn.pane.Markdown("", styles={"font-size": "10px", "color": "orange"})

        self.edit_btn = pn.widgets.Button(
            name="Edit boundaries", button_type="primary",
            width=140, height=30, margin=(4, 0, 0, 0),
            stylesheets=["""
                button.bk-btn {
                    background-color: #2563eb !important;
                    border-color: #2563eb !important;
                    color: #ffffff !important;
                    font-weight: 600 !important;
                    font-size: 12px !important;
                    border-radius: 6px !important;
                }
                button.bk-btn:hover { background-color: #1d4ed8 !important; }
            """],
        )
        self.edit_btn.on_click(self._open_bounds_editor)
        self.bounds_status = pn.pane.HTML("", styles={"font-size": "11px"})

        input_section = pn.Column(
            pn.Row(
                pn.pane.Markdown(f"**{self.label}**", styles={'font-size': '13px', 'font-weight': '600'}, margin=(0, 5, 0, 0)),
                self.label_status
            ),
            self.text_input,
            self.edit_btn,
            self.bounds_status,
            width=140,
        )

        main_group = pn.Row(
            self.number_pane,
            file_tooltip,
            self.img,
            pn.Spacer(width=10),
            self.audio_col,
            sizing_mode="stretch_width",
            align="center",
        )

        self.layout = pn.FlexBox(
            main_group,
            input_section,
            css_classes=['sample-row-card'],
            sizing_mode="stretch_width",
            align_items="center",
            flex_wrap="wrap",
            gap=10,
        )

    def _open_bounds_editor(self, event):
        """Swap the page for the editor; Validate and Cancel bring it back."""
        try:
            editor = BoundsEditor(
                self.repertoire_entry, self.idx,
                on_validate=self._commit_bounds,
                on_cancel=self.parent.close_modal,
                config=self.controler.config,
                neighbours=self._neighbours(),
            )
        except Exception as error:
            logger.exception("Could not open the bounds editor")
            self.bounds_status.object = (
                f"<span style='color:#dc2626;'>Editor error: {error}</span>"
            )
            return
        self.parent.open_modal(editor.layout)

    def _neighbours(self):
        """The other annotations of the same file, read off the corpus.

        The corpus rather than the misclassified segments: those hold only the
        problematic ones, so a boundary would be placed against an incomplete
        picture. Silence is left out, since growing a phrase into it is the
        correction this editor exists for.
        """
        df = self.controler.corpus.dataset
        entry = self.repertoire_entry
        silence_tag = self.controler.config.transforms.annots.silence_tag
        same_file = df[(df["notated_path"] == entry.notated_path)
                       & (df["label"] != silence_tag)]
        others = same_file[
            (same_file["onset_s"] != entry.onset_s)
            | (same_file["offset_s"] != entry.offset_s)
        ]
        # as far as the editor can widen its view, so a neighbour appears
        # as soon as it comes into sight
        near = others[(others["offset_s"] > entry.onset_s - MARGIN_MAX_S)
                      & (others["onset_s"] < entry.offset_s + MARGIN_MAX_S)]
        return list(near.itertuples())


    def _refresh_thumbnail(self, onset, offset):
        """Redraw the card with the bounds just validated.

        The correction is only pending — the corpus changes on Save — so the
        card is redrawn from the new bounds directly, rather than waiting for a
        rebuild that would come several edits later.
        """
        try:
            spec = self.controler.render_segment(
                self.repertoire_entry.notated_path, onset, offset
            )
        except Exception:
            logger.exception("Could not redraw the corrected segment")
            return
        self.img.object = spec[0]
        players = [p for p in self.audio_col if isinstance(p, pn.pane.Audio)]
        for player, audio in zip(players, (spec[1], spec[2])):
            player.object = audio

    def _commit_bounds(self, key, bounds):
        self.bounds_correction = (float(bounds[0]), float(bounds[1]))
        self._refresh_thumbnail(*self.bounds_correction)
        self.bounds_status.object = (
            f"<span style='color:#16a34a;'>bounds "
            f"{bounds[0]:.3f}–{bounds[1]:.3f} s pending</span>"
        )
        if not self.bounds_corrected:
            self.bounds_corrected = True
            self.parent.listen_correction(self.label, increment=1)
        self.parent.close_modal()

    @property
    def correction(self):
        return self.text_input.value

    def on_correction_notify(self, events):
        validation = self.parent.check_correction(self.text_input.value)
        if not validation:
            self.label_status.object = "New class"
        else:
            self.label_status.object = ""

        if self.text_input.value == "":
            self.corrected = False
            self.parent.listen_correction(self.label, increment=-1)
        else:
            if not self.corrected:
                self.corrected = True
                self.parent.listen_correction(self.label, increment=1)