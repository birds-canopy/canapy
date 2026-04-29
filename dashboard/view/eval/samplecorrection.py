# Author: Nathan Trouvain at 18/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pathlib

import panel as pn

from canapy.plots import plot_bokeh_label_count

from .classmerge import RepertoireView
from ..helpers import SubDash, Registry, SideBar, custom_tooltip

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

        self.registry = Registry()
        self.registry["sample"] = dict()
        self.registry["class"] = dict()
        
        self.total_corrected = 0

        self.class_selectors = self.build_class_selectors()

        self.save_btn = pn.widgets.Button(
            name="Save all", 
            sizing_mode="stretch_width", 
            button_type="primary",
            disabled=True 
        )
        self.save_btn.on_click(self.on_click_save)
        self.save_msg = pn.pane.HTML(styles=dict(color="green", background="white"))

        self.repertoire_view = RepertoireView(
            self, num_panel=1, orientation="column", num_samples=100, page_size=5,
        )

        fig, self.counts = plot_bokeh_label_count(self.controler.misclassified_segments)
        fig.min_border = 0
        bokeh_pane = pn.pane.Bokeh(fig, sizing_mode="stretch_both", max_height=350)

        self.sample_container = pn.Column(
            pn.pane.Alert("Select a class above to start correcting.", alert_type="light"),
            sizing_mode="stretch_both",
            scroll=True,
            min_height=300
        )

        repertoire_card = pn.Column(
            self.repertoire_view.layout,
            sizing_mode="stretch_width",
            css_classes=['sample-card']
        )

        self.layout = pn.Column(
            pn.Row(
                bokeh_pane,
                pn.Spacer(width=20),
                repertoire_card,
                sizing_mode="stretch_width",
                min_height=280
            ),
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
                        "border": "1px solid #dee2e6"
                    },
                    sizing_mode="stretch_width"
                ),
                pn.pane.Markdown("### 2. Correct samples", margin=(20, 0, 10, 0)),
                pn.Row(
                    self.save_msg,
                    self.save_btn,
                    sizing_mode="stretch_width",
                    align="center",
                    margin=(0, 0, 8, 0),
                ),
                sizing_mode="stretch_width"
            ),
            pn.Spacer(height=10),
            self.sample_container,
            sizing_mode="stretch_both",
        )

    def build_class_selectors(self):
        grid = pn.FlexBox(
            justify_content='start',
            gap=10,
            align_items='center'
        )
        misclass = self.controler.misclassified_segments
        for lbl in misclass.label.unique():
            n_error = len(misclass.query("label==@lbl"))
            display = ClassSelectionView(lbl, n_error, self)
            self.registry["class"][lbl] = display
            grid.append(display.layout)
        return grid

    def get_sample_corrector(self, label):
        if self.registry["sample"].get(label) is not None:
            return self.registry["sample"][label].layout
        else:
            sample_corrector = SampleCorrectorView(self, label)
            self.registry["sample"][label] = sample_corrector
            return sample_corrector.layout

    def highlight_selection(self, active_label):
        for lbl, view in self.registry["class"].items():
            if lbl == active_label:
                view.select_btn.button_type = "primary"
            else:
                view.select_btn.button_type = "default"

    def listen_class_selection(self, label):
        self.highlight_selection(label)
        
        sample_corrector_layout = self.get_sample_corrector(label)
        self.sample_container.objects = [sample_corrector_layout]

    def listen_correction(self, label, increment):
        self.registry["class"][label].receive_correction(increment)
        self.total_corrected += increment
        self.save_btn.disabled = self.total_corrected == 0

    def check_correction(self, label):
        return label in self.controler.classes if label != "" else True

    def on_click_save(self, events):
        new_corrections = {}
        for sample_corrector in self.registry["sample"].values():
            new_corrections.update(sample_corrector.corrections)
        self.controler.upload_corrections(new_corrections, "annot")
        self.save_msg.object = "Saved!"


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
    def __init__(self, parent, label):
        super().__init__(parent)

        self.misclassified_segments = self.controler.misclassified_segments.query(
            "label==@label"
        )
        self.label = label
        self.registry = Registry()
        self.layout = self.build_display()

    @property
    def corrections(self):
        return {
            i: display.correction
            for i, display in self.registry.items()
            if display.correction != ""
        }

    def build_display(self):
        segments = self.controler.load_repertoire(self.misclassified_segments)
        container = pn.Column(sizing_mode="stretch_width")
        for i, (segment, (idx, annots)) in enumerate(zip(
            segments, self.misclassified_segments.iterrows()
        )):
            display_num = i + 1
            if self.registry.get(idx) is None:
                display = SingleSampleCorrectorView(annots, segment, self.parent, display_num)
                self.registry[idx] = display
            else:
                display = self.registry[idx]
            container.append(display.layout)
        return container


class SingleSampleCorrectorView(SubDash):
    def __init__(self, repertoire_entry, spec, parent, display_num):
        super().__init__(parent)

        self.label = repertoire_entry.label
        self.repertoire_entry = repertoire_entry

        self.predictions = self.repertoire_entry.filter(regex="pred_.*")
        self.predictions.index = [p.split("_")[1] for p in self.predictions.index]

        self.text_input = pn.widgets.TextInput(placeholder="Correction", width=130)
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
            min_width=290,
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

        input_section = pn.Column(
            pn.Row(
                pn.pane.Markdown(f"**{self.label}**", styles={'font-size': '13px', 'font-weight': '600'}, margin=(0, 5, 0, 0)),
                self.label_status
            ),
            self.text_input,
            width=140,
        )

        self.layout = pn.Row(
            self.number_pane,
            file_tooltip,
            self.img,
            pn.Spacer(width=10),
            self.audio_col,
            pn.Spacer(width=20),
            input_section,
            css_classes=['sample-row-card'],
            sizing_mode="stretch_width",
            align="center",
        )

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