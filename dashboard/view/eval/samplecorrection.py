# Author: Nathan Trouvain at 18/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pathlib

import panel as pn

from canapy.plots import plot_bokeh_label_count

from .classmerge import RepertoireView
from ..helpers import SubDash, Registry


class SampleCorrectionDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        self.registry = Registry()
        self.registry["sample"] = dict()
        self.registry["class"] = dict()

        self.class_selectors = self.build_class_selectors()

        self.save_btn = pn.widgets.Button(
            name="Save all", width=500, button_type="primary"
        )
        self.save_btn.on_click(self.on_click_save)
        self.save_msg = pn.pane.HTML(styles=dict(color="green", background="white"))

        self.repertoire_view = RepertoireView(
            self, num_panel=1, orientation="row", num_samples=4
        )

        fig, self.counts = plot_bokeh_label_count(self.controler.misclassified_segments)

        self.layout = pn.Column(
            pn.Row(
                pn.pane.Bokeh(fig),
                pn.Column(self.repertoire_view.layout, self.save_msg, self.save_btn),
            ),
            self.class_selectors,
            pn.Spacer(),
        )

    @property
    def save_txt(self):
        return self.layout[0][1][1].object

    @save_txt.setter
    def save_txt(self, value):
        self.layout[0][1][1].object = value

    def build_class_selectors(self):
        grid = pn.GridBox(ncols=15)
        misclass = self.controler.misclassified_segments
        for lbl in misclass.label.unique():
            # n_error = self.controler.misclass_counts.loc[lbl]["counts"]
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

    def listen_class_selection(self, label):
        sample_corrector_layout = self.get_sample_corrector(label)
        self.layout[2] = sample_corrector_layout

    def listen_correction(self, label, increment):
        self.registry["class"][label].receive_correction(increment)

    def check_correction(self, label):
        return label in self.controler.classes if label != "" else True

    def on_click_save(self, events):
        new_corrections = {}
        for sample_corrector in self.registry["sample"].values():
            new_corrections.update(sample_corrector.corrections)
        self.controler.upload_corrections(new_corrections, "annot")
        self.save_txt = "Saved!"


class ClassSelectionView(SubDash):
    def __init__(self, label, num_error, parent):
        super().__init__(parent)

        self.label = label
        self.num_error = num_error
        self.num_corrected = 0

        self.select_btn = pn.widgets.Button(
            name=label, button_type="primary", width=100
        )
        self.select_btn.on_click(self.on_click_notify_display)
        self.corrected_msg = pn.pane.HTML(
            object=f"{self.num_corrected}/{num_error} corrected",
            styles=dict(background="white"),
        )

        self.layout = pn.Column(self.select_btn, self.corrected_msg)

    def on_click_notify_display(self, events):
        self.parent.listen_class_selection(self.label)

    def receive_correction(self, increment):
        self.num_corrected += increment

        style = {"color": "black", "background": "white"}
        if self.num_corrected == self.num_error:
            style = {"color": "green", "background": "white"}

        self.layout[1] = pn.pane.HTML(
            object=f"{self.num_corrected}/{self.num_error} corrected",
            styles=style,
        )


class SampleCorrectorView(SubDash):
    def __init__(self, parent, label):
        super().__init__(parent)

        self.misclassified_segments = self.controler.misclassified_segments.query(
            "label==@label"
        )

        self.label = label
        self.registry = Registry()
        self.layout = pn.Column(self.build_display(), scroll=True, height=400)

    @property
    def corrections(self):
        return {
            i: display.correction
            for i, display in self.registry.items()
            if display.correction != ""
        }

    def build_display(self):
        segments = self.controler.load_repertoire(self.misclassified_segments)
        grid = pn.GridBox(ncols=4)
        for segment, (idx, annots) in zip(
            segments, self.misclassified_segments.iterrows()
        ):
            if self.registry.get(idx) is None:
                display = SingleSampleCorrectorView(annots, segment, self.parent)
                self.registry[idx] = display
            else:
                display = self.registry[idx]
            grid.append(display.layout)
        return grid


class SingleSampleCorrectorView(SubDash):
    def __init__(self, repertoire_entry, spec, parent):
        super().__init__(parent)

        self.label = repertoire_entry.label
        self.repertoire_entry = repertoire_entry

        self.predictions = self.repertoire_entry.filter(regex="pred_.*")
        # Retrieve model names as index
        self.predictions.index = [p.split("_")[1] for p in self.predictions.index]

        self.text_input = pn.widgets.TextInput(width=75)
        self.corrected = False
        self.text_input.param.watch(self.on_correction_notify, "value")

        sampling_rate = round(self.controler.config.transforms.audio.sampling_rate)

        self.img = pn.pane.Matplotlib(spec[0])
        self.audio = pn.pane.Audio(spec[1], sample_rate=sampling_rate)
        self.short_audio = pn.pane.Audio(spec[2], sample_rate=sampling_rate)

        models_preds = ", ".join([f"{idx}: {p}" for idx, p in self.predictions.items()])

        notated_file = pathlib.Path(self.repertoire_entry.notated_path).stem

        file_tooltip = pn.widgets.TooltipIcon(
            value=f"From audio: {notated_file}"
                  f"\n start: {self.repertoire_entry.onset_s:.3f} s, "
                  f"end: {self.repertoire_entry.offset_s:.3f} s"
                  f"\n duration: {self.repertoire_entry.offset_s - self.repertoire_entry.onset_s:.3f} s"
            )

        self.infos = pn.pane.HTML(
            f"""<p>Models predictions:</p>
                <p>{models_preds}</p>
            """
        )

        self.label_title = pn.pane.HTML(
            f"{self.label}", styles={"background": "WhiteSmoke", "font-size": "2em"}
        )

        self.label_status = pn.pane.HTML(
            styles={"background": "WhiteSmoke", "color": "orange"}
        )

        self.layout = pn.Row(
            pn.Column(file_tooltip, self.infos, self.img, self.audio, self.short_audio),
            pn.Column(
                self.label_title,
                pn.pane.HTML("Correction :"),
                self.text_input,
                self.label_status,
            ),
            styles=dict(background="WhiteSmoke"),
            margin=10,
        )

    @property
    def correction(self):
        return self.layout[1][2].value

    @property
    def text_value(self):
        return self.layout[1][1].object

    @text_value.setter
    def text_value(self, value):
        self.layout[1][1].object = value

    @property
    def text_style(self):
        return self.layout[1][1].style

    @text_style.setter
    def text_style(self, value):
        self.layout[1][1].style = value

    @property
    def validation_value(self):
        return self.layout[1][2].object

    @validation_value.setter
    def validation_value(self, value):
        self.layout[1][2].object = value

    def on_correction_notify(self, events):
        validation = self.parent.check_correction(self.text_input.value)
        if not (validation):
            self.validation_value = """<p>Not found in repertory.</p>
                                        <p>Will create a new class.</p>"""
        else:
            self.validation_value = ""

        if self.text_input.value == "":
            self.corrected = False
            self.text_value = "Correction :"
            self.text_style = {"color": "black"}
            self.parent.listen_correction(self.label, increment=-1)
        else:
            if not (self.corrected):
                self.corrected = True
                self.text_value = "Corrected"
                self.text_style = {"color": "green"}
                self.parent.listen_correction(self.label, increment=1)
            else:
                pass
