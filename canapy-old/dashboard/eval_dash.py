from pathlib import Path

import panel as pn
import pandas as pd

from .helpers import SubDash
from .helpers import SideBar
from .helpers import Registry

MAX_SAMPLE_DISPLAY = 10


class EvalDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        self.sidebar = SideBar(self, "eval")
        self.merge_dashboard = ClassMergeDashboard(self)
        self.sample_dashboard = SampleCorrectionDashboard(self)

        self.pane_selection = pn.widgets.RadioButtonGroup(
            name="Pane Selection", options=["Class merge", "Sample correction"]
        )
        self.pane_selection.param.watch(self.on_switch_panel, "value")

        self.layout = pn.Row(
            self.sidebar.layout,
            pn.Column(self.pane_selection, self.merge_dashboard.layout),
        )

    def on_switch_panel(self, events):
        if self.pane_selection.value == "Class merge":
            self.layout[1][1] = self.merge_dashboard.layout
        else:
            self.layout[1][1] = self.sample_dashboard.layout


class SampleCorrectionDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        self.registry = Registry()
        self.registry["sample"] = dict()
        self.registry["class"] = dict()

        self.class_selectors = self.build_class_selectors()

        self.save_btn = pn.widgets.Button(
            name="Save all", width=500, button_type="success"
        )
        self.save_btn.on_click(self.on_click_save)
        self.save_msg = pn.pane.HTML(style={"color": "green"}, background="white")

        self.repertoire_view = RepertoireView(
            self, num_panel=1, orientation="row", num_samples=4
        )

        self.layout = pn.Column(
            pn.Row(
                pn.pane.Bokeh(self.controler.misclass_plot),
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
        for lbl in self.controler.misclass_labels:
            if lbl != "SIL":
                num_error = self.controler.misclass_counts.loc[lbl]["counts"]
                display = ClassSelectionView(lbl, num_error, self)
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
        return label in self.controler.dataset.vocab if label != "" else True

    def on_click_save(self, events):
        new_corrections = {}
        for sample_corrector in self.registry["sample"].values():
            new_corrections.update(sample_corrector.corrections)
        self.controler.new_corrections["sample"].update(new_corrections)
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
            object=f"{self.num_corrected}/{num_error} corrected", background="white"
        )

        self.layout = pn.Column(self.select_btn, self.corrected_msg)

    def on_click_notify_display(self, events):
        self.parent.listen_class_selection(self.label)

    def receive_correction(self, increment):
        self.num_corrected += increment

        style = {"color": "black"}
        if self.num_corrected == self.num_error:
            style = {"color": "green"}

        self.layout[1] = pn.pane.HTML(
            object=f"{self.num_corrected}/{self.num_error} corrected",
            background="white",
            style=style,
        )


class SampleCorrectorView(SubDash):
    def __init__(self, parent, label):

        super().__init__(parent)

        self.misclass_samples = self.controler.misclass_df[
            self.controler.misclass_df["syll"] == label
        ]
        self.misclass_preds = {
            k: v
            for k, v in self.controler.misclass_indexes.items()
            if k in self.misclass_samples.index.values.tolist()
        }

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
        specs = self.controler.load_repertoire(self.misclass_samples)
        grid = pn.GridBox(ncols=4)
        for sp, rep in zip(specs, self.misclass_samples.itertuples()):
            pred = self.misclass_preds[rep.Index]
            if self.registry.get(rep.Index) is None:
                display = SingleSampleCorrectorView(rep, pred, sp, self.parent)
                self.registry[rep.Index] = display
            else:
                display = self.registry[rep.Index]
            grid.append(display.layout)
        return grid


class SingleSampleCorrectorView(SubDash):
    def __init__(self, repertoire_entry, predictions, spec, parent):
        super().__init__(parent)

        self.label = repertoire_entry.syll
        self.repertoire_entry = repertoire_entry
        self.predictions = predictions

        self.text_input = pn.widgets.TextInput(max_width=100)
        self.corrected = False
        self.text_input.param.watch(self.on_correction_notify, "value")

        self.img = pn.pane.Matplotlib(spec[0])
        self.audio = pn.pane.Audio(spec[1], sample_rate=round(spec[3]))
        self.short_audio = pn.pane.Audio(spec[2], sample_rate=round(spec[3]))

        self.infos = pn.pane.HTML(
            f"""<p>{self.repertoire_entry.wave}</p>
                <p>start: {self.repertoire_entry.start:.2f} sec, end: {self.repertoire_entry.end:.2f} sec</p>
                <p>Models predictions: </p>
                <p>syn : {self.predictions[0]}, nsyn : {self.predictions[1]}, ensemble : {self.predictions[2]}</p>
            """
        )

        self.label_title = pn.pane.HTML(
            f"{self.label}", background="WhiteSmoke", style={"font-size": "2em"}
        )

        self.label_status = pn.pane.HTML(
            background="WhiteSmoke", style={"color": "orange"}
        )

        self.layout = pn.Row(
            pn.Column(self.infos, self.img, self.audio, self.short_audio),
            pn.Column(
                self.label_title,
                pn.pane.HTML("Correction :"),
                self.text_input,
                self.label_status,
            ),
            background="WhiteSmoke",
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


class ClassMergeDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        self.metrics = MetricsView(self)
        self.repertoire = RepertoireView(self, num_panel=2, orientation="column")
        self.corrector = CorrectorView(self)

        self.layout = pn.Row(
            self.metrics.layout, self.repertoire.layout, self.corrector.layout
        )


class MetricsView(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        self.layout = self.build_tabs()

    def build_tabs(self):

        tabs = pn.Tabs()
        for model, cm in self.controler.cms.items():
            p = self.controler.confusion_matrix(model)
            fig_pane = pn.pane.Bokeh(p)
            df = pd.DataFrame(self.controler.reports[model]).T
            format_dict = {
                "recall": "{:.2%}",
                "precision": "{:.2%}",
                "f1-score": "{:.2%}",
                "support": "{0:,.0f}",
            }
            df = df.style.format(format_dict)
            metrics = pn.pane.HTML(df.render())
            tabs.append((model, pn.Column(fig_pane, metrics)))

        return tabs


class RepertoireView(SubDash):
    def __init__(self, parent, num_panel, orientation, num_samples=MAX_SAMPLE_DISPLAY):
        super().__init__(parent)

        self.orientation = orientation
        self.num_samples = num_samples

        self.select_left = pn.widgets.Select(
            options=[lbl for lbl in self.controler.dataset.vocab if lbl != "SIL"],
            max_width=100,
        )
        self.sample_left = SampleView(self, label=self.select_left.value)
        self.select_left.param.watch(self.on_select_left, "value")
        self.registry = Registry()

        if num_panel == 2:
            self.select_right = pn.widgets.Select(
                options=[lbl for lbl in self.controler.dataset.vocab if lbl != "SIL"],
                max_width=100,
            )
            self.sample_right = SampleView(self, label=self.select_right.value)
            self.select_right.param.watch(self.on_select_right, "value")

            self.layout = pn.Row(
                pn.Column(self.select_left, self.sample_left),
                pn.Column(self.select_right, self.sample_right),
            )
        else:
            self.layout = pn.Row(pn.Column(self.select_left, self.sample_left))

    def on_select_left(self, events):
        label = self.select_left.value
        if self.registry.get(label) is None:
            if len(self.registry) > 10:
                self.registry.popitem()
            sample_view = SampleView(
                self,
                label=label,
                orientation=self.orientation,
                num_samples=self.num_samples,
            )
            self.registry[label] = sample_view
        self.layout[0][1] = self.registry[label].layout

    def on_select_right(self, events):
        label = self.select_right.value
        if self.registry.get(label) is None:
            if len(self.registry) > 10:
                self.registry.popitem()
            sample_view = SampleView(
                self,
                label=label,
                orientation=self.orientation,
                num_samples=self.num_samples,
            )
            self.registry[label] = sample_view
        self.layout[1][1] = self.registry[label].layout


class SampleView(SubDash):
    def __init__(
        self, parent, label=None, orientation="column", num_samples=MAX_SAMPLE_DISPLAY
    ):
        super().__init__(parent)

        self.num_samples = num_samples
        self.orientation = orientation
        self.layout = self.build_display(label)

    def build_display(self, label):

        selected_df = self.controler.dataset.df[
            self.controler.dataset.df["syll"] == label
        ]
        selected_df = selected_df.iloc[: self.num_samples]

        specs = self.controler.load_repertoire(selected_df)

        views = []
        for sp in specs:
            img = pn.pane.Matplotlib(sp[0], align="center")
            audio = pn.pane.Audio(sp[1], sample_rate=round(sp[3]), width=100)
            audio_short = pn.pane.Audio(sp[2], sample_rate=round(sp[3]), width=100)
            views.append(pn.Column(img, audio, audio_short))

        if self.orientation == "column":
            layout = pn.Column(*views)
        else:
            layout = pn.Row(*views)

        return layout


class CorrectorView(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        self.layout = self.build_display()

    def build_display(self):

        self.grid = pn.GridBox(ncols=3)
        for l in self.controler.dataset.vocab:
            # if self.controler.dataset.corrections["syll"].get(l) is not None:
            #    self.grid.append(pn.widgets.TextInput(value=self.controler.dataset.corrections['syll'].get(l),
            #                                          name=l,
            #                                          max_width=100))
            # else:
            self.grid.append(pn.widgets.TextInput(name=l, max_width=100))

        self.save_btn = pn.widgets.Button(
            name="Save corrections", button_type="primary"
        )
        self.save_btn.on_click(self.on_click_save)
        self.save_msg = pn.pane.HTML(background="white")

        return pn.Column(self.grid, self.save_btn, self.save_msg)

    def on_click_save(self, events):
        new_corrections = {}
        for text in self.grid:
            if text.value != "":
                new_corrections[text.name] = text.value
        self.controler.new_corrections["syll"].update(new_corrections)
        self.layout[2].object = "Saved!"
