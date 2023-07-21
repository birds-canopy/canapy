# Author: Nathan Trouvain at 18/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from pathlib import Path

import panel as pn
import pandas as pd

from canapy.plots import plot_bokeh_confusion_matrix

from ..helpers import SubDash
from ..helpers import Registry

pn.extension("tabulator")

MAX_SAMPLE_DISPLAY = 10


class ClassMergeDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        self.metrics = MetricsView(self)
        self.repertoire = RepertoireView(self, num_panel=2, orientation="column")
        self.corrector = CorrectorView(self)

        self.layout = pn.Row(
                self.metrics.layout,
                self.corrector.layout,
                self.repertoire.layout,
            )


def format_score_df(styler):
    styler.format(
        {
            "recall": "{:.2%}",
            "precision": "{:.2%}",
            "f1-score": "{:.2%}",
            "support": "{:,d}",
        }
    )
    styler.background_gradient(
        axis=None,
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
        subset=["recall", "precision", "f1-score"],
    )
    return styler


class MetricsView(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        self.layout = self.build_tabs()

    def build_tabs(self):
        tabs = pn.Tabs()
        for split, metrics in self.controler.metrics.items():
            sub_tabs = pn.Tabs()
            for name, cm in metrics["cm"].items():
                p = plot_bokeh_confusion_matrix(cm, self.controler.classes, title=name)
                fig_pane = pn.pane.Bokeh(p, sizing_mode="stretch_both")

                df = pd.DataFrame(metrics["report"][name]).T
                score_table = pn.widgets.Tabulator(df, disabled=True)
                score_table.style.pipe(format_score_df)

                sub_tabs.append(
                    (
                        name,
                        pn.Column(
                            pn.Column(fig_pane, width=600, height=600),
                            pn.Column(score_table, width=600),
                            sizing_mode="stretch_height",
                        ),
                    ),
                )

            tabs.append((split, sub_tabs))
        return pn.Row(tabs, sizing_mode="stretch_both")


class RepertoireView(SubDash):
    def __init__(self, parent, num_panel, orientation, num_samples=MAX_SAMPLE_DISPLAY):
        super().__init__(parent)

        self.orientation = orientation
        self.num_samples = num_samples

        self.select_left = pn.widgets.Select(
            options=[lbl for lbl in self.controler.classes if lbl != "SIL"],
            max_width=100,
        )
        self.sample_left = SampleView(self, label=self.select_left.value, orientation=self.orientation)
        self.select_left.param.watch(self.on_select_left, "value")
        self.registry = Registry()

        if num_panel == 2:
            self.select_right = pn.widgets.Select(
                options=[lbl for lbl in self.controler.classes if lbl != "SIL"],
                max_width=100,
            )
            self.sample_right = SampleView(self, label=self.select_right.value)
            self.select_right.param.watch(self.on_select_right, "value")

            self.layout = pn.Row(
                pn.Column(self.select_left, self.sample_left),
                pn.Column(self.select_right, self.sample_right),
                width=250,
                sizing_mode="stretch_height",
            )
        else:
            self.layout = pn.Row(
                pn.Column(
                    self.select_left, self.sample_left
                ),
                width=250,
                sizing_mode="stretch_height",
            )

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
        selected_df = self.controler.corpus.dataset.query("label == @label")
        selected_df = selected_df.iloc[: self.num_samples]
        specs = self.controler.load_repertoire(selected_df)

        sampling_rate = self.controler.config.transforms.audio.sampling_rate

        views = []
        for sp in specs:
            img = pn.pane.Matplotlib(
                sp[0], format="png", tight=True
            )
            audio = pn.pane.Audio(sp[1], sample_rate=round(sampling_rate), width=50)
            audio_short = pn.pane.Audio(
                sp[2], sample_rate=round(sampling_rate), width=300
            )
            views.append(
                pn.Column(img, audio, audio_short, max_height=300, max_width=300)
            )

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
        self.grid = pn.GridBox(ncols=2)
        for l in self.controler.classes:
            if l != self.controler.config.transforms.annots.silence_tag:
                self.grid.append(pn.widgets.TextInput(name=l, width=75))

        self.save_btn = pn.widgets.Button(
            name="Save corrections", button_type="primary"
        )
        self.save_btn.on_click(self.on_click_save)
        self.save_msg = pn.pane.HTML(styles=dict(background="WhiteSmoke"))

        return pn.Column(
            self.grid,
            self.save_btn,
            self.save_msg,
            width=200,
        )

    def on_click_save(self, events):
        new_corrections = {}
        for text in self.grid:
            if text.value != "":
                new_corrections[text.name] = text.value
        self.controler.upload_corrections(new_corrections, "class")
        self.layout[2].object = "Saved!"
