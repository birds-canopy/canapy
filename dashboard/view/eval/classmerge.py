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

MERGE_CSS = """
:host {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.dashboard-col {
    background-color: #ffffff;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    padding: 15px;
    border: 1px solid #e5e7eb;
    height: 100%;           
    display: flex;          
    flex-direction: column; 
    overflow: hidden;       
    box-sizing: border-box;
}
.inner-metric-card {
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 10px;
    height: 100%;
    overflow: hidden;
    display: flex;
    flex-direction: column;
}
.col-header {
    font-size: 16px;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 15px;
    border-bottom: 2px solid #f3f4f6;
    padding-bottom: 8px;
    text-transform: uppercase;
    flex-shrink: 0; 
}
.scrollable-content {
    overflow-y: auto;
    overflow-x: hidden;
    flex-grow: 1;
    padding-right: 5px;
    padding-bottom: 20px; 
}
.sample-card {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    margin-bottom: 15px;
}
.corrector-input input {
    text-align: center;
    font-size: 13px;
    color: #374151;
    font-weight: 500;
}
"""

class ClassMergeDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)
        
        pn.config.raw_css.append(MERGE_CSS)

        self.metrics = MetricsView(self)
        self.repertoire = RepertoireView(self, num_panel=2, orientation="column", num_samples=100)
        self.corrector = CorrectorView(self)
        
        self.layout = pn.Column(
            pn.Column(
                pn.pane.HTML("<div class='col-header'>1. Evaluation Metrics</div>"),
                self.metrics.layout,
                styles={
                    "background-color": "#ffffff",
                    "border-radius": "8px",
                    "box-shadow": "0 1px 3px rgba(0,0,0,0.1)",
                    "padding": "15px",
                    "border": "1px solid #e5e7eb",
                    "box-sizing": "border-box",
                },
                sizing_mode="stretch_width",
            ),
            pn.Spacer(height=15),
            pn.Row(
                pn.Column(
                    pn.pane.HTML("<div class='col-header' style='border-bottom: none;'>2. Class Correction</div>"),
                    self.corrector.layout,
                    css_classes=['dashboard-col'],
                    styles={"min-width": "300px", "max-width": "420px"},
                    sizing_mode="stretch_height",
                ),
                pn.Spacer(width=15),
                pn.Column(
                    pn.pane.HTML("<div class='col-header'>3. Audio Repertoire</div>"),
                    pn.Column(self.repertoire.layout, css_classes=['scrollable-content'], sizing_mode="stretch_both"),
                    css_classes=['dashboard-col'],
                    sizing_mode="stretch_both",
                ),
                sizing_mode="stretch_both",
                min_height=500,
            ),
            sizing_mode="stretch_both",
        )


def format_score_df(styler):
    styler.format(
        {
            "recall": "{:.1%}",
            "precision": "{:.1%}",
            "f1-score": "{:.1%}",
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
    styler.set_properties(**{'font-size': '12px', 'text-align': 'center'})
    return styler


class MetricsView(SubDash):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = self.build_tabs()

    def build_tabs(self):
        tabs = pn.Tabs(dynamic=True, sizing_mode="stretch_width")
        for split, metrics in self.controler.metrics.items():
            sub_tabs = pn.Tabs(dynamic=True)
            for name, cm in metrics["cm"].items():
                
                p = plot_bokeh_confusion_matrix(cm, self.controler.classes, title=None)
                p.sizing_mode = "stretch_both"
                p.min_border = 10
                
                fig_pane = pn.pane.Bokeh(
                    p, 
                    sizing_mode="stretch_both",
                    max_width=650,
                    max_height=650
                )
                
                heatmap_card = pn.Column(
                    fig_pane,
                    css_classes=['inner-metric-card'],
                    sizing_mode="stretch_both",
                    max_width=680,
                    max_height=680
                )

                df = pd.DataFrame(metrics["report"][name]).T
                score_table = pn.widgets.Tabulator(
                    df, 
                    disabled=True, 
                    theme='site', 
                    sizing_mode="stretch_both",
                    max_width=650,
                    max_height=650
                )
                score_table.style.pipe(format_score_df)

                stats_card = pn.Column(
                    score_table,
                    css_classes=['inner-metric-card'],
                    sizing_mode="stretch_both",
                    max_width=680,
                    max_height=680
                )

                sub_tabs.append(
                    (
                        name,
                        pn.Row(
                            heatmap_card,
                            pn.Spacer(width=15),
                            stats_card,
                            sizing_mode="stretch_width",
                            min_height=500,
                            max_height=700,
                        ),
                    ),
                )

            tabs.append((split.capitalize(), sub_tabs))
        return tabs


class RepertoireView(SubDash):
    def __init__(self, parent, num_panel, orientation, num_samples=MAX_SAMPLE_DISPLAY):
        super().__init__(parent)
        self.orientation = orientation
        self.num_samples = num_samples
        self.registry = Registry()

        self.select_left = pn.widgets.Select(
            name="Class A",
            options=[lbl for lbl in self.controler.classes if lbl != "SIL"],
            sizing_mode="stretch_width",
        )
        self.select_left.param.watch(self.on_select_left, "value")

        if num_panel == 2:
            self.select_right = pn.widgets.Select(
                name="Class B",
                options=[lbl for lbl in self.controler.classes if lbl != "SIL"],
                sizing_mode="stretch_width",
            )
            self.select_right.param.watch(self.on_select_right, "value")

            self._left_col = pn.Column(self.select_left, pn.Spacer(height=5), self._placeholder(), sizing_mode="stretch_width")
            self._right_col = pn.Column(self.select_right, pn.Spacer(height=5), self._placeholder(), sizing_mode="stretch_width")
            self.layout = pn.Row(
                self._left_col,
                pn.Spacer(width=10),
                self._right_col,
                sizing_mode="stretch_width",
            )
        else:
            self._left_col = pn.Column(self.select_left, pn.Spacer(height=5), self._placeholder(), sizing_mode="stretch_width")
            self.layout = self._left_col

    @staticmethod
    def _placeholder():
        return pn.pane.HTML(
            "<div style='color:#9ca3af;font-size:13px;padding:20px;text-align:center;'>"
            "Select a class above to load samples."
            "</div>"
        )

    def update_classes(self):
        new_classes = [lbl for lbl in self.controler.classes if lbl != "SIL"]
        self.registry.clean()
        self.controler._repertoire_cache.clear()
        self.select_left.options = new_classes
        self._left_col[2] = self._placeholder()
        if hasattr(self, 'select_right'):
            self.select_right.options = new_classes
            self._right_col[2] = self._placeholder()

    def on_select_left(self, events):
        label = events.new
        if self.registry.get(label) is None:
            if len(self.registry) > 10: self.registry.popitem()
            self.registry[label] = SampleView(self, label=label, orientation=self.orientation, num_samples=self.num_samples)
        self._left_col[2] = self.registry[label].layout

    def on_select_right(self, events):
        label = events.new
        if self.registry.get(label) is None:
            if len(self.registry) > 10: self.registry.popitem()
            self.registry[label] = SampleView(self, label=label, orientation=self.orientation, num_samples=self.num_samples)
        self._right_col[2] = self.registry[label].layout


class SampleView(SubDash):
    PAGE_SIZE = 4
    WINDOW_SIZE = 20

    def __init__(self, parent, label=None, orientation="column", num_samples=MAX_SAMPLE_DISPLAY):
        super().__init__(parent)
        self.orientation = orientation
        selected_df = self.controler.corpus.dataset.query("label == @label")
        self.selected_df = selected_df.iloc[:num_samples]
        n = len(self.selected_df)
        self._n_pages = max(1, -(-n // self.PAGE_SIZE))
        self._updating_pager = False

        self._content = pn.Column(sizing_mode="stretch_width")
        self._pager = pn.widgets.RadioButtonGroup(
            options=self._window(1),
            value=1,
            button_style="outline",
            button_type="default",
            visible=self._n_pages > 1,
        )
        self._pager.param.watch(self._on_page_change, "value")
        self.layout = pn.Column(
            self._content,
            pn.Row(pn.Spacer(), self._pager, pn.Spacer(), sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )
        self._render_page(0)

    def _window(self, current_page):
        if self._n_pages <= self.WINDOW_SIZE:
            return list(range(1, self._n_pages + 1))
        half = self.WINDOW_SIZE // 2
        start = max(1, current_page - half)
        end = min(self._n_pages, start + self.WINDOW_SIZE - 1)
        start = max(1, end - self.WINDOW_SIZE + 1)
        return list(range(start, end + 1))

    def _render_page(self, page_index):
        start = page_index * self.PAGE_SIZE
        batch = self.selected_df.iloc[start:start + self.PAGE_SIZE]
        if batch.empty:
            return
        specs = self.controler.load_repertoire(batch)
        sampling_rate = self.controler.config.transforms.audio.sampling_rate
        views = []
        for i, sp in enumerate(specs):
            display_num = start + i + 1
            visual_block = pn.Row(
                pn.pane.Markdown(f"**#{display_num}**", styles={'font-size': '11px', 'color': '#6b7280'}, width=30, align='center'),
                pn.pane.Matplotlib(sp[0], format="png", tight=True, sizing_mode="stretch_width", height=50),
                sizing_mode="stretch_width",
                styles={"min-width": "120px", "max-width": "320px"},
                margin=(0, 10, 0, 0),
            )
            audio_block = pn.Column(
                pn.pane.Audio(sp[1], sample_rate=round(sampling_rate), height=35, sizing_mode="stretch_width"),
                pn.Spacer(height=5),
                pn.pane.Audio(sp[2], sample_rate=round(sampling_rate), height=35, sizing_mode="stretch_width"),
                sizing_mode="stretch_width",
            )
            card_content = pn.FlexBox(
                visual_block, audio_block,
                align_items='center', justify_content='start',
                flex_wrap='wrap', gap=10, sizing_mode="stretch_width",
            )
            views.append(pn.Column(card_content, css_classes=['sample-card'], padding=20, sizing_mode="stretch_width"))
        self._content.objects = views

    def _on_page_change(self, event):
        if self._updating_pager:
            return
        page = event.new
        self._render_page(page - 1)
        if self._n_pages > self.WINDOW_SIZE:
            new_opts = self._window(page)
            if new_opts != self._pager.options:
                self._updating_pager = True
                try:
                    self._pager.param.update(options=new_opts, value=page)
                finally:
                    self._updating_pager = False


class CorrectorView(SubDash):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = self.build_display()

    def build_display(self):
        info = pn.pane.Markdown("Rename classes to merge them.", styles={'font-size': '13px', 'color': '#6b7280'})
        self.grid = pn.FlexBox(justify_content='center', gap=10)
        
        for l in self.controler.classes:
            if l != self.controler.config.transforms.annots.silence_tag:
                self.grid.append(pn.widgets.TextInput(name=l, placeholder=l, width=120, css_classes=['corrector-input']))

        self.save_btn = pn.widgets.Button(name="Apply", button_type="primary", sizing_mode="stretch_width")
        self.save_btn.on_click(self.on_click_save)
        self.save_msg = pn.pane.Alert("", alert_type="success", visible=False)

        return pn.Column(
            info, 
            pn.Column(
                self.grid, 
                scroll=True, 
                sizing_mode="stretch_both",
            ),
            pn.Column(
                self.save_msg, 
                self.save_btn,
                sizing_mode="stretch_width",
                margin=(10, 0, 0, 0)
            ),
            sizing_mode="stretch_both"
        )

    def rebuild_grid(self):
        silence_tag = self.controler.config.transforms.annots.silence_tag
        self.grid.objects = [
            pn.widgets.TextInput(name=l, placeholder=l, width=120, css_classes=['corrector-input'])
            for l in self.controler.classes
            if l != silence_tag
        ]

    def on_click_save(self, events):
        new_corrections = {text.name: text.value for text in self.grid if isinstance(text, pn.widgets.TextInput) and text.value != ""}
        if new_corrections:
            self.controler.apply_live_corrections(new_corrections, "class")
            self.rebuild_grid()
            self.parent.repertoire.update_classes()
            self.save_msg.object = "Applied!"; self.save_msg.visible = True