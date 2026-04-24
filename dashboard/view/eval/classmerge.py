# Author: Nathan Trouvain at 18/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import math
from pathlib import Path

import panel as pn
import pandas as pd

from canapy.plots import plot_bokeh_confusion_matrix

from ..helpers import SubDash, Registry, custom_tooltip

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
        self.repertoire = RepertoireView(self, num_panel=2, orientation="column")
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
                    "height": "fit-content",
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


_REPORT_TOOLTIP = (
    "<b>How to read:</b> Precision = of all predicted as X, how many were really X. "
    "Recall = of all real X, how many were found. F1 = balance between the two.<br><br>"
    "<b>Summary</b>: Accuracy = overall correct predictions. "
    "Macro avg = mean across classes (ignores class size). "
    "Weighted avg = mean weighted by number of samples per class.<br>"
    "<b>Per class</b>: same metrics broken down for each label. "
    "Colours go from red (0%) to green (100%)."
)

_SUMMARY_ROWS = {"accuracy", "macro avg", "weighted avg"}


def _make_score_table(df):
    table = pn.widgets.Tabulator(df, disabled=True, theme='site')
    table.style.pipe(_format_score_df)
    return table


def _format_score_df(styler):
    styler.format(
        {
            "recall": lambda v: f"{v:.1%}" if v == v else "—",
            "precision": lambda v: f"{v:.1%}" if v == v else "—",
            "f1-score": "{:.1%}",
            "support": "{:,.0f}",
        }
    )
    cols = [c for c in ["recall", "precision", "f1-score"] if c in styler.data.columns]
    styler.background_gradient(axis=None, vmin=0.0, vmax=1.0, cmap="RdYlGn", subset=cols)
    styler.set_properties(**{'font-size': '12px', 'text-align': 'center'})
    return styler


class MetricsView(SubDash):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = self.build_tabs()

    def build_tabs(self):
        tabs = pn.Tabs(dynamic=True, sizing_mode="stretch_width")
        for split, metrics in self.controler.metrics.items():
            sub_tabs = pn.Tabs(dynamic=True, sizing_mode="stretch_width")
            for name, cm in metrics["cm"].items():
                p = plot_bokeh_confusion_matrix(cm, self.controler.classes, title=None)
                p.min_border = 10

                heatmap_card = pn.Column(
                    pn.pane.Bokeh(p),
                    css_classes=['inner-metric-card'],
                    width=620,
                    height=540,
                )

                df = pd.DataFrame(metrics["report"][name]).T
                summary_idx = [i for i in df.index if i in _SUMMARY_ROWS]
                class_idx   = [i for i in df.index if i not in _SUMMARY_ROWS]

                _header_style = "font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.5px;display:block;margin-bottom:4px;"
                tooltip = custom_tooltip(_REPORT_TOOLTIP, direction="right", margin=(0, 0, 0, 6))

                stats_card = pn.Column(
                    pn.Row(
                        pn.Column(
                            pn.Row(
                                pn.pane.HTML(f"<span style='{_header_style}'>Summary</span>"),
                                tooltip,
                                align='center',
                                margin=(0, 0, 4, 0),
                            ),
                            _make_score_table(df.loc[summary_idx]),
                        ),
                        pn.Spacer(width=15),
                        pn.Column(
                            pn.pane.HTML(f"<span style='{_header_style}'>Per class</span>"),
                            _make_score_table(df.loc[class_idx]),
                        ),
                        align='start',
                    ),
                    css_classes=['inner-metric-card'],
                )

                sub_tabs.append(
                    (
                        name,
                        pn.Row(
                            heatmap_card,
                            pn.Spacer(width=15),
                            stats_card,
                            sizing_mode="stretch_width",
                        ),
                    ),
                )

            tabs.append((split.capitalize(), sub_tabs))
        return tabs


class RepertoireView(SubDash):
    def __init__(self, parent, num_panel, orientation, num_samples=MAX_SAMPLE_DISPLAY, page_size=None):
        super().__init__(parent)
        self.orientation = orientation
        self.num_samples = num_samples
        self.page_size = page_size
        self.registry = Registry()

        self.select_left = pn.widgets.Select(
            name="Class A",
            options=[lbl for lbl in self.controler.classes if lbl != "SIL"],
            sizing_mode="stretch_width",
        )
        self.select_left.param.watch(self.on_select_left, "value")

        _desc = pn.pane.HTML(
            "<div style='color:#6b7280; font-size:12px; margin-bottom:8px;'>"
            "Browse audio examples per class. The spectrogram shows the segment in context "
            "(dashed lines mark onset/offset). Two players per sample: "
            "<b>context window</b> (~1s around the segment), then <b>exact segment</b> only."
            "</div>"
        )

        if num_panel == 2:
            self.select_right = pn.widgets.Select(
                name="Class B",
                options=[lbl for lbl in self.controler.classes if lbl != "SIL"],
                sizing_mode="stretch_width",
            )
            self.select_right.param.watch(self.on_select_right, "value")

            self._left_col = pn.Column(self.select_left, pn.Spacer(height=5), self._placeholder(), sizing_mode="stretch_width")
            self._right_col = pn.Column(self.select_right, pn.Spacer(height=5), self._placeholder(), sizing_mode="stretch_width")
            self.layout = pn.Column(
                _desc,
                pn.Row(self._left_col, pn.Spacer(width=10), self._right_col, sizing_mode="stretch_width"),
                sizing_mode="stretch_width",
            )
        else:
            self._left_col = pn.Column(self.select_left, pn.Spacer(height=5), self._placeholder(), sizing_mode="stretch_width")
            self.layout = pn.Column(_desc, self._left_col, sizing_mode="stretch_width")

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
        if len(self.registry) > 10:
            self.registry.popitem()
        self.registry[label] = SampleView(self, label=label, orientation=self.orientation, num_samples=self.num_samples, page_size=self.page_size)
        self._left_col[2] = self.registry[label].layout

    def on_select_right(self, events):
        label = events.new
        if len(self.registry) > 10:
            self.registry.popitem()
        self.registry[label] = SampleView(self, label=label, orientation=self.orientation, num_samples=self.num_samples, page_size=self.page_size)
        self._right_col[2] = self.registry[label].layout


class SampleView(SubDash):
    def __init__(self, parent, label=None, orientation="column", num_samples=MAX_SAMPLE_DISPLAY, page_size=None):
        super().__init__(parent)
        self.orientation = orientation
        self.page_size = page_size
        selected_df = self.controler.corpus.dataset.query("label == @label")
        n = min(num_samples, len(selected_df))
        self.selected_df = selected_df.sample(n) if n > 0 else selected_df.iloc[:0]

        if page_size:
            self._page = 0
            self._n_pages = max(1, math.ceil(len(self.selected_df) / page_size))
            self._updating = False
            self._content = pn.Column(sizing_mode="stretch_width")
            self._prev_btn = pn.widgets.Button(name="<", width=32, height=30, button_type="default", margin=(0, 4))
            self._next_btn = pn.widgets.Button(name=">", width=32, height=30, button_type="default", margin=(0, 4))
            self._page_input = pn.widgets.IntInput(value=1, width=48, height=30, margin=(0, 2))
            self._total_label = pn.pane.HTML(
                f"<span style='font-size:13px;color:#374151;'>/ {self._n_pages}</span>",
                align="center", margin=(0, 6),
            )
            self._error_msg = pn.pane.HTML("", margin=(4, 0, 0, 0))
            self._prev_btn.on_click(self._on_prev)
            self._next_btn.on_click(self._on_next)
            self._page_input.param.watch(self._on_page_input, "value")
            pager = pn.Row(
                self._prev_btn, self._page_input, self._total_label, self._next_btn,
                align="center", margin=(8, 0, 4, 0),
            )
            self.layout = pn.Column(self._content, pager, self._error_msg, sizing_mode="stretch_width")
            self._render_page(0)
        else:
            self.layout = self._build_all(self.selected_df)

    def _make_cards(self, df, offset=0):
        if df.empty:
            return [pn.pane.HTML("<div style='color:#9ca3af;font-size:13px;padding:20px;text-align:center;'>No samples available.</div>")]
        specs = self.controler.load_repertoire(df)
        sampling_rate = self.controler.config.transforms.audio.sampling_rate
        views = []
        for i, sp in enumerate(specs):
            visual_block = pn.Row(
                pn.pane.Markdown(f"**#{offset+i+1}**", styles={'font-size': '11px', 'color': '#6b7280'}, width=30, align='center'),
                pn.pane.Matplotlib(sp[0], format="png", tight=True, sizing_mode="stretch_width", height=100),
                sizing_mode="stretch_width",
                styles={"min-width": "120px", "max-width": "320px"},
                margin=(0, 10, 0, 0),
            )
            if offset == 0 and i == 0:
                audio_block = pn.Column(
                    pn.pane.HTML("<span style='font-size:10px;color:#6b7280;'>🔊 Context window (~1s around segment)</span>"),
                    pn.pane.Audio(sp[1], sample_rate=round(sampling_rate), height=35, sizing_mode="stretch_width"),
                    pn.Spacer(height=3),
                    pn.pane.HTML("<span style='font-size:10px;color:#6b7280;'>🎯 Exact segment only</span>"),
                    pn.pane.Audio(sp[2], sample_rate=round(sampling_rate), height=35, sizing_mode="stretch_width"),
                    sizing_mode="stretch_width",
                )
            else:
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
        return views

    def _build_all(self, df):
        views = self._make_cards(df)
        layout_cls = pn.Column if self.orientation == "column" else pn.Row
        return layout_cls(*views, sizing_mode="stretch_width")

    def _render_page(self, page):
        self._page = page
        start = page * self.page_size
        batch = self.selected_df.iloc[start:start + self.page_size]
        self._content.objects = self._make_cards(batch, offset=start)
        self._updating = True
        try:
            self._page_input.value = page + 1
        finally:
            self._updating = False

    def _on_prev(self, event):
        self._error_msg.object = ""
        self._render_page((self._page - 1) % self._n_pages)

    def _on_next(self, event):
        self._error_msg.object = ""
        self._render_page((self._page + 1) % self._n_pages)

    def _on_page_input(self, event):
        if self._updating:
            return
        val = event.new
        if val is None:
            return
        if 1 <= val <= self._n_pages:
            self._error_msg.object = ""
            self._render_page(val - 1)
        else:
            self._error_msg.object = (
                f"<span style='color:#ef4444;font-size:11px;'>"
                f"Please enter a page number between 1 and {self._n_pages}.</span>"
            )


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