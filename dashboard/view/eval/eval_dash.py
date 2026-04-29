from pathlib import Path
import threading
import logging

import panel as pn
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from ..helpers import SubDash, SideBar, Registry, custom_tooltip

logger = logging.getLogger("canapy")

from .classmerge import ClassMergeDashboard
from .samplecorrection import SampleCorrectionDashboard

MAX_SAMPLE_DISPLAY = 10

# CSS Global (similaire à LoadData)
EVAL_CSS = """
.page-title {
    font-size: 32px;
    font-weight: 700;
    color: #212529;
    margin-bottom: 5px;
}
.page-subtitle {
    font-size: 18px;
    color: #868e96;
    margin-bottom: 30px;
}
.selector-bar {
    background: white;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}
"""

class EvalDashboard(SubDash):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        pn.config.raw_css.append(EVAL_CSS)

        self.sidebar = SideBar(self, "eval")
        self.merge_dashboard = ClassMergeDashboard(self)
        self.sample_dashboard = SampleCorrectionDashboard(self)

        _EXPORT_CONFIG_CSS = """
            button.bk-btn {
                background-color: #dcfce7 !important;
                color: #166534 !important;
                border: 1px solid #86efac !important;
                font-weight: bold !important;
                border-radius: 6px !important;
                font-size: 13px !important;
            }
            button.bk-btn:hover {
                background-color: #bbf7d0 !important;
            }
        """
        self._export_config_btn = pn.widgets.Button(
            name="Export Config",
            width=160,
            align="end",
            stylesheets=[_EXPORT_CONFIG_CSS],
        )
        self._export_config_tooltip = custom_tooltip(
            "Exports the current configuration (ESN hyperparameters, audio parameters, etc.)"
            " as a TOML file in <code style='background:#374151;padding:1px 4px;border-radius:3px;'>config/user/</code>."
            " You can reload it next time via Settings.",
            direction="left",
            align="end",
            margin=(0, 0),
        )
        self._export_config_msg = pn.pane.Markdown(
            "", styles={"font-size": "12px", "color": "#166534"}, visible=False, align="end"
        )
        self._export_config_btn.on_click(self._on_click_export_config)

        self.header = pn.Row(
            pn.Column(
                pn.pane.Markdown("# Model Evaluation", css_classes=['page-title'], margin=0),
                pn.pane.Markdown(
                    "Analyze performance and correct annotations. "
                    "Once you have evaluated the performances, you can either **fit again** "
                    "if you modified classes or samples, or directly **export** if you are "
                    "satisfied with the performances.",
                    css_classes=['page-subtitle'], margin=0
                ),
                sizing_mode="stretch_width",
            ),
            pn.Column(
                pn.Row(self._export_config_tooltip, self._export_config_btn, align="end"),
                self._export_config_msg,
                align="end",
                margin=(0, 0, 0, 10),
            ),
            sizing_mode="stretch_width",
            align="start",
        )

        self.pane_selection = pn.widgets.RadioButtonGroup(
            name="Pane Selection", 
            options=["Class merge", "Sample correction"],
            button_type="default",
            button_style="outline",
            align="center"
        )
        self.pane_selection.param.watch(self.on_switch_panel, "value")

        spec_section = self._build_class_spectrogram_section()
        main_content = pn.Column(
            self.header,
            spec_section,
            pn.Row(self.pane_selection, css_classes=['selector-bar'], sizing_mode="stretch_width"),
            self.merge_dashboard.layout,
            sizing_mode="stretch_both",
            margin=(20, 20, 20, 20)
        )

        self.layout = pn.Row(
            self.sidebar,
            main_content,
            sizing_mode="stretch_both",
            background="#f8fafc" 
        )

    def _build_class_spectrogram_section(self):
        self._spec_computed = False
        self.spec_toggle_btn = pn.widgets.Button(
            name="Class Spectrograms",
            button_type="primary",
            width=180,
        )
        self.spec_toggle_btn.on_click(self._on_toggle_spec_panel)
        self.spec_collapsed = pn.Row(
            self.spec_toggle_btn,
            pn.pane.Markdown(
                "Visualize one mel spectrogram per class to identify similar classes.",
                styles={"font-size": "13px", "color": "#374151"},
                align="center",
                margin=(0, 0, 0, 12),
            ),
            align="center",
            sizing_mode="stretch_width",
        )
        self.spec_progress = pn.widgets.Progress(
            value=0,
            max=100,
            sizing_mode="stretch_width",
            visible=False,
            bar_color="primary",
        )
        self.spec_status = pn.pane.Markdown(
            "",
            styles={"font-size": "12px", "color": "#6b7280"},
            visible=False,
        )
        self.spec_grid = pn.FlexBox(
            justify_content="start",
            gap=10,
            flex_wrap="wrap",
            sizing_mode="stretch_width",
        )
        self.spec_expanded = pn.Column(
            pn.Column(
                self.spec_progress,
                self.spec_status,
                sizing_mode="stretch_width",
            ),
            self.spec_grid,
            visible=False,
            sizing_mode="stretch_width",
        )
        return pn.Column(
            self.spec_collapsed,
            self.spec_expanded,
            css_classes=["dashboard-col"],
            sizing_mode="stretch_width",
            styles={"margin-bottom": "15px"},
        )

    def _on_toggle_spec_panel(self, event):
        self.spec_expanded.visible = not self.spec_expanded.visible
        if self.spec_expanded.visible and not self._spec_computed:
            self._on_compute_class_spectrograms(None)

    def _on_compute_class_spectrograms(self, event):
        self.spec_toggle_btn.loading = True
        self.spec_progress.value = 0
        self.spec_progress.visible = True
        self.spec_status.visible = True
        self.spec_status.object = "Starting..."
        self.spec_grid.objects = []

        def run():
            try:
                df = self.controler.corpus.dataset
                classes = sorted([c for c in df["label"].unique() if c != "SIL"])
                total = len(classes)
                for i, cls in enumerate(classes):
                    self.spec_status.object = f"Class {i + 1}/{total}: {cls}"
                    self.spec_progress.value = int(i / total * 100)
                    class_df = df[df["label"] == cls].copy()
                    class_df = class_df.assign(
                        duration=class_df["offset_s"] - class_df["onset_s"]
                    )
                    med_dur = class_df["duration"].median()
                    best_idx = (class_df["duration"] - med_dur).abs().idxmin()
                    sample_row = class_df.loc[[best_idx]]
                    try:
                        specs = self.controler.load_repertoire(sample_row)
                        if specs:
                            fig = specs[0][0]
                            card = pn.Column(
                                pn.pane.Markdown(
                                    f"**{cls}**",
                                    styles={"font-size": "11px", "text-align": "center"},
                                    margin=(0, 0, 2, 0),
                                ),
                                pn.pane.Matplotlib(
                                    fig,
                                    format="png",
                                    tight=True,
                                    sizing_mode="stretch_width",
                                    height=120,
                                ),
                                styles={
                                    "border": "1px solid #e5e7eb",
                                    "border-radius": "6px",
                                    "padding": "8px",
                                    "background": "#f8fafc",
                                },
                                width=240,
                            )
                            self.spec_grid.append(card)
                    except Exception as e:
                        logger.debug(f"Spectrogram error ({cls}): {e}")
                        self.spec_grid.append(
                            pn.pane.Markdown(
                                f"**{cls}**: _error_",
                                styles={"font-size": "11px", "color": "#dc2626"},
                                width=220,
                            )
                        )
                self._spec_computed = True
                self.spec_progress.value = 100
                self.spec_status.object = f"Done — {total} classes."
                self.spec_progress.visible = False
                self.spec_status.visible = False
            except Exception as e:
                logger.error(f"Class spectrogram computation error: {e}")
                self.spec_status.object = f"Error: {e}"
            finally:
                self.spec_toggle_btn.loading = False

        threading.Thread(target=run, daemon=True).start()

    def _on_click_export_config(self, event):
        try:
            path = self.controler.export_config()
            self._export_config_msg.object = f"Saved to `{path}`"
            self._export_config_msg.styles = {"font-size": "12px", "color": "#166534"}
        except Exception as e:
            self._export_config_msg.object = f"Error: {e}"
            self._export_config_msg.styles = {"font-size": "12px", "color": "#be123c"}
        self._export_config_msg.visible = True

    def on_switch_panel(self, events):
        if self.pane_selection.value == "Class merge":
            self.layout[1][3] = self.merge_dashboard.layout
        else:
            self.layout[1][3] = self.sample_dashboard.layout