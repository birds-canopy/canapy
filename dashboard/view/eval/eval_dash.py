from pathlib import Path

import panel as pn
import pandas as pd

from ..helpers import SubDash, SideBar, Registry, custom_tooltip

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

        main_content = pn.Column(
            self.header,
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
            self.layout[1][2] = self.merge_dashboard.layout
        else:
            self.layout[1][2] = self.sample_dashboard.layout