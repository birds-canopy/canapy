# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
import logging
import threading
import time
from pathlib import Path

import panel as pn

from ..helpers import SubDash, SideBar, pick_directory

logger = logging.getLogger("canapy-dashboard")

EXPORT_CSS = """
.export-card {
    background-color: #ffffff;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    padding: 18px 20px;
    border: 1px solid #e5e7eb;
    box-sizing: border-box;
}
.section-header {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #6b7280;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 6px;
    margin-bottom: 10px;
}
.field-label {
    font-size: 12px;
    font-weight: 600;
    color: #374151;
    margin-bottom: 2px;
}
"""


def _spinner_col(label, indicator, status):
    return pn.Column(
        pn.pane.HTML(f"<span style='font-size:12px; color:#6b7280;'>{label}</span>"),
        indicator,
        status,
        sizing_mode="stretch_width",
        align="center",
    )


def _idle():
    return "<p style='margin:4px 0 0 0; color:#9ca3af; font-size:12px;'>Idle</p>"


def _set_status(obj, status, duration=0.0):
    if status == "running":
        obj.object = "<p style='color:#2563eb; font-weight:600; margin:4px 0 0 0; font-size:12px;'>Training...</p>"
    elif status == "done":
        obj.object = (
            f"<p style='color:#16a34a; font-weight:600; margin:4px 0 0 0; font-size:12px;'>"
            f"Done {round(duration, 1)} s</p>"
        )
    elif status == "skipped":
        obj.object = "<p style='margin:4px 0 0 0; color:#9ca3af; font-size:12px;'>Skipped</p>"
    elif status == "idle":
        obj.object = _idle()


class ExportDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)
        pn.config.raw_css.append(EXPORT_CSS)

        self.sidebar = SideBar(self, "export")

        config_card = self._build_config_panel()
        monitor_card = self._build_monitor_panel()
        export_config_card = self._build_export_config_panel()

        header = pn.Column(
            pn.pane.Markdown("# Export", css_classes=["page-title"], margin=0),
            pn.pane.Markdown(
                "Retrain models on the full dataset and export them for annotation.",
                css_classes=["page-subtitle"],
                margin=0,
            ),
            margin=(0, 0, 12, 0),
        )

        self.layout = pn.Row(
            self.sidebar,
            pn.Column(
                header,
                pn.Row(
                    config_card,
                    pn.Spacer(width=16),
                    monitor_card,
                    sizing_mode="stretch_width",
                ),
                pn.Spacer(height=16),
                export_config_card,
                sizing_mode="stretch_both",
                margin=(20, 10),
            ),
            sizing_mode="stretch_both",
            background="#f8fafc",
        )

    def _build_config_panel(self):
        default_dir = str(self.controler.output_directory / "model") if self.controler.output_directory else ""

        self.dir_input = pn.widgets.TextInput(
            value=default_dir,
            placeholder="/path/to/export/folder",
            sizing_mode="stretch_width",
            height=38,
            margin=0,
        )
        self.browse_btn = pn.widgets.Button(
            name="Browse", button_type="default", width=90, height=38, margin=0
        )
        self.browse_btn.on_click(self._browse_dir)

        self.toggle_syn = pn.widgets.Toggle(
            name="Syn-ESN", value=True, button_type="primary", sizing_mode="stretch_width"
        )
        self.toggle_nsyn = pn.widgets.Toggle(
            name="NSyn-ESN", value=True, button_type="primary", sizing_mode="stretch_width"
        )
        for t in [self.toggle_syn, self.toggle_nsyn]:
            t.param.watch(self._on_toggle_change, "value")

        self.export_btn = pn.widgets.Button(
            name="Export trained models",
            button_type="success",
            height=45,
            sizing_mode="stretch_width",
            margin=(4, 0, 0, 0),
        )
        self.export_btn.on_click(self._on_export)

        self.info_msg = pn.pane.HTML("", sizing_mode="stretch_width")

        return pn.Column(
            pn.pane.HTML("<div class='section-header'>Output Directory</div>"),
            pn.Row(
                self.dir_input,
                self.browse_btn,
                sizing_mode="stretch_width",
                align="center",
                margin=0,
            ),
            pn.Spacer(height=12),
            pn.pane.HTML("<div class='section-header'>Select Models to Export</div>"),
            self.toggle_syn,
            pn.Spacer(height=4),
            self.toggle_nsyn,
            pn.Spacer(height=12),
            self.export_btn,
            pn.Spacer(height=6),
            self.info_msg,
            css_classes=["export-card"],
            sizing_mode="stretch_width",
            min_width=240,
            max_width=360,
        )

    def _build_monitor_panel(self):
        self.syn_indicator  = pn.indicators.LoadingSpinner(value=False, width=60, height=60)
        self.nsyn_indicator = pn.indicators.LoadingSpinner(value=False, width=60, height=60)
        self.syn_status     = pn.pane.HTML(_idle())
        self.nsyn_status    = pn.pane.HTML(_idle())

        return pn.Column(
            pn.pane.HTML("<div class='section-header'>Process Monitor</div>"),
            pn.Row(
                _spinner_col("Syn-ESN",  self.syn_indicator,  self.syn_status),
                _spinner_col("NSyn-ESN", self.nsyn_indicator, self.nsyn_status),
                sizing_mode="stretch_width",
            ),
            css_classes=["export-card"],
            sizing_mode="stretch_both",
        )

    def _build_export_config_panel(self):
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
        self._export_config_name_input = pn.widgets.TextInput(
            value=self.controler.get_default_export_config_name(),
            placeholder="filename.toml",
            sizing_mode="stretch_width",
            height=38,
            margin=0,
        )
        self._export_config_btn = pn.widgets.Button(
            name="Export Config",
            height=38,
            width=160,
            stylesheets=[_EXPORT_CONFIG_CSS],
            margin=0,
        )
        self._export_config_msg = pn.pane.HTML("", sizing_mode="stretch_width")
        self._export_config_btn.on_click(self._on_click_export_config)

        return pn.Column(
            pn.pane.HTML("<div class='section-header'>Export Configuration</div>"),
            pn.pane.HTML(
                "<p style='font-size:13px;color:#6b7280;margin:0 0 10px 0;'>"
                "Save the current hyperparameters and audio settings as a TOML file. "
                "This is independent from the model export above — you can reload this config next time via Settings."
                "</p>"
            ),
            pn.pane.HTML("<div class='field-label'>Config name</div>"),
            pn.Row(
                self._export_config_name_input,
                self._export_config_btn,
                align="center",
                margin=0,
            ),
            pn.Spacer(height=4),
            self._export_config_msg,
            css_classes=["export-card"],
            sizing_mode="stretch_width",
        )

    def _browse_dir(self, event):
        directory = pick_directory("Select Export Directory")
        if directory:
            self.dir_input.value = directory

    def _on_toggle_change(self, event):
        event.obj.button_type = "primary" if event.new else "default"

    def _commit_from_eval(self):
        """Finalise the transition from eval when the user actually starts the export."""
        if getattr(self.controler, '_came_from_eval_export', False):
            self.controler._came_from_eval_export = False
            self.controler.checkpoint()
            self.controler.apply_corrections()
            self.controler.next_iter()
            self.controler.export_corpus()

    def _on_export(self, event):
        chosen_syn  = self.toggle_syn.value
        chosen_nsyn = self.toggle_nsyn.value

        if not chosen_syn and not chosen_nsyn:
            self.info_msg.object = (
                "<span style='color:#d97706;font-size:12px;'>Select at least one model.</span>"
            )
            return

        out_dir = self.dir_input.value.strip()
        if not out_dir:
            self.info_msg.object = (
                "<span style='color:#dc2626;font-size:12px;'>Please set an output directory.</span>"
            )
            return

        self._commit_from_eval()
        self.export_btn.disabled = True
        self.export_btn.loading = True
        self.info_msg.object = "<span style='color:#6b7280;font-size:12px;'>Running...</span>"
        _set_status(self.syn_status,  "idle")
        _set_status(self.nsyn_status, "idle")

        def run():
            try:
                if chosen_syn:
                    _set_status(self.syn_status, "running")
                    self.syn_indicator.value = True
                    tic = time.time()
                    self.controler.train_syn_export()
                    _set_status(self.syn_status, "done", time.time() - tic)
                    self.syn_indicator.value = False
                else:
                    _set_status(self.syn_status, "skipped")

                if chosen_nsyn:
                    _set_status(self.nsyn_status, "running")
                    self.nsyn_indicator.value = True
                    tic = time.time()
                    self.controler.train_nsyn_export()
                    _set_status(self.nsyn_status, "done", time.time() - tic)
                    self.nsyn_indicator.value = False
                else:
                    _set_status(self.nsyn_status, "skipped")

                self.controler.model_root = Path(out_dir)
                self.sidebar.on_export_done()
                self.info_msg.object = (
                    f"<span style='color:#16a34a;font-weight:600;font-size:12px;'>"
                    f"Export complete → {out_dir}</span>"
                )
                logger.info(f"Export done → {out_dir}")
            except Exception as e:
                logger.exception("Export error")
                self.info_msg.object = f"<span style='color:#dc2626;font-size:12px;'>Error: {e}</span>"
                self.syn_indicator.value = False
                self.nsyn_indicator.value = False
            finally:
                self.export_btn.disabled = False
                self.export_btn.loading = False

        threading.Thread(target=run, daemon=True).start()

    def _on_click_export_config(self, event):
        try:
            path = self.controler.export_config(self._export_config_name_input.value)
            self._export_config_msg.object = (
                f"<span style='color:#16a34a;font-size:12px;'>Saved to <code>{path}</code></span>"
            )
            self._export_config_name_input.value = self.controler.get_default_export_config_name()
        except Exception as e:
            self._export_config_msg.object = (
                f"<span style='color:#dc2626;font-size:12px;'>Error: {e}</span>"
            )

    def begin(self):
        pass
