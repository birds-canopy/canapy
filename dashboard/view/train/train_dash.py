import logging
import threading
import time
import panel as pn

from ..helpers import SubDash
from ..helpers import SideBar

logger = logging.getLogger("canapy-dashboard")

TRAIN_CSS = """
.train-card {
    background-color: #ffffff;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    padding: 18px 20px;
    border: 1px solid #e5e7eb;
    box-sizing: border-box;
}
.page-title {
    font-size: 24px;
    font-weight: 700;
    margin: 0;
}
.page-subtitle {
    font-size: 14px;
    color: #6b7280;
    margin: 0;
}
"""


class TrainDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        pn.config.raw_css.append(TRAIN_CSS)

        self._pending_nav = None

        self._cancel_leave_btn = pn.widgets.Button(
            name="Cancel", button_type="default", width=100,
        )
        self._confirm_leave_btn = pn.widgets.Button(
            name="Leave & Stop Search", button_type="danger", width=180,
        )
        self._cancel_leave_btn.on_click(self._on_cancel_leave)
        self._confirm_leave_btn.on_click(self._on_confirm_leave)

        self._leave_warning = pn.Column(
            pn.pane.HTML(
                "<div style='background:#fef3c7;border:2px solid #f59e0b;"
                "border-radius:8px;padding:14px 18px;font-size:14px;'>"
                "<b>⚠ Hyperparameter search in progress</b><br>"
                "If you leave this page, the search will be stopped. Continue?"
                "</div>"
            ),
            pn.Row(self._cancel_leave_btn, self._confirm_leave_btn, margin=(8, 0, 0, 0)),
            visible=False,
            margin=(0, 0, 12, 0),
        )

        self.sidebar = SideBar(self, "train")
        self.traindash = TrainerDashboard(self)
        self.annotdash = AnnotatorDashboard(self)

        hp_section = self._build_hp_section()

        is_retrain = self.controler._iter > 1
        header = pn.Column(
            pn.pane.Markdown(
                f"# Train again — iteration {self.controler._iter}" if is_retrain else "# Train",
                css_classes=["page-title"], margin=0,
            ),
            pn.pane.Markdown(
                "Train and evaluate the annotator models.",
                css_classes=["page-subtitle"],
                margin=0,
            ),
            margin=(0, 0, 12, 0),
        )

        self.layout = pn.Row(
            self.sidebar,
            pn.Column(
                self._leave_warning,
                header,
                hp_section,
                self.traindash,
                sizing_mode="stretch_both",
                margin=(20, 10),
            ),
            sizing_mode="stretch_both",
            background="#f8fafc",
        )

    def request_navigate_away(self, nav_fn):
        if self.controler.is_optimization_running:
            self._pending_nav = nav_fn
            self._leave_warning.visible = True
        else:
            nav_fn()

    def _on_cancel_leave(self, event):
        self._pending_nav = None
        self._leave_warning.visible = False

    def _on_confirm_leave(self, event):
        self._leave_warning.visible = False
        fn = self._pending_nav
        self._pending_nav = None
        self.traindash._opt_running = False
        self.controler.stop_optimization()
        if fn is not None:
            fn()

    def _build_hp_section(self):
        td = self.traindash

        self._hp_toggle_btn = pn.widgets.Button(
            name="Optimize Hyperparameters",
            button_type="primary",
            width=230,
        )
        self._hp_toggle_btn.on_click(self._on_toggle_hp_panel)

        hp_collapsed = pn.Row(
            self._hp_toggle_btn,
            td.params_display,
            align="center",
            sizing_mode="stretch_width",
        )

        self._hp_info_alert = pn.pane.Alert(
            "Runs a hyperparameter search on the ESN reservoir. "
            "This may take several minutes. Configure dataset percentage and number of jobs in Settings.",
            alert_type="info",
            margin=(0, 0, 10, 0),
        )

        self._hp_params_pane = pn.pane.HTML(
            self._build_params_text(),
            margin=(0, 0, 10, 0),
        )

        is_retrain = self.controler._iter > 1

        self._hp_expanded = pn.Column(
            self._hp_info_alert,
            self._hp_params_pane,
            pn.Row(td.optimize_btn, td.stop_search_btn, td.opt_indicator, align="center", margin=(0, 0, 6, 0)),
            td.opt_progress,
            td.opt_progress_text,
            td.opt_status,
            visible=is_retrain,
            sizing_mode="stretch_width",
            margin=(10, 0, 0, 0),
        )

        return pn.Column(
            hp_collapsed,
            self._hp_expanded,
            css_classes=["train-card"],
            sizing_mode="stretch_width",
            styles={"margin-bottom": "12px"},
        )

    def _build_params_text(self):
        c = self.controler
        pct = int(getattr(c, "opt_max_percentage", 0.3) * 100)
        n_jobs = getattr(c, "opt_n_jobs", 4)
        n_evals = getattr(c, "opt_max_evals", 100)
        parallel = getattr(c, "opt_parallel", False)
        mode = "Random Parallel" if parallel else "Bayesian sequential"
        chip = "background:#eff6ff;color:#1d4ed8;border:1px solid #93c5fd;border-radius:4px;padding:2px 8px;font-size:12px;font-family:monospace;"
        return (
            f"<div style='display:flex;gap:8px;flex-wrap:wrap;align-items:center;'>"
            f"<span style='{chip}'>{n_evals} trials</span>"
            f"<span style='{chip}'>{n_jobs} jobs</span>"
            f"<span style='{chip}'>{pct}% data</span>"
            f"<span style='{chip}'>{mode}</span>"
            f"</div>"
        )

    def _on_toggle_hp_panel(self, event):
        self._hp_expanded.visible = not self._hp_expanded.visible
        if self._hp_expanded.visible:
            name = self.controler._config_display_name or "default"
            if self.controler._settings_dirty:
                name += " *(custom)*"
            self.traindash.params_display.object = f"**Current params:** {name}"
            self._hp_params_pane.object = self._build_params_text()


def _spinner_col(label, indicator, status):
    return pn.Column(
        pn.pane.HTML(f"<span style='font-size:12px; color:#6b7280;'>{label}</span>"),
        indicator,
        status,
        sizing_mode="stretch_width",
        align="center",
    )


class TrainerDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        _btn_label = (
            "Train again with eval corrections"
            if self.controler._iter > 1 else
            "Start Training"
        )
        self.train_btn = pn.widgets.Button(
            name=_btn_label,
            button_type="success",
            sizing_mode="stretch_width",
            margin=(16, 0, 0, 0),
        )
        self.train_btn.on_click(self.on_click_train)

        # Train spinners
        self.syn_train_indicator  = pn.indicators.LoadingSpinner(value=False, width=60, height=60)
        self.nsyn_train_indicator = pn.indicators.LoadingSpinner(value=False, width=60, height=60)
        self.syn_train_status     = pn.pane.HTML(_idle())
        self.nsyn_train_status    = pn.pane.HTML(_idle())

        # Annotate spinners
        self.syn_annot_indicator  = pn.indicators.LoadingSpinner(value=False, width=60, height=60)
        self.nsyn_annot_indicator = pn.indicators.LoadingSpinner(value=False, width=60, height=60)
        self.ens_annot_indicator  = pn.indicators.LoadingSpinner(value=False, width=60, height=60)
        self.syn_annot_status     = pn.pane.HTML(_idle())
        self.nsyn_annot_status    = pn.pane.HTML(_idle())
        self.ens_annot_status     = pn.pane.HTML(_idle())

        # HP optimization components (used by TrainDashboard._build_hp_section)
        self.optimize_btn = pn.widgets.Button(
            name="Run Search",
            button_type="primary",
            width=140,
        )
        self.optimize_btn.on_click(self.on_click_optimize)

        self.stop_search_btn = pn.widgets.Button(
            name="Stop Search",
            button_type="danger",
            width=130,
            visible=False,
        )
        self.stop_search_btn.on_click(self.on_click_stop_search)

        self.opt_indicator = pn.indicators.LoadingSpinner(value=False, width=40, height=40)
        self.opt_status = pn.pane.HTML(
            "<span style='color:#6b7280; font-size:13px;'>Recommended for new data.</span>"
        )
        self.opt_progress = pn.widgets.Progress(
            value=0,
            max=100,
            sizing_mode="stretch_width",
            bar_color="primary",
            visible=False,
        )
        self.opt_progress_text = pn.pane.HTML(
            "",
            styles={"font-size": "12px", "color": "#6b7280"},
            visible=False,
        )
        self._opt_running = False
        self.params_display = pn.pane.Markdown(
            f"**Current params:** {self.controler._config_display_name or 'default'}",
            styles={"font-size": "13px", "color": "#374151"},
            align="center",
            margin=(0, 0, 0, 16),
        )
        self.layout = pn.Column(
            pn.Row(
                _spinner_col("Syn — train",     self.syn_train_indicator,  self.syn_train_status),
                _spinner_col("NSyn — train",    self.nsyn_train_indicator, self.nsyn_train_status),
                _spinner_col("Syn — annotate",  self.syn_annot_indicator,  self.syn_annot_status),
                _spinner_col("NSyn — annotate", self.nsyn_annot_indicator, self.nsyn_annot_status),
                _spinner_col("Ensemble — annotate", self.ens_annot_indicator, self.ens_annot_status),
                sizing_mode="stretch_width",
            ),
            self.train_btn,
            css_classes=["train-card"],
            sizing_mode="stretch_width",
            min_width=400,
        )

    def on_click_stop_search(self, event):
        self.stop_search_btn.disabled = True
        self.controler.stop_optimization()

    def on_click_optimize(self, event):
        self.optimize_btn.disabled = True
        self.optimize_btn.loading = True
        self.stop_search_btn.visible = True
        self.stop_search_btn.disabled = False
        self.opt_status.object = (
            "<span style='color:#d97706; font-size:13px;'><b>Running hyperparameter search...</b></span>"
        )
        self.opt_indicator.value = True
        self.opt_progress.value = 0
        self.opt_progress.visible = True
        self.opt_progress_text.object = "Trial 0 / ?"
        self.opt_progress_text.visible = True
        self._opt_running = True
        tic = time.time()

        def _safe_ui(fn):
            try:
                fn()
            except Exception:
                pass

        def _on_progress(done, total):
            if not self._opt_running:
                return
            pct = int(done / total * 100) if total > 0 else 0
            try:
                with pn.io.unlocked():
                    self.opt_progress.value = pct
                    self.opt_progress_text.object = f"Trial {done} / {total}"
            except Exception:
                pass

        def run():
            try:
                best_params = self.controler.optimize_models(progress_callback=_on_progress)
                duration = time.time() - tic
                if best_params is None:
                    _safe_ui(lambda: setattr(self.opt_status, 'object',
                        "<span style='color:#dc2626; font-size:13px;'>"
                        "<b>Optimization was interrupted.</b>"))
                else:
                    lines = []
                    for k, v in best_params.items():
                        lines.append(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")
                    _safe_ui(lambda: setattr(self.params_display, 'object',
                        "🤓 **Best params found:**\n```\n" + "\n".join(lines) + "\n```"))
                    _safe_ui(lambda: setattr(self.opt_status, 'object',
                        f"<span style='color:#16a34a; font-size:13px;'>"
                        f"<b>Done</b> in {round(duration, 1)} s</span>"))
                    _safe_ui(lambda: setattr(self.optimize_btn, 'visible', False))
                    _safe_ui(lambda: setattr(self.parent._hp_info_alert, 'visible', False))
            except Exception as e:
                _safe_ui(lambda: setattr(self.opt_status, 'object',
                    f"<span style='color:#dc2626; font-size:13px;'>Error: {e}</span>"))
            finally:
                self._opt_running = False
                _safe_ui(lambda: setattr(self.opt_indicator, 'value', False))
                _safe_ui(lambda: setattr(self.opt_indicator, 'visible', False))
                _safe_ui(lambda: setattr(self.optimize_btn, 'disabled', False))
                _safe_ui(lambda: setattr(self.optimize_btn, 'loading', False))
                _safe_ui(lambda: setattr(self.stop_search_btn, 'visible', False))
                _safe_ui(lambda: setattr(self.opt_progress, 'visible', False))
                _safe_ui(lambda: setattr(self.opt_progress_text, 'visible', False))

        threading.Thread(target=run, daemon=True).start()

    def on_click_train(self, event):
        self.train_btn.disabled = True
        self.train_btn.loading = True

        _set(self.syn_train_status, "running")
        self.syn_train_indicator.value = True
        tic = time.time()
        self.controler.train_syn()
        _set(self.syn_train_status, "done", time.time() - tic)
        self.syn_train_indicator.value = False

        _set(self.nsyn_train_status, "running")
        self.nsyn_train_indicator.value = True
        tic = time.time()
        self.controler.train_nsyn()
        _set(self.nsyn_train_status, "done", time.time() - tic)
        self.nsyn_train_indicator.value = False

        logger.info("Trained!")
        self.train_btn.loading = False
        self.parent.annotdash.begin()


class AnnotatorDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

    def begin(self):
        td = self.parent.traindash

        _set(td.syn_annot_status, "running")
        td.syn_annot_indicator.value = True
        tic = time.time()
        self.controler.annotate_syn()
        _set(td.syn_annot_status, "done", time.time() - tic)
        td.syn_annot_indicator.value = False

        _set(td.nsyn_annot_status, "running")
        td.nsyn_annot_indicator.value = True
        tic = time.time()
        self.controler.annotate_nsyn()
        _set(td.nsyn_annot_status, "done", time.time() - tic)
        td.nsyn_annot_indicator.value = False

        _set(td.ens_annot_status, "running")
        td.ens_annot_indicator.value = True
        tic = time.time()
        self.controler.annotate_ensemble()
        _set(td.ens_annot_status, "done", time.time() - tic)
        td.ens_annot_indicator.value = False

        logger.info("Annotated!")
        td.train_btn.name = "Trained !"
        td.train_btn.disabled = True
        self.parent.sidebar.enable_next()


# --- helpers ---

def _idle():
    return "<p style='margin:4px 0 0 0; color:#9ca3af; font-size:12px;'>Idle</p>"


def _set(obj, status, duration=0.0):
    if status == "running":
        obj.object = "<p style='color:#2563eb; font-weight:600; margin:4px 0 0 0; font-size:12px;'>Running...</p>"
    elif status == "done":
        obj.object = (
            f"<p style='color:#16a34a; font-weight:600; margin:4px 0 0 0; font-size:12px;'>"
            f"Done {round(duration, 1)} s</p>"
        )
