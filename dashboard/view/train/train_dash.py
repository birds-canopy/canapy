import logging
import time
import panel as pn

from ..helpers import SubDash
from ..helpers import SideBar

logger = logging.getLogger("canapy-dashboard")

class TrainDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        self.sidebar = SideBar(self, "train")
        self.traindash = TrainerDashboard(self)
        self.annotdash = AnnotatorDashboard(self)

        self.layout = pn.Row(
            self.sidebar,
            self.traindash,
            self.annotdash,
            sizing_mode="stretch_both",
            background="#ffffff",
        )

class TrainerDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        self.train_btn = pn.widgets.Button(name="Start training")
        self.train_btn.on_click(self.on_click_train)

        self.syn_indicator = pn.indicators.LoadingSpinner(
            value=False, width=100, height=100
        )
        self.nsyn_indicator = pn.indicators.LoadingSpinner(
            value=False, width=100, height=100
        )

        self.syn_status = pn.pane.HTML("<h2>Idle</h2>")
        self.nsyn_status = pn.pane.HTML("<h2>Idle</h2>")

        # Composants pour l'optimisation (déplacés en bas)
        self.optimize_btn = pn.widgets.Button(
            name="Auto-Tune Reservoir",
            button_type="primary"
        )
        self.optimize_btn.on_click(self.on_click_optimize)
        self.opt_indicator = pn.indicators.LoadingSpinner(value=False, width=50, height=50)
        self.opt_status = pn.pane.HTML("<i>Recommended for new data</i>")
        self.params_display = pn.pane.Markdown("**Current Params:** Default", sizing_mode="stretch_width")

        self.export_config_btn = pn.widgets.Button(
            name="Export config",
            button_type="success",
            visible=False,
        )
        self.export_config_btn.on_click(self.on_click_export_config)
        self.export_config_status = pn.pane.HTML("", visible=False)

        self.layout = pn.Column(
            pn.Row(
                pn.Column(
                    pn.pane.HTML("Syn training:"),
                    self.syn_indicator,
                    self.syn_status,
                    sizing_mode="stretch_width",
                    align="center",
                ),
                pn.Column(
                    pn.pane.HTML("NSyn training:"),
                    self.nsyn_indicator,
                    self.nsyn_status,
                    sizing_mode="stretch_width",
                    align="center",
                ),
                sizing_mode="stretch_width",
            ),
            self.train_btn,
            pn.layout.Divider(),
            pn.pane.Markdown("### Hyperparameter Optimization"),
            pn.Row(self.optimize_btn, self.opt_indicator, sizing_mode="stretch_width"),
            self.opt_status,
            self.params_display,
            self.export_config_btn,
            self.export_config_status,
            sizing_mode="stretch_width",
            margin=(20, 30),
            min_width=280,
        )

    def switch_status(self, obj, status, duration=0.0):
        if status == "training":
            obj.object = "<h2>Training...</h2>"
            obj.style = {"color": "blue"}
        elif status == "optimizing":
            obj.object = "<b>Running hyperparameters search...</b>"
            obj.style = {"color": "orange"}
        elif status == "done":
            obj.object = f"<h2>Done !</h2> in {round(duration, 2)} sec."
            obj.style = {"color": "green"}
        elif status == "opt_done":
            obj.object = f"<b>Optimization Finished!</b> ({round(duration, 2)} sec)"
            obj.style = {"color": "green"}

    def on_click_optimize(self, events):
        self.optimize_btn.disabled = True
        self.export_config_btn.visible = False
        self.export_config_status.visible = False
        self.switch_status(self.opt_status, "optimizing")
        self.opt_indicator.value = True
        tic = time.time()
        try:
            best_params = self.controler.optimize_models()
            if best_params is None:
                self.opt_status.object = (
                    "<span style='color:red'><b>Optimization failed or was interrupted.</b> "
                    "Check the logs for details.</span>"
                )
            else:
                lines = []
                for k, v in best_params.items():
                    lines.append(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")
                self.params_display.object = "**Best parameters found:**\n```\n" + "\n".join(lines) + "\n```"
                self.switch_status(self.opt_status, "opt_done", duration=time.time() - tic)
                self.export_config_btn.visible = True
        except Exception as e:
            self.opt_status.object = f"<span style='color:red'>Error: {e}</span>"
        self.opt_indicator.value = False
        self.optimize_btn.disabled = False

    def on_click_export_config(self, events):
        try:
            path = self.controler.export_config()
            self.export_config_status.object = (
                f"<span style='color:green'>Config saved to <code>{path}</code></span>"
            )
        except Exception as e:
            self.export_config_status.object = (
                f"<span style='color:red'>Export failed: {e}</span>"
            )
        self.export_config_status.visible = True

    def on_click_train(self, events):
        self.train_btn.disabled = True
        self.switch_status(self.syn_status, "training")
        self.syn_indicator.value = True
        tic = time.time()
        self.controler.train_syn()
        self.switch_status(self.syn_status, "done", duration=time.time() - tic)
        self.syn_indicator.value = False

        self.switch_status(self.nsyn_status, "training")
        self.nsyn_indicator.value = True
        tic = time.time()
        self.controler.train_nsyn()
        self.switch_status(self.nsyn_status, "done", duration=time.time() - tic)
        self.nsyn_indicator.value = False

        logger.info("Trained!")
        self.train_btn.disabled = False
        self.parent.annotdash.begin()

class AnnotatorDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        self.syn_indicator = pn.indicators.LoadingSpinner(value=False, width=100, height=100)
        self.nsyn_indicator = pn.indicators.LoadingSpinner(value=False, width=100, height=100)
        self.ens_indicator = pn.indicators.LoadingSpinner(value=False, width=100, height=100)

        self.syn_status = pn.pane.HTML("<h2>Idle</h2>")
        self.nsyn_status = pn.pane.HTML("<h2>Idle</h2>")
        self.ens_status = pn.pane.HTML("<h2>Idle</h2>")

        self.layout = pn.Row(
            pn.Column(pn.pane.HTML("Syn annotation:"), self.syn_indicator, self.syn_status,
                      sizing_mode="stretch_width", align="center"),
            pn.Column(pn.pane.HTML("NSyn annotations:"), self.nsyn_indicator, self.nsyn_status,
                      sizing_mode="stretch_width", align="center"),
            pn.Column(pn.pane.HTML("Ensemble annotations:"), self.ens_indicator, self.ens_status,
                      sizing_mode="stretch_width", align="center"),
            sizing_mode="stretch_width",
            margin=(20, 30),
            min_width=280,
        )

    def switch_status(self, obj, status, duration=0.0):
        if status == "annotating":
            obj.object = "<h2>Annotating...</h2>"
            obj.style = {"color": "blue"}
        if status == "done":
            obj.object = f"<h2>Done !</h2> in {round(duration, 2)} sec."
            obj.style = {"color": "green"}

    def begin(self):
        self.switch_status(self.syn_status, "annotating")
        self.syn_indicator.value = True
        tic = time.time()
        self.controler.annotate_syn()
        self.switch_status(self.syn_status, "done", duration=time.time() - tic)
        self.syn_indicator.value = False

        self.switch_status(self.nsyn_status, "annotating")
        self.nsyn_indicator.value = True
        tic = time.time()
        self.controler.annotate_nsyn()
        self.switch_status(self.nsyn_status, "done", duration=time.time() - tic)
        self.nsyn_indicator.value = False

        self.switch_status(self.ens_status, "annotating")
        self.ens_indicator.value = True
        tic = time.time()
        self.controler.annotate_ensemble()
        self.switch_status(self.ens_status, "done", duration=time.time() - tic)
        self.ens_indicator.value = False

        logger.info("Annotated!")
        self.parent.sidebar.enable_next()