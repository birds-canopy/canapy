import time

import panel as pn

from .helpers import SubDash
from .helpers import SideBar


class TrainDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        self.sidebar = SideBar(self, "train")
        self.traindash = TrainerDashboard(self)
        self.annotdash = AnnotatorDashboard(self)

        self.layout = pn.Row(
            self.sidebar.layout, self.traindash.layout, self.annotdash.layout
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

        self.layout = pn.Column(
            pn.Row(
                pn.Column(
                    pn.pane.HTML("Syn training:"), self.syn_indicator, self.syn_status
                ),
                pn.Column(
                    pn.pane.HTML("NSyn training:"),
                    self.nsyn_indicator,
                    self.nsyn_status,
                ),
            ),
            self.train_btn,
        )

    def switch_status(self, obj, status, duration=None):
        if status == "training":
            obj.object = "<h2>Training...</h2>"
            obj.style = {"color": "blue"}
        if status == "done":
            obj.object = f"<h2>Done !</h2> in {round(duration, 2)} sec."
            obj.style = {"color": "green"}

    def on_click_train(self, events):

        self.switch_status(self.syn_status, "training")
        self.syn_indicator.value = True

        tic = time.time()
        self.controler.train_syn()
        toc = time.time()

        self.switch_status(self.syn_status, "done", duration=toc - tic)
        self.syn_indicator.value = False

        self.switch_status(self.nsyn_status, "training")
        self.nsyn_indicator.value = True

        tic = time.time()
        self.controler.train_nsyn()
        toc = time.time()

        self.switch_status(self.nsyn_status, "done", duration=toc - tic)
        self.nsyn_indicator.value = False

        self.parent.annotdash.begin()


class AnnotatorDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        self.syn_indicator = pn.indicators.LoadingSpinner(
            value=False, width=100, height=100
        )
        self.nsyn_indicator = pn.indicators.LoadingSpinner(
            value=False, width=100, height=100
        )
        self.ens_indicator = pn.indicators.LoadingSpinner(
            value=False, width=100, height=100
        )

        self.syn_status = pn.pane.HTML("<h2>Idle</h2>")
        self.nsyn_status = pn.pane.HTML("<h2>Idle</h2>")
        self.ens_status = pn.pane.HTML("<h2>Idle</h2>")

        self.layout = pn.Row(
            pn.Column(
                pn.pane.HTML("Syn annotation:"), self.syn_indicator, self.syn_status
            ),
            pn.Column(
                pn.pane.HTML("NSyn annotations:"), self.nsyn_indicator, self.nsyn_status
            ),
            pn.Column(
                pn.pane.HTML("Ensemble annotations:"),
                self.ens_indicator,
                self.ens_status,
            ),
        )

    def switch_status(self, obj, status, duration=None):
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
        toc = time.time()

        self.switch_status(self.syn_status, "done", duration=toc - tic)
        self.syn_indicator.value = False

        self.switch_status(self.nsyn_status, "annotating")
        self.nsyn_indicator.value = True

        tic = time.time()
        self.controler.annotate_nsyn()
        toc = time.time()

        self.switch_status(self.nsyn_status, "done", duration=toc - tic)
        self.nsyn_indicator.value = False

        self.switch_status(self.ens_status, "annotating")
        self.ens_indicator.value = True

        tic = time.time()
        self.controler.annotate_ensemble()
        toc = time.time()

        self.switch_status(self.ens_status, "done", duration=toc - tic)
        self.ens_indicator.value = False

        self.parent.sidebar.enable_next()
