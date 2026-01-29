import time

import panel as pn

from ..helpers import SubDash
from ..helpers import SideBar


class ExportDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        self.sidebar = SideBar(self, "export")

        self.export_info = pn.pane.HTML(
            f"Models will be exported to "
            f"{str(self.controler.output_directory / 'model')}"
        )

        self.syn_indicator = pn.indicators.LoadingSpinner(
            value=False, width=100, height=100
        )
        self.nsyn_indicator = pn.indicators.LoadingSpinner(
            value=False, width=100, height=100
        )

        self.syn_status = pn.pane.HTML("<h2>Idle</h2>")
        self.nsyn_status = pn.pane.HTML("<h2>Idle</h2>")

        self.layout = pn.Row(
            self.sidebar.layout,
            pn.Column(
                self.export_info,
                pn.Row(
                    pn.Column(
                        pn.pane.HTML("Syn training:"),
                        self.syn_indicator,
                        self.syn_status,
                    ),
                    pn.Column(
                        pn.pane.HTML("NSyn training:"),
                        self.nsyn_indicator,
                        self.nsyn_status,
                    ),
                ),
            ),
        )

    def begin(self):
        self.switch_status(self.syn_status, "training")
        self.syn_indicator.value = True

        tic = time.time()
        self.controler.train_syn_export()
        toc = time.time()

        self.switch_status(self.syn_status, "done", duration=toc - tic)
        self.syn_indicator.value = False

        self.switch_status(self.nsyn_status, "training")
        self.nsyn_indicator.value = True

        tic = time.time()
        self.controler.train_nsyn_export()
        toc = time.time()

        self.switch_status(self.nsyn_status, "done", duration=toc - tic)
        self.nsyn_indicator.value = False

        self.sidebar.enable_next()

    def switch_status(self, obj, status, duration=None):
        if status == "training":
            obj.object = "<h2>Training...</h2>"
            obj.style = {"color": "blue"}
        if status == "done":
            obj.object = f"<h2>Exported !</h2> in {round(duration, 2)} sec."
            obj.style = {"color": "green"}
