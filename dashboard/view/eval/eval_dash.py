from pathlib import Path

import panel as pn
import pandas as pd

from ..helpers import SubDash
from ..helpers import SideBar
from ..helpers import Registry

from .classmerge import ClassMergeDashboard
from .samplecorrection import SampleCorrectionDashboard


MAX_SAMPLE_DISPLAY = 10


class EvalDashboard(SubDash):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.sidebar = SideBar(self, "eval")
        self.merge_dashboard = ClassMergeDashboard(self)
        self.sample_dashboard = SampleCorrectionDashboard(self)

        self.pane_selection = pn.widgets.RadioButtonGroup(
            name="Pane Selection", options=["Class merge", "Sample correction"]
        )
        self.pane_selection.param.watch(self.on_switch_panel, "value")

        self.layout = pn.Row(
            self.sidebar,
            pn.Column(
                self.pane_selection,
                self.merge_dashboard.layout,
            ),
            sizing_mode="stretch_both",
        )

    def on_switch_panel(self, events):
        if self.pane_selection.value == "Class merge":
            self.layout[1][1] = self.merge_dashboard.layout
        else:
            self.layout[1][1] = self.sample_dashboard.layout
