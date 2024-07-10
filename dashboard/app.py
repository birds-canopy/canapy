import logging

from typing import Optional

import panel as pn

from .controler import Controler
from .view import SideBar
from .view import (
    HomeDashboard, 
    UploadDashboard,
    AnnotateDashboard,
    ExportDashboard,
    TrainDashboard,
    EvalDashboard
)
from .view.helpers import Registry


MAX_SAMPLE_DISPLAY = 10


logger = logging.getLogger("canapy-dashboard")


class CanapyDashboard(pn.viewable.Viewer):
    """Canapy dashboard application."""

    def __init__(self, port: Optional[int] = None):
        self.port = port
        self.controler = Controler(app=self)

        self.views = {
            "train": TrainDashboard,
            "eval": EvalDashboard,
            "export": ExportDashboard,
            "home": HomeDashboard,
            "load_annotate": AnnotateDashboard,
            "load_train": UploadDashboard
        }

        self.sidebar = SideBar(app=self)
        self.layout = pn.Row(
            self.sidebar,
            pn.Spacer(sizing_mode="stretch_both"), sizing_mode="stretch_both"
        )

        # Open homepage on startup
        self.switch_panel(to_view="home")

    def __panel__(self):
        return self.layout

    def switch_panel(self, to_view: str):
        """Change current view to another dashboard."""
        Registry.clean_all()
        self.sidebar.clear_controls()

        dashboard = self.views[to_view](parent=self)
        self.layout[1] = dashboard

    def show(self, **kwargs):
        """Launch application."""
        logger.info("Starting server...")
        super().show(title="Canapy", port=self.port, threaded=True, open=True)

    def stop(self):
        """Terminate application."""
        logger.info("Server shut down.")
        exit()
