import logging

from .controler.base import Controler

from pathlib import Path
from typing import List, Optional

import attr
import panel as pn

from canapy.utils import as_path

from .view.welcome.welcome_dash import WelcomeDashboard
from .view.helpers import Registry

MAX_SAMPLE_DISPLAY = 10

logger = logging.getLogger("canapy-dashboard")

@attr.define
class DashboardTest(pn.viewable.Viewer):

    def __attrs_post_init__(self):
        self.subdash = None
        self.current_view = pn.Row(
           WelcomeDashboard(self)
        )

    def __panel__(self):
        return self.current_view

    def show(self, **kwargs):
        logger.info("Starting server...")
        super().show(title="Canapy", port=9321, threaded=True, open=True)

    def stop(self):
        logger.info("Server shut down.")
        exit()