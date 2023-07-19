import logging

from pathlib import Path
from typing import List, Optional

import attr
import panel as pn

from .controler import Controler

from .view.train.train_dash import TrainDashboard
from .view.eval.eval_dash import EvalDashboard
from .view.export.export_dash import ExportDashboard
from .view.helpers import Registry

MAX_SAMPLE_DISPLAY = 10


logger = logging.getLogger("canapy-dashboard")


@attr.define
class CanapyDashboard(pn.viewable.Viewer):
    data_directory: Path = attr.field(converter=Path)
    output_directory: Path = attr.field(converter=Path)
    config_path: Path = attr.field(converter=Path)
    port: Optional[int] = attr.field()
    annot_format: str = attr.field(default="marron1csv")
    audio_ext: str = attr.field(default=".wav")
    annotators: List = attr.field(default=["syn-esn", "nsyn-esn", "ensemble"])

    def __attrs_post_init__(self):

        self.controler = Controler(
            data_directory=self.data_directory,
            output_directory=self.output_directory,
            config_path=self.config_path,
            dashboard=self,
            annot_format=self.annot_format,
            audio_ext=self.audio_ext,
            annotators=self.annotators,
        )

        self.layouts = {
            "train": TrainDashboard,
            "eval": EvalDashboard,
            "export": ExportDashboard,
        }

        self.subdash = None

        self.layout = pn.Row(pn.Spacer())

        self.switch_panel()

    def __panel__(self):
        return self.layout

    def switch_panel(self):
        Registry.clean_all()
        self.subdash = self.layouts[self.controler.step](self)
        self.layout[0] = self.subdash.layout

        if self.controler.step == "export":
            self.subdash.begin()

    def show(self, **kwargs):
        logger.info("Starting server...")
        # self._server_instance = self.layout.show(
        #     port=self.port, title="Canapy", threaded=True
        # )
        super().show(title="Canapy", port=self.port, threaded=False, open=True)

    def stop(self):
        super().stop()
        logger.info("Server shut down.")
