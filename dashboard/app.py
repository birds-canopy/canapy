import logging
from pathlib import Path
from typing import List, Optional

import attr
import panel as pn

from canapy.utils import as_path
from .controler import Controler
from .view.train.train_dash import TrainDashboard
from .view.eval.eval_dash import EvalDashboard
from .view.export.export_dash import ExportDashboard
from .view.helpers import Registry
from .view.preprocess.preprocess_dash import PreprocessDashboard
from .view.home.home_dash import HomeDashboard
from .view.annotate.annotate_dash import AnnotateDashboard
from .view.loaddata.load_data_dash import LoadDataDashboard  

MAX_SAMPLE_DISPLAY = 10
logger = logging.getLogger("canapy-dashboard")


@attr.define
class CanapyDashboard(pn.viewable.Viewer):
    # modif 20/01
    annots_directory: Optional[Path] = attr.field(default=None, converter=as_path)
    audio_directory: Optional[Path] = attr.field(default=None, converter=as_path)
    output_directory: Optional[Path] = attr.field(default=None, converter=as_path)
    # modif 20/01
    spec_directory: Optional[Path] = attr.field(default=None, converter=as_path)
    config_path: Optional[Path] = attr.field(default=None, converter=as_path)
    port: Optional[int] = attr.field(default=None)
    annot_format: str = attr.field(default="marron1csv")
    audio_ext: str = attr.field(default=".wav")
    annotators: List = attr.field(default=["syn-esn", "nsyn-esn", "ensemble"])

    def __attrs_post_init__(self):
        # modif 20/01 : output dir par défaut pour canapy dash
        if self.output_directory is None:
            self.output_directory = Path.cwd() / "canapy_output"
        
        self.spec_directory = (
            self.output_directory / "spectrograms"
            if self.spec_directory is None
            else self.spec_directory
        )

        self.controler = Controler(
            annots_directory=self.annots_directory,
            audio_directory=self.audio_directory,
            output_directory=self.output_directory,
            spec_directory=self.spec_directory,
            config_path=self.config_path,
            dashboard=self,
            annot_format=self.annot_format,
            audio_ext=self.audio_ext,
            annotators=self.annotators,
        )

        self.views = {
            "preprocess": PreprocessDashboard,
            "train": TrainDashboard,
            "eval": EvalDashboard,
            "export": ExportDashboard,
            "home": HomeDashboard,
            "annotate": AnnotateDashboard,
            "loaddata": LoadDataDashboard,  
        }

        self.subdash = None
        self.current_view = pn.Row(
            pn.Spacer(sizing_mode="stretch_both"), sizing_mode="stretch_both"
        )
        self.switch_panel()

    def __panel__(self):
        return self.current_view

    def switch_panel(self):
        Registry.clean_all()
        self.subdash = self.views[self.controler.step](self)
        self.current_view[0] = self.subdash.layout
        
        if self.controler.step == "export":
            self.subdash.begin()

    def switch_to_load_data(self):
        """Navigate to Load Data view"""
        self.controler._step = "loaddata"
        self.switch_panel()
    
    def switch_to_home(self):
        """Navigate back to Home view"""
        self.controler._step = "home"
        self.switch_panel()

    def show(self, **kwargs):
        logger.info("Starting server...")
        super().show(title="Canapy", port=self.port, threaded=True, open=True)

    def stop(self):
        logger.info("Server shut down.")
        exit()