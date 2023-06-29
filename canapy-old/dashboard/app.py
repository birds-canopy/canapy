import panel as pn

from .controler import Controler

from .train_dash import TrainDashboard
from .eval_dash import EvalDashboard
from .export_dash import ExportDashboard
from .helpers import Registry

MAX_SAMPLE_DISPLAY = 10


class Canapy(object):
    def __init__(
        self,
        data,
        output,
        config=None,
        corrections=None,
        port=None,
        repertoire=None,
        vocab=None,
        audioformat=None,
        rate=None,
    ):

        self.port = port
        self.server_instance = None
        self.controler = Controler(
            data, output, self, audioformat=audioformat, rate=rate
        )
        self.labels = self.controler.dataset.vocab

        self.layouts = {
            "train": TrainDashboard,
            "eval": EvalDashboard,
            "export": ExportDashboard,
        }

        self.layout = pn.Row(pn.Spacer())
        self.switch_panel()

    def switch_panel(self):
        Registry.clean_all()
        self.subdash = self.layouts[self.controler.step](self)
        self.layout[0] = self.subdash.layout

        if self.controler.step == "export":
            self.subdash.begin()

    def serve(self):
        print(f"Starting server...")
        self.server_instance = self.layout.show(
            port=self.port, title="Canapy", threaded=True
        )

    def stop(self):
        self.server_instance.stop()
