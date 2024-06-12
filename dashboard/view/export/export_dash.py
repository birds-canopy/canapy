import time
import logging
import panel as pn

from ..helpers import SubDash
from ..helpers import SideBar

from ..train.train_dash import Card

logger = logging.getLogger("canapy-dashboard")

class ExportDashboard(SubDash):

    def __init__(self, parent):
        super().__init__(parent)

        self.export_info = pn.pane.HTML(
            f"Models will be exported to "
            f"{str(self.controler.output_directory.resolve() / 'model')}"
        )

        self.sidebar = SideBar(self, "export")

        self.cards = {
            annotator: Card(self, annotator)
            for annotator in self.controler.annotators
        }

        self.layout = pn.Row(
            self.sidebar,
            pn.Column(
            self.export_info,
            pn.Row(
                *self.cards.values()
            )
        ))

    def begin(self):

        for annotator in self.controler.annotators:

            card = self.cards[annotator]

            card.switch_status("annotation")

            tic = time.time()
            self.controler.train(annotator,export=True,save=True)
            toc = time.time()

            card.switch_status("done", duration=toc - tic)

        logger.info("Annotated!")

        self.sidebar.enable_next()