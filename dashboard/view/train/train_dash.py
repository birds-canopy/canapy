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

        self.layout = pn.Row(self.sidebar, self.traindash, self.annotdash)


class TrainerDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        self.train_btn = pn.widgets.Button(name="Start training")
        self.train_btn.on_click(self.on_click_train)

        self.cards = {
            annotator: Card(self,annotator)
            for annotator in self.controler.annotators
            if annotator != "ensemble"
        }

        self.layout = pn.Column(
            pn.Row(
            *self.cards.values()
            ),
            self.train_btn,
        )

    def on_click_train(self, events):

        self.train_btn.disabled = True

        for annotator in self.controler.annotators:

            if annotator == "ensemble" :
                continue

            card = self.cards[annotator]

            card.switch_status("training")

            tic = time.time()
            self.controler.train(annotator)
            toc = time.time()

            card.switch_status("done", duration=toc - tic)

        logger.info("Trained!")

        self.parent.annotdash.begin()
class Card(SubDash):
    def __init__(self,parent,model_name):
        super(Card, self).__init__(parent)

        self.status = pn.pane.HTML("<h2>Idle</h2>")

        self.indicator = pn.indicators.LoadingSpinner(
            value=False, width=100, height=100
        )
        self.layout = pn.Column(
            pn.pane.HTML(model_name), self.indicator, self.status
        )

    def switch_status(self, status, duration=0):
        if status == "training":
            self.status.object = "<h2>Training...</h2>"
            self.status.style = {"color": "blue"}
            self.indicator.value=True

        if status == "annotation":
            self.status.object = "<h2>Annotating...</h2>"
            self.status.style = {"color": "blue"}
            self.indicator.value=True

        if status == "done":
            self.status.object = f"<h2>Done !</h2> in {round(duration, 2)} sec."
            self.status.style = {"color": "green"}
            self.indicator.value=False
class AnnotatorDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        self.cards = {
            annotator: Card(self, annotator)
            for annotator in self.controler.annotators
        }

        self.layout = pn.Column(
            pn.Row(
                *self.cards.values()
            )
        )

    def begin(self):

        for annotator in self.controler.annotators:

            card = self.cards[annotator]

            card.switch_status("annotation")

            tic = time.time()
            self.controler.annotate(annotator,split="train")
            self.controler.annotate(annotator,split="test")
            toc = time.time()

            card.switch_status("done", duration=toc - tic)

        logger.info("Annotated!")

        self.parent.sidebar.enable_next()
