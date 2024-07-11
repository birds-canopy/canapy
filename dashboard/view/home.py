import logging
from pathlib import Path

import panel as pn

from . import View

pn.extension()

logger = logging.getLogger("canapy-dashboard")

LOGO_PATH = Path(__file__).absolute().parent / "assets" / "Logo_canapy.png"

stylesheet = """
.bk-btn-primary,
.bk-btn-danger {
    font-weight: bold;
    font-size: 18px;
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
}
"""


class HomeDashboard(View):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logo = pn.pane.Image(LOGO_PATH, width=200, align="center")

        self.title = pn.pane.Markdown(
            """
            <h1 style="text-align:center; font-size:80px">Canapy</h1>
            """,
            align="center",
        )

        text_train = """
        **Here, you can upload your annotations and audio \
        files to train a model of your choice on your data.** 
        """

        text_annotate = """
        **There, you can upload a trained model and some \
        audio files to automatically annotate these audio files.**
        """

        self.train_btn = pn.widgets.Button(
            button_type="primary",
            name="Training",
            width=150,
            height=75,
            margin=(0, 25, 0, 0),
            stylesheets=[stylesheet]
        )
        self.train_btn.on_click(self.on_click_train)

        self.annotate_btn = pn.widgets.Button(
            button_type="primary",
            name="Annotate",
            width=150,
            height=75,
            margin=(0, 0, 0, 25),
            stylesheets=[stylesheet]
        )
        self.annotate_btn.on_click(self.on_click_annotate)

        self.layout = pn.Column(
                pn.Row(self.title, logo, align="center"),
                pn.Row(self.train_btn, self.annotate_btn, align="center"),
                pn.Row(
                    pn.Column(
                        pn.pane.Markdown("# Training", align="center"),
                        pn.layout.Divider(),
                        text_train,
                        styles=dict(background="whitesmoke"),
                        width=750,
                    ),
                    pn.Column(
                        pn.pane.Markdown("# Annotate", align="center"),
                        pn.layout.Divider(),
                        text_annotate,
                        styles=dict(background="whitesmoke"),
                        margin=(0, 0, 0, 50),
                        width=750,
                    ),
                    align="center",
                    margin=(100, 0, 0, 0),
                ),
                align="center",
                sizing_mode="stretch_width",
            )

    def on_click_train(self, event):
        """Move on to train dashboard."""
        self.controler.next_step(to_step="load_train")
        logger.info("Entering training data upload.")

    def on_click_annotate(self, event):
        """Move on to annotation dashboard."""
        self.controler.next_step(to_step="load_annotate")
        logger.info("Entering unannotated data upload.")


if __name__ == "__main__":
    dashboard = HomeDashboard()
    pn.serve(dashboard.layout)
