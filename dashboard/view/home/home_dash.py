import logging
import os
import panel as pn

pn.extension()

logger = logging.getLogger("canapy-dashboard")


class SideBar(pn.viewable.Viewer):
    def __init__(self, parent, title):
        super().__init__()
        self.parent = parent
        self.title = title

        self.quit_btn = pn.widgets.Button(
            name="Quit",
            button_type="danger",
            icon="square-rounded-x",
        )
        self.quit_btn.on_click(self.on_click_stop)

        self.layout = pn.Column(
            pn.pane.Markdown(f"## {self.title}", align='center'),
            self.quit_btn,
            width=150,
            sizing_mode="stretch_height",
            styles={"background": "WhiteSmoke"}
        )

    def on_click_stop(self, events):
        confirm_script = """
        <script>
            if (confirm("Are you sure you want to stop the server and close this tab?")) {
                window.close();
            }
        </script>
        """
        self.layout.append(pn.pane.HTML(confirm_script))

    def __panel__(self):
        return self.layout


class HomeDashboard(pn.viewable.Viewer):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent

        current_directory = os.path.dirname(os.path.abspath(__file__))
        relative_path = os.path.join(current_directory, '..', '..', '..', 'Logo_canapy.png')
        logo_path = os.path.normpath(relative_path)
        logo = pn.pane.Image(logo_path, width=200, align='center')

        self.sidebar = SideBar(self, "Home")

        self.title = pn.pane.Markdown(
            """
            <h1 style="text-align:center; font-size:80px">Canapy</h1>
            """,
            align='center'
        )

        text_train = """
        **Here, you can upload your annotations and audio files to train a model of your choice on your data.** 
        """

        text_annotate = """
        **There, you can upload a trained model and some audio files to automatically annotate these audio files.**
        """

        pn.config.raw_css.append("""
        .bk-btn-primary,
        .bk-btn-danger {
            font-weight: bold;
            font-size: 18px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }
        """)

        self.train_btn = pn.widgets.Button(button_type='primary', name="Training",
                                           width=150, height=75, margin=(0, 25, 0, 0))
        self.train_btn.on_click(self.on_click_train)

        self.annotate_btn = pn.widgets.Button(button_type='primary', name='Annotate',
                                              width=150, height=75, margin=(0, 0, 0, 25))
        self.annotate_btn.on_click(self.on_click_annotate)

        self.layout = pn.Row(
            self.sidebar,
            pn.Column(
                pn.Row(
                    self.title,
                    logo,
                    align='center'
                ),
                pn.Row(
                    self.train_btn,
                    self.annotate_btn,
                    align='center'
                ),
                pn.Row(
                    pn.Column(
                        pn.pane.Markdown("# Training", align='center'),
                        pn.layout.Divider(),
                        text_train,
                        styles=dict(background='whitesmoke'),
                        width=750
                    ),
                    pn.Column(
                        pn.pane.Markdown("# Annotate", align='center'),
                        pn.layout.Divider(),
                        text_annotate,
                        styles=dict(background='whitesmoke'),
                        margin=(0, 0, 0, 50),
                        width=750
                    ),
                    align='center',
                    margin=(100, 0, 0, 0)
                ),
                align='center',
                sizing_mode='stretch_width',
            ),
            align='center'
        )

    def on_click_train(self, event):
        self.parent.controler.next_step(to="load_train")
        print("Train clicked")

    def on_click_annotate(self, event):
        self.parent.controler.next_step(to="load_annotate")
        print("Annotate clicked")

    def __panel__(self):
        return self.layout


if __name__ == "__main__":
    dashboard = HomeDashboard()
    pn.serve(dashboard.layout)