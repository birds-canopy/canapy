from typing import List

import panel as pn

from canapy.utils.tempstorage import close_tempfiles

#from ..controler import Controler

class View(pn.viewable.Viewer):
    """Helper class to quickly define and control Canapy UI elements."""
    def __init__(self, parent: "View", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent
        self.controler: Controler = parent.controler
        self.sidebar = parent.sidebar

        self.layout = pn.Spacer()

    def __panel__(self):
        return self.layout


class SideBar(pn.viewable.Viewer):
    """Application side bar, present on  all UIs."""

    _app: pn.viewable.Viewable = None
    """Main application."""

    def __init__(self, app: pn.viewable.Viewable, title: str = "Home"):
        super().__init__()

        pn.config.raw_css.append("""
        .bk-btn-warning,
        .bk-btn-danger{
            font-weight: bold;
            font-size: 18px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }
        .GreyCard {
            border-radius: 15px;
            background-color: #F7F7F7;
            padding: 10px;
        }
        """)

        self._app = app

        self.title_pane = pn.pane.HTML(f"<h1>{title.capitalize()}</h1>",align='center')

        self.quit_btn = pn.widgets.Button(
            name="Quit",
            button_type="danger",
            width=90,align='center'
        )
        self.quit_btn.on_click(self.on_click_stop)

        self.back_btn = pn.widgets.Button(
            name="Back",
            button_type="warning",
            width=90,align='center'
        )
        self.back_btn.on_click(self.on_click_back)

        self.layout = pn.Column(
            self.title_pane,
            self.quit_btn,
            width=150,
            sizing_mode="stretch_height",
            styles={"background": "WhiteSmoke"},
        )

        self._controls = []

    def __panel__(self):
        return self.layout
    
    def change_title(self, title: str):
        """Update sidebar title."""
        self.title_pane.object = f"<h1>{title.capitalize()}</h1>"
        if title == "Annotate" or title == "Upload":
            self.layout.insert(1, self.back_btn)


    def clear_controls(self):
        """Remove all controls from sidebar (except quit button)."""
        for control in self._controls:
            self.layout.remove(control)

    def add_controls(self, controls: List[pn.viewable.Viewable]):
        """Add a list of controls to sidebar (buttons, widgets...)"""
        for control in reversed(controls):
            self.layout.insert(1, control)

    def on_click_back(self, events):
        """Get to the previous dashboard."""
        self.controler.next_step(to_step="home")


    def on_click_stop(self, events):
        """Stop application."""
        confirm_script = """
        <script>
            if (confirm("Are you sure you want to stop the server and close this tab?")) {
                window.close();
            }
        </script>
        """
        self.layout.append(pn.pane.HTML(confirm_script))
        close_tempfiles()
        self._app.stop()
