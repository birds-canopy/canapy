import panel as pn

from dashboard.controler import Controler


class SubDash(pn.viewable.Viewer):
    def __init__(self, parent, **kwargs):
        super().__init__(**kwargs)
        self.parent = parent
        self.controler: Controler = parent.controler
        self.layout = pn.Spacer()

    def __panel__(self):
        return self.layout


class Registry(object):
    instances = list()

    @classmethod
    def clean_all(cls):
        for instance in cls.instances:
            instance.clean()

    def __init__(self):
        self.data = dict()
        Registry.instances.append(self)

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def get(self, item):
        return self.data.get(item)

    def popitem(self):
        return self.data.popitem()

    def clean(self):
        self.data = dict()


class SideBar(SubDash):
    def __init__(self, parent, title):
        super().__init__(parent)

        self.title = title

        self.title_pane = pn.pane.HTML(f"<h1>{self.title.capitalize()}</h1>")

        self.next_btn = pn.widgets.Button(name="Next step", disabled=True)
        self.next_btn.on_click(self.on_click_next)

        self.quit_btn = pn.widgets.Button(
            name="Quit",
            button_type="warning",
            icon="square-rounded-x",
        )
        self.quit_btn.on_click(self.on_click_stop)

        self.layout = pn.Column(
            self.title_pane,
            self.next_btn,
            pn.Spacer(height=100),
            self.quit_btn,
            width=100,
            sizing_mode="stretch_height",
            styles={"background": "WhiteSmoke"},
        )

        if self.title == "eval":
            self.export_btn = pn.widgets.Button(name="Export", button_type="primary")
            self.export_btn.on_click(self.on_click_export)

            self.layout = pn.Column(
                self.title_pane,
                self.next_btn,
                self.export_btn,
                pn.Spacer(height=100),
                self.quit_btn,
                width=100,
                sizing_mode="stretch_height",
                styles={"background": "WhiteSmoke"},
            )

            self.enable_next()

        if self.title == "export":
            self.next_btn = pn.widgets.Button(
                name="End", disabled=True, button_type="success"
            )
            self.next_btn.on_click(self.on_click_stop)

            self.layout = pn.Column(
                self.title_pane,
                self.next_btn,
                pn.Spacer(height=100),
                self.quit_btn,
                width=100,
                sizing_mode="stretch_height",
                styles={"background": "WhiteSmoke"},
            )

    def on_click_next(self, events):
        self.controler.next_step()

    def on_click_export(self, events):
        self.controler.next_step(export=True)

    def on_click_stop(self, events):
        self.layout.append(
            pn.pane.Alert(
                "Server closed ! You can now close the browser tab.",
                alert_type="success",
            )
        )
        self.controler.stop_app()

    def enable_next(self):
        self.next_btn.disabled = False
        self.next_btn.button_type = "success"
