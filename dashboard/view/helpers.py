import panel as pn
from ..controler import Controler

SIDEBAR_CSS = """
.sidebar-container {
    background-color: #f8fafc; /* Gris très clair */
    border-right: 1px solid #e2e8f0;
    padding: 20px 10px;
    height: 100%;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.sidebar-title h1 {
    color: #475569;
    font-size: 18px !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    text-align: center;
    margin-bottom: 30px;
}
/* Style des boutons de navigation */
.nav-btn button.bk-btn {
    background-color: transparent !important;
    color: #64748b !important;
    box-shadow: none !important;
    border: 1px solid transparent !important;
    text-align: left !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding-left: 15px !important;
    transition: all 0.2s ease;
}
.nav-btn button.bk-btn:hover:not(:disabled) {
    background-color: #e2e8f0 !important;
    color: #1e293b !important;
    border-radius: 6px;
}
.nav-btn button.bk-btn:disabled {
    color: #cbd5e1 !important;
    cursor: not-allowed;
}
/* Bouton Quit spécifique (rouge au survol) */
.quit-btn button.bk-btn:hover {
    color: #ef4444 !important;
    background-color: #fee2e2 !important;
}
/* Séparateur visuel */
.nav-divider {
    border-top: 1px solid #e2e8f0;
    margin: 15px 0;
}
"""

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
    def __init__(self, parent, title, disabled=True):
        super().__init__(parent)
        
        pn.config.raw_css.append(SIDEBAR_CSS)

        self.title = title

        is_home_page = (self.title == "Home")
        
        self.home_btn = pn.widgets.Button(
            name="Home", 
            button_type="primary", 
            disabled=is_home_page, 
            align="center",
            sizing_mode="stretch_width",
            css_classes=['nav-btn']
        )
        self.home_btn.on_click(self.on_click_home)

        self.next_btn = pn.widgets.Button(
            name="Next Step  →",
            disabled=disabled,
            align="center",
            sizing_mode="stretch_width",
            css_classes=['nav-btn']
        )
        self.next_btn.on_click(self.on_click_next)

        self.quit_btn = pn.widgets.Button(
            name="✕  Quit App",
            button_type="warning",
            align="center",
            sizing_mode="stretch_width",
            css_classes=['nav-btn', 'quit-btn'] 
        )
        self.quit_btn.on_click(self.on_click_stop)

        widgets_list = [pn.Spacer(height=20)]

        if self.title == "Home":
            widgets_list.extend([
                pn.Spacer(height=20),
                self.quit_btn
            ])
            
        else:
            if not self.controler.is_ready:
                self.next_btn.disabled = True
            
            if self.title == "Preprocessing":
                #modif 27/01 Axel Preprocess en home
                if self.controler.home_path == "edit":
                    self.next_btn.name = "Export Changes"
                    self.next_btn.stylesheets = [""" 
                        button.bk-btn { 
                            background-color: #dcfce7 !important; 
                            color: #166534 !important; 
                            border: 1px solid #86efac !important;
                            font-weight: bold !important;
                        } 
                    """]
                self.enable_next()

            if self.title == "eval":
                self.export_btn = pn.widgets.Button(
                    name="Export", 
                    button_type="primary",
                    sizing_mode="stretch_width",
                    css_classes=['nav-btn']
                )
                self.export_btn.on_click(self.on_click_export)
                widgets_list.append(self.export_btn)
                self.enable_next()

            if self.title == "export":
                self.next_btn.name = "End Workflow"
                self.next_btn.button_type = "success"
                self.next_btn.disabled = False
                self.next_btn.on_click(self.on_click_stop) 
                self.home_btn.disabled = False
            
            if self.title != "Load Data":
                widgets_list.append(self.next_btn)
            
            widgets_list.append(pn.pane.HTML("<div class='nav-divider'></div>"))
            widgets_list.append(self.home_btn)
            
            widgets_list.append(pn.Spacer(height=20))
            widgets_list.append(self.quit_btn)

        self.layout = pn.Column(
            *widgets_list,
            width=220, 
            sizing_mode="stretch_height",
            css_classes=['sidebar-container'] 
        )

    def on_click_next(self, events):
        self.controler.next_step()

    def on_click_export(self, events):
        self.controler.next_step(export=True)
        self.home_btn.disabled = False

    def on_click_home(self, events):
        self.controler.home_path = None
        self.controler._step = "home"
        self.controler.dashboard.switch_panel()

    def on_click_stop(self, events):
        self.layout.append(
            pn.pane.Alert(
                "Server closed. You can close this tab.",
                alert_type="success",
            )
        )
        self.controler.stop_app()

    def enable_next(self):
        self.next_btn.disabled = False
        if self.controler.home_path != "edit":
            self.next_btn.stylesheets = [""" 
                button.bk-btn { 
                    background-color: #dcfce7 !important; 
                    color: #166534 !important; 
                    border: 1px solid #86efac !important;
                } 
            """]