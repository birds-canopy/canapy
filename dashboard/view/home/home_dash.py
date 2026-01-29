# Author: Axel Arnaud at xx/xx/xxxx <axel.arnaud<at>inria.fr>
# Licence: MIT License
# Copyright: Axel Arnaud
import panel as pn
from ..helpers import SubDash, SideBar

SCALED_CSS = """
:host {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.dashboard-container {
    max-width: 1000px;
    margin: 0 auto;
}
.section-header {
    font-size: 18px;
    font-weight: 600;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 20px;
    border-bottom: 2px solid #f1f3f5;
    padding-bottom: 10px;
}
button.bk-btn {
    font-size: 16px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}
.bk-alert {
    font-size: 16px !important;
}
"""

class HomeDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)
        
        pn.config.raw_css.append(SCALED_CSS)

        self.sidebar = SideBar(self, "Home")


        self.has_data = self.controler.is_ready
        

        c_path = getattr(self.controler, 'config_path', None)
        m_root = getattr(self.controler, 'model_root', None)
        self.has_config = (c_path is not None) or (m_root is not None)


        self.logo = pn.pane.PNG(
            "images/Logo_canapy.png",
            height=120,
            align="center"
        )

        if self.has_data and self.has_config:
            status_text = "System Fully Ready: Data & Configuration loaded."
            status_type = "success"
        elif self.has_data:
            status_text = "Partial Ready: Data loaded. (Training enabled, Annotation disabled due to missing Config)"
            status_type = "warning"
        else:
            status_text = "System Standby: No data loaded."
            status_type = "danger"
            
        self.status = pn.pane.Alert(
            status_text,
            alert_type=status_type,
            sizing_mode="stretch_width",
        )

        self.btn_load = pn.widgets.Button(
            name="Load Dataset",
            button_type="default",
            sizing_mode="stretch_width",
            height=60,
        )
        self.btn_load.on_click(self.load_data)

        self.btn_preprocess = pn.widgets.Button(
            name="Edit Dataset",
            button_type="primary",
            sizing_mode="stretch_width",
            height=100,
            disabled=not self.has_data,
        )
        self.btn_preprocess.on_click(self.go_preprocess_edit)

        self.btn_train = pn.widgets.Button(
            name="Train on labeled data",
            button_type="success",
            sizing_mode="stretch_width",
            height=100,
            disabled=not self.has_data,
        )
        self.btn_train.on_click(self.go_train)

        label_btn_name = "Annotate unlabeled data"
        if self.has_data and not self.has_config:
            label_btn_name += " (Missing Config)"

        self.btn_labelize = pn.widgets.Button(
            name=label_btn_name,
            button_type="success",
            sizing_mode="stretch_width",
            height=100,
            disabled=not (self.has_data and self.has_config),
        )
        self.btn_labelize.on_click(self.go_labelize)


        workflows_grid = pn.GridBox(
            self.btn_preprocess,
            self.btn_train,
            self.btn_labelize,
            ncols=3,
            sizing_mode="stretch_width",
            style={'gap': '20px'}
        )

        main_content = pn.Column(
            pn.Row(self.logo, align="center"),
            pn.Spacer(height=40),
            
            self.status,
            pn.Spacer(height=40),
            
            pn.pane.Markdown("<div class='section-header'>Data Management</div>"),
            self.btn_load,
            pn.Spacer(height=40),
            
            pn.pane.Markdown("<div class='section-header'>Workflows</div>"),
            workflows_grid,
            
            css_classes=['dashboard-container'],
            sizing_mode="stretch_width",
        )

        self.layout = pn.Row(
            self.sidebar,
            pn.Spacer(width=50),
            pn.Column(
                pn.Spacer(height=60),
                main_content, 
                pn.Spacer(height=60),
                sizing_mode="stretch_width"
            ),
            pn.Spacer(width=50),
            sizing_mode="stretch_both",
            background="#ffffff"
        )

    def load_data(self, _):
        self.parent.switch_to_load_data()

    def go_preprocess_edit(self, _):
        """
        Nouveau handler pour le mode 'Quick Edit'.
        On définit un home_path spécifique 'edit' pour signaler au controller 
        que l'on veut juste éditer et exporter, pas suivre le pipeline d'entraînement.
        """
        if not self.has_data:
            self.status.object = "Please load data first (-d)"
            self.status.alert_type = "danger"
            return
        
        self.controler.home_path = "edit" 
        self.controler.load_page("preprocess")

    def go_train(self, _):
        if not self.has_data:
            self.status.object = "Please load data first (-d)"
            self.status.alert_type = "danger"
            return
        self.controler.home_path = "preprocess"
        self.controler.next_step()

    def go_labelize(self, _):
        if not self.has_data:
            self.status.object = "Please load data first (-d)"
            self.status.alert_type = "danger"
            return
        if not self.has_config:
            self.status.object = "Configuration or Model required for annotation (-c)"
            self.status.alert_type = "danger"
            return
        self.controler.home_path = "annotate"
        self.controler.next_step()