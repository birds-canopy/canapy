import panel as pn
from ..helpers import SubDash, SideBar

TUTORIAL_TEXT = """
*Welcome to Canapy* - a user friendly auto annotator for animal vocalizations. 
Here you can .....
- Décrire le fonctionnement rapidement (syn-nsyn)
Schéma ? (utiliser celui qu'on utilisera pour l'article)
- Comment annoter automatiquement (entrainer sur un dataset du meme animal/ type d'animal si pas de variété inter individu)
...
"""

FAQ_TEXT = """
*Question 1*
réponse 1
*Question 2*
réponse 2
...
"""

HOME_CSS = """
:host {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.home-section {
    background: #f8fafc;
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 24px;
    border: 1px solid #e2e8f0;
}
.home-section h2 {
    font-size: 20px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 12px;
}
"""


class HomeDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        pn.config.raw_css.append(HOME_CSS)

        self.sidebar = SideBar(self, "Home")

        logo = pn.pane.PNG(
            "images/Logo_canapy.png",
            height=200,
            align="center",
            margin=(40, 0, 50, 0),
        )

        tutorial_section = pn.Column(
            pn.pane.Markdown("## Tutorial"),
            pn.pane.Markdown(TUTORIAL_TEXT, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
            css_classes=["home-section"],
        )

        faq_section = pn.Column(
            pn.pane.Markdown("## FAQ"),
            pn.pane.Markdown(FAQ_TEXT, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
            css_classes=["home-section"],
        )

        main_content = pn.Column(
            logo,
            tutorial_section,
            faq_section,
            sizing_mode="stretch_width",
            styles={"max-width": "800px", "margin": "0 auto"},
        )

        self.layout = pn.Row(
            self.sidebar,
            pn.Column(
                pn.Spacer(height=20),
                main_content,
                pn.Spacer(height=40),
                sizing_mode="stretch_width",
                height_policy="max",
                styles={"padding": "0 40px", "overflow-y": "auto", "height": "100%"},
            ),
            sizing_mode="stretch_both",
            margin=0,
        )
