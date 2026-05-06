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
**Can Canapy automatically annotate my recordings without any manual work?**

Not entirely: Canapy needs a small set of hand-labeled recordings to learn from; roughly 10 to 60 minutes depending on the species. Once trained, it can automatically annotate hundreds of hours of new audio files without any further human input.

---

**I have recordings from multiple species. Do I need to start from scratch each time?**

Yes, one model is trained per species (or per individual if vocalizations vary greatly). However, Canapy comes with pre-configured presets for several well-studied species: canary, Bengalese finch, zebra finch, mouse, infant marmoset, and soon more. If your species is on the list, you start with a solid baseline right away.

---

**If the model makes mistakes, do I have to redo everything?**

No. After each training run, Canapy shows you where the model went wrong: which vocalizations it confuses with which. You can then correct the mislabeled examples directly in the interface and retrain. In practice, maximum 3–4 iterations of this loop should be enough to reach good performance. No need to re-annotate from scratch.

---

**Do I need to know how to code to use Canapy?**

No. Canapy is driven by an interactive dashboard that opens in your web browser, just like a website. A single command in a terminal is enough to launch it, and all steps — loading data, cleaning, training, annotating, and exporting — are done by clicking through the interface.
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
            "images/logo_canapy_detailled.png",
            height=250,
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
