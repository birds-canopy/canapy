# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
import panel as pn
from ..helpers import SubDash, SideBar, _IMAGES_DIR

WELCOME_TEXT = """
**Welcome to Canapy** - a user friendly auto annotator for animal vocalizations.
Here you can modify and correct your annotated-by-hand dataset to train models to auto annotate large datasets.
...

📦 [GitHub Repository](https://github.com/birds-canopy/canapy/tree/main)

🧠 [Our team : Mnemosyne INRIA](https://team.inria.fr/mnemosyne/fr/)
"""

POWERED_BY_HTML = """
<div style="
    display: flex;
    align-items: center;
    gap: 10px;
    padding-top: 14px;
    margin-top: 14px;
    border-top: 1px solid #e2e8f0;
    font-size: 12px;
    color: #94a3b8;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
">
    <span style="letter-spacing: 0.05em; text-transform: uppercase; font-weight: 600;">Powered by</span>
    <a href="https://reservoirpy.readthedocs.io/" target="_blank" style="
        display: inline-flex; align-items: center; gap: 5px;
        background: #eff6ff; color: #3b82f6;
        padding: 3px 10px; border-radius: 99px;
        font-weight: 600; font-size: 12px;
        text-decoration: none; border: 1px solid #bfdbfe;
        transition: background 0.15s;
    ">⚡ ReservoirPy</a>
    <a href="https://crowsetta.readthedocs.io/" target="_blank" style="
        display: inline-flex; align-items: center; gap: 5px;
        background: #f0fdf4; color: #16a34a;
        padding: 3px 10px; border-radius: 99px;
        font-weight: 600; font-size: 12px;
        text-decoration: none; border: 1px solid #bbf7d0;
        transition: background 0.15s;
    ">🐦 Crowsetta</a>
</div>
"""

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
    margin-bottom: 16px;
    border: 1px solid #e2e8f0;
}
.home-welcome {
    background: #ffffff;
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 16px;
    border: 1px solid #e2e8f0;
}
"""


def _collapsible_section(title, content_md, collapsed=True):
    toggle_btn = pn.widgets.Button(
        name=title,
        button_type="light",
        sizing_mode="stretch_width",
        stylesheets=[f"""
            :host button {{
                text-align: left;
                font-size: 15px;
                font-weight: 700;
                color: #1e293b;
                background: transparent;
                border: none;
                padding: 0;
                cursor: pointer;
            }}
            :host button::before {{
                content: "{'▶' if collapsed else '▼'}";
                font-size: 11px;
                margin-right: 8px;
                color: #6b7280;
            }}
        """],
    )
    content = pn.Column(
        pn.pane.Markdown(content_md, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
        visible=not collapsed,
        margin=(12, 0, 0, 0),
    )

    OPEN_CSS = """
        :host button { text-align: left; font-size: 15px; font-weight: 700;
            color: #1e293b; background: transparent; border: none; padding: 0; cursor: pointer; }
        :host button::before { content: "▼"; font-size: 11px; margin-right: 8px; color: #6b7280; }
    """
    CLOSED_CSS = """
        :host button { text-align: left; font-size: 15px; font-weight: 700;
            color: #1e293b; background: transparent; border: none; padding: 0; cursor: pointer; }
        :host button::before { content: "▶"; font-size: 11px; margin-right: 8px; color: #6b7280; }
    """

    def on_toggle(event):
        content.visible = not content.visible
        toggle_btn.stylesheets = [OPEN_CSS if content.visible else CLOSED_CSS]

    toggle_btn.on_click(on_toggle)

    return pn.Column(
        toggle_btn,
        content,
        sizing_mode="stretch_width",
        css_classes=["home-section"],
    )


class HomeDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)

        pn.config.raw_css.append(HOME_CSS)

        self.sidebar = SideBar(self, "Home")

        logo = pn.pane.PNG(
            str(_IMAGES_DIR / "logo_canapy_detailled.png"),
            height=250,
            align="center",
            margin=(40, 0, 50, 0),
        )

        welcome_section = pn.Column(
            pn.pane.Markdown(WELCOME_TEXT, sizing_mode="stretch_width"),
            pn.pane.HTML(POWERED_BY_HTML, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
            css_classes=["home-welcome"],
        )

        tutorial_section = _collapsible_section("Tutorial", TUTORIAL_TEXT, collapsed=True)
        faq_section = _collapsible_section("FAQ", FAQ_TEXT, collapsed=False)

        main_content = pn.Column(
            logo,
            welcome_section,
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
