# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
import os
from pathlib import Path

import panel as pn
from ..helpers import SubDash, SideBar, _IMAGES_DIR, pick_directory

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
Canapy learns to annotate animal vocalizations from a **small set of hand-labeled
recordings** (≈ 10–60 min), then annotates hours of new audio for you. It trains
[Reservoir Computing](https://reservoirpy.readthedocs.io/) models (Echo State Networks).

**The workflow** follows the sidebar, top to bottom:

1. **Load data** — point Canapy to your audio (mono WAV) and annotations
   (*marron1csv* CSV: `wave, start, end, syll`).
2. **Settings** *(optional)* — load a species preset (canary, finch, mouse…) or a
   config you exported, or tune the parameters by hand.
3. **Preprocess** — clean the dataset: merge acoustically similar classes, correct
   mislabeled samples, trim silences.
4. **Train** — three models are trained: `syn` (uses song context), `nsyn`
   (context-free) and `ensemble` (majority vote). An automatic hyperparameter
   search is available beforehand if needed.
5. **Eval** — read the confusion matrix and per-class metrics, fix the mistakes,
   and retrain. **3–4 Train → Eval iterations** are usually enough.
6. **Export** — save the trained models (and optionally your config) once you're happy.
7. **Annotate** — load unlabeled audio + your exported models, run the annotation,
   and export the results.

**Two ways to start**
- *Train a new model* → begin at **Load data** and follow the pipeline.
- *Already have a model?* → load it in **Load data** and jump straight to **Annotate**.

One model is trained per species (or per individual if vocalizations vary a lot).

---
📖 Need more detail? See the [extended documentation](https://github.com/birds-canopy/canapy/blob/main/README_extended.md).
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
.home-workdir-callout {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border: 2px solid #f59e0b;
    border-radius: 14px;
    padding: 26px 30px;
    margin-bottom: 20px;
    box-shadow: 0 6px 20px rgba(245, 158, 11, 0.25);
    animation: workdir-pulse 2.2s ease-in-out infinite;
}
@keyframes workdir-pulse {
    0%, 100% { box-shadow: 0 6px 20px rgba(245, 158, 11, 0.25); }
    50%      { box-shadow: 0 6px 28px rgba(245, 158, 11, 0.55); }
}
.home-workdir-set {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #22c55e;
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 16px;
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

        workdir_section = self._build_workdir_section()

        tutorial_section = _collapsible_section("Tutorial", TUTORIAL_TEXT, collapsed=True)
        faq_section = _collapsible_section("FAQ", FAQ_TEXT, collapsed=False)

        main_content = pn.Column(
            logo,
            workdir_section,
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

    def _build_workdir_section(self):
        """Working-directory selector. Until a directory is validated, the whole
        sidebar stays locked (see SideBar gating). Once set, all Canapy data —
        output/, the config/ folder (presets + exported configs), etc. — lives
        inside this directory, which is reachable even on a pip install."""
        c = self.controler

        if c.working_directory_set:
            change_btn = pn.widgets.Button(
                name="Change", width=90, height=32,
                stylesheets=[
                    "button.bk-btn { background:#ffffff !important; color:#15803d !important; "
                    "border:1px solid #86efac !important; font-weight:600 !important; "
                    "font-size:12px !important; border-radius:7px !important; box-shadow:none !important; } "
                    "button.bk-btn:hover { background:#f0fdf4 !important; }"
                ],
                align="center",
            )
            change_btn.on_click(self._on_change_workdir)
            return pn.Row(
                pn.pane.HTML(
                    "<div style='display:flex;align-items:center;gap:12px;'>"
                    "<span style='font-size:20px;'>📂</span>"
                    "<div style='line-height:1.35;'>"
                    "<span style='font-size:11px;font-weight:700;text-transform:uppercase;"
                    "letter-spacing:0.05em;color:#16a34a;'>✓ Working directory</span><br>"
                    f"<code style='font-size:13px;color:#374151;'>{c.working_directory}</code>"
                    "</div></div>",
                    sizing_mode="stretch_width",
                    align="center",
                ),
                change_btn,
                css_classes=["home-workdir-set"],
                sizing_mode="stretch_width",
            )

        # Pre-fill with the directory Canapy was launched from.
        prefill = str(Path(os.getcwd()))

        self._workdir_input = pn.widgets.TextInput(
            value=prefill,
            placeholder="Path to your working directory...",
            sizing_mode="stretch_width",
            stylesheets=[
                "input { border:2px solid #f59e0b !important; border-radius:8px !important; "
                "font-size:14px !important; padding:8px 10px !important; }"
            ],
        )
        browse_btn = pn.widgets.Button(
            name="📁 Browse…", button_type="default", width=120, height=42,
        )
        browse_btn.on_click(self._on_browse_workdir)
        validate_btn = pn.widgets.Button(
            name="Validate →", width=140, height=42,
            stylesheets=[
                "button.bk-btn { background:#f59e0b !important; color:#1f2937 !important; "
                "border:none !important; font-weight:800 !important; font-size:15px !important; "
                "border-radius:8px !important; box-shadow:0 2px 6px rgba(245,158,11,0.4) !important; } "
                "button.bk-btn:hover { background:#d97706 !important; }"
            ],
        )
        validate_btn.on_click(self._on_validate_workdir)

        self._workdir_msg = pn.pane.HTML("", sizing_mode="stretch_width")

        return pn.Column(
            pn.pane.HTML(
                "<div style='display:flex;align-items:center;gap:10px;'>"
                "<span style='font-size:13px;font-weight:800;text-transform:uppercase;"
                "letter-spacing:0.06em;color:#b45309;background:#fde68a;"
                "padding:4px 10px;border-radius:999px;'>🔒 Action required</span></div>"
                "<div style='font-size:22px;font-weight:800;color:#78350f;margin-top:12px;'>"
                "Choose a working directory</div>"
                "<div style='font-size:14px;color:#92400e;margin-top:8px;line-height:1.5;'>"
                "Canapy stores everything here: "
                "trained models, exported annotations, and the "
                "<code>config/</code> folder with species presets and your exported "
                "configurations. Pick a folder you can easily find again.</div>"
            ),
            pn.Spacer(height=16),
            pn.Row(self._workdir_input, browse_btn, validate_btn, sizing_mode="stretch_width"),
            self._workdir_msg,
            css_classes=["home-workdir-callout"],
            sizing_mode="stretch_width",
        )

    def _on_browse_workdir(self, _):
        initial = self._workdir_input.value.strip() or os.getcwd()
        directory = pick_directory("Select Working Directory", initialdir=initial)
        if directory:
            self._workdir_input.value = directory

    def _on_validate_workdir(self, _):
        path = self._workdir_input.value.strip()
        if not path:
            self._workdir_msg.object = (
                "<span style='font-size:12px;color:#dc2626;'>Please enter a directory path.</span>"
            )
            return
        try:
            self.controler.set_working_directory(path)
        except OSError as e:
            self._workdir_msg.object = (
                f"<span style='font-size:12px;color:#dc2626;'>Could not use this directory: {e}</span>"
            )
            return
        # Re-render Home: the sidebar unlocks now that a working directory is set.
        self.controler.dashboard.switch_panel()

    def _on_change_workdir(self, _):
        self.controler.working_directory = None
        self.controler.dashboard.switch_panel()
