# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
import subprocess
import sys
import uuid

import panel as pn
from ..controler import Controler


def custom_tooltip(text, direction="right", width=240, margin=(0, 10), align="center"):
    """Custom HTML tooltip icon. direction='left' opens left of the icon, 'right' opens right."""
    uid = f"ctt-{uuid.uuid4().hex[:8]}"
    pos = (
        "right:calc(100% + 8px); top:50%; transform:translateY(-50%);"
        if direction == "left"
        else "left:calc(100% + 8px); top:50%; transform:translateY(-50%);"
    )
    text_html = text.replace("\n", "<br>")
    html = (
        f'<div class="{uid}" style="position:relative;display:inline-block;cursor:help;">'
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:18px;height:18px;border-radius:50%;background:#e5e7eb;color:#6b7280;'
        f'font-size:11px;font-weight:700;font-family:sans-serif;border:1px solid #d1d5db;">?</span>'
        f'<div style="display:none;position:absolute;{pos}background:#1f2937;color:#f9fafb;'
        f'font-size:12px;font-family:sans-serif;line-height:1.5;padding:8px 10px;border-radius:6px;'
        f'width:{width}px;z-index:9999;box-shadow:0 4px 12px rgba(0,0,0,0.25);white-space:normal;">'
        f'{text_html}</div></div>'
        f'<style>.{uid}:hover > div{{display:block !important;}}</style>'
    )
    return pn.pane.HTML(html, align=align, margin=margin)


def dataset_info_badge(controler):
    df = controler.corpus.dataset
    silence_tag = controler.config.transforms.annots.silence_tag
    total_s = (df["offset_s"] - df["onset_s"]).sum()
    hours = int(total_s // 3600)
    minutes = int((total_s % 3600) // 60)
    seconds = total_s % 60
    if hours > 0:
        dur_str = f"{hours}h {minutes}m {seconds:.0f}s"
    elif minutes > 0:
        dur_str = f"{minutes}m {seconds:.0f}s"
    else:
        dur_str = f"{seconds:.1f}s"
    n_samples = len(df[df["label"] != silence_tag])
    return pn.pane.HTML(
        f"<span style='font-size:12px;color:#6b7280;'>"
        f"<b>{n_samples}</b> samples (excl. silence) &nbsp;·&nbsp; <b>{dur_str}</b> total"
        f"</span>",
        margin=(4, 0, 0, 0),
    )


def pick_directory(title="Select Directory", initialdir=None):
    """Open a folder picker dialog. Uses osascript on macOS to avoid the
    NSWindow-must-be-on-main-thread crash when called from a Panel callback."""
    if sys.platform == "darwin":
        if initialdir:
            script = (
                f'POSIX path of (choose folder with prompt "{title}" '
                f'default location (POSIX file "{initialdir}"))'
            )
        else:
            script = f'POSIX path of (choose folder with prompt "{title}")'
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip().rstrip("/")
        return None
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        directory = filedialog.askdirectory(title=title, initialdir=initialdir or "")
        root.destroy()
        return directory or None


def pick_file(title="Select File", filetypes=None, initialdir=None):
    """Open a file picker dialog. Uses osascript on macOS to avoid the
    NSWindow-must-be-on-main-thread crash when called from a Panel callback."""
    if sys.platform == "darwin":
        if initialdir:
            script = (
                f'POSIX path of (choose file with prompt "{title}" '
                f'default location (POSIX file "{initialdir}"))'
            )
        else:
            script = f'POSIX path of (choose file with prompt "{title}")'
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title=title, filetypes=filetypes or [], initialdir=initialdir or ""
        )
        root.destroy()
        return path or None

PAGE_HEADER_CSS = """
.page-title {
    font-size: 24px;
    font-weight: 700;
    margin: 0;
}
.page-subtitle {
    font-size: 14px;
    color: #6b7280;
    margin: 0;
}
audio {
    max-width: 100%;
    min-width: 0;
    box-sizing: border-box;
}
.sample-card > *, .sample-row-card > * {
    min-width: 0;
    flex-shrink: 1;
}
.sample-card, .sample-row-card {
    overflow: hidden;
}
"""

SIDEBAR_BASE_CSS = """
.sidebar-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 210px;
    height: 100vh;
    overflow-y: hidden;
    z-index: 1000;
    background-color: #f8fafc;
    border-right: 2px solid #e2e8f0;
    padding: 10px 12px;
    box-sizing: border-box;
}
.nav-divider {
    border-top: 1px solid #e2e8f0;
    margin: 8px 0;
}
body {
    padding-left: 210px;
    box-sizing: border-box;
}
"""

_DISABLED_BTN_CSS = """
button.bk-btn {
    background-color: #f1f5f9 !important;
    color: #94a3b8 !important;
    border: 1px solid #e2e8f0 !important;
    cursor: not-allowed !important;
    font-weight: 400 !important;
    font-size: 13px !important;
    border-radius: 6px !important;
    text-align: left !important;
    box-shadow: none !important;
}
"""

_ACCESSIBLE_BTN_CSS = """
button.bk-btn {
    background-color: #f0fdf4 !important;
    color: #166534 !important;
    border: 1px solid #86efac !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border-radius: 6px !important;
    text-align: left !important;
    box-shadow: none !important;
}
button.bk-btn:hover {
    background-color: #dcfce7 !important;
}
"""

_CURRENT_BTN_CSS = """
button.bk-btn {
    background-color: #dcfce7 !important;
    color: #166534 !important;
    border: 2px solid #111827 !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    border-radius: 6px !important;
    text-align: left !important;
    box-shadow: none !important;
}
"""

_NEXT_STEP_BTN_CSS = """
button.bk-btn {
    background-color: #eff6ff !important;
    color: #1d4ed8 !important;
    border: 1px solid #93c5fd !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border-radius: 6px !important;
    text-align: left !important;
    box-shadow: none !important;
}
button.bk-btn:hover {
    background-color: #dbeafe !important;
}
"""

_HOME_BTN_CSS = """
button.bk-btn {
    background-color: #f5f5f5 !important;
    color: #555555 !important;
    border: 1px solid #cccccc !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border-radius: 6px !important;
    text-align: left !important;
    box-shadow: none !important;
}
button.bk-btn:hover {
    background-color: #e8e8e8 !important;
    color: #333333 !important;
}
"""

_QUIT_BTN_CSS = """
button.bk-btn {
    background-color: #fff1f2 !important;
    color: #be123c !important;
    border: 1px solid #fda4af !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border-radius: 6px !important;
    text-align: left !important;
    box-shadow: none !important;
}
button.bk-btn:hover {
    background-color: #ffe4e6 !important;
}
"""

_TITLE_TO_KEY = {
    "home": "home",
    "load data": "loaddata",
    "loaddata": "loaddata",
    "settings": "settings",
    "preprocessing": "preprocess",
    "preprocess": "preprocess",
    "train": "train",
    "fit": "train",
    "eval": "eval",
    "export": "export",
    "annotation": "annotate",
    "annotate": "annotate",
}


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
    def __init__(self, parent, title, disabled=True, **kwargs):
        super().__init__(parent)

        pn.config.raw_css.append(SIDEBAR_BASE_CSS)
        pn.config.raw_css.append(PAGE_HEADER_CSS)

        self.title = title
        c = self.controler
        current_key = _TITLE_TO_KEY.get(title.lower(), title.lower())
        is_home = (current_key == "home")

        has_data = c.is_ready
        fit_done = getattr(c, 'fit_done', False)
        eval_done = getattr(c, 'eval_done', False)
        export_done = getattr(c, 'export_done', False)
        has_model = (getattr(c, 'model_root', None) is not None) or export_done
        in_pipeline = current_key in ("train", "eval", "export")

        accessible = {
            "loaddata":   not in_pipeline,
            "settings":   has_data and not fit_done,
            "preprocess": has_data,
            "train":      has_data,
            "eval":       fit_done,
            "export":     eval_done,
            "annotate":   has_model,
        }

        if is_home:
            accessible["settings"] = has_data
            accessible["eval"] = False
            accessible["export"] = False

        suggest_eval         = fit_done and not eval_done and not is_home
        suggest_train_export = eval_done and not export_done and not is_home

        def _style(page_key):
            if not accessible[page_key]:
                return _DISABLED_BTN_CSS
            if current_key == page_key:
                return _CURRENT_BTN_CSS
            if (page_key == "eval" and suggest_eval) or \
               (page_key in ("train", "export") and suggest_train_export):
                return _NEXT_STEP_BTN_CSS
            return _ACCESSIBLE_BTN_CSS

        def _btn(label, page_key):
            return pn.widgets.Button(
                name=label,
                disabled=not accessible[page_key],
                sizing_mode="stretch_width",
                stylesheets=[_style(page_key)],
                align="center",
            )

        btn_loaddata   = _btn("Load Data",     "loaddata")
        btn_settings   = _btn("Settings",      "settings")
        btn_preprocess = _btn("Preprocessing", "preprocess")
        btn_train      = _btn("Fit (& re-Fit)",           "train")
        btn_eval       = _btn("Eval (& re-Process)",          "eval")
        btn_export     = _btn("Export",        "export")
        btn_annotate   = _btn("Annotate",      "annotate")

        self._btn_loaddata  = btn_loaddata
        self._btn_settings  = btn_settings
        self._btn_preprocess = btn_preprocess
        self._btn_train    = btn_train
        self._btn_eval     = btn_eval
        self._btn_export   = btn_export
        self._btn_annotate = btn_annotate

        def _nav_loaddata(e):
            self.controler._step = "loaddata"
            self.controler.dashboard.switch_panel()

        def _nav_settings(e):
            self.controler.go_settings()

        def _nav_preprocess(e):
            self.controler.load_page("preprocess")

        def _show_spinner():
            self._spinner.value = True
            self._spinner.visible = True

        def _nav_fit(e):
            if getattr(self.controler, "_audio_params_dirty", False):
                self.controler._clear_audio_cache()
                self.controler._audio_params_dirty = False
            step = self.controler.step
            if step == "preprocess":
                _show_spinner()
                self.controler.next_step()
            elif step == "eval":
                # Navigate to train but defer next_iter() until the user actually starts training,
                # so they can still go back to eval/export without losing their work.
                self.controler._came_from_eval = True
                self.controler._step = "train"
                self.controler.dashboard.switch_panel()
            else:
                if step == "home":
                    self.controler.fit_done = False
                    self.controler.eval_done = False
                    self.controler.export_done = False
                self.controler._step = "train"
                self.controler.dashboard.switch_panel()

        def _nav_eval(e):
            step = self.controler.step
            came_from_eval = getattr(self.controler, '_came_from_eval', False)
            if step == "train" and self.controler.fit_done and not came_from_eval:
                _show_spinner()
                self.controler.next_step()
            else:
                self.controler._came_from_eval = False
                self.controler._came_from_eval_export = False
                self.controler._step = "eval"
                self.controler.dashboard.switch_panel()

        def _nav_export(e):
            if not self.controler.export_done:
                # Defer checkpoint/apply_corrections/next_iter/export_corpus until
                # the user actually clicks Start Export, so they can still go back to eval.
                self.controler._came_from_eval = False
                self.controler._came_from_eval_export = True
                self.controler._step = "export"
                self.controler.dashboard.switch_panel()
            else:
                self.controler._step = "export"
                self.controler.dashboard.switch_panel()

        def _nav_annotate(e):
            self.controler.home_path = "annotate"
            self.controler._step = "annotate"
            self.controler.dashboard.switch_panel()

        def _guarded(fn):
            """Intercept navigation away from train if optimization is running."""
            def _wrapped(e):
                if (getattr(self.controler, 'is_optimization_running', False)
                        and hasattr(self.parent, 'request_navigate_away')):
                    self.parent.request_navigate_away(lambda: fn(e))
                else:
                    fn(e)
            return _wrapped

        self._nav_settings  = _nav_settings
        self._nav_preprocess = _nav_preprocess
        self._nav_fit      = _nav_fit
        self._nav_eval     = _nav_eval
        self._nav_export   = _nav_export
        self._nav_annotate = _nav_annotate

        btn_loaddata.on_click(_guarded(_nav_loaddata))
        if has_data:
            btn_settings.on_click(_guarded(_nav_settings))
            btn_preprocess.on_click(_guarded(_nav_preprocess))
            btn_train.on_click(_nav_fit)  # stays on train — no guard needed
        if fit_done:
            btn_eval.on_click(_guarded(_nav_eval))
        if eval_done:
            btn_export.on_click(_guarded(_nav_export))
        if has_model:
            btn_annotate.on_click(_guarded(_nav_annotate))

        quit_btn = pn.widgets.Button(
            name="✕  Quit App",
            sizing_mode="stretch_width",
            align="center",
            stylesheets=[_QUIT_BTN_CSS],
        )
        quit_btn.on_click(self.on_click_stop)

        items = []

        if not is_home:
            items.append(
                pn.pane.PNG(
                    "images/logo_canapy_simple.png",
                    height=69,
                    align="center",
                    margin=(10, 5, 15, 5),
                )
            )
        else:
            items.append(pn.Spacer(height=10))

        spinner = pn.indicators.LoadingSpinner(
            value=False,
            visible=False,
            size=24,
            align="center",
            margin=(6, 0, 2, 0),
        )
        self._spinner = spinner

        items += [
            btn_loaddata,
            pn.Spacer(height=4),
            btn_settings,
            pn.Spacer(height=4),
            btn_preprocess,
            pn.Spacer(height=4),
            btn_train,
            pn.Spacer(height=4),
            btn_eval,
            pn.Spacer(height=4),
            btn_export,
            pn.Spacer(height=4),
            btn_annotate,
            pn.layout.Spacer(sizing_mode="stretch_height"),
            spinner,
            pn.pane.HTML("<div class='nav-divider'></div>"),
        ]

        if not is_home:
            home_btn = pn.widgets.Button(
                name="Home",
                sizing_mode="stretch_width",
                align="center",
                stylesheets=[_HOME_BTN_CSS],
            )
            home_btn.on_click(_guarded(self.on_click_home))
            items.append(home_btn)
            items.append(pn.Spacer(height=4))

        items.append(quit_btn)
        items.append(pn.Spacer(height=8))

        self.layout = pn.Column(
            *items,
            width=210,
            sizing_mode="stretch_height",
            css_classes=['sidebar-container'],
        )

    def on_data_loaded(self):
        """Called after a dataset is successfully loaded — unlocks settings, preprocessing, fit."""
        for btn, cb in [
            (self._btn_settings,   self._nav_settings),
            (self._btn_preprocess, self._nav_preprocess),
            (self._btn_train,      self._nav_fit),
        ]:
            btn.disabled = False
            btn.stylesheets = [_ACCESSIBLE_BTN_CSS]
            btn.on_click(cb)

    def enable_next(self):
        """Called when fit (training + annotation) completes."""
        self.controler.fit_done = True
        self._btn_eval.disabled = False
        self._btn_eval.stylesheets = [_NEXT_STEP_BTN_CSS]
        self._btn_eval.on_click(self._nav_eval)

    def on_export_done(self):
        """Called when export training completes."""
        self.controler.export_done = True
        self.controler.model_root = self.controler.output_directory / "model"
        self._btn_annotate.disabled = False
        self._btn_annotate.stylesheets = [_ACCESSIBLE_BTN_CSS]
        self._btn_annotate.on_click(self._nav_annotate)

    def on_click_home(self, events):
        self.controler.home_path = None
        self.controler._step = "home"
        self.controler.dashboard.switch_panel()

    def on_click_stop(self, events):
        self.controler.stop_optimization()
        self.layout.append(
            pn.pane.Alert(
                "Server closed. You can close this tab.",
                alert_type="success",
            )
        )
        self.controler.stop_app()
