import pathlib
import logging
import panel as pn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import librosa
from ..helpers import SubDash, Registry, SideBar

pn.extension("tabulator")

logger = logging.getLogger("canapy")

MAX_SAMPLE_DISPLAY = 10

PREPROCESS_CSS = """
:host {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.page-title {
    font-size: 30px;
    font-weight: 700;
    color: #212529;
    margin-bottom: 5px;
}
.page-subtitle {
    font-size: 16px;
    color: #868e96;
    margin-bottom: 20px;
}
.selector-bar {
    background: white;
    padding: 10px 15px;
    border-radius: 6px;
    border: 1px solid #e5e7eb;
    margin-bottom: 15px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.dashboard-col {
    background-color: #ffffff;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    padding: 15px;
    border: 1px solid #e5e7eb;
    height: 100%;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-sizing: border-box;
}
.col-header {
    font-size: 16px;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 15px;
    border-bottom: 2px solid #f3f4f6;
    padding-bottom: 8px;
    text-transform: uppercase;
    flex-shrink: 0;
}
.scrollable-content {
    overflow-y: auto;
    overflow-x: hidden;
    flex-grow: 1;
    padding-right: 5px;
    padding-bottom: 20px;
}
.sample-card {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    margin-bottom: 15px;
    overflow: hidden;
    box-sizing: border-box;
}
.sample-card audio {
    max-width: 100%;
    width: 100%;
    box-sizing: border-box;
}
.corrector-input input {
    text-align: center;
    font-size: 13px;
    color: #374151;
    font-weight: 500;
    background: transparent;
    max-width: 100%;
    box-sizing: border-box;
}
.selector-card {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 5px;
    transition: all 0.2s;
    cursor: pointer;
}
.selector-card:hover {
    border-color: #3b82f6;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.stat-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 4px;
    padding: 8px 10px;
    font-size: 11px;
    color: #1e3a8a;
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-width: 100px;
}
.stats-container {
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 15px;
}
.sample-row-card {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
"""

def compute_gated_stats(y, sr, threshold_ratio=0.2):
    """
    Quick per-sample computation (mean) for individual display
    """
    if len(y) == 0:
        return 0, 0
    N_FFT = 2048
    HOP_LEN = 512
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LEN))
    rms = librosa.feature.rms(S=S)[0]
    rms_max = np.max(rms) if len(rms) > 0 else 0
    if rms_max == 0:
        return 0, 0
    threshold = rms_max * threshold_ratio
    mask = rms > threshold
    if np.sum(mask) < 3:
        return 0, 0
    cent_frames = librosa.feature.spectral_centroid(S=S, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN)[0]
    
    avg_cent = np.mean(cent_frames[mask])
    
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    dom_freq_idx = np.argmax(S, axis=0)
    dom_freq_hz = freqs[dom_freq_idx]
    dt = HOP_LEN / sr
    delta_freq = np.diff(dom_freq_hz)
    instant_slopes = delta_freq / dt
    valid_slope_mask = mask[:-1] & mask[1:]
    if np.sum(valid_slope_mask) < 2:
        return int(avg_cent), 0
    valid_slopes = instant_slopes[valid_slope_mask]
    lower_bound = np.percentile(valid_slopes, 10)
    upper_bound = np.percentile(valid_slopes, 90)
    cleaned_slopes = valid_slopes[(valid_slopes >= lower_bound) & (valid_slopes <= upper_bound)]
    if len(cleaned_slopes) == 0:
        cleaned_slopes = valid_slopes
        
    avg_slope = np.mean(cleaned_slopes)
    return int(avg_cent), int(avg_slope)

class PreprocessDashboard(SubDash):
    def __init__(self, parent):
        super().__init__(parent)
        pn.config.raw_css.append(PREPROCESS_CSS)
        self.sidebar = SideBar(self, "Preprocessing", disabled=False)
        self.header = pn.Column(
            pn.pane.Markdown("# Preprocessing", css_classes=["page-title"], margin=0),
            pn.pane.Markdown(
                "Inspect dataset, merge classes, and correct labels.",
                css_classes=["page-subtitle"],
                margin=0,
            ),
        )
        self.pane_selection = pn.widgets.RadioButtonGroup(
            name="Pane Selection",
            options=["Class Merge", "Sample Correction"],
            button_type="default",
            button_style="outline",
            align="center",
            value="Class Merge",
        )
        self.pane_selection.param.watch(self.on_switch_panel, "value")
        self.merge_tool = MergeTool(self)
        self.correction_tool = CorrectionTool(self)
        self.content_area = pn.Column(self.merge_tool.layout, sizing_mode="stretch_both")
        main_content = pn.Column(
            self.header,
            pn.Row(self.pane_selection, css_classes=["selector-bar"], sizing_mode="stretch_width"),
            self.content_area,
            sizing_mode="stretch_both",
            margin=(20, 20, 20, 20),
        )
        self.layout = pn.Row(
            self.sidebar,
            main_content,
            sizing_mode="stretch_both",
            background="#f8fafc",
        )

    def on_switch_panel(self, event):
        if event.new == "Class Merge":
            self.content_area[:] = [self.merge_tool.layout]
        else:
            self.correction_tool.refresh_selectors()
            self.content_area[:] = [self.correction_tool.layout]

class MergeTool(SubDash):
    def __init__(self, parent):
        super().__init__(parent)
        self.repertoire = RepertoireView(self, num_panel=2, orientation="column")
        self.corrector = CorrectorView(self)
        self.layout = pn.Row(
            pn.Column(
                pn.pane.HTML("<div class='col-header' style='border-bottom: none;'>1. Class Correction</div>"),
                self.corrector.layout,
                css_classes=["dashboard-col"],
                styles={"min-width": "260px", "max-width": "420px"},
                sizing_mode="stretch_height",
            ),
            pn.Spacer(width=15),
            pn.Column(
                pn.pane.HTML("<div class='col-header'>2. Audio Repertoire</div>"),
                pn.Column(self.repertoire.layout, css_classes=["scrollable-content"], sizing_mode="stretch_both"),
                css_classes=["dashboard-col"],
                sizing_mode="stretch_both",
            ),
            sizing_mode="stretch_both",
            min_height=500,
        )

class CorrectorView(SubDash):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = self.build_display()

    def build_display(self):
        info = pn.pane.Markdown(
            "Rename classes to merge them.",
            styles={"font-size": "13px", "color": "#6b7280"},
        )
        self.grid = pn.FlexBox(justify_content="center", gap=10)
        for l in self.controler.classes:
            if l != self.controler.config.transforms.annots.silence_tag:
                self.grid.append(
                    pn.widgets.TextInput(
                        name=l,
                        placeholder=l,
                        width=120,
                        css_classes=["corrector-input"],
                    )
                )
        self.save_btn = pn.widgets.Button(
            name="Apply", button_type="primary", sizing_mode="stretch_width"
        )
        self.save_btn.on_click(self.on_click_save)
        self.save_msg = pn.pane.Alert("", alert_type="success", visible=False)
        return pn.Column(
            info,
            pn.Column(self.grid, scroll=True, sizing_mode="stretch_both"),
            pn.Column(
                self.save_msg,
                self.save_btn,
                sizing_mode="stretch_width",
                margin=(10, 0, 0, 0),
            ),
            sizing_mode="stretch_both",
        )

    def on_click_save(self, events):
        new_corrections = {
            text.name: text.value
            for text in self.grid
            if isinstance(text, pn.widgets.TextInput) and text.value != ""
        }
        if new_corrections:
            self.controler.apply_live_corrections(new_corrections, "class")
            self.save_msg.object = "Saved & Memory Updated!"
            self.save_msg.visible = True

class RepertoireView(SubDash):
    def __init__(self, parent, num_panel, orientation, num_samples=MAX_SAMPLE_DISPLAY):
        super().__init__(parent)
        self.orientation = orientation
        self.num_samples = num_samples
        self.registry = Registry()
        self.select_left = pn.widgets.Select(
            name="Class A",
            options=[lbl for lbl in self.controler.classes if lbl != "SIL"],
            sizing_mode="stretch_width",
        )
        self.sample_left = SampleView(self, label=self.select_left.value, orientation=self.orientation)
        self.select_left.param.watch(self.on_select_left, "value")
        if num_panel == 2:
            self.select_right = pn.widgets.Select(
                name="Class B",
                options=[lbl for lbl in self.controler.classes if lbl != "SIL"],
                sizing_mode="stretch_width",
            )
            self.sample_right = SampleView(self, label=self.select_right.value, orientation=self.orientation)
            self.select_right.param.watch(self.on_select_right, "value")
            self.layout = pn.Row(
                pn.Column(self.select_left, pn.Spacer(height=5), self.sample_left, sizing_mode="stretch_both"),
                pn.Spacer(width=10),
                pn.Column(self.select_right, pn.Spacer(height=5), self.sample_right, sizing_mode="stretch_both"),
                sizing_mode="stretch_both",
            )
        else:
            self.layout = pn.Column(self.select_left, self.sample_left, sizing_mode="stretch_both")

    def on_select_left(self, events):
        label = self.select_left.value
        if self.registry.get(label) is None:
            if len(self.registry) > 10:
                self.registry.popitem()
            self.registry[label] = SampleView(
                self,
                label=label,
                orientation=self.orientation,
                num_samples=self.num_samples,
            )
        self.layout[0][2] = self.registry[label].layout

    def on_select_right(self, events):
        label = self.select_right.value
        if self.registry.get(label) is None:
            if len(self.registry) > 10:
                self.registry.popitem()
            self.registry[label] = SampleView(
                self,
                label=label,
                orientation=self.orientation,
                num_samples=self.num_samples,
            )
        self.layout[2][2] = self.registry[label].layout

class SampleView(SubDash):
    def __init__(self, parent, label=None, orientation="column", num_samples=MAX_SAMPLE_DISPLAY):
        super().__init__(parent)
        self.num_samples = num_samples
        self.orientation = orientation
        self.layout = self.build_display(label)

    def build_display(self, label):
        if not label:
            return pn.Spacer()
        selected_df = self.controler.corpus.dataset.query("label == @label")
        selected_df = selected_df.iloc[: self.num_samples]
        specs = self.controler.load_repertoire(selected_df)
        sampling_rate = self.controler.config.transforms.audio.sampling_rate
        views = []
        for i, sp in enumerate(specs):
            visual_block = pn.Row(
                pn.pane.Markdown(
                    f"**#{i+1}**",
                    styles={"font-size": "11px", "color": "#6b7280"},
                    width=30,
                    align="center",
                ),
                pn.pane.Matplotlib(
                    sp[0],
                    format="png",
                    tight=True,
                    sizing_mode="stretch_width",
                    height=50,
                ),
                sizing_mode="stretch_width",
                styles={"min-width": "120px", "max-width": "320px"},
                margin=(0, 10, 0, 0),
            )
            audio_block = pn.Column(
                pn.pane.Audio(
                    sp[1],
                    sample_rate=round(sampling_rate),
                    height=35,
                    sizing_mode="stretch_width",
                ),
                pn.Spacer(height=5),
                pn.pane.Audio(
                    sp[2],
                    sample_rate=round(sampling_rate),
                    height=35,
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            )
            card_content = pn.FlexBox(
                visual_block,
                audio_block,
                align_items="center",
                justify_content="start",
                flex_wrap="wrap",
                gap=10,
                sizing_mode="stretch_width",
            )
            views.append(
                pn.Column(
                    card_content,
                    css_classes=["sample-card"],
                    padding=20,
                    sizing_mode="stretch_width",
                )
            )
        layout_cls = pn.Column if self.orientation == "column" else pn.Row
        return layout_cls(*views, scroll=True, sizing_mode="stretch_both")

class CorrectionTool(SubDash):
    def __init__(self, parent):
        super().__init__(parent)
        self.registry = Registry()
        self.registry["sample"] = {}
        self.registry["class"] = {}
        self.total_corrected = 0
        self.selector_area = pn.Column(
            scroll=True,
            styles={
                "background": "#f8f9fa",
                "padding": "15px",
                "border-radius": "8px",
                "border": "1px solid #dee2e6",
            },
            sizing_mode="stretch_width",
        )
        self.build_class_selectors()
        self.stats_table = pn.widgets.Tabulator(
            pd.DataFrame(columns=["Class", "Count", "Med Dur", "Med Centroid", "Med Slope"]),
            show_index=False,
            disabled=True,
            height=200,
            width=320,
            sizing_mode="stretch_width",
            configuration={"columnDefaults": {"headerSort": False}},
        )
        self.stats_tooltip = pn.widgets.TooltipIcon(
            value=(
                "Count: number of samples in the class\n"
                "Med Dur: median segment duration\n"
                "Med Centroid: median spectral centroid (energy-weighted mean frequency)\n"
                "Med Slope: median temporal slope of dominant frequency"
            )
        )
        self.compute_btn = pn.widgets.Button(
            name="Calculate Stats",
            button_type="primary",
            width=160,
            height=60,
            align="center",
        )
        self.compute_btn.on_click(self.run_global_stats_calculation)
        self.compute_status = pn.pane.Markdown(
            "<i>Click to compute global stats</i>",
            styles={"font-size": "11px", "color": "grey"},
        )
        self.stats_dashboard = self.build_stats_layout()
        self.save_btn = pn.widgets.Button(
            name="Save all",
            sizing_mode="stretch_width",
            button_type="primary",
            disabled=True,
        )
        self.save_btn.on_click(self.on_click_save)
        self.save_msg = pn.pane.HTML(styles=dict(color="green", background="white"))
        self.sample_container = pn.Column(
            pn.pane.Alert(
                "Select a class above to inspect samples.", alert_type="light"
            ),
            scroll=True,
            sizing_mode="stretch_both",
        )
        self.layout = pn.Column(
            pn.Column(
                pn.pane.Markdown(
                    "### 1. Global Analysis & Selection",
                    margin=(0, 0, 10, 0),
                ),
                self.stats_dashboard,
                self.selector_area,
                sizing_mode="stretch_width",
                margin=(0, 0, 20, 0),
            ),
            pn.Column(
                pn.pane.Markdown(
                    "### 2. Correct Samples",
                    margin=(0, 0, 10, 0),
                ),
                self.sample_container,
                sizing_mode="stretch_both",
                min_height=300,
            ),
            pn.Column(
                pn.pane.Markdown(
                    "### 3. Save corrections",
                    margin=(10, 0, 5, 0),
                ),
                self.save_msg,
                self.save_btn,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_both",
        )

    def refresh_selectors(self):
        self.registry["class"] = {}
        self.build_class_selectors()

    def build_stats_layout(self):
        df = self.controler.corpus.dataset.copy()
        if "duration" not in df.columns:
            df["duration"] = df["offset_s"] - df["onset_s"]
        all_classes = sorted([c for c in df["label"].unique() if c != "SIL"])
        MAX_PER_PAGE = 40
        self.class_chunks = [
            all_classes[i : i + MAX_PER_PAGE]
            for i in range(0, len(all_classes), MAX_PER_PAGE)
        ]
        self.plot_pane = pn.pane.Matplotlib(
            tight=True, sizing_mode="stretch_width", min_height=250
        )
        self.stat_selector = pn.widgets.RadioButtonGroup(
            name="Metric",
            options=["Duration", "Centroid", "Slope"],
            value="Duration",
            button_type="light",
            button_style="solid",
            styles={"font-size": "12px"},
        )
        self.stat_selector.param.watch(self.update_boxplot_view, "value")
        self.plot_pager = pn.widgets.RadioButtonGroup(
            name="Page",
            options=list(range(1, len(self.class_chunks) + 1)),
            value=1,
            button_style="outline",
            button_type="default",
            visible=(len(self.class_chunks) > 1),
        )
        self.plot_pager.param.watch(self.update_boxplot_view, "value")
        self.update_boxplot_view()
        right_panel = pn.Column(
            pn.Row(
                pn.pane.Markdown(
                    "**Class Statistics**",
                    styles={"font-size": "13px", "font-weight": "bold"},
                ),
                self.stats_tooltip,
                self.compute_status,
                sizing_mode="stretch_width",
            ),
            self.stats_table,
            self.compute_btn,
            sizing_mode="stretch_width",
            max_width=430,
            margin=(0, 0, 0, 15),
        )
        return pn.Row(
            pn.Column(
                pn.Row(pn.Spacer(), self.stat_selector, pn.Spacer()),
                self.plot_pane,
                pn.Row(pn.Spacer(), self.plot_pager, pn.Spacer()),
                sizing_mode="stretch_width",
            ),
            right_panel,
            sizing_mode="stretch_width",
            css_classes=["stats-container"],
            margin=(0, 0, 15, 0),
        )

    def update_boxplot_view(self, event=None):
        page_index = self.plot_pager.value - 1
        current_classes = self.class_chunks[page_index] if self.class_chunks else []
        metric = self.stat_selector.value
        df = self.controler.corpus.dataset
        temp_df = df[df["label"].isin(current_classes)].copy()
        if "duration" not in temp_df.columns:
            temp_df["duration"] = temp_df["offset_s"] - temp_df["onset_s"]
        data_to_plot = []
        y_label = ""
        if metric == "Duration":
            data_to_plot = [
                temp_df[temp_df["label"] == c]["duration"].values
                for c in current_classes
            ]
            y_label = "Time (s)"
        elif metric == "Centroid":
            y_label = "Frequency (Hz)"
            if "centroid" in temp_df.columns:
                for c in current_classes:
                    vals = (
                        temp_df[temp_df["label"] == c]["centroid"]
                        .dropna()
                        .values
                    )
                    data_to_plot.append(vals if len(vals) > 0 else [])
            else:
                data_to_plot = [[] for _ in current_classes]
        elif metric == "Slope":
            y_label = "Mean Slope (Hz/s)"
            if "slope" in temp_df.columns:
                for c in current_classes:
                    vals = (
                        temp_df[temp_df["label"] == c]["slope"]
                        .dropna()
                        .values
                    )
                    data_to_plot.append(vals if len(vals) > 0 else [])
            else:
                data_to_plot = [[] for _ in current_classes]
        fig, ax = plt.subplots(figsize=(6, 4))
        has_data = any(len(d) > 0 for d in data_to_plot)
        if has_data:
            ax.boxplot(
                data_to_plot,
                labels=current_classes,
                patch_artist=True,
                boxprops=dict(facecolor="#3b82f6", color="#1f2937"),
                medianprops=dict(color="white"),
            )
        else:
            if metric != "Duration":
                ax.text(
                    0.5,
                    0.5,
                    "Click 'Calculate Stats' to view data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="gray",
                )
            ax.set_xticklabels(current_classes)
        title_suffix = (
            f" (Page {self.plot_pager.value}/{len(self.class_chunks)})"
            if len(self.class_chunks) > 1
            else ""
        )
        ax.set_title(f"{metric} Distribution{title_suffix}", fontsize=10)
        ax.set_ylabel(y_label, fontsize=9)
        ax.tick_params(
            axis="x",
            rotation=90 if len(current_classes) > 10 else 45,
            labelsize=8,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        self.plot_pane.object = fig

    def run_global_stats_calculation(self, event):
        self.compute_btn.loading = True
        self.compute_status.object = "Calculating Centroid & Slope..."
        sampling_rate = self.controler.config.transforms.audio.sampling_rate
        unique_labels = self.controler.corpus.dataset["label"].unique()
        classes = sorted([c for c in unique_labels if c != "SIL"])
        if "centroid" not in self.controler.corpus.dataset.columns:
            self.controler.corpus.dataset["centroid"] = np.nan
        if "slope" not in self.controler.corpus.dataset.columns:
            self.controler.corpus.dataset["slope"] = np.nan
        stats_data = []
        try:
            if "duration" not in self.controler.corpus.dataset.columns:
                self.controler.corpus.dataset["duration"] = (
                    self.controler.corpus.dataset["offset_s"]
                    - self.controler.corpus.dataset["onset_s"]
                )
            for c in classes:
                full_class_df = self.controler.corpus.dataset.query("label == @c")
                count = len(full_class_df)
                
                avg_dur = (
                    full_class_df["duration"].median()
                    if not full_class_df.empty
                    else 0
                )
                
                sub_df = full_class_df.sample(frac=1).head(200)
                centroid_list = []
                slope_list = []
                if not sub_df.empty:
                    try:
                        repertoire = self.controler.load_repertoire(sub_df)
                        for (idx, row), item in zip(
                            sub_df.iterrows(), repertoire
                        ):
                            audio_data = item[1]
                            if not isinstance(audio_data, np.ndarray):
                                audio_data = np.array(audio_data)
                            audio_float = audio_data.astype(np.float32)
                            max_val = np.max(np.abs(audio_float))
                            if max_val > 1.0:
                                audio_float /= 32768.0
                            if len(audio_float) > 0:
                                gated_cent, gated_slope = compute_gated_stats(
                                    audio_float, sampling_rate
                                )
                                if gated_cent > 0:
                                    centroid_list.append(gated_cent)
                                    self.controler.corpus.dataset.at[
                                        idx, "centroid"
                                    ] = gated_cent
                                    slope_list.append(gated_slope)
                                    self.controler.corpus.dataset.at[
                                        idx, "slope"
                                    ] = gated_slope
                    except Exception as e:
                        logger.error(f"Stats Error Class {c}: {e}")
                
                avg_centroid = (
                    int(np.median(centroid_list)) if centroid_list else 0
                )
                avg_slope = int(np.median(slope_list)) if slope_list else 0
                
                stats_data.append(
                    {
                        "Class": c,
                        "Count": count,
                        "Med Dur": f"{avg_dur:.2f}s",
                        "Med Centroid": f"{avg_centroid / 1000:.2f} kHz",
                        "Med Slope": f"{avg_slope / 1000:.2f} kHz/s",
                    }
                )
            self.stats_table.value = pd.DataFrame(stats_data)
            self.compute_status.object = "Done."
            self.update_boxplot_view()
        except Exception as e:
            self.compute_status.object = f"Error: {str(e)}"
        finally:
            self.compute_btn.loading = False

    def build_class_selectors(self):
        grid = pn.FlexBox(justify_content="start", gap=10, align_items="center")
        unique_labels = self.controler.corpus.dataset["label"].unique()
        classes = sorted([c for c in unique_labels if c != "SIL"])
        dataset = self.controler.corpus.dataset
        for lbl in classes:
            n_samples = len(dataset.query("label==@lbl"))
            display = ClassSelectionView(lbl, n_samples, self)
            self.registry["class"][lbl] = display
            grid.append(display.layout)
        self.selector_area.objects = [grid]

    def get_sample_list_view(self, label):
        if self.registry["sample"].get(label) is not None:
            return self.registry["sample"][label].layout
        view = SampleListView(self, label)
        self.registry["sample"][label] = view
        return view.layout

    def highlight_selection(self, active_label):
        for lbl, view in self.registry["class"].items():
            view.select_btn.button_type = (
                "primary" if lbl == active_label else "default"
            )

    def listen_class_selection(self, label):
        self.highlight_selection(label)
        view_layout = self.get_sample_list_view(label)
        self.sample_container.objects = [view_layout]

    def listen_correction(self, label, increment):
        self.registry["class"][label].receive_correction(increment)
        self.total_corrected += increment
        self.save_btn.disabled = self.total_corrected == 0

    def check_correction(self, label):
        return label in self.controler.classes if label != "" else True

    def on_click_save(self, events):
        new_corrections = {}
        for sample_view in self.registry["sample"].values():
            new_corrections.update(sample_view.corrections)
        if new_corrections:
            self.controler.apply_live_corrections(new_corrections, "annot")
            self.save_msg.object = "Saved successfully!"
            self.save_msg.visible = True

class ClassSelectionView(SubDash):
    def __init__(self, label, total_count, parent):
        super().__init__(parent)
        self.label = label
        self.total_count = total_count
        self.num_corrected = 0
        self.select_btn = pn.widgets.Button(
            name=label,
            button_type="default",
            height=35,
            sizing_mode="stretch_width",
        )
        self.select_btn.on_click(self.on_click_notify_display)
        self.corrected_msg = pn.pane.Markdown(
            f"**{self.num_corrected}** / {self.total_count}",
            styles={
                "text-align": "center",
                "font-size": "11px",
                "color": "#6b7280",
            },
            margin=(5, 0),
        )
        self.layout = pn.Column(
            self.select_btn,
            self.corrected_msg,
            width=110,
            css_classes=["selector-card"],
        )

    def on_click_notify_display(self, events):
        self.parent.listen_class_selection(self.label)

    def receive_correction(self, increment):
        self.num_corrected += increment
        color = "green" if self.num_corrected > 0 else "#6b7280"
        self.corrected_msg.object = f"**{self.num_corrected}** / {self.total_count}"
        self.corrected_msg.styles["color"] = color

class SampleListView(SubDash):
    def __init__(self, parent, label):
        super().__init__(parent)
        self.label = label
        self.dataset_slice = self.controler.corpus.dataset.query("label==@label")
        self.sliced_df = self.dataset_slice
        self.registry = Registry()
        self.layout = self.build_display()

    @property
    def corrections(self):
        return {
            i: display.correction
            for i, display in self.registry.items()
            if display.correction != ""
        }

    def build_display(self):
        if self.sliced_df.empty:
            return pn.pane.Markdown("_No samples found._")
        segments = self.controler.load_repertoire(self.sliced_df)
        container = pn.Column(sizing_mode="stretch_width")
        for i, (segment, (idx, row)) in enumerate(
            zip(segments, self.sliced_df.iterrows())
        ):
            display_num = i + 1
            if self.registry.get(idx) is None:
                display = SingleSampleRow(
                    row, segment, self.parent, idx, display_num
                )
                self.registry[idx] = display
            else:
                display = self.registry[idx]
            container.append(display.layout)
        return container

class SingleSampleRow(SubDash):
    def __init__(self, row_data, spec, parent, idx, display_num):
        super().__init__(parent)
        self.label = row_data.label
        self.idx = idx
        self.corrected = False
        sampling_rate = round(self.controler.config.transforms.audio.sampling_rate)

        self.number_pane = pn.pane.Markdown(
            f"**#{display_num}**",
            styles={'font-size': '16px', 'color': '#9ca3af', 'text-align': 'right'},
            width=40, align='center', margin=(0, 10, 0, 0)
        )

        self.img = pn.pane.Matplotlib(
            spec[0], format="png", tight=True, height=70,
            sizing_mode="stretch_width", min_width=100, max_width=220,
            margin=(0, 0), align='center'
        )

        self.audio_col = pn.Column(
            pn.pane.Audio(spec[1], sample_rate=sampling_rate, height=30, sizing_mode="stretch_width"),
            pn.pane.Audio(spec[2], sample_rate=sampling_rate, height=30, sizing_mode="stretch_width"),
            sizing_mode="stretch_width", min_width=290, max_width=340,
            align='center', margin=(0, 0, 0, 0),
            styles={"overflow": "hidden"},
        )

        duration = row_data.offset_s - row_data.onset_s
        audio_data = spec[1]
        
        cent_str = "-"
        slope_str = "-"
        
        if audio_data is not None and len(audio_data) > 0:
            try:
                if not isinstance(audio_data, np.ndarray):
                    audio_data = np.array(audio_data)
                    
                audio_float = audio_data.astype(np.float32)
                max_val = np.max(np.abs(audio_float))
                
                if max_val > 1.0:
                     audio_float /= 32768.0

                cent_val, slope_val = compute_gated_stats(audio_float, sampling_rate)

                if cent_val > 0:
                    cent_str = f"{cent_val / 1000:.2f} kHz"
                    slope_str = f"{slope_val / 1000:.2f} kHz/s"
                    
            except Exception as e:
                cent_str = "Err"; slope_str = "Err"
                logger.error(f"Error computing sample stats: {e}")

        self.stats_col = pn.Column(
            pn.pane.Markdown(f"**Duration:** {duration:.2f}s", margin=(2,0)),
            pn.pane.Markdown(f"**Centroid:** {cent_str}", margin=(2,0)),
            pn.pane.Markdown(f"**Slope:** {slope_str}", margin=(2,0)),
            width=120, align='center', margin=(10, 10), css_classes=['stat-box']
        )

        notated_file = pathlib.Path(row_data.notated_path).stem
        file_tooltip = pn.widgets.TooltipIcon(
            value=f"File: {notated_file}\nIndex: {idx}\nTime: {row_data.onset_s:.2f}s - {row_data.offset_s:.2f}s",
            margin=(10, 15), align='center'
        )

        self.text_input = pn.widgets.TextInput(placeholder="Correction", width=140)
        self.text_input.param.watch(self.on_correction_notify, "value")
        self.label_status = pn.pane.Markdown("", styles={"font-size": "11px", "color": "orange"})

        input_section = pn.Column(
            pn.Row(
                pn.pane.Markdown(f"**{self.label}**", styles={'font-size': '14px'}, margin=(0, 5, 0, 0)),
                self.label_status
            ),
            self.text_input,
            width=150, align='center'
        )
        
        self.layout = pn.Row(
            self.number_pane, file_tooltip, self.img, self.audio_col,
            pn.Spacer(width=20), self.stats_col, pn.Spacer(width=10), input_section,
            css_classes=['sample-card'],
            sizing_mode="stretch_width",
            align="center",
        )

    @property
    def correction(self):
        return self.text_input.value

    def on_correction_notify(self, events):
        validation = self.parent.check_correction(self.text_input.value)
        self.label_status.object = "" if validation else "New class"
        if self.text_input.value == "":
            if self.corrected:
                self.corrected = False
                self.parent.listen_correction(self.label, increment=-1)
        else:
            if not self.corrected:
                self.corrected = True
                self.parent.listen_correction(self.label, increment=1)