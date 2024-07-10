import logging
import os
import time

import panel as pn
import crowsetta
import pandas as pd

from collections import defaultdict
from pathlib import Path
from math import pi

import soundfile
from bokeh.palettes import Category20c
from bokeh.plotting import figure
from bokeh.transform import cumsum, factor_cmap

pn.extension('floatpanel')

logger = logging.getLogger("canapy-dashboard")


class SideBar(pn.viewable.Viewer):
    def __init__(self, parent, title):
        super().__init__()
        self.parent = parent

        self.back_btn = pn.widgets.Button(button_type='warning', icon="arrow-back-up", name="Back")
        self.back_btn.on_click(self.on_click_back)

        self.quit_btn = pn.widgets.Button(name="Quit", button_type="danger", icon="square-rounded-x")
        self.quit_btn.on_click(self.on_click_stop)

        self.settings_button = pn.widgets.Button(button_type='primary', icon="settings", name="Settings")
        self.settings_button.on_click(self.on_click_settings)

        self.train_settings = pn.Column(
            pn.widgets.IntInput(name='Seed', value=42, step=2, start=0, end=1000),
            pn.widgets.DiscreteSlider(name='Time Precision',
                                      options=[0.0004, 0.0006, 0.0008, 0.001, 0.0012, 0.0014, 0.0016], value=0.001),
            pn.widgets.DiscreteSlider(name='Min Label Duration', options=[0.01, 0.02, 0.04, 0.06, 0.08], value=0.02),
            pn.widgets.TextInput(name='Lonely Labels', value="cri,TRASH"),
            pn.widgets.DiscreteSlider(name='Min Silence Gap',
                                      options=[0.0004, 0.0006, 0.0008, 0.001, 0.0012, 0.0014, 0.0016], value=0.001),
            pn.widgets.TextInput(name='Silence Tag', value="SIL"),
            pn.widgets.IntInput(name='Sampling Rate', value=44100, step=500),
            pn.widgets.IntInput(name='N_MFCC', value=13, step=1),
            pn.widgets.FloatSlider(name='Hop Length', start=0, end=0.1, step=0.005, value=0.01),
            pn.widgets.FloatSlider(name='Win Length', start=0, end=0.1, step=0.005, value=0.02),
            pn.widgets.IntInput(name='N_FFT', value=2048, step=100),
            pn.widgets.IntInput(name='F Min', value=500, step=10),
            pn.widgets.IntInput(name='F Max', value=8000, step=50),
            pn.widgets.IntInput(name='Lifter', value=40, step=2),
            pn.widgets.TextInput(name='Padding', value="wrap"),
            pn.widgets.TextInput(name='Output Directory', value=os.path.join(os.getcwd(), "bird1_output"))
        )

        self.settingspanel = pn.layout.FloatPanel(
            self.train_settings,
            name='Settings',
            margin=20,
            visible=False,
            config={"headerControls": {"close": "remove", "maximize": "remove"}},
            height=700
        )

        self.layout = pn.Column(
            pn.pane.Markdown(f"## {title}", align='center'),
            self.settings_button,
            self.back_btn,
            self.quit_btn,
            self.settingspanel,
            width=150,
            sizing_mode="stretch_height",
            styles={"background": "WhiteSmoke"}
        )

    def on_click_stop(self, events):
        confirm_script = """
        <script>
            if (confirm("Are you sure you want to stop the server and close this tab?")) {
                window.close();
            }
        </script>
        """
        self.layout.append(pn.pane.HTML(confirm_script))

    def on_click_back(self, events):
        print("Back")

    def on_click_settings(self, event):
        self.settingspanel.visible = not self.settingspanel.visible

    def __panel__(self):
        return self.layout


class ModelCheckboxes(pn.viewable.Viewer):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.syn_checkbox = pn.widgets.Checkbox(name='Syntactic Model')
        self.nsyn_checkbox = pn.widgets.Checkbox(name='Non Syntactic Model')
        self.ensemble_checkbox = pn.widgets.Checkbox(name='Ensemble Model')

        model_text = """
        - **Syntactic Model**: Use this model if you want precise annotations for complete songs, taking into account the actual order of phrases and the syntactic structure of the bird songs. The best results are observed with this model.
        - **Non Syntactic Model**: Choose this model if you need annotations based solely on the individual characteristics of syllables, without considering the context or order of phrases.
        - **Ensemble Model**: This model combines the outputs of the previous two models using a voting system, offering a compromise between the two approaches.
        Select this model to benefit from the advantages of both methods.

        Choose based on whether you prioritize syntactic structure (syn), category balance (nsyn), or a combination of both (ensemble)."""

        self.model_accordion = pn.Accordion(('Which model should I choose ?', model_text), width=1000)

        self.layout = pn.Column(
            pn.pane.Markdown(f"## Model selection :"),
            pn.Row(self.syn_checkbox, self.nsyn_checkbox, self.ensemble_checkbox),
            self.model_accordion
        )

    def __panel__(self):
        return self.layout


class UploadDashboard(pn.viewable.Viewer):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent

        pn.config.raw_css.append("""
        .bk-btn-primary,
        .bk-btn-warning,
        .bk-btn-danger,
        .bk-btn-success {
            font-weight: bold;
            font-size: 18px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }
        """)

        title = pn.pane.Markdown(
            "<h1 style='text-align:center; font-size:50px'>Dataset upload</h1>",
            align='center',
            width=750,
        )

        self.sidebar = SideBar(self, "Upload")
        self.modelcheckboxes = ModelCheckboxes(self)

        self.validate_btn = pn.widgets.Button(name="Validate", button_type="success")
        self.validate_btn.on_click(self.on_click_validate)

        data_accordion_text = """Select one folder containing both the annotations and the audio files of your dataset. Alternatively, you can select one folder for the annotations and another for the audio files."""
        self.data_accordion = pn.Accordion(('Which folder(s) to select ?', data_accordion_text), width=1000)

        annot_formats = crowsetta.formats.FORMATS
        format_names = list(annot_formats.keys())
        annot_formats = ", ".join(format_names)

        audio_formats = soundfile.available_formats()
        format_names = list(audio_formats.keys())
        audio_formats = ", ".join(format_names)

        format_accordion_text = f"""Annotation formats accepted : \n{annot_formats}\nAudio formats accepted : \n{audio_formats}, NPY"""
        self.format_accordion = pn.Accordion(('Which formats are accepted ?', format_accordion_text), width=1000)

        self.train_btn = pn.widgets.Button(name="Train", align='start', button_type="success", disabled=True)
        self.train_btn.on_click(self.on_click_train)

        self.notification = pn.pane.Alert(alert_type='warning', visible=False, width=1000)

        self.file_selector = pn.widgets.FileSelector(directory=os.getcwd(), root_directory="/",
                                                     name="Select a directory", width=1000)

        self.stats = pn.pane.Markdown(f"""
        ## Data stats :
        ### Nombre de classes : ...
        ### Labels des classes : ...
        ### Durée totale de l'audio : ...
        ### Durée totale du silence : ...
        ### Nombre de fichiers audios annotés : ...
        """)

        data = {
            '...': ['...', '...', '...', '...', '...'],
        }
        df = pd.DataFrame(data)
        df_example = df.head(5)
        self.dataframe = pn.pane.DataFrame(df_example, height=200)

        self.bokeh_pane = pn.pane.Bokeh(figure(height=390, title="Class Repartition", toolbar_location=None,
                                               tools="hover", tooltips="@class: @value", x_range=(-0.5, 1.0)),
                                        theme="dark_minimal", visible=True)

        self.central_layout = pn.Row(
            pn.Column(
                title,
                self.modelcheckboxes,
                pn.pane.Markdown(f"## Data selection:"),
                self.notification,
                self.file_selector,
                self.validate_btn,
                self.data_accordion,
                self.format_accordion
            ),
            pn.Column(
                pn.pane.Markdown(f"## Data head :"), self.dataframe, self.stats, self.bokeh_pane, self.train_btn,
                margin=(0, 0, 0, 50), align="start"
            ),
            margin=(20, 0, 0, 0)
        )

        self.layout = pn.Row(
            self.sidebar,
            self.central_layout,
            title="Canapy"
        )

    def on_click_validate(self, event):

        if self.file_selector is not None:
            selected_folders = self.file_selector.value

            if len(selected_folders) > 2 or len(selected_folders) == 0:
                self.notification.object = f"Oups! {len(selected_folders)} éléments sélectionnés (2 max)."
                self.notification.alert_type = 'info'
                self.notification.visible = True
                self.train_btn.disabled = True
                return

            for file in selected_folders:
                if not Path(file).is_dir():
                    logger.error("Ce n'est pas un dossier: %s", file)
                    self.notification.object = f"Oups! {file} n'est pas un dossier"
                    self.notification.alert_type = 'info'
                    self.notification.visible = True
                    self.train_btn.disabled = True
                    return

            formats = crowsetta.formats.FORMATS
            format_extensions = []

            for format_name, format_class in formats.items():
                extensions = getattr(format_class, 'ext', None)
                if extensions:
                    if isinstance(extensions, (list, tuple)):
                        for ext in extensions:
                            format_extensions.append((format_name, ext))
                    else:
                        format_extensions.append((format_name, extensions))

            extensions_by_folder = self.get_extensions(selected_folders)

            annot_folder = None
            audio_folder = None

            for folder, extensions in extensions_by_folder.items():
                if len(extensions) > 2:
                    self.notification.object = f"Oups! Le nombre d'extensions est incorrect: {len(extensions)} (2 max)."
                    self.notification.alert_type = 'info'
                    self.notification.visible = True
                    self.train_btn.disabled = True
                    return

                for ext in extensions:
                    if ext == ".csv":
                        annot_folder = folder
                    if ext == ".wav":
                        audio_folder = folder

            if not annot_folder or not audio_folder:
                self.notification.object = f"Oups! Il manque des données."
                self.notification.alert_type = 'info'
                self.notification.visible = True
                self.train_btn.disabled = True
                return

            self.notification.object = "La sélection est validée."
            self.notification.alert_type = 'success'
            self.notification.visible = True
            self.update_data(*self.csv_parser(annot_folder))
            self.bokeh_pane.visible = True
            self.train_btn.disabled = False

    def get_extensions(self, directories):
        extensions_by_folder = {}
        for directory in directories:
            extensions = set()
            for file_path in Path(directory).rglob('*'):
                if file_path.is_file():
                    extensions.add(file_path.suffix)
            extensions_by_folder[directory] = extensions
        return extensions_by_folder

    def csv_parser(self, selected_folder):
        all_classes = set()
        num_annotated_files = 0
        csv_head = 0
        class_counts = defaultdict(int)
        total_duration = 0
        total_silence_duration = 0

        data_frames = []

        for filename in os.listdir(selected_folder):
            if filename.endswith('.csv'):
                num_annotated_files += 1
                file_path = os.path.join(selected_folder, filename)

                df = pd.read_csv(file_path)
                data_frames.append(df)

                unique_classes = df['syll'].unique()
                all_classes.update(unique_classes)

                syll_counts = df['syll'].value_counts()

                for syll, count in syll_counts.items():
                    class_counts[syll] += count

                silence_duration = df[df['syll'] == 'SIL']['end'] - df[df['syll'] == 'SIL']['start']
                total_silence_duration += silence_duration.sum()

                max_end_in_file = df['end'].max()
                total_duration += max_end_in_file

                if len(data_frames) == 1:
                    csv_head = df.head(10)

        class_repartition = dict(class_counts)
        num_classes_total = len(all_classes)
        class_labels_total = list(all_classes)

        return (csv_head, class_repartition, num_classes_total, class_labels_total,
                total_duration, num_annotated_files, total_silence_duration)

    def update_data(self, csv_head, class_repartition, num_classes_total, class_labels_total, total_duration,
                    num_annotated_files, total_silence_duration):

        total_duration_str = time.strftime("%H:%M:%S", time.gmtime(total_duration))
        total_silence_duration_str = time.strftime("%H:%M:%S", time.gmtime(total_silence_duration))
        data_stats = f"""
        ## Data stats :
        ### Nombre de classes : {num_classes_total}
        ### Labels des classes : {", ".join(sorted(class_labels_total))}
        ### Durée totale de l'audio : {total_duration_str}
        ### Durée totale du silence : {total_silence_duration_str}
        ### Nombre de fichiers audios annotés : {num_annotated_files} 
        """
        self.stats.object = data_stats

        self.dataframe.object = csv_head

        data = pd.Series(class_repartition).reset_index(name='value').rename(columns={'index': 'class'})
        data = data[data['class'] != 'SIL']
        data['angle'] = data['value'] / data['value'].sum() * 2 * pi
        data['color'] = (Category20c[20] * ((len(data) // 20) + 1))[:len(data)]
        data = data.sort_values(by='value', ascending=False)

        self.p = figure(height=390, title="Class Repartition", toolbar_location=None,
                        tools="hover", tooltips="@class: @value", x_range=(-0.5, 1.0))

        self.p.wedge(x=0, y=1, radius=0.4,
                     start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                     line_color="white",
                     fill_color=factor_cmap('class', palette=data['color'], factors=data['class']),
                     legend_field='class', source=data)
        self.p.axis.axis_label = None
        self.p.axis.visible = False
        self.p.grid.grid_line_color = None

        self.bokeh_pane.object = self.p
        self.bokeh_pane.visible = True

    def on_click_train(self, events):
        print("Train")

    def __panel__(self):
        return self.layout


if __name__ == '__main__':
    dashboard = UploadDashboard()
    pn.serve(dashboard.layout)