import logging
import os
import time
import random
from collections import defaultdict
from pathlib import Path

import panel as pn
import crowsetta
import pandas as pd
import soundfile
import matplotlib.pyplot as plt

from . import View
from .settings import SettingsView

logger = logging.getLogger("canapy-dashboard")


class ModelCheckboxes(View):
    def __init__(self, parent):
        super().__init__(parent)

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


class UploadDashboard(View):
    def __init__(self, parent=None):
        super().__init__(parent)

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

        self.sidebar.change_title("Upload")
        
        self.settings_view = SettingsView(parent=self)

        # self.sidebar = SideBar(self, "Upload")
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

         # Barplot Example
        fig, ax = plt.subplots(figsize=(7, 4))
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.title('Class repartition', fontweight='bold')
        plt.tight_layout()
        self.barplot_pane = pn.pane.Matplotlib(fig, height=350, disabled=True)

        self.layout = pn.Row(
            pn.Column(
                        pn.Column(
                            title,
                            self.modelcheckboxes,
                            pn.pane.Markdown(f"## Data selection :"),
                            self.file_selector,
                            pn.Row(self.validate_btn, self.notification),
                            self.data_accordion,
                            self.format_accordion,
                            width=1000,
                            margin=(0, 0, 0, 5),
                            css_classes=["Settings"]),
                        pn.Column(
                                self.settings_view
                            )
                    ),
            pn.Column(
                pn.pane.Markdown(f"## Data head :"), self.dataframe, self.stats, self.barplot_pane, self.train_btn,
                css_classes=["Settings"],
                margin=(0, 0, 0, 20), align="start"
            ),
            margin=(20, 0, 0, 20)
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
                if extensions is not None:
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

                # TODO: don't restrict this too much!
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
            
            self.controler.create_corpus(audio_directory=audio_folder,
                annots_directory=annot_folder,
                spec_directory="./", # TODO: replace by output directory
                config=None,  # TODO: get configuration from file if needed
                annot_format="marron1csv", # TODO: get this from the search above
                audio_ext=".wav", # TODO: idem
            )

            self.notification.object = "La sélection est validée."
            self.notification.alert_type = 'success'
            self.notification.visible = True
            # TODO: use controler.corpus.dataset here, not csv_parser
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
                ### Number of classes : {num_classes_total}
                ### Class labels : {", ".join(sorted(class_labels_total))}
                ### Total audio duration : {total_duration_str}
                ### Total silence duration : {total_silence_duration_str}
                ### Number of annotated audio files : {num_annotated_files} 
                """

        self.stats.object = data_stats

        self.dataframe.object = csv_head

        if 'SIL' in class_repartition:
            del class_repartition['SIL']

        sorted_items = sorted(class_repartition.items(), key=lambda item: item[1])
        classes = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]

        colors = ['#' + ''.join(random.choices('0123456789abcdef', k=6)) for _ in range(len(classes))]

        plt.figure(figsize=(7, 4))
        bars = plt.bar(classes, counts, color=colors)
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.title('Class repartition', fontweight='bold')

        for bar, count in zip(bars, counts):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, f'{count}', ha='center', va='bottom')

        plt.xticks(rotation=90)
        plt.tight_layout()

        self.barplot_pane.object = plt.gcf()

    def on_click_train(self, events):
        logger.info("Entering train dashboard.")
