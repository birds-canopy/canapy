import logging
import os
import time
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import panel as pn
import crowsetta
import pandas as pd
import soundfile
import matplotlib.pyplot as plt

from . import View
from .settings import SettingsView
#from ..controler import Controler

logger = logging.getLogger("canapy-dashboard")


class ModelCheckboxes(View):
    def __init__(self, parent):
        super().__init__(parent)

        self.syn_checkbox = pn.widgets.Checkbox(name='Syntactic Model', value=True)
        self.nsyn_checkbox = pn.widgets.Checkbox(name='Non Syntactic Model')
        self.ensemble_checkbox = pn.widgets.Checkbox(name='Ensemble Model')

        model_text = """
        - **Syntactic Model**: Use this model if you want precise annotations for complete songs, taking into account the actual order of phrases and the syntactic structure of the bird songs. **The best results are observed with this model.**
        - **Non Syntactic Model**: Choose this model if you need annotations based solely on the individual characteristics of syllables, without considering the context or order of phrases.
        - **Ensemble Model**: This model combines the outputs of the previous two models using a voting system, offering a compromise between the two approaches. Select this model to benefit from the advantages of both methods.

        Choose based on whether you prioritize syntactic structure (syn), category balance (nsyn), or a combination of both (ensemble)."""

        self.model_accordion = pn.Accordion(('Which model should I choose ?', model_text), width=900)

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
        .GreyCard {
            border-radius: 15px;
            background-color: #F7F7F7;
            padding: 10px;
        }
        """)

        title = pn.pane.Markdown(
            "<h1 style='text-align:center; font-size:50px'>Dataset upload</h1>",
            align='center',
            width=1000,
            css_classes=["GreyCard"],
        )

        self.sidebar.change_title("Upload")

        self.settings_view = SettingsView(parent=self)

        self.modelcheckboxes = ModelCheckboxes(self)
        self.update_ensemble_checkbox()
        self.modelcheckboxes.syn_checkbox.param.watch(self.update_ensemble_checkbox, 'value')
        self.modelcheckboxes.nsyn_checkbox.param.watch(self.update_ensemble_checkbox, 'value')
        self.modelcheckboxes.ensemble_checkbox.param.watch(self.update_ensemble_checkbox, 'value')

        self.select_format = pn.widgets.Select(name='Annotation format',
                                               options=list(crowsetta.formats.FORMATS.keys()),
                                               value='marron1csv')

        self.validate_btn = pn.widgets.Button(name="Validate", button_type="success", margin=(15, 0, 10, 20))
        self.validate_btn.on_click(self.on_click_validate)

        data_accordion_text = """Select one folder containing both the annotations and the audio files of your dataset. Alternatively, you can select one folder for the annotations and another for the audio files."""
        self.data_accordion = pn.Accordion(('Which folder(s) to select ?', data_accordion_text), width=900)

        annot_formats = crowsetta.formats.FORMATS
        format_names = list(annot_formats.keys())
        annot_formats = ", ".join(format_names)

        audio_formats = soundfile.available_formats()
        format_names = list(audio_formats.keys())
        audio_formats = ", ".join(format_names)

        self.audio_format = None
        self.annot_format = None

        format_accordion_text = f"""Annotation formats accepted : \n{annot_formats}\nAudio formats accepted : \n{audio_formats}, NPY"""
        self.format_accordion = pn.Accordion(('Which formats are accepted ?', format_accordion_text), width=900)

        self.train_btn = pn.widgets.Button(name="Train", align='start', button_type="success", disabled=True)
        self.train_btn.on_click(self.on_click_train)

        self.notification = pn.pane.Alert(alert_type='warning', visible=False, width=500)

        self.file_selector = pn.widgets.FileSelector(directory=os.getcwd(), root_directory="/",
                                                     name="Select a directory", width=900)

        self.stats = pn.pane.Markdown(f"""
        ## Data stats :
        ### Number of classes : ...
        ### Class labels : ...
        ### Total audio duration : ...
        ### Total silence duration : ...
        ### Number of annotated audio files : ...
        """)

        data = {
            '...': ['...', '...', '...', '...', '...'],
        }
        df = pd.DataFrame(data)
        df_example = df.head(5)
        self.dataframe = pn.pane.DataFrame(df_example, index=False, height=200, width=600)

        # Barplot Examples
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Count')
        ax1.set_title('Class repartition', fontweight='bold')
        plt.tight_layout()
        self.count_barplot_pane = pn.pane.Matplotlib(fig1, height=350, disabled=True)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Total duration (s)')
        ax2.set_title('Class duration', fontweight='bold')
        plt.tight_layout()
        self.time_barplot_pane = pn.pane.Matplotlib(fig2, height=350, disabled=True)

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.set_xlabel('Classes')
        ax3.set_ylabel('Duration (s)')
        ax3.set_title('Class duration distribution', fontweight='bold')
        plt.tight_layout()
        self.violin_plot_pane = pn.pane.Matplotlib(fig3, height=350, disabled=True)

        self.data_selection_accordion = pn.Accordion(('Data selection', pn.Column(
            self.modelcheckboxes,
            pn.pane.Markdown(f"## Data selection :"),
            self.file_selector,
            pn.Row(self.select_format, self.validate_btn, self.notification),
            self.data_accordion,
            self.format_accordion,
        )), width=975, active=[0])

        self.layout = pn.Row(
            pn.Column(
                title,
                pn.Column(self.data_selection_accordion,
                          margin=(20, 0, 0, 0),
                          width=1000,
                          css_classes=["GreyCard"]),
                pn.Column(self.settings_view,
                          width=1000,
                          margin=(20, 0, 0, 0),
                          css_classes=["GreyCard"]),
                width=1000,
                margin=(0, 0, 0, 5),
            ),
            pn.Column(
                pn.pane.Markdown(f"## Data head :"), self.dataframe, self.stats, self.count_barplot_pane,
                self.time_barplot_pane, self.violin_plot_pane, self.train_btn,
                css_classes=["GreyCard"],
                margin=(0, 0, 0, 20), align="start"
            ),
            margin=(20, 0, 0, 20)
        )

    def update_ensemble_checkbox(self, event=None):
        syn_checked = self.modelcheckboxes.syn_checkbox.value
        nsyn_checked = self.modelcheckboxes.nsyn_checkbox.value
        ensemble_checked = self.modelcheckboxes.ensemble_checkbox.value

        if syn_checked and nsyn_checked:
            self.modelcheckboxes.ensemble_checkbox.disabled = False
        else:
            self.modelcheckboxes.ensemble_checkbox.disabled = True
            self.modelcheckboxes.ensemble_checkbox.value = False

        Controler.annotator_names = []
        if syn_checked:
            Controler.annotator_names.append("syn")
        if nsyn_checked:
            Controler.annotator_names.append("nsyn")
        if ensemble_checked:
            Controler.annotator_names.append("ensemble")

        #TODO:Update controler annotators

    def on_click_validate(self, event):

        if self.file_selector is not None:
            selected_folders = self.file_selector.value

            if len(selected_folders) > 2 or len(selected_folders) == 0:
                self.notification.object = f"Oops! {len(selected_folders)} folder(s) selected (maximum of 2)."
                self.notification.alert_type = 'info'
                self.notification.visible = True
                self.train_btn.disabled = True
                return

            for file in selected_folders:
                if not Path(file).is_dir():
                    self.notification.object = f"Oops! {file} is not a folder."
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
                    self.notification.object = f"Oops! Incorrect number of extensions: {len(extensions)} (maximum of 2)."
                    self.notification.alert_type = 'info'
                    self.notification.visible = True
                    self.train_btn.disabled = True
                    return

                for ext in extensions:
                    if ext in [".csv", ".txt", ".xml", ".not", ".mat", ".TextGrid", ".phn", ".PHN", ".wrd", ".WRD"]:
                        annot_folder = folder
                        self.annot_format = ext
                    if ext in [".wav", ".aiff", ".aifc", ".au", ".snd", ".raw", ".paf", ".iff", ".svx", ".nist",
                               ".sf", ".voc", ".w64", ".mat4", ".mat5", ".pvf", ".xi", ".htk", ".caf", ".sd2", ".flac"]:
                        audio_folder = folder
                        self.audio_format = ext

            if not annot_folder or not audio_folder:
                self.notification.object = f"Oops! Some data is missing."
                self.notification.alert_type = 'info'
                self.notification.visible = True
                self.train_btn.disabled = True
                return

            # self.controler.create_corpus(audio_directory=audio_folder,
            #                              annots_directory=annot_folder,
            #                              spec_directory=self.file_selector.value,
            #                              config=self.settings_view.settings_widgets,  # TODO: get configuration from file if needed
            #                              annot_format=self.annot_format,
            #                              audio_ext=self.audio_format,
            #                              )

            annotator_list = [
                "syn-esn" if self.modelcheckboxes.syn_checkbox.value else None,
                "nsyn-esn" if self.modelcheckboxes.nsyn_checkbox.value else None,
                "ensemble" if self.modelcheckboxes.ensemble_checkbox.value else None
            ]
            self.controler.annotator_names = [annotator for annotator in annotator_list if annotator is not None]

            self.notification.visible = False

            # TODO: use controler.corpus.dataset here, not csv_parser
            self.update_data(*self.csv_parser(annot_folder))
            self.count_barplot_pane.visible = True
            self.time_barplot_pane.visible = True
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
        class_durations = defaultdict(float)
        class_instance_durations = defaultdict(list)  # Pour stocker les durées des instances par classe

        data_frames = []

        for filename in os.listdir(selected_folder):
            if filename.endswith('.csv'):
                num_annotated_files += 1
                file_path = os.path.join(selected_folder, filename)

                df = pd.read_csv(file_path)
                df['start'] = pd.to_numeric(df['start'], errors='coerce')
                df['end'] = pd.to_numeric(df['end'], errors='coerce')
                data_frames.append(df)

                unique_classes = df['syll'].unique()
                all_classes.update(unique_classes)

                syll_counts = df['syll'].value_counts()

                for syll, count in syll_counts.items():
                    class_counts[syll] += count

                    # Collecter les durées individuelles des instances
                    instance_durations = df[df['syll'] == syll]['end'] - df[df['syll'] == syll]['start']
                    class_instance_durations[syll].extend(instance_durations.tolist())

                for syll in unique_classes:
                    syll_duration = df[df['syll'] == syll]['end'] - df[df['syll'] == syll]['start']
                    class_durations[syll] += syll_duration.sum()

                silence_duration = df[df['syll'] == self.settings_view.silence_tag.value]['end'] - \
                                   df[df['syll'] == self.settings_view.silence_tag.value]['start']
                total_silence_duration += silence_duration.sum()

                max_end_in_file = df['end'].max()
                total_duration += max_end_in_file

                if len(data_frames) == 1:
                    if 'Unnamed: 0' in df.columns:
                        df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
                    csv_head = df.head(10)

        class_repartition = dict(class_counts)
        num_classes_total = len(all_classes)
        class_labels_total = [str(label) for label in list(all_classes)]

        return (csv_head, class_repartition, num_classes_total, class_labels_total,
                total_duration, num_annotated_files, total_silence_duration,
                dict(class_durations), dict(class_instance_durations))

    def update_data(self, csv_head, class_repartition, num_classes_total, class_labels_total, total_duration,
                    num_annotated_files, total_silence_duration, class_durations, class_instance_durations):

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

        if self.settings_view.silence_tag.value in class_repartition:
            del class_repartition[self.settings_view.silence_tag.value]
        if self.settings_view.silence_tag.value in class_durations:
            del class_durations[self.settings_view.silence_tag.value]

        sorted_items = sorted(class_repartition.items(), key=lambda item: item[1])
        classes = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]

        # Génération des couleurs
        colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))

        # Création d'un dictionnaire classe -> couleur
        class_to_color = {cls: col for cls, col in zip(classes, colors)}

        plt.figure(figsize=(10, 5) if len(classes) > 35 else (7, 4))
        bars = plt.bar(classes, counts, color=[class_to_color[cls] for cls in classes])
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.title('Class frequency', fontweight='bold')

        for bar, count in zip(bars, counts):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, f'{count}', ha='center', va='bottom')

        plt.xticks(rotation=90)
        plt.tight_layout()

        # Save the figure in the pane
        self.count_barplot_pane.object = plt.gcf()

        sorted_duration_items = sorted(class_durations.items(), key=lambda item: item[1])
        duration_classes = [item[0] for item in sorted_duration_items]
        durations = [item[1] for item in sorted_duration_items]

        plt.figure(figsize=(10, 5) if len(duration_classes) > 35 else (7, 4))

        # Utilisation des mêmes couleurs pour les mêmes classes
        duration_bars = plt.bar(duration_classes, durations, color=[class_to_color[cls] for cls in duration_classes])
        plt.xlabel('Classes')
        plt.ylabel('Total duration (s)')
        plt.title('Class duration', fontweight='bold')

        for bar, duration in zip(duration_bars, durations):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, f'{round(duration)}', ha='center', va='bottom')

        plt.xticks(rotation=90)
        plt.tight_layout()

        # Save the figure in the pane
        self.time_barplot_pane.object = plt.gcf()

        # Violin plot pour la durée des classes
        data = [class_instance_durations[cls] for cls in duration_classes if cls in class_instance_durations]

        plt.figure(figsize=(10, 5) if len(duration_classes) > 35 else (7, 4))
        vp = plt.violinplot(data, showmeans=False, showmedians=True)

        # Ajuster les couleurs des violins
        for i, cls in enumerate(duration_classes):
            if cls in class_instance_durations:
                for b in vp['bodies'][i::len(duration_classes)]:
                    b.set_facecolor(class_to_color[cls])
                    b.set_edgecolor('black')
                    b.set_alpha(1)

        plt.xticks(np.arange(1, len(duration_classes) + 1), duration_classes, rotation=90)
        plt.xlabel('Classes')
        plt.ylabel('Duration (s)')
        plt.title('Class duration distribution', fontweight='bold')
        plt.tight_layout()

        # Save the figure in the pane
        self.violin_plot_pane.object = plt.gcf()

    def on_click_train(self, events):
        #initialize_models(self)
        logger.info("Entering train dashboard.")
        self.controler.next_step(to_step="train")
