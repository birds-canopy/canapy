import logging
import os
import pickle
import wave
import numpy as np
import librosa
from pathlib import Path

import panel as pn
import soundfile

from . import View
from .settings import SettingsView

pn.extension('floatpanel')

logger = logging.getLogger("canapy-dashboard")


class AnnotateDashboard(View):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.sidebar.change_title("Annotate")

        self.settings_view = SettingsView(parent=self)

        title = pn.pane.Markdown(
            """
            <h1 style="text-align:center; font-size:50px">Annotation</h1>
            """,
            width=1000,
            align='center',
        )

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

        self.audio_btn = pn.widgets.Button(button_type='primary',button_style='outline', name='Load audio 🎵', width=150, height=75)
        self.audio_btn.on_click(self.on_click_audio)

        self.model_btn = pn.widgets.Button(button_type='primary', name='Load model 🤖', width=150, height=75)
        self.model_btn.on_click(self.on_click_model)

        self.annotate_btn = pn.widgets.Button(button_type='primary', name='Annotate', disabled=True, width=150, height=75)
        self.annotate_btn.on_click(self.on_click_annotate)

        self.notification = pn.pane.Alert(visible=False, width=900, margin=(20, 0, 0, 0), align="center")

        data_accordion_text = """Select one folder containing the audio files of your dataset."""
        self.data_accordion = pn.Accordion(('Which folder(s) to select ?', data_accordion_text),
                                           width=375, margin=(20, 0, 0, 5))

        self.audio_selector = pn.widgets.FileSelector(
            directory=os.getcwd(),
            root_directory="/",
            name="Select a directory",
            width=975,
            align='center',
            margin=(20, 0, 0, 0)
        )

        self.valid_audio_btn = pn.widgets.Button(
            button_type='success',
            name='Validate',
            width=150,
            height=50,
            margin=(20, 0, 20, 30)
        )
        self.valid_audio_btn.on_click(self.on_click_valid_audio)

        self.loading_audio = pn.indicators.LoadingSpinner(size=20, name='Loading...', visible=False, align='center',
                                                          margin=(15, 0, 0, 20))

        self.audio_stats = pn.pane.Markdown(f"""
        ## Audio stats :
        ### Number of audio files : ...
        ### Total audio duration : ...
        ### Sampling rate : ...
        ### File extension : ... 
        """, align="start")

        self.model_settings = pn.pane.Markdown(f"""
        ## Model settings :
        ### Model type : ...
        ### Units : ...
        ### Spectral Radius : ...
        ### Leakage : ...
        ### MFCC input scaling : ...
        ### MFCC derivatives input scaling : ...
        ### MFCC second derivatives input scaling : ...
        ### Ridge regularisation : ...
        ### Backend : ...
        ### Workers for the backend : ...
        """, align="start")

        self.model_selector = pn.widgets.FileInput(
            directory=os.getcwd(),
            root_directory="/",
            name="Select a model",
            align='center',
            margin=(50, 0, 0, 0),
            visible=False
        )

        self.valid_model_btn = pn.widgets.Button(
            button_type='success',
            name='Validate',
            width=150,
            height=50,
            align="center",
            margin=(50, 0, 20, 0),
            visible=False
        )
        self.valid_model_btn.on_click(self.on_click_valid_model)

        self.audio_context = pn.Row(self.data_accordion, self.valid_audio_btn, self.loading_audio,
                                    align="start")

        self.data_selection_accordion = pn.Accordion(('Data selection', pn.Column(
            pn.Row(
                pn.Column(self.audio_btn, align='center'),
                pn.Column(self.model_btn, align='center', margin=(0, 0, 0, 100)),
                pn.Column(self.annotate_btn, align='center', margin=(0, 0, 0, 100)),
                margin=(20, 0, 0, 0),
                align='center'
            ),
            self.notification,
            pn.Row(self.audio_selector, align="center"),
            self.audio_context,
            self.model_selector,
            self.valid_model_btn,align='center')), width=975, active=[0])

        self.layout = pn.Row(
            pn.Column(
                pn.Column(
                    pn.Column(
                        title,
                        align='center',
                        margin=(20, 0, 0, 20),
                        width=1000,
                        css_classes=["GreyCard"],
                    ),
                    pn.Column(
                        self.data_selection_accordion,
                        align='center',
                        margin=(20, 0, 0, 20),
                        width=1000,
                        css_classes=["GreyCard"]),
                ),
                pn.Column(self.settings_view, css_classes=["GreyCard"], margin=(20, 0, 0, 20)),
            ),
            pn.Column(
                self.audio_stats,
                self.model_settings,
                align='start', css_classes=["GreyCard"],
                sizing_mode="stretch_width", margin=(20, 20, 0, 20), height=850
            ),
            title="Canapy"
        )

    def on_click_audio(self, event):
        self.model_selector.visible = False
        self.valid_model_btn.visible = False
        self.model_btn.button_style = 'solid'
        self.audio_btn.button_style = 'outline'
        self.audio_selector.visible = True
        self.valid_audio_btn.visible = True
        self.audio_context.visible = True

    def on_click_model(self, event):
        self.audio_context.visible = False
        self.audio_selector.visible = False
        self.valid_audio_btn.visible = False
        self.model_btn.button_style = 'outline'
        self.audio_btn.button_style = 'solid'
        self.model_selector.visible = True
        self.valid_model_btn.visible = True

    def on_click_valid_audio(self, event):
        self.loading_audio.value = True
        self.loading_audio.visible = True
        if self.audio_selector is not None:
            selected_folders = self.audio_selector.value
            extensions_by_folder, audio_count, audio_duration, sample_rates = self.get_extensions(selected_folders)
            hours, minutes, seconds = self.seconds_to_hms(audio_duration)
            audio_duration = f"{hours:02}:{minutes:02}:{seconds:02}"
            extensions = [ext for folder, ext in extensions_by_folder.items()]

            if len(set(sample_rates)) != 1:
                self.audio_btn.button_type = "warning"
                self.notification.object = "Different sampling rates have been detected"
                self.notification.alert_type = 'warning'
                self.notification.visible = True

            if extensions == [{'.wav'}] or extensions == [{'.npy'}]:
                self.audio_btn.button_type = "success"
                self.notification.visible = False
                self.update_stats(audio_count, audio_duration, extensions, sample_rates[0])
            else:
                self.audio_btn.button_type = "warning"
                self.notification.object = "Invalid audio folder selected"
                self.notification.alert_type = 'warning'
                self.notification.visible = True

        self.annotate_btn.disabled = not (
                self.audio_btn.button_type == "success" and self.model_btn.button_type == "success")

        self.loading_audio.value = False
        self.loading_audio.visible = False

    def get_extensions(self, directories):

        audio_formats = soundfile.available_formats()
        format_names = list(audio_formats.keys()) + ['NPY']
        audio_formats = ['.' + fmt.lower() for fmt in format_names]
        sample_rates = []
        file_counter = 0
        audio_duration = 0
        extensions_by_folder = {}
        for directory in directories:
            extensions = set()
            for file_path in Path(directory).rglob('*'):
                if file_path.is_file():
                    extensions.add(file_path.suffix)
                    if file_path.suffix in audio_formats:
                        duration, sr = self.get_audio_duration_and_sr(file_path)
                        audio_duration += duration
                        sample_rates.append(sr)
                        file_counter += 1
            extensions_by_folder[directory] = extensions

        return extensions_by_folder, file_counter, audio_duration, sample_rates

    def get_audio_duration_and_sr(self, file_path):
        y, sr = librosa.load(file_path, sr=None)  # Load the audio file with its original sample rate
        duration = librosa.get_duration(y=y, sr=sr)
        return duration, sr

    def seconds_to_hms(self, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return int(hours), int(minutes), int(seconds)

    def on_click_valid_model(self, event):

        self.model_selector.save('model.pkl')

        with open('model.pkl', 'rb') as file:
            data = pickle.load(file)

        print(data)

        if self.model_selector.filename in ["syn-esn", "nsyn-esn", "ensemble"]:
            self.model_btn.button_type = "success"
            self.notification.visible = False
        else:
            print("Wrong file")
            self.model_btn.button_type = "warning"
            self.notification.object = "Invalid model selected"
            self.notification.alert_type = 'warning'
            self.notification.visible = True

        self.annotate_btn.disabled = not (
                self.audio_btn.button_type == "success" and self.model_btn.button_type == "success")

    def update_stats(self, audio_count, audio_duration, extensions, sampling_rate):
        self.audio_stats.object = f"""
        ## Audio stats :
        ### Number of audio files : {audio_count}
        ### Total audio duration : {audio_duration}
        ### Sampling rate : {sampling_rate}
        ### File extension : {extensions[0]}    
        """

    def on_click_annotate(self, event):
        logger.info("Entering export dashboard.")
        self.controler.next_step(to_step="export")

    def __panel__(self):
        return self.layout


if __name__ == "__main__":
    dashboard = AnnotateDashboard()
    pn.serve(dashboard.layout)
