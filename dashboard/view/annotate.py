import logging
import os
import wave
import numpy as np
from pathlib import Path

import panel as pn

pn.extension('floatpanel')

logger = logging.getLogger("canapy-dashboard")


class SideBar(pn.viewable.Viewer):
    def __init__(self, parent, title):
        super().__init__()
        self.parent = parent

        self.back_btn = pn.widgets.Button(button_type='warning', icon="arrow-back-up", name="Back")
        self.back_btn.on_click(self.on_click_back)

        self.quit_btn = pn.widgets.Button(
            name="Quit",
            button_type="danger",
            icon="square-rounded-x",
        )
        self.quit_btn.on_click(self.on_click_stop)

        self.settings_button = pn.widgets.Button(button_type='primary', icon="settings",
                                                 name="Settings", css_classes=['no-style'])
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

    def on_click_settings(self, event):
        self.settingspanel.visible = not self.settingspanel.visible

    def on_click_back(self, event):
        self.parent.controler.next_step(to="home")

    def on_click_stop(self, events):
        confirm_script = """
        <script>
            if (confirm("Are you sure you want to stop the server and close this tab?")) {
                window.close();
            }
        </script>
        """
        self.layout.append(pn.pane.HTML(confirm_script))

    def __panel__(self):
        return self.layout


class AnnotateDashboard(pn.viewable.Viewer):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent

        self.sidebar = SideBar(self, "Annotate")

        title = pn.pane.Markdown(
            """
            <h1 style="text-align:center; font-size:50px">Annotation</h1>
            """,
            align='center',
            width=1000,
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
        """)

        self.audio_btn = pn.widgets.Button(button_type='primary', name='Load audios',
                                           disabled=True, width=150, height=75)
        self.audio_btn.on_click(self.on_click_audio)

        self.model_btn = pn.widgets.Button(button_type='primary', name='Load model', width=150, height=75)
        self.model_btn.on_click(self.on_click_model)

        self.annotate_btn = pn.widgets.Button(button_type='success', name='Annotate', width=150, height=75)
        self.annotate_btn.on_click(self.on_click_annotate)

        self.audio_check = pn.widgets.Button(name="🎵", size="4em", description="Audio", align='center')
        self.model_check = pn.widgets.Button(name="🤖", size="4em", description="Model", align='center')

        self.notification = pn.pane.Alert(visible=False, width=1000, margin=(20, 0, 0, 0), align="center")

        data_accordion_text = """Select one folder containing the audio files of your dataset."""
        self.data_accordion = pn.Accordion(('Which folder(s) to select ?', data_accordion_text),
                                           width=375, margin=(20, 0, 0, 5))

        self.audio_selector = pn.widgets.FileSelector(
            directory=os.getcwd(),
            root_directory="/",
            name="Select a directory",
            width=1000,
            align='center',
            margin=(20, 0, 0, 0)
        )

        self.valid_audio_btn = pn.widgets.Button(
            button_type='success',
            name='Validate',
            width=150,
            height=50,
            margin=(20, 0, 0, 0)
        )
        self.valid_audio_btn.on_click(self.on_click_valid_audio)

        self.stats = pn.pane.Markdown(f"""
        ## Audio stats :
        ### Number of audio files : ...
        ### Total audio duration : ...
        ### File extension : ... 
        """, margin=(0, 0, 0, 0), align="start")

        self.model_selector = pn.widgets.FileInput(
            directory=os.getcwd(),
            root_directory="/",
            name="Select a model",
            align='center',
            margin=(20, 0, 0, 0),
            visible=False
        )

        self.valid_model_btn = pn.widgets.Button(
            button_type='success',
            name='Validate',
            width=150,
            height=50,
            align="center",
            margin=(20, 0, 0, 0),
            visible=False
        )
        self.valid_model_btn.on_click(self.on_click_valid_model)

        self.audio_context = pn.Row(self.data_accordion, pn.Spacer(width=480),self.valid_audio_btn, align="start")

        self.layout = pn.Row(
            self.sidebar,
            pn.Column(
                pn.Column(
                    title,
                    align='center',
                    margin=(75, 0, 0, 0),
                ),
                pn.Row(
                    pn.Column(self.audio_btn, align='center', margin=(0, 0, 0, 50)),
                    pn.Column(self.model_btn, align='center', margin=(0, 0, 0, 100)),
                    pn.Column(self.audio_check, self.model_check, align='center', margin=(0, 0, 0, 100)),
                    pn.Column(self.annotate_btn, align='center'),
                    margin=(20, 0, 0, 0),
                    align='center'
                ),
                self.notification,
                pn.Row(self.audio_selector, align="center"),
                self.audio_context,
                self.model_selector,
                self.valid_model_btn,
                margin=(0, 0, 0, 300)
            ),
            pn.Column(
                self.stats, align='center', margin=(0, 0, 0, 50)
            ),
            title="Canapy"
        )

        if self.audio_check.button_type == "success" and self.model_check.button_type == "success":
            self.annotate_btn.disabled = False
        else:
            self.annotate_btn.disabled = True

    def on_click_audio(self, event):
        self.audio_btn.disabled = True
        self.model_btn.disabled = False
        self.model_selector.visible = False
        self.valid_model_btn.visible = False
        self.audio_selector.visible = True
        self.valid_audio_btn.visible = True
        self.audio_context.visible = True

    def on_click_model(self, event):
        self.audio_btn.disabled = False
        self.model_btn.disabled = True
        self.audio_context.visible = False
        self.audio_selector.visible = False
        self.valid_audio_btn.visible = False
        self.model_selector.visible = True
        self.valid_model_btn.visible = True

    def on_click_valid_audio(self, event):
        if self.audio_selector is not None:
            selected_folders = self.audio_selector.value
            extensions_by_folder, audio_count, audio_duration = self.get_extensions(selected_folders)
            hours, minutes, seconds = self.seconds_to_hms(audio_duration)
            audio_duration = f"{hours:02}:{minutes:02}:{seconds:02}"
            extensions = [ext for folder, ext in extensions_by_folder.items()]
            print(extensions)
            if extensions == [{'.wav'}] or extensions == [{'.npy'}]:
                self.audio_check.button_type = "success"
                self.notification.visible = False
                self.update_stats(audio_count, audio_duration, extensions)
            else:
                self.audio_check.button_type = "warning"
                self.notification.object = "Invalid audio folder selected"
                self.notification.alert_type = 'warning'
                self.notification.visible = True

    def get_extensions(self, directories):
        file_counter = 0
        audio_duration = 0
        extensions_by_folder = {}
        sample_rate = 44100
        for directory in directories:
            extensions = set()
            for file_path in Path(directory).rglob('*'):
                if file_path.is_file():
                    extensions.add(file_path.suffix)
                    converted_file_path = str(file_path).replace("\\", "\\\\")
                    if file_path.suffix == ".wav":
                        audio_duration += self.get_wav_duration(converted_file_path)
                        file_counter += 1
                    elif file_path.suffix == ".npy":
                        audio_duration += self.get_npy_duration(converted_file_path, sample_rate)
                        file_counter += 1
            extensions_by_folder[directory] = extensions
        return extensions_by_folder, file_counter, audio_duration

    def get_wav_duration(self, file_path):
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration

    def get_npy_duration(self, file_path, sample_rate):
        data = np.load(file_path)
        num_samples = data.shape[0]
        duration = num_samples / float(sample_rate)
        return duration

    def seconds_to_hms(self, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return int(hours), int(minutes), int(seconds)

    def on_click_valid_model(self, event):
        if self.model_selector.filename in ["syn-esn", "nsyn-esn", "ensemble"]:
            self.model_check.button_type = "success"
            self.notification.visible = False
        else:
            print("Wrong file")
            self.model_check.button_type = "warning"
            self.notification.object = "Invalid model selected"
            self.notification.alert_type = 'warning'
            self.notification.visible = True

    def update_stats(self, audio_count, audio_duration, extensions):
        self.stats.object = f"""
        ## Audio stats :
        ### Number of audio files : {audio_count}
        ### Total audio duration : {audio_duration}
        ### File extension : {extensions[0]}    
        """

    def on_click_annotate(self, event):
        print("Annotate")

    def __panel__(self):
        return self.layout


if __name__ == "__main__":
    dashboard = AnnotateDashboard()
    pn.serve(dashboard.layout)
