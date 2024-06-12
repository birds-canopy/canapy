import logging
import time
import os
import panel as pn
import crowsetta

from ..helpers import SubDash
from ..helpers import SideBar
from pathlib import Path

logger = logging.getLogger("canapy-dashboard")

class WelcomeDashboard(pn.viewable.Viewer):

    def __init__(self, parent=None):
        super().__init__()

        self.parent = parent

        self.search_btn = pn.widgets.Button(name="Select file")
        self.validate_btn = pn.widgets.Button(name="Validate")

        self.notification = pn.pane.Alert(alert_type='danger', margin=(10, 15),visible=False)

        self.file_selector = None

        self.search_btn.on_click(self.on_click_browse)
        self.validate_btn.on_click(self.on_click_validate)

        self.menu_audio = pn.widgets.MenuButton(name='Dropdown', items=menu_items, button_type='primary')
        self.menu_annot = pn.widgets.MenuButton(name='Dropdown', items=menu_items, button_type='primary')

        self.layout = pn.Column("Titre", self.search_btn, self.validate_btn, self.notification)

    def on_click_browse(self,events):
        if self.file_selector is None:
            self.file_selector = pn.widgets.FileSelector(directory=os.getcwd(),root_directory="/",name="Select a directory")
            self.layout.append(self.file_selector)

    def on_click_validate(self, event):
        if self.file_selector is not None:
            selected_files = self.file_selector.value

            if len(selected_files)>2 or len(selected_files)==0 :
                self.notification.object = f"Erreur: {len(selected_files)} éléments sélectionnés (2 max)."
                self.notification.alert_type = 'danger'
                self.notification.visible = True
                return

            for file in selected_files:
                if not Path(file).is_dir():
                    logger.error("Ce n'est pas un dossier: %s", file)
                    self.notification.object = f"Erreur: {file} n'est pas un dossier"
                    self.notification.alert_type = 'danger'
                    self.notification.visible = True
                    return

            formats = crowsetta.formats.FORMATS

            format_extensions = []

            for format_name, format_class in formats.items():
                extensions = getattr(format_class, 'ext', None)
                if extensions:
                    if isinstance(extensions, (list,tuple)):
                        for ext in extensions:
                            format_extensions.append((format_name, ext))
                    else:
                        format_extensions.append((format_name, extensions))

            extensions_by_folder = self.get_extensions(selected_files)
            for folder, extensions in extensions_by_folder.items():

                if len(extensions) > 2:
                    self.notification.object = f"Erreur: Nombre d'extensions incorrect:{len(extensions)} (2 max)."
                    self.notification.alert_type = 'danger'
                    self.notification.visible = True
                    return

                valid_formats = []
                for ext in extensions:
                    for format_name, format_ext in format_extensions:
                        if ext == format_ext:
                            valid_formats.append(format_name)
                            annot_folder = folder
                    if ext in [".npy",".wav"] :
                        annot_audio = folder

            menu_ext_audio = [('Option A', 'a'), ('Option B', 'b'), ('Option C', 'c')]
            menu_ext_annot = [('Option A', 'a'), ('Option B', 'b'), ('Option C', 'c')]



            self.notification.object = "La sélection est validée."
            self.notification.alert_type = 'success'
            self.notification.visible = True

    def get_extensions(self, directories):
        extensions_by_folder = {}
        for directory in directories:
            extensions = set()
            for file_path in Path(directory).rglob('*'):
                if file_path.is_file():
                    extensions.add(file_path.suffix)
            extensions_by_folder[directory] = extensions
        return extensions_by_folder

    def __panel__(self):
        return self.layout

    def browsing(self):
        files = pn.widgets.FileSelector('~')






