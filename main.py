import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from src.ui.ui_launcher import LauncherWindow
from src.ui.ui_model_loader import ModelLoaderPage
from src.ui.ui_dicom_converter import DicomConverterPage
from src.ui.ui_inference import InferencePage
from src.ui.ui_nifti_to_stl import NiftiToStlPage
from src.ui.ui_simulation import MammographyProSim

class AppController(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Mammography Pro System')
        self.resize(1100, 800)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.launcher = None
        self.loader = None
        self.converter = None
        self.inference_page = None
        self.mesher = None
        self.simulation = None

    def start(self):
        self.launcher = LauncherWindow()
        self.launcher.models_loaded.connect(self.switch_to_loader)
        self.stack.addWidget(self.launcher)
        self.show()

    def switch_to_loader(self):
        if AppController.models_already_installed():
            self.switch_to_converter()
        else:
            self.loader = ModelLoaderPage()
            self.loader.models_ready.connect(self.switch_to_converter)
            self.stack.addWidget(self.loader)
            self.stack.setCurrentWidget(self.loader)

    def switch_to_converter(self):
        self.converter = DicomConverterPage()
        self.converter.conversion_completed.connect(self.switch_to_inference)
        self.stack.addWidget(self.converter)
        self.stack.setCurrentWidget(self.converter)

    def switch_to_inference(self, nifti_folder):
        self.inference_page = InferencePage(nifti_folder)
        self.inference_page.inference_completed.connect(self.switch_to_mesher)
        self.stack.addWidget(self.inference_page)
        self.stack.setCurrentWidget(self.inference_page)

    def switch_to_mesher(self, hybrid_nifti_path):
        self.mesher = NiftiToStlPage(hybrid_nifti_path)
        self.mesher.processing_completed.connect(self.switch_to_main)
        self.stack.addWidget(self.mesher)
        self.stack.setCurrentWidget(self.mesher)

    def switch_to_main(self, stl_folder_path):
        self.simulation = MammographyProSim(stl_folder=stl_folder_path)
        self.stack.addWidget(self.simulation)
        self.stack.setCurrentWidget(self.simulation)
        self.showMaximized()

    def switch_to_main(self, stl_folder_path):
        print(f'>> Meshes ready in: {stl_folder_path}')
        self.mesher.close()
        self.main_window = MammographyProSim(stl_folder=stl_folder_path)
        self.main_window.showMaximized()

    def models_already_installed():
        print('Checking if models are already installed...')
        breastsegnet_path = os.path.abspath('./models/nnUNet_weights/Dataset910_BreastSegNet')
        print(breastsegnet_path)
        breastsegnet_ok = os.path.exists(breastsegnet_path) and len(os.listdir(breastsegnet_path)) > 0
        nnunet_path = os.path.abspath('./models/nnUNet_weights')
        nnunet_ok = os.path.exists(nnunet_path) and len(os.listdir(nnunet_path)) > 0
        return breastsegnet_ok and nnunet_ok
if __name__ == '__main__':
    app = QApplication(sys.argv)
    controller = AppController()
    controller.start()
    sys.exit(app.exec())