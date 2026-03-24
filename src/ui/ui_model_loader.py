import sys
import os
import time
import zipfile
import requests
import gdown
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QWidget, QApplication, QMessageBox
from PySide6.QtGui import QFont, QColor
from PySide6.QtCore import Qt, QThread, Signal, QObject

class ModelDownloadWorker(QObject):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def run(self):
        try:
            if not os.path.exists('./models'):
                os.makedirs('./models')
            self.download_breastsegnet()
            self.download_nnunet()
            self.status.emit('All downloads finished successfully.')
            self.progress.emit(100)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def download_breastsegnet(self):
        target_folder = os.path.abspath('./BreastSegNet_models')
        file_id = '1o9lcFrPDA2UGNolvsyvzOE4YvNFojz1q'
        zip_name = 'BreastSegNet_weights.zip'
        if os.path.exists(target_folder) and len(os.listdir(target_folder)) > 0:
            self.status.emit('BreastSegNet models found. Skipping.')
            self.progress.emit(30)
            return
        self.status.emit('Downloading BreastSegNet (G-Drive)...')
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, zip_name, quiet=False, fuzzy=True)
        self.progress.emit(15)
        if os.path.exists(zip_name):
            self.status.emit('Extracting BreastSegNet...')
            with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove(zip_name)
        self.progress.emit(30)

    def download_nnunet(self):
        download_path = os.path.abspath('./models/nnUNet_weights')
        url = 'https://zenodo.org/records/11998679/files/Dataset009_Breast.zip?download=1'
        target_file = os.path.join('./models', 'Dataset009_Breast.zip')
        if os.path.exists(download_path) and len(os.listdir(download_path)) > 0:
            self.status.emit('nnU-Net weights found. Skipping.')
            self.progress.emit(100)
            return
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        self.status.emit('Starting nnU-Net download (Zenodo)...')
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            with requests.get(url, headers=headers, stream=True) as response:
                if response.status_code == 403:
                    raise Exception('Access Denied (403).')
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                wrote = 0
                start_time = time.time()
                last_update_time = time.time()
                with open(target_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            wrote += len(chunk)
                            f.write(chunk)
                            current_time = time.time()
                            if current_time - last_update_time > 0.1:
                                elapsed_total = current_time - start_time
                                if elapsed_total > 0:
                                    speed = wrote / 1024 / 1024 / elapsed_total
                                    mb_wrote = wrote / 1024 / 1024
                                    mb_total = total_size / 1024 / 1024
                                    if total_size > 0:
                                        raw_percent = wrote / total_size
                                        ui_percent = 30 + int(raw_percent * 70)
                                        self.progress.emit(ui_percent)
                                        status_msg = f'Downloading AI Assets: {mb_wrote:.1f}MB / {mb_total:.1f}MB ({speed:.1f} MB/s)'
                                        self.status.emit(status_msg)
                                last_update_time = current_time
            self.status.emit('Extracting files... (This may take a moment)')
            if zipfile.is_zipfile(target_file):
                with zipfile.ZipFile(target_file, 'r') as zip_ref:
                    zip_ref.extractall(download_path)
                os.remove(target_file)
            self.progress.emit(100)
        except Exception as e:
            self.error.emit(f'Download Error: {str(e)}')
            raise e

class ModelLoaderPage(QMainWindow):
    models_ready = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.showFullScreen()
        self.setStyleSheet('background-color: #1E1F22;')
        self.setup_ui()
        self.start_download()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addStretch(1)
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(100, 0, 100, 0)
        content_layout.setSpacing(15)
        lbl_title = QLabel('AI RESOURCE INSTALLATION')
        lbl_title.setFont(QFont('Segoe UI', 32, QFont.Bold))
        lbl_title.setStyleSheet('color: #E6E8EB; letter-spacing: 2px;')
        lbl_title.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(lbl_title)
        lbl_subtitle = QLabel('3D BREAST SIMULATION - MODEL REPOSITORY')
        lbl_subtitle.setFont(QFont('Segoe UI', 12, QFont.Medium))
        lbl_subtitle.setStyleSheet('color: #B4B8BE; letter-spacing: 1px;')
        lbl_subtitle.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(lbl_subtitle)
        content_layout.addSpacing(40)
        self.pbar = QProgressBar()
        self.pbar.setFixedHeight(6)
        self.pbar.setTextVisible(False)
        self.pbar.setStyleSheet('\n            QProgressBar {\n                background-color: #141516;\n                border: 1px solid #34363B;\n                border-radius: 3px;\n            }\n            QProgressBar::chunk {\n                background-color: #0F7F6E;\n                border-radius: 3px;\n            }\n        ')
        content_layout.addWidget(self.pbar)
        info_layout = QHBoxLayout()
        self.lbl_status = QLabel('Connecting to server...')
        self.lbl_status.setFont(QFont('Segoe UI', 10))
        self.lbl_status.setStyleSheet('color: #7A7F87;')
        self.lbl_status.setAlignment(Qt.AlignLeft)
        self.lbl_percent = QLabel('0%')
        self.lbl_percent.setFont(QFont('Segoe UI', 10, QFont.Bold))
        self.lbl_percent.setStyleSheet('color: #0F7F6E;')
        self.lbl_percent.setAlignment(Qt.AlignRight)
        info_layout.addWidget(self.lbl_status)
        info_layout.addWidget(self.lbl_percent)
        content_layout.addLayout(info_layout)
        main_layout.addWidget(content_container)
        main_layout.addStretch(1)
        lbl_footer = QLabel('Secured Connection - Resource Manager v2.0')
        lbl_footer.setFont(QFont('Segoe UI', 9))
        lbl_footer.setStyleSheet('color: #34363B; margin-bottom: 20px;')
        lbl_footer.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(lbl_footer)

    def start_download(self):
        self.thread = QThread()
        self.worker = ModelDownloadWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self.lbl_status.setText)
        self.worker.progress.connect(self.update_ui_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def update_ui_progress(self, value):
        self.pbar.setValue(value)
        self.lbl_percent.setText(f'{value}%')

    def on_finished(self):
        self.lbl_status.setText('All resources installed.')
        QMessageBox.information(self, 'Success', 'AI Models installed successfully.')
        self.models_ready.emit()
        self.close()

    def on_error(self, err_msg):
        self.lbl_status.setStyleSheet('color: #ef5350;')
        QMessageBox.critical(self, 'Download Error', f'Failed to download models:\n{err_msg}')
        self.close()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont('Segoe UI')
    font.setHintingPreference(QFont.PreferFullHinting)
    app.setFont(font)
    win = ModelLoaderPage()
    sys.exit(app.exec())