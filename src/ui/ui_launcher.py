import sys
import time
import random
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QApplication
from PySide6.QtGui import QFont, QPalette, QColor
from PySide6.QtCore import Qt, QThread, Signal, QObject

class LoaderWorker(QObject):
    progress = Signal(int)
    log_text = Signal(str)
    finished = Signal()

    def run(self):
        steps = ['Initializing CUDA kernels...', 'Allocating VRAM...', 'Loading Base-UNet weights (250MB)...', 'Loading SAM (Segment Anything Model) adapter...', 'Verifying TensorRT engines...', 'Calibrating physics engine...', 'System Ready.']
        for i, step in enumerate(steps):
            time.sleep(random.uniform(0.3, 0.8))
            self.log_text.emit(step)
            prog = int((i + 1) / len(steps) * 100)
            self.progress.emit(prog)
        time.sleep(0.5)
        self.finished.emit()

class LauncherWindow(QMainWindow):
    models_loaded = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle('3D Breast Simulation - Initialization')
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.showFullScreen()
        self.setStyleSheet('background-color: #1E1F22;')
        self.setup_ui()
        self.start_loading()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addStretch(1)
        content_container = QWidget()
        content_container.setStyleSheet('background-color: transparent;')
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(100, 0, 100, 0)
        content_layout.setSpacing(15)
        self.lbl_title = QLabel('3D BREAST SIMULATION')
        self.lbl_title.setFont(QFont('Segoe UI', 32, QFont.Bold))
        self.lbl_title.setStyleSheet('color: #E6E8EB; letter-spacing: 2px;')
        self.lbl_title.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.lbl_title)
        self.lbl_subtitle = QLabel('DIAGNOSTIC & SURGICAL PLANNING SUITE')
        self.lbl_subtitle.setFont(QFont('Segoe UI', 12, QFont.Medium))
        self.lbl_subtitle.setStyleSheet('color: #B4B8BE; letter-spacing: 1px;')
        self.lbl_subtitle.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.lbl_subtitle)
        content_layout.addSpacing(40)
        self.pbar = QProgressBar()
        self.pbar.setFixedHeight(6)
        self.pbar.setTextVisible(False)
        self.pbar.setStyleSheet('\n            QProgressBar {\n                background-color: #141516;\n                border: 1px solid #34363B;\n                border-radius: 3px;\n            }\n            QProgressBar::chunk {\n                background-color: #0F7F6E;\n                border-radius: 3px;\n            }\n        ')
        content_layout.addWidget(self.pbar)
        info_layout = QHBoxLayout()
        self.lbl_status = QLabel('Loading system modules...')
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
        lbl_footer = QLabel('Authorized Use Only - Clinical ID: 884-XJ-09')
        lbl_footer.setFont(QFont('Segoe UI', 9))
        lbl_footer.setStyleSheet('color: #34363B; margin-bottom: 20px;')
        lbl_footer.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(lbl_footer)

    def start_loading(self):
        self.thread = QThread()
        self.worker = LoaderWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def update_progress(self, value):
        self.pbar.setValue(value)
        self.lbl_percent.setText(f'{value}%')
        if value < 30:
            self.lbl_status.setText('Initializing core kernel...')
        elif value < 70:
            self.lbl_status.setText('Loading simulation physics...')
        elif value < 90:
            self.lbl_status.setText('Verifying diagnostic engines...')
        else:
            self.lbl_status.setText('Finalizing...')

    def on_finished(self):
        self.models_loaded.emit()
        self.close()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont('Segoe UI')
    font.setHintingPreference(QFont.PreferFullHinting)
    app.setFont(font)
    win = LauncherWindow()
    sys.exit(app.exec())