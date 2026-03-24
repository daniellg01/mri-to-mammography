import sys
import os
import time
import gc
import tempfile
import shutil
import numpy as np
import nibabel as nib
import trimesh
from skimage import measure
from PySide6.QtCore import Qt, QThread, Signal, QObject, QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QProgressBar, QPushButton, QRadioButton, QHBoxLayout, QFrame, QGridLayout, QGraphicsOpacityEffect
from PySide6.QtGui import QFont, QColor, QPalette
COLORS = {'background': '#1E1F22', 'surface': '#2B2D31', 'primary': '#0F7F6E', 'primary_dim': '#08453C', 'text_main': '#E6E8EB', 'text_sec': '#7A7F87', 'danger': '#D32F2F', 'border': '#3F4148'}
STYLESHEET = f"""\n    QMainWindow {{ background-color: {COLORS['background']}; }}\n    QWidget {{ font-family: 'Segoe UI', 'Inter', sans-serif; color: {COLORS['text_main']}; }}\n    \n    /* Contenedores */\n    QFrame#Container {{\n        background-color: {COLORS['surface']};\n        border: 1px solid {COLORS['border']};\n        border-radius: 5px;\n    }}\n\n    /* Títulos */\n    QLabel#Title {{\n        font-size: 18px;\n        font-weight: 600;\n        letter-spacing: 2px;\n        color: {COLORS['text_main']};\n        text-transform: uppercase;\n    }}\n    \n    QLabel#StatusLabel {{\n        font-size: 14px;\n        color: {COLORS['primary']};\n        font-weight: bold;\n    }}\n    \n    QLabel#SubLabel {{\n        font-size: 12px;\n        color: {COLORS['text_sec']};\n    }}\n\n    /* Radio Buttons estilo "Chips" */\n    QRadioButton {{\n        color: {COLORS['text_sec']};\n        spacing: 8px;\n        padding: 5px;\n    }}\n    QRadioButton::indicator {{\n        width: 14px;\n        height: 14px;\n        border-radius: 8px;\n        border: 2px solid {COLORS['text_sec']};\n    }}\n    QRadioButton::indicator:checked {{\n        border: 2px solid {COLORS['primary']};\n        background-color: {COLORS['primary']};\n    }}\n\n    /* Botones */\n    QPushButton {{\n        background-color: {COLORS['primary_dim']};\n        color: {COLORS['primary']};\n        border: 1px solid {COLORS['primary']};\n        border-radius: 4px;\n        padding: 10px 20px;\n        font-weight: 600;\n        font-size: 13px;\n        letter-spacing: 0.5px;\n    }}\n    QPushButton:hover {{\n        background-color: {COLORS['primary']};\n        color: #FFFFFF;\n    }}\n    QPushButton:disabled {{\n        background-color: {COLORS['surface']};\n        color: {COLORS['text_sec']};\n        border-color: {COLORS['border']};\n    }}\n\n    /* Barra de Progreso Minimalista */\n    QProgressBar {{\n        background-color: {COLORS['surface']};\n        border: none;\n        border-radius: 3px;\n        height: 6px;\n        text-align: center;\n    }}\n    QProgressBar::chunk {{\n        background-color: {COLORS['primary']};\n        border-radius: 3px;\n    }}\n"""
MODEL_HYBRID_CONFIG = {'FGT': 1, 'Vessels': 2, 'Muscle': 3, 'Bone': 4, 'Fat': 5, 'Skin': 6, 'Nipple': 7, 'Lymphatics': 8, 'Ligaments': 9}

class StageIndicator(QWidget):

    def __init__(self, name, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        self.status_dot = QLabel('●')
        self.status_dot.setStyleSheet(f"color: {COLORS['text_sec']}; font-size: 10px;")
        self.lbl_name = QLabel(name)
        self.lbl_name.setStyleSheet(f"color: {COLORS['text_sec']}; font-size: 13px;")
        layout.addWidget(self.status_dot)
        layout.addWidget(self.lbl_name)
        layout.addStretch()
        self.status = 'PENDING'

    def set_status(self, status):
        self.status = status
        if status == 'PROCESSING':
            self.status_dot.setStyleSheet(f"color: {COLORS['primary']}; font-size: 14px;")
            self.lbl_name.setStyleSheet(f"color: {COLORS['text_main']}; font-weight: bold;")
        elif status == 'DONE':
            self.status_dot.setText('✓')
            self.status_dot.setStyleSheet(f"color: {COLORS['primary']}; font-size: 14px;")
            self.lbl_name.setStyleSheet(f"color: {COLORS['text_sec']}; text-decoration: none;")
        elif status == 'SKIPPED':
            self.status_dot.setText('-')
            self.status_dot.setStyleSheet(f"color: {COLORS['border']};")
            self.lbl_name.setStyleSheet(f"color: {COLORS['border']};")
        elif status == 'ERROR':
            self.status_dot.setText('✕')
            self.status_dot.setStyleSheet(f"color: {COLORS['danger']};")

class NiftiToStlWorker(QObject):
    progress = Signal(int)
    status_text = Signal(str)
    stage_update = Signal(str, str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, nifti_path, use_ram_mode=True, base_output_dir='patient_repository'):
        super().__init__()
        self.nifti_path = nifti_path
        self.use_ram_mode = use_ram_mode
        self.base_output_dir = base_output_dir
        self.temp_dir = None

    def run(self):
        try:
            self.status_text.emit('INITIALIZING ENGINE')
            self.progress.emit(5)
            if not os.path.exists(self.nifti_path):
                raise FileNotFoundError('Input file not found')
            filename_base = os.path.basename(self.nifti_path).split('.')[0]
            patient_id = filename_base.replace('_HYBRID', '').replace('_0000', '')
            output_folder = os.path.join(self.base_output_dir, f'Patient_{patient_id}')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            self.status_text.emit('LOADING VOLUMETRIC DATA')
            img = nib.load(self.nifti_path)
            affine = img.affine
            if self.use_ram_mode:
                raw_data = img.get_fdata()
                mask_data = np.squeeze(raw_data)
                del raw_data
            else:
                self.status_text.emit('MAPPING DISK CACHE')
                self.temp_dir = tempfile.mkdtemp()
                temp_file = os.path.join(self.temp_dir, 'vol_cache.dat')
                temp_arr = img.get_fdata()
                temp_arr = np.squeeze(temp_arr)
                dtype = np.uint8
                mask_data = np.memmap(temp_file, dtype=dtype, mode='w+', shape=temp_arr.shape)
                mask_data[:] = temp_arr.astype(dtype)[:]
                mask_data.flush()
                del temp_arr
                gc.collect()
            if self.use_ram_mode:
                unique_labels = np.unique(mask_data).astype(int)
            else:
                unique_labels = np.unique(mask_data)
            total_tissues = len(MODEL_HYBRID_CONFIG)
            processed_count = 0
            generated_files = []
            for tissue_name, label_id in MODEL_HYBRID_CONFIG.items():
                if label_id not in unique_labels:
                    self.stage_update.emit(tissue_name, 'SKIPPED')
                    processed_count += 1
                    continue
                self.status_text.emit(f'PROCESSING: {tissue_name.upper()}')
                self.stage_update.emit(tissue_name, 'PROCESSING')
                try:
                    tissue_mask = mask_data == label_id
                    if np.count_nonzero(tissue_mask) < 50:
                        self.stage_update.emit(tissue_name, 'SKIPPED')
                        continue
                    verts, faces, normals, values = measure.marching_cubes(tissue_mask, level=0.5, step_size=1)
                    verts = (affine[:3, :3] @ verts.T).T + affine[:3, 3]
                    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                    if self.use_ram_mode:
                        try:
                            trimesh.smoothing.filter_laplacian(mesh, iterations=2)
                        except:
                            pass
                    out_name = f'{patient_id}_{tissue_name}.stl'
                    mesh.export(os.path.join(output_folder, out_name))
                    generated_files.append(out_name)
                    self.stage_update.emit(tissue_name, 'DONE')
                except Exception as e:
                    print(e)
                    self.stage_update.emit(tissue_name, 'ERROR')
                if 'tissue_mask' in locals():
                    del tissue_mask
                if 'mesh' in locals():
                    del mesh
                if 'verts' in locals():
                    del verts
                gc.collect()
                processed_count += 1
                prog_val = 20 + int(processed_count / total_tissues * 80)
                self.progress.emit(prog_val)
            if not generated_files:
                print('Warning: No STL files generated (Empty Segmentation)')
                self.status_text.emit('COMPLETED (NO TISSUES)')
                self.progress.emit(100)
                self.finished.emit(output_folder)
            else:
                if self.temp_dir:
                    try:
                        del mask_data
                        gc.collect()
                        shutil.rmtree(self.temp_dir)
                    except:
                        pass
                self.status_text.emit('COMPLETED')
                self.progress.emit(100)
                self.finished.emit(output_folder)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class NiftiToStlPage(QMainWindow):
    processing_completed = Signal(str)

    def __init__(self, nifti_path):
        super().__init__()
        self.nifti_path = nifti_path
        self.setWindowTitle('Medical Analysis Interface')
        self.setWindowState(Qt.WindowMaximized)
        self.setStyleSheet(STYLESHEET)
        self.tissue_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(25)
        header_layout = QHBoxLayout()
        lbl_title = QLabel('RECONSTRUCTION ANALYSIS')
        lbl_title.setObjectName('Title')
        self.lbl_file_info = QLabel(f'Source: {os.path.basename(self.nifti_path)}')
        self.lbl_file_info.setObjectName('SubLabel')
        self.lbl_file_info.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header_layout.addWidget(lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(self.lbl_file_info)
        main_layout.addLayout(header_layout)
        status_frame = QFrame()
        status_frame.setObjectName('Container')
        status_layout = QVBoxLayout(status_frame)
        status_layout.setContentsMargins(30, 30, 30, 30)
        self.lbl_main_status = QLabel('READY TO START')
        self.lbl_main_status.setObjectName('StatusLabel')
        self.lbl_main_status.setAlignment(Qt.AlignCenter)
        self.lbl_main_status.setStyleSheet(f"font-size: 24px; color: {COLORS['text_main']};")
        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        status_layout.addWidget(self.lbl_main_status)
        status_layout.addSpacing(15)
        status_layout.addWidget(self.pbar)
        main_layout.addWidget(status_frame)
        lbl_stages = QLabel('DIAGNOSTIC STAGES')
        lbl_stages.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {COLORS['text_sec']}; letter-spacing: 1px;")
        main_layout.addWidget(lbl_stages)
        stages_frame = QFrame()
        stages_frame.setObjectName('Container')
        stages_layout = QGridLayout(stages_frame)
        stages_layout.setContentsMargins(20, 20, 20, 20)
        row, col = (0, 0)
        for tissue in MODEL_HYBRID_CONFIG.keys():
            widget = StageIndicator(tissue)
            self.tissue_widgets[tissue] = widget
            stages_layout.addWidget(widget, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1
        main_layout.addWidget(stages_frame)
        main_layout.addStretch()
        controls_layout = QHBoxLayout()
        self.rb_ram = QRadioButton('RAM Mode (Fast)')
        self.rb_ram.setChecked(True)
        self.rb_disk = QRadioButton('Disk Mode (Safe)')
        self.btn_action = QPushButton('INITIALIZE SEQUENCE')
        self.btn_action.setFixedWidth(200)
        self.btn_action.clicked.connect(self.start_sequence)
        controls_layout.addWidget(self.rb_ram)
        controls_layout.addWidget(self.rb_disk)
        controls_layout.addStretch()
        controls_layout.addWidget(self.btn_action)
        main_layout.addLayout(controls_layout)

    def start_sequence(self):
        self.btn_action.setEnabled(False)
        self.rb_ram.setEnabled(False)
        self.rb_disk.setEnabled(False)
        self.lbl_main_status.setStyleSheet(f"font-size: 24px; color: {COLORS['primary']};")
        use_ram = self.rb_ram.isChecked()
        self.thread = QThread()
        self.worker = NiftiToStlWorker(self.nifti_path, use_ram_mode=use_ram)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.pbar.setValue)
        self.worker.status_text.connect(self.lbl_main_status.setText)
        self.worker.stage_update.connect(self.update_stage_ui)
        self.worker.finished.connect(self.on_success)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def update_stage_ui(self, tissue_name, status):
        if tissue_name in self.tissue_widgets:
            self.tissue_widgets[tissue_name].set_status(status)

    def on_success(self, output_folder):
        self.output_folder = output_folder
        self.btn_action.setText('OPEN VIEWER')
        self.btn_action.setEnabled(True)
        self.btn_action.clicked.disconnect()
        self.btn_action.clicked.connect(lambda: self.processing_completed.emit(self.output_folder))
        self.pbar.setStyleSheet(f"\n            QProgressBar {{ background-color: {COLORS['surface']}; border: none; border-radius: 3px; height: 6px; }}\n            QProgressBar::chunk {{ background-color: {COLORS['primary']}; }}\n        ")

    def on_error(self, err):
        self.lbl_main_status.setText('SEQUENCE HALTED')
        self.lbl_main_status.setStyleSheet(f"font-size: 24px; color: {COLORS['danger']};")
        self.btn_action.setText('RETRY')
        self.btn_action.setEnabled(True)
        self.rb_ram.setEnabled(True)
        self.rb_disk.setEnabled(True)
        print(f'Error Log: {err}')
if __name__ == '__main__':
    app = QApplication(sys.argv)
    dummy_path = 'test_dummy.nii.gz'
    if not os.path.exists(dummy_path):
        data = np.zeros((50, 50, 50), dtype=np.uint8)
        data[10:40, 10:40, 10:40] = 4
        nib.save(nib.Nifti1Image(data, np.eye(4)), dummy_path)
    window = NiftiToStlPage(dummy_path)
    window.show()
    sys.exit(app.exec())