import sys
import os
import shutil
import subprocess
import glob
import numpy as np
import nibabel as nib
import gc
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QMessageBox, QFrame, QSpacerItem, QSizePolicy
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer
try:
    from src.config.theme_config import THEME_COLORS, STYLESHEET
except ImportError:
    THEME_COLORS = {'bg': '#1E1F22', 'panel': '#2B2D31', 'teal': '#0F7F6E', 'teal_hover': '#149682', 'text_main': '#E6E8EB', 'text_dim': '#949BA4', 'border': '#3F4148', 'success': '#00C853', 'danger': '#FF5252'}
    STYLESHEET = ''
try:
    import torch
    HAS_TORCH = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    CUDA_AVAILABLE = False
env = os.environ.copy()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
WEIGHTS_PATH = os.path.join(BASE_DIR, 'models', 'nnUNet_weights')
if CUDA_AVAILABLE:
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.pop('NNUNETV2_FORCE_CPU', None)
    DEVICE = 'cuda'
    print(f'[GPU ENABLED] Usando GPU: {torch.cuda.get_device_name(0)}')
else:
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['NNUNETV2_FORCE_CPU'] = '1'
    DEVICE = 'cpu'
    print('[CPU MODE] No se detectó GPU, usando CPU')
env['nnUNet_results'] = WEIGHTS_PATH
env['nnUNet_raw'] = os.path.join(BASE_DIR, 'models', 'nnUNet_raw')
env['nnUNet_preprocessed'] = os.path.join(BASE_DIR, 'models', 'nnUNet_preprocessed')
TISSUE_MAP = {1: 'Glandular (FGT)', 2: 'Vessels', 3: 'Muscle', 4: 'Bone', 5: 'Fat', 6: 'Skin (Hybrid)', 7: 'Nipple', 8: 'Lymph Nodes', 9: 'Ligaments'}

class InferenceWorker(QObject):
    progress = Signal(int)
    status = Signal(str)
    console_log = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, input_folder):
        super().__init__()
        self.input_folder = input_folder
        self.temp_folder = os.path.join(input_folder, 'temp_ai_outputs')

    def fix_model_structure(self):
        if not os.path.exists(WEIGHTS_PATH):
            os.makedirs(WEIGHTS_PATH)
        lost_folder = os.path.join(BASE_DIR, 'BreastSegNet_models')
        if os.path.exists(lost_folder):
            for p_data in glob.glob(os.path.join(lost_folder, 'Dataset*')):
                dest = os.path.join(WEIGHTS_PATH, os.path.basename(p_data))
                if not os.path.exists(dest):
                    shutil.move(p_data, dest)
        candidates = glob.glob(os.path.join(WEIGHTS_PATH, '**', 'Dataset*'), recursive=True)
        found = []
        for path in candidates:
            if os.path.isdir(path):
                folder_name = os.path.basename(path)
                parent = os.path.dirname(path)
                if os.path.abspath(parent) != os.path.abspath(WEIGHTS_PATH):
                    dest = os.path.join(WEIGHTS_PATH, folder_name)
                    if not os.path.exists(dest):
                        shutil.move(path, dest)
                    found.append(folder_name)
                else:
                    found.append(folder_name)
        return list(set(found))

    def detect_config(self, dataset_name):
        path = os.path.join(WEIGHTS_PATH, dataset_name)
        if not os.path.exists(path):
            return ('2d', None)
        for f in os.listdir(path):
            if 'ResEncUNet' in f:
                parts = f.split('__')
                return (parts[2], parts[1]) if len(parts) > 2 else ('2d', parts[1])
        for f in os.listdir(path):
            if 'nnUNetTrainer' in f:
                parts = f.split('__')
                return (parts[2], parts[1]) if len(parts) >= 3 else ('2d', parts[1] if len(parts) >= 2 else None)
        return ('2d', None)

    def get_available_folds(self, dataset_name, config, plan):
        dataset_path = os.path.join(WEIGHTS_PATH, dataset_name)
        trainer_folder = None
        for f in os.listdir(dataset_path):
            if plan and plan in f and (config in f):
                trainer_folder = f
                break
            elif not plan and 'nnUNetPlans' in f and (config in f):
                trainer_folder = f
                break
        if not trainer_folder:
            cands = [f for f in os.listdir(dataset_path) if 'nnUNetTrainer' in f]
            if cands:
                trainer_folder = cands[0]
            else:
                return []
        full_path = os.path.join(dataset_path, trainer_folder)
        folds = []
        if os.path.exists(full_path):
            for item in os.listdir(full_path):
                if item.startswith('fold_') and os.path.isdir(os.path.join(full_path, item)):
                    folds.append(item.replace('fold_', ''))
        return folds

    def run_nnunet_streaming(self, dataset, out_dir, cfg, plan=None):
        cmd = ['nnUNetv2_predict', '-i', self.input_folder, '-o', out_dir, '-d', dataset, '-c', cfg, '--disable_tta', '-device', DEVICE]
        if plan and 'nnUNetPlans' not in plan:
            cmd.extend(['-p', plan])
        folds = self.get_available_folds(dataset, cfg, plan)
        if folds:
            cmd.append('-f')
            cmd.extend(folds)
            self.status.emit(f'Config: {cfg} | Folds: {folds}')
        else:
            self.status.emit(f'Config: {cfg} | Folds: Auto')
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True, shell=False, env=env)
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                clean = line.strip()
                if clean:
                    self.console_log.emit(f'{clean}')
        if process.returncode != 0:
            raise RuntimeError(f'Error nnU-Net: {process.returncode}')

    def fuse_masks_robust(self, path_skin, path_duke, output_path):
        self.status.emit('PREPARING DISK FUSION (Low RAM)...')
        gc.collect()
        if not os.path.exists(path_duke):
            raise FileNotFoundError('Missing Duke result')
        img_duke = nib.load(path_duke)
        img_skin = nib.load(path_skin) if os.path.exists(path_skin) else None
        affine = img_duke.affine
        header = img_duke.header
        shape = img_duke.shape
        z_depth = shape[2]
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)
        temp_map_file = os.path.join(self.temp_folder, 'temp_fusion_array.dat')
        final_hybrid = np.memmap(temp_map_file, dtype=np.uint8, mode='w+', shape=shape)
        self.status.emit(f'PROCESSING {z_depth} SLICES ON DISK...')
        found_labels = set()
        for z in range(z_depth):
            slice_duke = np.asanyarray(img_duke.dataobj[..., z]).astype(np.uint8)
            if img_skin:
                slice_skin = np.asanyarray(img_skin.dataobj[..., z]).astype(np.uint8)
                mask_inject = (slice_skin > 0) & (slice_duke == 0)
                slice_duke[mask_inject] = 6
                del slice_skin, mask_inject
            final_hybrid[..., z] = slice_duke
            unique_in_slice = np.unique(slice_duke)
            for u in unique_in_slice:
                if u != 0:
                    found_labels.add(u)
            del slice_duke
            if z % 10 == 0:
                prog = 85 + int(z / z_depth * 10)
                self.progress.emit(prog)
                self.console_log.emit(f'Merging slice {z}/{z_depth}...')
                if z % 50 == 0:
                    final_hybrid.flush()
                QThread.msleep(1)
        self.status.emit(f'PACKAGING NIFTI FILE...')
        final_hybrid.flush()
        clean_header = header.copy()
        clean_header.set_data_dtype(np.uint8)
        new_img = nib.Nifti1Image(final_hybrid, affine, clean_header)
        nib.save(new_img, output_path)
        del final_hybrid
        del new_img
        del img_duke
        if img_skin:
            del img_skin
        gc.collect()
        try:
            if os.path.exists(temp_map_file):
                os.remove(temp_map_file)
        except Exception as e:
            print(f'Warning: Could not remove temp {temp_map_file}: {e}')
        found_names = [TISSUE_MAP.get(l, f'Label {l}') for l in sorted(list(found_labels))]
        return found_names

    def create_dummy(self, src, dst):
        i = nib.load(src)
        nib.save(nib.Nifti1Image(np.zeros(i.shape, dtype=np.uint8), i.affine), dst)

    def run(self):
        try:
            models = self.fix_model_structure()
            duke = next((m for m in models if '910' in m or '501' in m), None)
            skin = next((m for m in models if '009' in m), None)
            self.console_log.emit(f'INTERNAL MODEL: {duke}')
            self.console_log.emit(f'SKIN MODEL: {skin}')
            nifti = [f for f in os.listdir(self.input_folder) if f.endswith('.nii.gz') and 'HYBRID' not in f and ('temp' not in f)]
            if not nifti:
                raise FileNotFoundError('No input NIfTI found.')
            input_f = nifti[0]
            if not input_f.endswith('_0000.nii.gz'):
                new_n = input_f.replace('.nii.gz', '_0000.nii.gz')
                os.rename(os.path.join(self.input_folder, input_f), os.path.join(self.input_folder, new_n))
                input_f = new_n
            base_out = input_f.replace('_0000.nii.gz', '.nii.gz')
            dir_duke = os.path.join(self.temp_folder, 'duke_out')
            dir_skin = os.path.join(self.temp_folder, 'skin_out')
            os.makedirs(dir_duke, exist_ok=True)
            os.makedirs(dir_skin, exist_ok=True)
            self.progress.emit(10)
            target_duke = os.path.join(dir_duke, base_out)
            if duke:
                cfg, plan = self.detect_config(duke)
                self.status.emit(f'STRUCTURAL ANALYSIS ({cfg})...')
                try:
                    self.run_nnunet_streaming(duke, dir_duke, cfg, plan)
                except Exception as e:
                    self.status.emit(f'Internal Error: {e}')
                    self.create_dummy(os.path.join(self.input_folder, input_f), target_duke)
            else:
                self.create_dummy(os.path.join(self.input_folder, input_f), target_duke)
            self.progress.emit(50)
            target_skin = os.path.join(dir_skin, base_out)
            if skin:
                cfg, plan = self.detect_config(skin)
                self.status.emit(f'SURFACE ANALYSIS ({cfg})...')
                try:
                    self.run_nnunet_streaming(skin, dir_skin, cfg, plan)
                except Exception as e:
                    self.status.emit(f'Skin Error: {e}')
                    self.create_dummy(os.path.join(self.input_folder, input_f), target_skin)
            else:
                self.create_dummy(os.path.join(self.input_folder, input_f), target_skin)
            self.status.emit('MERGING RESULTS...')
            self.progress.emit(85)
            output_folder = os.path.join(self.input_folder, 'ai_output')
            os.makedirs(output_folder, exist_ok=True)
            final_filename = base_out.replace('.nii.gz', '_HYBRID.nii.gz')
            final_path = os.path.join(output_folder, final_filename)
            found_tissues = self.fuse_masks_robust(target_skin, target_duke, final_path)
            if not os.path.exists(final_path):
                raise FileNotFoundError(f'File not saved at: {final_path}')
            tissues_str = ', '.join(found_tissues)
            self.status.emit(f'ANALYSIS COMPLETED.')
            self.console_log.emit(f'File: {final_filename}')
            self.progress.emit(100)
            QThread.msleep(500)
            self.finished.emit(final_path)
        except Exception as e:
            self.error.emit(f'CRITICAL ERROR: {str(e)}')

class InferencePage(QMainWindow):
    inference_completed = Signal(str)

    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.setWindowTitle('AI Processing')
        self.setStyleSheet(f"\n            QMainWindow {{\n                background-color: {THEME_COLORS['bg']};\n            }}\n            QWidget {{\n                font-family: 'Segoe UI', 'Roboto', sans-serif;\n                color: {THEME_COLORS['text_main']};\n            }}\n        ")
        self.setup_ui()
        self.showMaximized()

    def setup_ui(self):
        central_widget = QWidget()
        central_widget.setStyleSheet(f"background-color: {THEME_COLORS['bg']};")
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        header = QFrame()
        header.setFixedHeight(70)
        header.setStyleSheet(f"\n            QFrame {{\n                background-color: {THEME_COLORS['panel']};\n                border-bottom: 1px solid {THEME_COLORS['border']};\n            }}\n        ")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 0, 30, 0)
        lbl_title = QLabel('AI TISSUE SEGMENTATION')
        lbl_title.setFont(QFont('Segoe UI', 18, QFont.Bold))
        lbl_title.setStyleSheet(f"color: {THEME_COLORS['teal']}; letter-spacing: 2px;")
        header_layout.addWidget(lbl_title)
        header_layout.addStretch()
        main_layout.addWidget(header)
        content = QWidget()
        content.setStyleSheet(f"background-color: {THEME_COLORS['bg']};")
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(30)
        content_layout.setContentsMargins(60, 50, 60, 50)
        center_container = QWidget()
        center_layout = QVBoxLayout(center_container)
        center_layout.setSpacing(30)
        center_layout.setAlignment(Qt.AlignCenter)
        processing_card = QFrame()
        processing_card.setFixedSize(650, 350)
        processing_card.setStyleSheet(f"\n            QFrame {{\n                background-color: {THEME_COLORS['panel']};\n                border: 1px solid {THEME_COLORS['border']};\n                border-radius: 16px;\n            }}\n        ")
        card_layout = QVBoxLayout(processing_card)
        card_layout.setSpacing(25)
        card_layout.setContentsMargins(50, 40, 50, 40)
        card_layout.setAlignment(Qt.AlignCenter)
        self.lbl_main = QLabel('Ready to Start')
        self.lbl_main.setFont(QFont('Segoe UI', 22, QFont.Bold))
        self.lbl_main.setStyleSheet(f"color: {THEME_COLORS['text_main']};")
        self.lbl_main.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.lbl_main)
        self.bar = QProgressBar()
        self.bar.setValue(0)
        self.bar.setFixedHeight(12)
        self.bar.setTextVisible(False)
        self.bar.setStyleSheet(f"\n            QProgressBar {{ \n                background: {THEME_COLORS['bg']}; \n                border-radius: 6px;\n                border: none;\n            }}\n            QProgressBar::chunk {{ \n                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, \n                    stop:0 {THEME_COLORS['teal']}, \n                    stop:1 {THEME_COLORS['teal_hover']}); \n                border-radius: 6px; \n            }}\n        ")
        card_layout.addWidget(self.bar)
        self.lbl_sub = QLabel('Waiting for input...')
        self.lbl_sub.setFont(QFont('Consolas', 11))
        self.lbl_sub.setStyleSheet(f"color: {THEME_COLORS['text_dim']};")
        self.lbl_sub.setAlignment(Qt.AlignCenter)
        self.lbl_sub.setWordWrap(True)
        card_layout.addWidget(self.lbl_sub)
        center_layout.addWidget(processing_card, alignment=Qt.AlignCenter)
        self.btn = QPushButton('START PROCESSING')
        self.btn.setFixedHeight(55)
        self.btn.setFixedWidth(650)
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setStyleSheet(f"\n            QPushButton {{ \n                background-color: {THEME_COLORS['success']}; \n                color: white; \n                border: none;\n                border-radius: 10px; \n                font-weight: bold; \n                font-size: 15px;\n                letter-spacing: 1px;\n            }}\n            QPushButton:hover {{\n                background-color: #00E676;\n            }}\n            QPushButton:disabled {{ \n                background-color: #333; \n                color: #555; \n            }}\n        ")
        self.btn.clicked.connect(self.run_process)
        center_layout.addWidget(self.btn, alignment=Qt.AlignCenter)
        info_label = QLabel(f'Processing folder: {os.path.basename(self.folder)}')
        info_label.setStyleSheet(f"color: {THEME_COLORS['text_dim']}; font-size: 12px;")
        info_label.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(info_label)
        content_layout.addStretch()
        content_layout.addWidget(center_container, alignment=Qt.AlignCenter)
        content_layout.addStretch()
        main_layout.addWidget(content, stretch=1)

    def update_main_status(self, text):
        self.lbl_main.setText(text)

    def update_sub_status(self, text):
        if len(text) > 100:
            text = text[:97] + '...'
        self.lbl_sub.setText(text)

    def on_process_finished(self, path):
        self.lbl_main.setText('Analysis Complete')
        self.lbl_main.setStyleSheet(f"color: {THEME_COLORS['success']};")
        self.lbl_sub.setText(f'Results saved: {os.path.basename(path)}')
        self.bar.setValue(100)
        self.btn.setText('DONE')
        self.btn.setStyleSheet(f"\n            QPushButton {{ \n                background-color: {THEME_COLORS['teal']}; \n                color: white; \n                border: none;\n                border-radius: 10px; \n                font-weight: bold; \n                font-size: 15px;\n            }}\n        ")
        self.inference_completed.emit(path)
        QTimer.singleShot(1500, self.close)

    def run_process(self):
        self.btn.setEnabled(False)
        self.btn.setText('PROCESSING...')
        self.btn.setStyleSheet(f'\n            QPushButton {{ \n                background-color: #333; \n                color: #888; \n                border: none;\n                border-radius: 10px; \n                font-weight: bold; \n                font-size: 15px;\n            }}\n        ')
        self.bar.setValue(0)
        self.lbl_sub.setText('Initializing AI core...')
        self.th = QThread()
        self.wk = InferenceWorker(self.folder)
        self.wk.moveToThread(self.th)
        self.th.started.connect(self.wk.run)
        self.wk.status.connect(self.update_main_status)
        self.wk.console_log.connect(self.update_sub_status)
        self.wk.progress.connect(self.bar.setValue)
        self.wk.finished.connect(self.on_process_finished)
        self.wk.finished.connect(self.th.quit)
        self.th.finished.connect(self.th.deleteLater)
        self.wk.error.connect(self.on_error)
        self.th.start()

    def on_error(self, error_msg):
        self.lbl_main.setText('Processing Failed')
        self.lbl_main.setStyleSheet(f"color: {THEME_COLORS['danger']};")
        self.lbl_sub.setText(str(error_msg)[:100])
        QMessageBox.critical(self, 'Error', error_msg)
        self.btn.setEnabled(True)
        self.btn.setText('RETRY')
        self.btn.setStyleSheet(f"\n            QPushButton {{ \n                background-color: {THEME_COLORS['danger']}; \n                color: white; \n                border: none;\n                border-radius: 10px; \n                font-weight: bold; \n                font-size: 15px;\n            }}\n            QPushButton:hover {{\n                background-color: #FF7070;\n            }}\n        ")
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InferencePage(os.getcwd())
    window.show()
    sys.exit(app.exec())