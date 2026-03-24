import sys
import os
import shutil
import zipfile
import time
import stat
import gc
import dicom2nifti
import pydicom
import nibabel as nib
import numpy as np
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QProgressBar, QMessageBox, QFrame, QSpacerItem, QSizePolicy, QApplication
from PySide6.QtGui import QFont, QScreen
from PySide6.QtCore import Qt, QThread, Signal, QObject
try:
    from src.config.theme_config import THEME_COLORS, STYLESHEET
except ImportError:
    THEME_COLORS = {'bg': '#1E1F22', 'panel': '#2B2D31', 'teal': '#0F7F6E', 'teal_hover': '#149682', 'text_main': '#E6E8EB', 'text_dim': '#949BA4', 'border': '#3F4148', 'success': '#00C853', 'danger': '#FF5252'}
    STYLESHEET = ''

class DicomConverterWorker(QObject):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, zip_path, output_dir):
        super().__init__()
        self.zip_path = zip_path
        self.output_dir = output_dir
        self.temp_extract_dir = os.path.join(output_dir, 'temp_dicom')

    def _remove_readonly(self, func, path, excinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    def _robust_cleanup(self, folder_path):
        gc.collect()
        max_retries = 5
        for i in range(max_retries):
            try:
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path, onerror=self._remove_readonly)
                return
            except PermissionError:
                if i < max_retries - 1:
                    time.sleep(0.5)
                else:
                    print(f'Warning: Windows bloqueó el borrado de {folder_path}')

    def _fallback_dicom_to_nifti(self, dicom_folder, output_dir):
        converted_count = 0
        series_dict = {}
        for root, dirs, files in os.walk(dicom_folder):
            dicom_files = []
            for f in files:
                fpath = os.path.join(root, f)
                try:
                    ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                    if hasattr(ds, 'SeriesInstanceUID'):
                        dicom_files.append(fpath)
                except:
                    continue
            if dicom_files:
                for fpath in dicom_files:
                    try:
                        ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                        series_uid = str(ds.SeriesInstanceUID) if hasattr(ds, 'SeriesInstanceUID') else 'unknown'
                        if series_uid not in series_dict:
                            series_dict[series_uid] = []
                        series_dict[series_uid].append(fpath)
                    except:
                        continue
        for series_uid, files in series_dict.items():
            try:
                slices = []
                for f in files:
                    try:
                        ds = pydicom.dcmread(f)
                        if hasattr(ds, 'pixel_array'):
                            slices.append(ds)
                    except:
                        continue
                if not slices:
                    continue
                try:
                    slices.sort(key=lambda x: int(x.InstanceNumber) if hasattr(x, 'InstanceNumber') else 0)
                except:
                    try:
                        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]) if hasattr(x, 'ImagePositionPatient') else 0)
                    except:
                        pass
                if len(slices) == 1:
                    pixel_data = slices[0].pixel_array[np.newaxis, :, :]
                else:
                    pixel_data = np.stack([s.pixel_array for s in slices], axis=0)
                if hasattr(slices[0], 'RescaleSlope') and hasattr(slices[0], 'RescaleIntercept'):
                    pixel_data = pixel_data * float(slices[0].RescaleSlope) + float(slices[0].RescaleIntercept)
                ds = slices[0]
                try:
                    pixel_spacing = [float(x) for x in ds.PixelSpacing] if hasattr(ds, 'PixelSpacing') else [1.0, 1.0]
                    slice_thickness = float(ds.SliceThickness) if hasattr(ds, 'SliceThickness') else 1.0
                    if len(slices) > 1 and hasattr(slices[0], 'ImagePositionPatient') and hasattr(slices[1], 'ImagePositionPatient'):
                        pos0 = np.array([float(x) for x in slices[0].ImagePositionPatient])
                        pos1 = np.array([float(x) for x in slices[1].ImagePositionPatient])
                        slice_spacing = np.linalg.norm(pos1 - pos0)
                        if slice_spacing > 0:
                            slice_thickness = slice_spacing
                    affine = np.eye(4)
                    affine[0, 0] = pixel_spacing[0]
                    affine[1, 1] = pixel_spacing[1]
                    affine[2, 2] = slice_thickness
                    if hasattr(ds, 'ImagePositionPatient'):
                        pos = [float(x) for x in ds.ImagePositionPatient]
                        affine[0, 3] = pos[0]
                        affine[1, 3] = pos[1]
                        affine[2, 3] = pos[2]
                except Exception as e:
                    print(f'Warning: Could not compute affine matrix: {e}')
                    affine = np.eye(4)
                pixel_data = np.transpose(pixel_data, (2, 1, 0))
                nifti_img = nib.Nifti1Image(pixel_data.astype(np.float32), affine)
                series_desc = str(ds.SeriesDescription).replace(' ', '_').replace('/', '_') if hasattr(ds, 'SeriesDescription') else 'series'
                series_desc = ''.join((c if c.isalnum() or c == '_' else '_' for c in series_desc))
                output_file = os.path.join(output_dir, f'{series_desc}_{series_uid[:8]}.nii.gz')
                nib.save(nifti_img, output_file)
                converted_count += 1
                print(f'Fallback converted: {series_desc}')
            except Exception as e:
                print(f'Warning: Could not convert series {series_uid[:8]}: {e}')
                continue
        return converted_count

    def run(self):
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            existing_niftis = [f for f in os.listdir(self.output_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
            for f in existing_niftis:
                try:
                    os.remove(os.path.join(self.output_dir, f))
                except:
                    pass
            self._robust_cleanup(self.temp_extract_dir)
            if not os.path.exists(self.temp_extract_dir):
                os.makedirs(self.temp_extract_dir)
            self.status.emit('Extracting files...')
            self.progress.emit(10)
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_extract_dir)
            self.progress.emit(30)
            found_stl = False
            for root, dirs, files in os.walk(self.temp_extract_dir):
                for file in files:
                    if file.lower().endswith('.stl'):
                        found_stl = True
                        break
            if found_stl:
                raise Exception('ERROR: El ZIP contiene archivos .STL. Esta herramienta convierte DICOM (.dcm) a NIfTI. Si ya tienes los STL, no uses este convertidor.')
            self.status.emit('Converting DICOM series to NIfTI...')
            dicom2nifti_success = False
            try:
                dicom2nifti.convert_directory(self.temp_extract_dir, self.output_dir, compression=True, reorient=True)
                dicom2nifti_success = True
            except Exception as convert_error:
                print(f'dicom2nifti failed: {convert_error}')
                print('Attempting fallback conversion with pydicom + nibabel...')
            self.progress.emit(60)
            generated_files = [f for f in os.listdir(self.output_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
            if not generated_files:
                self.status.emit('Using fallback converter...')
                try:
                    fallback_count = self._fallback_dicom_to_nifti(self.temp_extract_dir, self.output_dir)
                    if fallback_count > 0:
                        print(f'Fallback converter successfully converted {fallback_count} series')
                except Exception as fallback_error:
                    print(f'Fallback conversion error: {fallback_error}')
            self.progress.emit(80)
            generated_files = [f for f in os.listdir(self.output_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
            if not generated_files:
                raise Exception('FALLO: No se generaron archivos NIfTI. Verifica que el ZIP contenga carpetas con series DICOM (.dcm) válidas y no carpetas vacías o archivos sueltos sin formato.')
            self.status.emit('Cleaning up...')
            self._robust_cleanup(self.temp_extract_dir)
            self.progress.emit(100)
            self.status.emit('Conversion Complete.')
            self.finished.emit(self.output_dir)
        except Exception as e:
            self._robust_cleanup(self.temp_extract_dir)
            self.error.emit(str(e))

class DicomConverterPage(QMainWindow):
    conversion_completed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('DICOM to NIfTI Converter')
        self.setStyleSheet(f"\n            QMainWindow {{\n                background-color: {THEME_COLORS['bg']};\n            }}\n            QWidget {{\n                font-family: 'Segoe UI', 'Roboto', sans-serif;\n                color: {THEME_COLORS['text_main']};\n            }}\n        ")
        self.setup_ui()
        self.selected_zip = None
        self.showMaximized()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        header = QFrame()
        header.setFixedHeight(70)
        header.setStyleSheet(f"\n            QFrame {{\n                background-color: {THEME_COLORS['panel']};\n                border-bottom: 1px solid {THEME_COLORS['border']};\n            }}\n        ")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 0, 30, 0)
        lbl_title = QLabel('🔄 DICOM TO NIFTI CONVERTER')
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
        center_layout.setSpacing(25)
        center_layout.setAlignment(Qt.AlignCenter)
        lbl_desc = QLabel('Upload a .zip file containing a DICOM series (.dcm) to convert it to NIfTI format')
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet(f"color: {THEME_COLORS['text_dim']}; font-size: 14px;")
        lbl_desc.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(lbl_desc)
        upload_card = QFrame()
        upload_card.setFixedSize(600, 280)
        upload_card.setStyleSheet(f"\n            QFrame {{\n                background-color: {THEME_COLORS['panel']};\n                border: 2px dashed {THEME_COLORS['border']};\n                border-radius: 16px;\n            }}\n            QFrame:hover {{\n                border-color: {THEME_COLORS['teal']};\n            }}\n        ")
        upload_card_layout = QVBoxLayout(upload_card)
        upload_card_layout.setSpacing(20)
        upload_card_layout.setContentsMargins(40, 40, 40, 40)
        upload_card_layout.setAlignment(Qt.AlignCenter)
        icon_label = QLabel('📁')
        icon_label.setFont(QFont('Segoe UI Emoji', 48))
        icon_label.setAlignment(Qt.AlignCenter)
        upload_card_layout.addWidget(icon_label)
        self.upload_btn = QPushButton('  SELECT ZIP FILE  ')
        self.upload_btn.setFixedHeight(50)
        self.upload_btn.setFixedWidth(280)
        self.upload_btn.setCursor(Qt.PointingHandCursor)
        self.upload_btn.setStyleSheet(f"\n            QPushButton {{\n                background-color: {THEME_COLORS['teal']};\n                border: none;\n                border-radius: 8px;\n                font-weight: bold;\n                font-size: 14px;\n                color: white;\n                letter-spacing: 1px;\n            }}\n            QPushButton:hover {{\n                background-color: {THEME_COLORS['teal_hover']};\n            }}\n            QPushButton:disabled {{\n                background-color: #333;\n                color: #666;\n            }}\n        ")
        self.upload_btn.clicked.connect(self.select_file)
        upload_card_layout.addWidget(self.upload_btn, alignment=Qt.AlignCenter)
        self.lbl_file = QLabel('No file selected')
        self.lbl_file.setAlignment(Qt.AlignCenter)
        self.lbl_file.setStyleSheet(f"color: {THEME_COLORS['text_dim']}; font-style: italic; font-size: 13px;")
        upload_card_layout.addWidget(self.lbl_file)
        center_layout.addWidget(upload_card, alignment=Qt.AlignCenter)
        progress_section = QFrame()
        progress_section.setFixedWidth(600)
        progress_section.setStyleSheet(f"\n            QFrame {{\n                background-color: {THEME_COLORS['panel']};\n                border-radius: 12px;\n                padding: 20px;\n            }}\n        ")
        progress_layout = QVBoxLayout(progress_section)
        progress_layout.setSpacing(15)
        progress_layout.setContentsMargins(30, 25, 30, 25)
        self.lbl_status = QLabel('Waiting for file selection...')
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet(f"color: {THEME_COLORS['text_dim']}; font-size: 14px;")
        progress_layout.addWidget(self.lbl_status)
        self.pbar = QProgressBar()
        self.pbar.setFixedHeight(10)
        self.pbar.setTextVisible(False)
        self.pbar.setStyleSheet(f"\n            QProgressBar {{ \n                background: {THEME_COLORS['bg']}; \n                border-radius: 5px;\n                border: none;\n            }}\n            QProgressBar::chunk {{ \n                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, \n                    stop:0 {THEME_COLORS['teal']}, \n                    stop:1 {THEME_COLORS['teal_hover']}); \n                border-radius: 5px; \n            }}\n        ")
        progress_layout.addWidget(self.pbar)
        center_layout.addWidget(progress_section, alignment=Qt.AlignCenter)
        self.btn_convert = QPushButton('▶  START CONVERSION')
        self.btn_convert.setFixedHeight(55)
        self.btn_convert.setFixedWidth(600)
        self.btn_convert.setEnabled(False)
        self.btn_convert.setCursor(Qt.PointingHandCursor)
        self.btn_convert.setStyleSheet(f"\n            QPushButton {{ \n                background-color: {THEME_COLORS['success']}; \n                color: white; \n                border: none;\n                border-radius: 10px; \n                font-weight: bold; \n                font-size: 15px;\n                letter-spacing: 1px;\n            }}\n            QPushButton:hover {{\n                background-color: #00E676;\n            }}\n            QPushButton:disabled {{ \n                background-color: #333; \n                color: #555; \n            }}\n        ")
        self.btn_convert.clicked.connect(self.start_conversion)
        center_layout.addWidget(self.btn_convert, alignment=Qt.AlignCenter)
        content_layout.addStretch()
        content_layout.addWidget(center_container, alignment=Qt.AlignCenter)
        content_layout.addStretch()
        main_layout.addWidget(content, stretch=1)

    def select_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select DICOM Zip', '', 'Zip Files (*.zip)')
        if fname:
            self.selected_zip = fname
            self.lbl_file.setText(f'📄 {os.path.basename(fname)}')
            self.lbl_file.setStyleSheet(f"color: {THEME_COLORS['text_main']}; font-weight: bold; font-size: 14px;")
            self.btn_convert.setEnabled(True)
            self.lbl_status.setText('✓ File ready. Click START to begin conversion.')
            self.lbl_status.setStyleSheet(f"color: {THEME_COLORS['teal']}; font-size: 14px;")

    def start_conversion(self):
        if not self.selected_zip:
            return
        self.btn_convert.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.lbl_status.setText('⏳ Initializing conversion...')
        self.lbl_status.setStyleSheet(f"color: {THEME_COLORS['text_dim']}; font-size: 14px;")
        self.pbar.setValue(0)
        output_path = os.path.join(os.getcwd(), 'converted_nifti')
        self.thread = QThread()
        self.worker = DicomConverterWorker(self.selected_zip, output_path)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.pbar.setValue)
        self.worker.status.connect(self.lbl_status.setText)
        self.worker.finished.connect(self.on_success)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_success(self, output_path):
        self.lbl_status.setText('✅ Conversion Completed Successfully!')
        self.lbl_status.setStyleSheet(f"color: {THEME_COLORS['success']}; font-size: 14px; font-weight: bold;")
        QMessageBox.information(self, 'Success', f'Files converted to:\n{output_path}')
        self.conversion_completed.emit(output_path)
        self.btn_convert.setEnabled(True)
        self.upload_btn.setEnabled(True)

    def on_error(self, err_msg):
        self.lbl_status.setText('❌ Conversion Failed')
        self.lbl_status.setStyleSheet(f"color: {THEME_COLORS['danger']}; font-size: 14px; font-weight: bold;")
        QMessageBox.critical(self, 'Error', str(err_msg))
        self.btn_convert.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.pbar.setValue(0)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = DicomConverterPage()
    win.show()
    sys.exit(app.exec())