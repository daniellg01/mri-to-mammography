import sys
import os
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSlider, QLabel, QGroupBox, QScrollArea, QRadioButton, QFrame, QTabWidget, QProgressDialog, QFileDialog, QMessageBox
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QFont
from PySide6.QtCore import Qt, QThread, Signal, QObject
import pyvista as pv
from pyvistaqt import QtInteractor
from src.config.theme_config import THEME_COLORS, STYLESHEET
from src.physics.physics_config import NIST_PHYSICS
from src.physics.physics_engine import PhysicsEngine
from src.utils.geometry_utils import transform_meshes_logic
from src.physics.simulation_worker import SimulationWorker
from src.ui.ui_analysis import AnalysisDialog
from src.ui.ui_manual_crop import ManualCropDialog

class MammographyProSim(QMainWindow):

    def __init__(self, stl_folder=None):
        super().__init__()
        self.tissues = {}
        self.stl_folder = stl_folder
        self.setWindowTitle('Mammography Pro - NIST Precision Clinical Suite (High-Fidelity Physics)')
        self.resize(1900, 950)
        self.show_compressed = False
        self.raw_cc = None
        self.raw_mlo = None
        self.disp_cc = None
        self.disp_mlo = None
        self.setup_ui()
        self.load_data()
        self.refresh_view()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QHBoxLayout(central)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.setup_sidebar()
        view_3d_container = QVBoxLayout()
        view_3d_container.setContentsMargins(20, 20, 20, 20)
        lbl_3d = QLabel('REAL-TIME POSITIONING (3D)')
        lbl_3d.setStyleSheet(f"font-size: 14px; color: {THEME_COLORS['teal']}; font-weight: bold; letter-spacing: 2px;")
        view_3d_container.addWidget(lbl_3d)
        self.plotter = QtInteractor(self)
        self.plotter.set_background(THEME_COLORS['bg'])
        view_3d_container.addWidget(self.plotter)
        self.main_layout.addLayout(view_3d_container, stretch=2)
        self.setup_image_panel()

    def setup_sidebar(self):
        scroll = QScrollArea()
        scroll.setFixedWidth(360)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(25, 30, 25, 30)
        layout.setSpacing(25)
        lbl_title = QLabel('SIMULATION\nCONTROLS')
        lbl_title.setStyleSheet('font-size: 20px; font-weight: 800; color: white; letter-spacing: 2px;')
        layout.addWidget(lbl_title)
        layout.addWidget(self.create_separator())
        side_group = QGroupBox('PATIENT LATERALITY')
        s_lay = QVBoxLayout(side_group)
        self.btn_left = QRadioButton('Left Breast (L)')
        self.btn_right = QRadioButton('Right Breast (R)')
        self.btn_right.setChecked(True)
        for b in [self.btn_left, self.btn_right]:
            s_lay.addWidget(b)
            b.toggled.connect(self.refresh_view)
        layout.addWidget(side_group)
        geom_group = QGroupBox('ACQUISITION GEOMETRY')
        geom_lay = QVBoxLayout(geom_group)
        self.sld_chest = self.create_slider('Chest Wall Offset (Y-Axis)', -150, 50, -50, geom_lay)
        layout.addWidget(geom_group)
        gravity_group = QGroupBox('TISSUE GRAVITY')
        gravity_lay = QVBoxLayout(gravity_group)
        self.sld_gravity = self.create_slider('Gravity Effect', 0, 200, 100, gravity_lay)
        gravity_info = QLabel('Simula cómo la gravedad\ndeforma el tejido blando\nsegún posición del paciente.\n0=Sin gravedad, 100=Normal')
        gravity_info.setStyleSheet(f"\n            color: {THEME_COLORS['text_dim']};\n            font-size: 10px;\n            padding: 5px;\n        ")
        gravity_lay.addWidget(gravity_info)
        layout.addWidget(gravity_group)
        view_group = QGroupBox('3D PLATE VIEW')
        view_lay = QVBoxLayout(view_group)
        self.btn_view_cc = QRadioButton('Show CC Plates')
        self.btn_view_mlo = QRadioButton('Show MLO Plates')
        self.btn_view_cc.setChecked(True)
        for b in [self.btn_view_cc, self.btn_view_mlo]:
            view_lay.addWidget(b)
            b.toggled.connect(self.refresh_view)
        layout.addWidget(view_group)
        info_nist = QLabel('NIST XCOM PHYSICS v2.0\n----------------------\n• Flux: 5x10^5 photons/px\n• Depth: 16-bit Raw Data\n• Noise Model: Poisson (Shot)\n• Target SNR: > 10.0')
        info_nist.setStyleSheet(f"\n            color: {THEME_COLORS['text_dim']}; \n            background: rgba(43, 45, 49, 0.5); \n            padding: 15px; \n            border-left: 3px solid {THEME_COLORS['teal']};\n            font-family: 'Consolas', monospace;\n            font-size: 11px;\n            line-height: 16px;\n        ")
        layout.addWidget(info_nist)
        self.btn_preview = QPushButton('PREVIEW COMPRESSION')
        self.btn_preview.setCheckable(True)
        self.btn_preview.setStyleSheet(f"\n            QPushButton {{ \n                background: transparent; \n                border: 1px solid {THEME_COLORS['text_dim']};\n                color: {THEME_COLORS['text_main']};\n            }}\n            QPushButton:checked {{\n                border-color: #E67E22;\n                color: #E67E22;\n                background: rgba(230, 126, 34, 0.1);\n            }}\n        ")
        self.btn_preview.toggled.connect(self.toggle_compression_preview)
        layout.addWidget(self.btn_preview)
        self.btn_run = QPushButton('INITIALIZE ACQUISITION')
        self.btn_run.setStyleSheet(f"\n            background-color: {THEME_COLORS['teal']}; \n            color: white; \n            border: none;\n            height: 50px; \n            font-size: 13px;\n            border-radius: 4px;\n        ")
        self.btn_run.clicked.connect(self.start_generation)
        layout.addWidget(self.btn_run)
        self.btn_qa = QPushButton('QA ANALYSIS REPORT')
        self.btn_qa.setEnabled(False)
        self.btn_qa.setStyleSheet(f"\n            QPushButton:enabled {{\n                border: 1px solid {THEME_COLORS['teal']};\n                color: {THEME_COLORS['teal']};\n            }}\n            QPushButton:hover:enabled {{\n                background-color: rgba(15, 127, 110, 0.1);\n            }}\n        ")
        self.btn_qa.clicked.connect(self.open_qa_modal)
        layout.addWidget(self.btn_qa)
        self.btn_manual_crop = QPushButton('✂ CROP MANUAL')
        self.btn_manual_crop.setEnabled(False)
        self.btn_manual_crop.setToolTip('Ajuste manual de la región de interés (útil para mamas pequeñas)')
        self.btn_manual_crop.setStyleSheet(f"\n            QPushButton {{\n                background: transparent;\n                border: 1px solid #E67E22;\n                color: #E67E22;\n                height: 35px;\n                font-size: 12px;\n            }}\n            QPushButton:enabled:hover {{\n                background-color: rgba(230, 126, 34, 0.1);\n            }}\n            QPushButton:disabled {{\n                border-color: {THEME_COLORS['text_dim']};\n                color: {THEME_COLORS['text_dim']};\n            }}\n        ")
        self.btn_manual_crop.clicked.connect(self.open_manual_crop)
        layout.addWidget(self.btn_manual_crop)
        self.btn_download = QPushButton('⬇ DOWNLOAD IMAGES')
        self.btn_download.setEnabled(False)
        self.btn_download.setToolTip('Descarga CC y MLO con filtro Fourier aplicado')
        self.btn_download.setStyleSheet(f"\n            QPushButton {{\n                background: transparent;\n                border: 1px solid #3498DB;\n                color: #3498DB;\n                height: 35px;\n                font-size: 12px;\n            }}\n            QPushButton:enabled:hover {{\n                background-color: rgba(52, 152, 219, 0.1);\n            }}\n            QPushButton:disabled {{\n                border-color: {THEME_COLORS['text_dim']};\n                color: {THEME_COLORS['text_dim']};\n            }}\n        ")
        self.btn_download.clicked.connect(self.download_images_with_fourier)
        layout.addWidget(self.btn_download)
        layout.addStretch()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        self.main_layout.addWidget(scroll)

    def create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet(f"background-color: {THEME_COLORS['border']}; max-height: 1px; border: none;")
        return line

    def setup_image_panel(self):
        panel = QFrame()
        panel.setFixedWidth(650)
        panel.setStyleSheet(f"background-color: {THEME_COLORS['panel']}; border-left: 1px solid {THEME_COLORS['border']};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        lbl_cc_title = QLabel('CRANIOCAUDAL (CC)')
        lbl_cc_title.setStyleSheet(f"font-size: 12px; color: {THEME_COLORS['teal']}; font-weight: bold;")
        lbl_cc_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_cc_title)
        self.img_cc = self.create_img_label('Waiting for CC Projection...')
        self.img_cc.setFixedSize(300, 370)
        layout.addWidget(self.img_cc, alignment=Qt.AlignCenter)
        lbl_mlo_title = QLabel('MEDIOLATERAL OBLIQUE (MLO)')
        lbl_mlo_title.setStyleSheet(f"font-size: 12px; color: {THEME_COLORS['teal']}; font-weight: bold;")
        lbl_mlo_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_mlo_title)
        self.img_mlo = self.create_img_label('Waiting for MLO Projection...')
        self.img_mlo.setFixedSize(300, 370)
        layout.addWidget(self.img_mlo, alignment=Qt.AlignCenter)
        self.tabs = QTabWidget()
        self.tabs.setVisible(False)
        self.tabs.addTab(QWidget(), 'CC')
        self.tabs.addTab(QWidget(), 'MLO')
        self.main_layout.addWidget(panel)

    def create_img_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"\n            color: {THEME_COLORS['text_dim']}; \n            border: 1px dashed {THEME_COLORS['border']}; \n            background: {THEME_COLORS['bg']}; \n            font-size: 12px;\n            letter-spacing: 0.5px;\n        ")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedSize(300, 750)
        return lbl

    def create_slider(self, label, mi, ma, v, layout):
        container = QWidget()
        l = QVBoxLayout(container)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(5)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {THEME_COLORS['text_dim']}; font-size: 11px;")
        header_layout.addWidget(lbl)
        value_lbl = QLabel(f'{v}')
        value_lbl.setStyleSheet(f"color: {THEME_COLORS['teal']}; font-size: 11px; font-weight: bold;")
        header_layout.addWidget(value_lbl)
        header_layout.addStretch()
        l.addLayout(header_layout)
        s = QSlider(Qt.Horizontal)
        s.setRange(mi, ma)
        s.setValue(v)
        s.valueChanged.connect(lambda val: value_lbl.setText(f'{val}'))
        s.valueChanged.connect(self.refresh_view)
        l.addWidget(s)
        layout.addWidget(container)
        return s

    def load_data(self):
        path = self.stl_folder if self.stl_folder else './export_models'
        if not os.path.exists(path):
            print(f'Error: Path not found {path}')
            return
        detected_pid = None
        files = os.listdir(path)
        for f in files:
            if f.endswith('_FGT.stl'):
                detected_pid = f.replace('_FGT.stl', '')
                break
        if not detected_pid:
            for f in files:
                if f.endswith('_Grasa.stl'):
                    detected_pid = f.replace('_Grasa.stl', '')
                    break
        print(f'Loading Physics Models from: {path}')
        print(f'Detected Patient ID: {detected_pid}')
        for name in NIST_PHYSICS.keys():
            if detected_pid:
                filename = f'{detected_pid}_{name}.stl'
            else:
                filename = f'BreastDx-01-0002_{name}.stl'
            f_path = os.path.join(path, filename)
            if os.path.exists(f_path):
                try:
                    m = pv.read(f_path)
                    m.points = m.points.astype(np.float32)
                    try:
                        m = m
                    except:
                        pass
                    m.translate(-np.array(m.center))
                    self.tissues[name] = m
                    print(f' -> Loaded: {name}')
                except Exception as e:
                    print(f'Failed to load {name}: {e}')

    def get_transformed_meshes(self, is_mlo=False):
        return transform_meshes_logic(self.tissues, self.btn_right.isChecked(), 40, is_mlo)

    def start_generation(self):
        self.progress = QProgressDialog('Inicializando Física...', 'Cancelar', 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setStyleSheet(f"\n            QProgressDialog {{ background-color: {THEME_COLORS['bg']}; color: {THEME_COLORS['text_main']}; }}\n            QLabel {{ color: {THEME_COLORS['text_main']}; font-weight: bold; }}\n            QProgressBar {{\n                border: 1px solid {THEME_COLORS['border']};\n                border-radius: 2px;\n                background-color: {THEME_COLORS['panel']};\n                text-align: center;\n                color: white;\n            }}\n            QProgressBar::chunk {{\n                background-color: {THEME_COLORS['teal']};\n                width: 10px; \n            }}\n        ")
        self.progress.setAutoClose(False)
        self.progress.setValue(0)
        self.progress.show()
        self.thread = QThread()
        gravity_val = self.sld_gravity.value() / 100.0
        self.worker = SimulationWorker(self.tissues, self.btn_right.isChecked(), 40, self.sld_chest.value(), 145, 45, mlo_rot_y=45, mlo_rot_x=25, gravity_strength=gravity_val)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.error.connect(self.on_generation_error)
        self.worker.progress.connect(self.update_progress_label)
        self.worker.progress_int.connect(self.progress.setValue)
        self.thread.start()

    def on_generation_finished(self, cc_disp, cc_raw, mlo_disp, mlo_raw):
        self.progress.setValue(100)
        self.progress.close()
        self.raw_cc = cc_raw
        self.raw_mlo = mlo_raw
        self.disp_cc = cc_disp
        self.disp_mlo = mlo_disp
        self.display_image(cc_disp, self.img_cc)
        self.display_image(mlo_disp, self.img_mlo)
        self.btn_qa.setEnabled(True)
        self.btn_manual_crop.setEnabled(True)
        self.btn_download.setEnabled(True)
        print('Projections completed.')

    def update_progress_label(self, text):
        if self.progress:
            self.progress.setLabelText(text)

    def on_generation_error(self, err_msg):
        self.progress.close()
        print(f'Error during generation: {err_msg}')

    def open_qa_modal(self):
        current_idx = self.tabs.currentIndex()
        target_data = self.raw_cc if current_idx == 0 else self.raw_mlo
        title = 'QA ANALYSIS: CRANIOCAUDAL (CC) - 16 BIT RAW' if current_idx == 0 else 'QA ANALYSIS: MEDIOLATERAL OBLIQUE (MLO) - 16 BIT RAW'
        if target_data is None:
            return
        dlg = AnalysisDialog(target_data, title)
        dlg.exec()

    def open_manual_crop(self):
        if self.raw_cc is None or self.raw_mlo is None:
            print('No hay imágenes generadas para recortar.')
            return
        dlg = ManualCropDialog(cc_image=self.disp_cc, mlo_image=self.disp_mlo, cc_raw=self.raw_cc, mlo_raw=self.raw_mlo, parent=self)
        if dlg.exec():
            cc_disp, cc_raw, mlo_disp, mlo_raw = dlg.get_cropped_images()
            if cc_raw is not None:
                self.raw_cc = cc_raw
                self.disp_cc = cc_disp
                self.display_image(cc_disp, self.img_cc)
                print(f'[CROP MANUAL] CC actualizado: {cc_raw.shape}')
            if mlo_raw is not None:
                self.raw_mlo = mlo_raw
                self.disp_mlo = mlo_disp
                self.display_image(mlo_disp, self.img_mlo)
                print(f'[CROP MANUAL] MLO actualizado: {mlo_raw.shape}')
            print('[CROP MANUAL] Imágenes actualizadas exitosamente.')

    def apply_fourier_filter(self, image):
        img_float = image.astype(np.float64)
        f_transform = np.fft.fft2(img_float)
        f_shift = np.fft.fftshift(f_transform)
        rows, cols = image.shape
        crow, ccol = (rows // 2, cols // 2)
        cutoff = min(rows, cols) * 0.4
        order = 2
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        butterworth = 1 / (1 + (distance / cutoff) ** (2 * order))
        f_filtered = f_shift * butterworth
        f_ishift = np.fft.ifftshift(f_filtered)
        img_filtered = np.fft.ifft2(f_ishift)
        img_filtered = np.abs(img_filtered)
        img_filtered = np.clip(img_filtered, 0, 65535).astype(np.uint16)
        return img_filtered

    def download_images_with_fourier(self):
        from datetime import datetime
        if self.raw_cc is None or self.raw_mlo is None:
            print('No hay imágenes generadas para descargar.')
            return
        folder = QFileDialog.getExistingDirectory(self, 'Seleccionar carpeta de destino', os.path.expanduser('~'))
        if not folder:
            return
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            side = 'L' if self.btn_left.isChecked() else 'R'
            print('Aplicando filtro Fourier a CC...')
            cc_filtered = self.apply_fourier_filter(self.raw_cc)
            print('Aplicando filtro Fourier a MLO...')
            mlo_filtered = self.apply_fourier_filter(self.raw_mlo)
            from PIL import Image
            cc_path = os.path.join(folder, f'{side}_CC_fourier_{timestamp}.png')
            Image.fromarray(cc_filtered).save(cc_path)
            print(f'Guardado: {cc_path}')
            mlo_path = os.path.join(folder, f'{side}_MLO_fourier_{timestamp}.png')
            Image.fromarray(mlo_filtered).save(mlo_path)
            print(f'Guardado: {mlo_path}')
            QMessageBox.information(self, 'Descarga completada', f'Imágenes guardadas en:\n{folder}\n\n• {side}_CC_fourier_{timestamp}.png\n• {side}_MLO_fourier_{timestamp}.png')
        except Exception as e:
            print(f'Error al guardar imágenes: {e}')
            QMessageBox.critical(self, 'Error', f'No se pudieron guardar las imágenes:\n{e}')

    def display_image(self, img_array, label):
        if img_array is None:
            return
        img_data = np.ascontiguousarray(img_array)
        h, w, c = img_data.shape
        q_img = QImage(img_data.data, w, h, c * w, QImage.Format_RGB888).copy()
        painter = QPainter(q_img)
        painter.setPen(QColor(255, 255, 255))
        font = QFont('Arial', 24, QFont.Bold)
        painter.setFont(font)
        side = 'L' if self.btn_left.isChecked() else 'R'
        view = 'MLO' if 'MLO' in label.objectName() or label == self.img_mlo else 'CC'
        painter.drawText(50, 80, f'{side}-{view}')
        font_small = QFont('Consolas', 12)
        painter.setFont(font_small)
        painter.drawText(50, 110, '28 kVp  High Flux')
        painter.drawText(50, 130, 'Target: Mo/Mo')
        painter.end()
        pix = QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pix)
        label.setText('')

    def draw_paddles(self, z_min, z_max, y_limit, is_mlo=False, mlo_angle=45):
        p_depth, p_width = (400, 400)
        center_y = y_limit + p_depth / 2.0
        if is_mlo:
            compression_axis = self.get_mlo_compression_axis()
            meshes = self.get_transformed_meshes(is_mlo=False)
            filtered_points = []
            for m in meshes.values():
                if m is not None and hasattr(m, 'n_points') and (m.n_points > 0):
                    pts = m.points
                    mask = pts[:, 1] > y_limit
                    if np.any(mask):
                        filtered_points.append(pts[mask])
            if filtered_points:
                combined = np.vstack(filtered_points)
                projections = np.dot(combined, compression_axis)
                proj_min = np.min(projections)
                proj_max = np.max(projections)
                center_x = np.mean(combined[:, 0])
                center_z = np.mean(combined[:, 2])
                center = np.array([center_x, center_y, center_z])
                margin = 5
                bot_offset = proj_min - margin
                bot_center = center + compression_axis * (bot_offset - np.dot(center, compression_axis))
                top_offset = proj_max + margin
                top_center = center + compression_axis * (top_offset - np.dot(center, compression_axis))
            else:
                all_points = []
                for m in meshes.values():
                    if m is not None and hasattr(m, 'n_points') and (m.n_points > 0):
                        all_points.append(m.points)
                if all_points:
                    combined = np.vstack(all_points)
                    projections = np.dot(combined, compression_axis)
                    proj_min = np.min(projections)
                    proj_max = np.max(projections)
                    center = np.array([np.mean(combined[:, 0]), center_y, np.mean(combined[:, 2])])
                else:
                    proj_min, proj_max = (z_min, z_max)
                    center = np.array([0, center_y, (z_min + z_max) / 2])
                margin = 5
                bot_offset = proj_min - margin
                bot_center = center + compression_axis * (bot_offset - np.dot(center, compression_axis))
                top_offset = proj_max + margin
                top_center = center + compression_axis * (top_offset - np.dot(center, compression_axis))
            p_bot = pv.Plane(center=bot_center, direction=compression_axis, i_size=p_depth, j_size=p_width)
            p_top = pv.Plane(center=top_center, direction=compression_axis, i_size=p_depth, j_size=p_width)
        else:
            meshes = self.get_transformed_meshes(is_mlo=False)
            filtered_points = []
            for m in meshes.values():
                if m is not None and hasattr(m, 'n_points') and (m.n_points > 0):
                    pts = m.points
                    mask = pts[:, 1] > y_limit
                    if np.any(mask):
                        filtered_points.append(pts[mask])
            if filtered_points:
                combined = np.vstack(filtered_points)
                z_min_actual = np.min(combined[:, 2])
                z_max_actual = np.max(combined[:, 2])
            else:
                z_min_actual = z_min
                z_max_actual = z_max
            margin = 5
            p_bot = pv.Plane(center=(0, center_y, z_min_actual - margin), i_size=p_depth, j_size=p_width)
            p_top = pv.Plane(center=(0, center_y, z_max_actual + margin), i_size=p_depth, j_size=p_width)
        self.plotter.add_mesh(p_bot, color='#34495e', opacity=0.4, name='bot_p')
        self.plotter.add_mesh(p_top, color='#7f8c8d', opacity=0.2, name='top_p')

    def draw_coordinate_axes(self):
        for axis_name in ['axis_x', 'axis_y', 'axis_z', 'axis_label_x', 'axis_label_y', 'axis_label_z']:
            try:
                self.plotter.remove_actor(axis_name)
            except:
                pass
        axis_length = 80
        arrow_x = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=axis_length, tip_length=0.15, tip_radius=0.08, shaft_radius=0.03)
        self.plotter.add_mesh(arrow_x, color='red', name='axis_x')
        self.plotter.add_point_labels([(axis_length + 10, 0, 0)], ['X'], font_size=20, text_color='red', name='axis_label_x', shape=None)
        arrow_y = pv.Arrow(start=(0, 0, 0), direction=(0, 1, 0), scale=axis_length, tip_length=0.15, tip_radius=0.08, shaft_radius=0.03)
        self.plotter.add_mesh(arrow_y, color='green', name='axis_y')
        self.plotter.add_point_labels([(0, axis_length + 10, 0)], ['Y'], font_size=20, text_color='green', name='axis_label_y', shape=None)
        arrow_z = pv.Arrow(start=(0, 0, 0), direction=(0, 0, 1), scale=axis_length, tip_length=0.15, tip_radius=0.08, shaft_radius=0.03)
        self.plotter.add_mesh(arrow_z, color='blue', name='axis_z')
        self.plotter.add_point_labels([(0, 0, axis_length + 10)], ['Z'], font_size=20, text_color='blue', name='axis_label_z', shape=None)

    def get_mlo_compression_axis(self):
        rot_y = np.radians(45)
        axis = np.array([np.sin(rot_y), 0.0, np.cos(rot_y)])
        return axis / np.linalg.norm(axis)

    def refresh_view(self):
        self.img_cc.clear()
        self.img_cc.setText('Waiting for CC...')
        self.img_mlo.clear()
        self.img_mlo.setText('Waiting for MLO...')
        self.btn_qa.setEnabled(False)
        self.plotter.clear()
        meshes = self.get_transformed_meshes(is_mlo=False)
        valid_meshes = {}
        for name, m in meshes.items():
            if m is not None and hasattr(m, 'n_points') and (m.n_points > 0):
                valid_meshes[name] = m
            else:
                print(f'[SKIP VIEW] {name}: mesh is empty or invalid')
        if not valid_meshes:
            print('[WARNING] No valid meshes for 3D view')
            return
        y_limit = self.sld_chest.value()
        gravity_val = self.sld_gravity.value() / 100.0
        if gravity_val > 0:
            gravity_dir = np.array([0.0, 0.0, -1.0])
            valid_meshes = PhysicsEngine.apply_gravity_to_assembly(valid_meshes, gravity_direction=gravity_dir, anchor_axis=1, anchor_limit=y_limit, gravity_strength=gravity_val)
        z_min, z_max = PhysicsEngine.get_compressible_bounds(list(valid_meshes.values()), y_limit)
        current_z_max = z_max
        display_meshes = {}
        for name, m in valid_meshes.items():
            try:
                if self.show_compressed:
                    m = m.clip(normal=[0, 1, 0], origin=(0, y_limit, 0), invert=False)
                    m = m.clip(normal='z', origin=(0, 0, z_min), invert=False)
                    deformed, top_z = PhysicsEngine.apply_compression(name, m, force=130, elasticity=10, z_min=z_min, z_max=z_max)
                    deformed = deformed.clip(normal='z', origin=(0, 0, top_z), invert=True)
                    display_meshes[name] = deformed
                    if top_z < current_z_max:
                        current_z_max = top_z
                else:
                    display_meshes[name] = m
            except Exception as e:
                print(f'[ERROR VIEW] {name}: {str(e)}')
                display_meshes[name] = m
        if self.show_compressed and 'Skin' in display_meshes and (display_meshes['Skin'].n_points > 0):
            skin_mesh = display_meshes['Skin']
            for name in list(display_meshes.keys()):
                if name == 'Skin':
                    continue
                m = display_meshes[name]
                if m.n_points == 0:
                    continue
                try:
                    result = m.compute_implicit_distance(skin_mesh)
                    distances = result.point_data['implicit_distance']
                    inside_mask = distances <= 1.0
                    if np.any(inside_mask):
                        m_filtered = m.extract_points(inside_mask, adjacent_cells=True)
                        if m_filtered.n_points > 0:
                            display_meshes[name] = m_filtered
                except Exception as e:
                    print(f'   {name}: filtro piel falló ({e})')
        for name, final_mesh in display_meshes.items():
            try:
                if final_mesh.n_points > 0:
                    cfg = NIST_PHYSICS.get(name, {'color': '#FFFFFF', 'opacity': 0.5})
                    self.plotter.add_mesh(final_mesh, color=cfg['color'], opacity=cfg['opacity'], name=name)
                else:
                    print(f'[SKIP VIEW] {name}: mesh is empty')
            except Exception as e:
                print(f'[ERROR VIEW] {name}: {str(e)}')
                continue
        if len(display_meshes) > 0:
            display_top = current_z_max if self.show_compressed else z_max
            mlo_angle = 45
            is_mlo_view = self.btn_view_mlo.isChecked()
            self.draw_paddles(z_min, display_top, y_limit, is_mlo=is_mlo_view, mlo_angle=mlo_angle)
            self.draw_coordinate_axes()
            self.plotter.view_isometric()
            self.plotter.reset_camera()

    def toggle_compression_preview(self, checked):
        self.show_compressed = checked
        if checked:
            self.btn_preview.setText('HIDE COMPRESSION (3D)')
        else:
            self.btn_preview.setText('PREVIEW COMPRESSION (3D)')
        self.refresh_view()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    font = QFont('Segoe UI', 10)
    font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(font)
    win = MammographyProSim()
    win.show()
    sys.exit(app.exec())