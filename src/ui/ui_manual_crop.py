import numpy as np
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget, QSlider, QGroupBox, QMessageBox
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QCursor
from PySide6.QtCore import Qt, QRect, QPoint, Signal
from src.config.theme_config import THEME_COLORS

class CropLabel(QLabel):
    crop_changed = Signal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt.CrossCursor))
        self.selection_start = None
        self.selection_end = None
        self.is_selecting = False
        self.selection_rect = None
        self.original_pixmap = None
        self.display_scale = 1.0

    def setPixmap(self, pixmap):
        self.original_pixmap = pixmap
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.display_scale = scaled.width() / pixmap.width()
        super().setPixmap(scaled)

    def get_original_rect(self):
        if self.selection_rect is None:
            return None
        pixmap = self.pixmap()
        if pixmap is None:
            return None
        offset_x = (self.width() - pixmap.width()) // 2
        offset_y = (self.height() - pixmap.height()) // 2
        x = (self.selection_rect.x() - offset_x) / self.display_scale
        y = (self.selection_rect.y() - offset_y) / self.display_scale
        w = self.selection_rect.width() / self.display_scale
        h = self.selection_rect.height() / self.display_scale
        return QRect(int(x), int(y), int(w), int(h))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_selecting = True
            self.selection_start = event.pos()
            self.selection_end = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.selection_end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_selecting:
            self.is_selecting = False
            self.selection_end = event.pos()
            x1, y1 = (self.selection_start.x(), self.selection_start.y())
            x2, y2 = (self.selection_end.x(), self.selection_end.y())
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            if w > 20 and h > 20:
                self.selection_rect = QRect(x, y, w, h)
                self.crop_changed.emit(self.selection_rect)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selection_start is not None and self.selection_end is not None:
            painter = QPainter(self)
            pen = QPen(QColor(THEME_COLORS['teal']))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            x1, y1 = (self.selection_start.x(), self.selection_start.y())
            x2, y2 = (self.selection_end.x(), self.selection_end.y())
            rect = QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            painter.drawRect(rect)
            painter.setBrush(QColor(0, 0, 0, 100))
            painter.setPen(Qt.NoPen)
            full_rect = self.rect()
            painter.drawRect(0, 0, full_rect.width(), rect.top())
            painter.drawRect(0, rect.bottom(), full_rect.width(), full_rect.height() - rect.bottom())
            painter.drawRect(0, rect.top(), rect.left(), rect.height())
            painter.drawRect(rect.right(), rect.top(), full_rect.width() - rect.right(), rect.height())
            painter.setPen(QColor(255, 255, 255))
            orig_rect = self.get_original_rect()
            if orig_rect:
                dim_text = f'{orig_rect.width()} x {orig_rect.height()} px'
                painter.drawText(rect.x() + 5, rect.y() + 20, dim_text)

    def clear_selection(self):
        self.selection_start = None
        self.selection_end = None
        self.selection_rect = None
        self.update()

class ManualCropDialog(QDialog):

    def __init__(self, cc_image: np.ndarray, mlo_image: np.ndarray, cc_raw: np.ndarray, mlo_raw: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle('MANUAL CROP - Seleccione la Región de Interés')
        self.setMinimumSize(1200, 700)
        self.setStyleSheet(f"background-color: {THEME_COLORS['bg']}; color: {THEME_COLORS['text_main']};")
        self.cc_display = cc_image
        self.mlo_display = mlo_image
        self.cc_raw = cc_raw
        self.mlo_raw = mlo_raw
        self.cc_crop_rect = None
        self.mlo_crop_rect = None
        self.result_cc_raw = None
        self.result_mlo_raw = None
        self.result_cc_display = None
        self.result_mlo_display = None
        self.setup_ui()
        self.load_images()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        title = QLabel('CROP MANUAL DE IMÁGENES')
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {THEME_COLORS['teal']};")
        layout.addWidget(title)
        instructions = QLabel('Dibuje un rectángulo sobre cada imagen para seleccionar la región de la mama.\nHaga clic y arrastre para seleccionar. El área seleccionada se usará para los cálculos.')
        instructions.setStyleSheet(f"color: {THEME_COLORS['text_dim']}; font-size: 12px;")
        layout.addWidget(instructions)
        images_layout = QHBoxLayout()
        cc_group = QGroupBox('CRANIOCAUDAL (CC)')
        cc_group.setStyleSheet(f"\n            QGroupBox {{\n                font-weight: bold;\n                color: {THEME_COLORS['teal']};\n                border: 1px solid {THEME_COLORS['border']};\n                border-radius: 4px;\n                margin-top: 10px;\n                padding-top: 10px;\n            }}\n            QGroupBox::title {{\n                subcontrol-origin: margin;\n                left: 10px;\n                padding: 0 5px;\n            }}\n        ")
        cc_layout = QVBoxLayout(cc_group)
        self.cc_label = CropLabel()
        self.cc_label.setFixedSize(500, 550)
        self.cc_label.setAlignment(Qt.AlignCenter)
        self.cc_label.setStyleSheet(f"border: 1px solid {THEME_COLORS['border']}; background: black;")
        self.cc_label.crop_changed.connect(lambda r: self.on_crop_changed('cc', r))
        cc_layout.addWidget(self.cc_label)
        self.cc_info = QLabel('Selección: Sin definir')
        self.cc_info.setStyleSheet(f"color: {THEME_COLORS['text_dim']}; font-size: 11px;")
        cc_layout.addWidget(self.cc_info)
        btn_clear_cc = QPushButton('Limpiar Selección CC')
        btn_clear_cc.clicked.connect(lambda: self.clear_selection('cc'))
        btn_clear_cc.setStyleSheet(f"\n            QPushButton {{\n                background: transparent;\n                border: 1px solid {THEME_COLORS['text_dim']};\n                color: {THEME_COLORS['text_main']};\n                padding: 5px 10px;\n            }}\n            QPushButton:hover {{\n                border-color: {THEME_COLORS['teal']};\n            }}\n        ")
        cc_layout.addWidget(btn_clear_cc)
        images_layout.addWidget(cc_group)
        mlo_group = QGroupBox('MEDIOLATERAL OBLIQUE (MLO)')
        mlo_group.setStyleSheet(cc_group.styleSheet())
        mlo_layout = QVBoxLayout(mlo_group)
        self.mlo_label = CropLabel()
        self.mlo_label.setFixedSize(500, 550)
        self.mlo_label.setAlignment(Qt.AlignCenter)
        self.mlo_label.setStyleSheet(f"border: 1px solid {THEME_COLORS['border']}; background: black;")
        self.mlo_label.crop_changed.connect(lambda r: self.on_crop_changed('mlo', r))
        mlo_layout.addWidget(self.mlo_label)
        self.mlo_info = QLabel('Selección: Sin definir')
        self.mlo_info.setStyleSheet(f"color: {THEME_COLORS['text_dim']}; font-size: 11px;")
        mlo_layout.addWidget(self.mlo_info)
        btn_clear_mlo = QPushButton('Limpiar Selección MLO')
        btn_clear_mlo.clicked.connect(lambda: self.clear_selection('mlo'))
        btn_clear_mlo.setStyleSheet(btn_clear_cc.styleSheet())
        mlo_layout.addWidget(btn_clear_mlo)
        images_layout.addWidget(mlo_group)
        layout.addLayout(images_layout)
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        btn_cancel = QPushButton('Cancelar')
        btn_cancel.clicked.connect(self.reject)
        btn_cancel.setStyleSheet(f"\n            QPushButton {{\n                background: transparent;\n                border: 1px solid {THEME_COLORS['text_dim']};\n                color: {THEME_COLORS['text_main']};\n                padding: 10px 30px;\n                font-size: 13px;\n            }}\n        ")
        buttons_layout.addWidget(btn_cancel)
        btn_apply = QPushButton('Aplicar Crop')
        btn_apply.clicked.connect(self.apply_crop)
        btn_apply.setStyleSheet(f"\n            QPushButton {{\n                background-color: {THEME_COLORS['teal']};\n                color: white;\n                border: none;\n                padding: 10px 30px;\n                font-size: 13px;\n                font-weight: bold;\n            }}\n            QPushButton:hover {{\n                background-color: #0d6b5c;\n            }}\n        ")
        buttons_layout.addWidget(btn_apply)
        layout.addLayout(buttons_layout)

    def load_images(self):
        if self.cc_display is not None:
            cc_pixmap = self.numpy_to_pixmap(self.cc_display)
            self.cc_label.setPixmap(cc_pixmap)
        if self.mlo_display is not None:
            mlo_pixmap = self.numpy_to_pixmap(self.mlo_display)
            self.mlo_label.setPixmap(mlo_pixmap)

    def numpy_to_pixmap(self, img_array):
        if img_array is None:
            return QPixmap()
        img_data = np.ascontiguousarray(img_array)
        h, w, c = img_data.shape
        q_img = QImage(img_data.data, w, h, c * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(q_img)

    def on_crop_changed(self, view, rect):
        label = self.cc_label if view == 'cc' else self.mlo_label
        info = self.cc_info if view == 'cc' else self.mlo_info
        orig_rect = label.get_original_rect()
        if orig_rect:
            if view == 'cc':
                self.cc_crop_rect = orig_rect
            else:
                self.mlo_crop_rect = orig_rect
            info.setText(f'Selección: ({orig_rect.x()}, {orig_rect.y()}) - {orig_rect.width()} x {orig_rect.height()} px')
            info.setStyleSheet(f"color: {THEME_COLORS['teal']}; font-size: 11px;")

    def clear_selection(self, view):
        if view == 'cc':
            self.cc_label.clear_selection()
            self.cc_crop_rect = None
            self.cc_info.setText('Selección: Sin definir')
            self.cc_info.setStyleSheet(f"color: {THEME_COLORS['text_dim']}; font-size: 11px;")
        else:
            self.mlo_label.clear_selection()
            self.mlo_crop_rect = None
            self.mlo_info.setText('Selección: Sin definir')
            self.mlo_info.setStyleSheet(f"color: {THEME_COLORS['text_dim']}; font-size: 11px;")

    def apply_crop(self):
        if self.cc_crop_rect is None and self.mlo_crop_rect is None:
            QMessageBox.warning(self, 'Sin selección', 'Debe seleccionar al menos una región para aplicar el crop.')
            return
        if self.cc_crop_rect is not None and self.cc_raw is not None:
            x, y = (self.cc_crop_rect.x(), self.cc_crop_rect.y())
            w, h = (self.cc_crop_rect.width(), self.cc_crop_rect.height())
            y_max = min(y + h, self.cc_raw.shape[0])
            x_max = min(x + w, self.cc_raw.shape[1])
            y = max(0, y)
            x = max(0, x)
            self.result_cc_raw = self.cc_raw[y:y_max, x:x_max].copy()
            self.result_cc_display = self.cc_display[y:y_max, x:x_max].copy()
            print(f'[CROP MANUAL CC] Aplicado: ({x}, {y}) -> ({x_max}, {y_max})')
            print(f'[CROP MANUAL CC] Resultado: {self.result_cc_raw.shape}')
        else:
            self.result_cc_raw = self.cc_raw
            self.result_cc_display = self.cc_display
        if self.mlo_crop_rect is not None and self.mlo_raw is not None:
            x, y = (self.mlo_crop_rect.x(), self.mlo_crop_rect.y())
            w, h = (self.mlo_crop_rect.width(), self.mlo_crop_rect.height())
            y_max = min(y + h, self.mlo_raw.shape[0])
            x_max = min(x + w, self.mlo_raw.shape[1])
            y = max(0, y)
            x = max(0, x)
            self.result_mlo_raw = self.mlo_raw[y:y_max, x:x_max].copy()
            self.result_mlo_display = self.mlo_display[y:y_max, x:x_max].copy()
            print(f'[CROP MANUAL MLO] Aplicado: ({x}, {y}) -> ({x_max}, {y_max})')
            print(f'[CROP MANUAL MLO] Resultado: {self.result_mlo_raw.shape}')
        else:
            self.result_mlo_raw = self.mlo_raw
            self.result_mlo_display = self.mlo_display
        self.accept()

    def get_cropped_images(self):
        return (self.result_cc_display, self.result_cc_raw, self.result_mlo_display, self.result_mlo_raw)