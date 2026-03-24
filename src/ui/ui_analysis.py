import numpy as np
from PySide6.QtWidgets import QDialog, QVBoxLayout, QGroupBox, QGridLayout, QLabel, QPushButton, QHBoxLayout, QTextEdit
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib
from src.config.theme_config import THEME_COLORS
import csv
import io

class AnalysisDialog(QDialog):

    def __init__(self, img_data, title='Mammography QA Analysis'):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1200, 900)
        self.setStyleSheet(f"\n            background-color: {THEME_COLORS['bg']}; \n            color: {THEME_COLORS['text_main']};\n            font-family: 'Segoe UI', Arial, sans-serif;\n        ")
        if len(img_data.shape) == 3:
            self.raw_data = np.mean(img_data, axis=2).astype(np.float64)
        else:
            self.raw_data = img_data.astype(np.float64)
        self.calculate_mammography_qa_metrics()
        self.setup_ui()

    def calculate_mammography_qa_metrics(self):
        np.random.seed(42)
        threshold = 2500
        mask_tissue = self.raw_data > threshold
        tissue_vals = self.raw_data[mask_tissue] if np.any(mask_tissue) else self.raw_data
        roi_noise = tissue_vals
        h, w = self.raw_data.shape
        y_indices, x_indices = np.where(mask_tissue)
        if len(y_indices) > 1000:
            cy_t = int(np.mean(y_indices))
            cx_t = int(np.mean(x_indices))
            y_min_t, y_max_t = (np.min(y_indices), np.max(y_indices))
            x_min_t, x_max_t = (np.min(x_indices), np.max(x_indices))
            dy = (y_max_t - y_min_t) // 3
            dx = (x_max_t - x_min_t) // 3
            self.roi_centers = [(cy_t, cx_t, 'Inner Center'), (y_min_t + dy, x_min_t + dx, 'Core NW'), (y_min_t + dy, x_max_t - dx, 'Core NE'), (y_max_t - dy, x_min_t + dx, 'Core SW'), (y_max_t - dy, x_max_t - dx, 'Core SE')]
        else:
            self.roi_centers = [(h // 2, w // 2, 'Center')]
        roi_size = min(60, h // 15, w // 15)
        rois_uniformity = []
        for cy, cx, label in self.roi_centers:
            y_start = max(0, int(cy - roi_size // 2))
            y_end = min(h, int(cy + roi_size // 2))
            x_start = max(0, int(cx - roi_size // 2))
            x_end = min(w, int(cx + roi_size // 2))
            rois_uniformity.append(self.raw_data[y_start:y_end, x_start:x_end])
        self.mean_signal = np.mean(tissue_vals)
        h, w = self.raw_data.shape
        sz = 10
        mask_air = self.raw_data < 1500
        y_air, x_air = np.where(mask_air)
        stds = []
        if len(y_air) > 500:
            for _ in range(200):
                idx = np.random.randint(0, len(y_air))
                ry, rx = (y_air[idx], x_air[idx])
                if ry < h - sz and rx < w - sz:
                    patch = self.raw_data[ry:ry + sz, rx:rx + sz]
                    stds.append(np.std(patch))
        if len(stds) < 50:
            y_t, x_t = np.where(mask_tissue)
            for _ in range(300):
                idx = np.random.randint(0, len(y_t))
                ry, rx = (y_t[idx], x_t[idx])
                if ry < h - sz and rx < w - sz:
                    patch = self.raw_data[ry:ry + sz, rx:rx + sz]
                    stds.append(np.std(patch))
        if len(stds) > 10:
            best_std = np.percentile(stds, 5)
        else:
            best_std = 30.0
        self.noise_std = max(best_std, 1.0)
        self.bg_mean = np.mean(tissue_vals)
        self.snr = self.mean_signal / self.noise_std
        if len(tissue_vals) > 100:
            p70 = np.percentile(tissue_vals, 70)
            p30 = np.percentile(tissue_vals, 30)
            mask_fgt = (self.raw_data > p70) & mask_tissue
            mask_fat = (self.raw_data > threshold) & (self.raw_data < p30)
            fgt_vals = self.raw_data[mask_fgt]
            fat_vals = self.raw_data[mask_fat]
            if len(fgt_vals) > 50 and len(fat_vals) > 50:
                mu_fgt = np.mean(fgt_vals)
                mu_fat = np.mean(fat_vals)
                sigma_fat = np.std(fat_vals)
                self.cnr = abs(mu_fgt - mu_fat) / max(sigma_fat, 1.0)
                self.mu_fgt = mu_fgt
                self.mu_fat = mu_fat
                self.sigma_fat = sigma_fat
            else:
                self.cnr = self.snr * 0.5
                self.mu_fgt = self.mean_signal
                self.mu_fat = self.mean_signal * 0.7
                self.sigma_fat = self.noise_std
        else:
            self.cnr = self.snr * 0.5
            self.mu_fgt = self.mean_signal
            self.mu_fat = self.mean_signal * 0.7
            self.sigma_fat = self.noise_std
        roi_means = [np.mean(roi) for roi in rois_uniformity]
        self.uniformity_means = roi_means
        self.non_uniformity = 100 * (max(roi_means) - min(roi_means)) / max(roi_means)
        self.rdi = self.mean_signal / 60000 * 1000 + 1500
        max_val = np.max(self.raw_data)
        min_val = np.min(self.raw_data[self.raw_data > 800]) if np.any(self.raw_data > 800) else 800
        self.dynamic_range_levels = max_val - min_val
        self.dynamic_range_db = 20 * np.log10(max_val / self.noise_std)
        self.bit_depth_usage_pct = 100 * max_val / 65535
        self.bit_depth_effective = self.dynamic_range_db / 6.02
        self.contrast_resolution = self.cnr * 20
        self.targets = {'mean_signal': {'min': 20000, 'max': 55000, 'unit': 'PV', 'weight': 1.0}, 'noise_std': {'max': 200, 'unit': 'PV', 'weight': 1.5}, 'snr': {'min': 100, 'unit': '', 'weight': 2.0}, 'cnr': {'min': 3, 'unit': '', 'weight': 1.5}, 'non_uniformity': {'max': 50, 'unit': '%', 'weight': 1.5}, 'rdi': {'min': 1500, 'max': 3000, 'unit': '', 'weight': 1.0}, 'dynamic_range_db': {'min': 60, 'unit': 'dB', 'weight': 0.5}, 'bit_depth_effective': {'min': 10, 'unit': 'bits', 'weight': 0.5}, 'contrast_resolution': {'min': 60, 'unit': 'steps', 'weight': 1.5}}
        self.qa_score = self.calculate_qa_score()

    def calculate_qa_score(self):
        score = 0
        max_score = 0
        metrics_to_check = [('snr', self.snr, 'min'), ('cnr', self.cnr, 'min'), ('non_uniformity', self.non_uniformity, 'max'), ('dynamic_range_db', self.dynamic_range_db, 'min'), ('bit_depth_effective', self.bit_depth_effective, 'min'), ('contrast_resolution', self.contrast_resolution, 'min')]
        for metric_name, value, check_type in metrics_to_check:
            target = self.targets[metric_name]
            weight = target['weight']
            max_score += weight * 10
            if check_type == 'min':
                target_val = target['min']
                if value >= target_val:
                    score += weight * 10
                else:
                    ratio = max(0, value / target_val)
                    score += weight * 10 * ratio
            else:
                target_val = target['max']
                if value <= target_val:
                    score += weight * 10
                else:
                    ratio = max(0, target_val / value)
                    score += weight * 10 * ratio
        if max_score > 0:
            final_score = score / max_score * 100
        else:
            final_score = 0
        return final_score

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        score_group = QGroupBox('QA Score Summary')
        score_layout = QGridLayout()
        score_label = QLabel(f'{self.qa_score:.0f}')
        score_label.setStyleSheet(f"\n            font-size: 48px;\n            font-weight: bold;\n            color: {self.get_score_color(self.qa_score)};\n            qproperty-alignment: 'AlignCenter';\n        ")
        score_text = QLabel(self.get_score_text(self.qa_score))
        score_text.setStyleSheet("font-size: 14px; color: #888; qproperty-alignment: 'AlignCenter';")
        score_layout.addWidget(score_label, 0, 0, 2, 1)
        score_layout.addWidget(score_text, 2, 0)
        quick_metrics = [('SNR', f'{self.snr:.1f}', self.targets['snr']['min'], '>'), ('CNR', f'{self.cnr:.2f}', self.targets['cnr']['min'], '>'), ('Uniformity', f'{self.non_uniformity:.1f}%', self.targets['non_uniformity']['max'], '<'), ('Dose Index', f'{self.rdi:.0f}', f"{self.targets['rdi']['min']}-{self.targets['rdi']['max']}", 'range')]
        for i, (name, value, target, comparison) in enumerate(quick_metrics):
            lbl_name = QLabel(name)
            lbl_value = QLabel(value)

            def extract_numeric(val_str):
                import re
                match = re.search('[-+]?\\d*\\.?\\d+', val_str)
                return float(match.group()) if match else 0.0
            val_num = extract_numeric(value)
            if comparison == '>':
                is_ok = val_num >= target
            elif comparison == '<':
                is_ok = val_num <= target
            else:
                min_val, max_val = map(float, target.split('-'))
                is_ok = min_val <= val_num <= max_val
            color = THEME_COLORS['success'] if is_ok else THEME_COLORS['danger']
            lbl_value.setStyleSheet(f'color: {color}; font-weight: bold; font-size: 16px;')
            score_layout.addWidget(lbl_name, i, 1)
            score_layout.addWidget(lbl_value, i, 2)
        score_group.setLayout(score_layout)
        layout.addWidget(score_group)
        metrics_group = QGroupBox('Detailed QA Metrics')
        metrics_layout = QGridLayout()
        metrics_layout.setVerticalSpacing(10)
        metrics_layout.setHorizontalSpacing(20)
        headers = ['Parameter', 'Measured', 'Target', 'Status', 'Weight']
        for col, header in enumerate(headers):
            lbl = QLabel(f'<b>{header}</b>')
            lbl.setStyleSheet('color: #666; font-size: 11px; border-bottom: 1px solid #444;')
            metrics_layout.addWidget(lbl, 0, col)

        def extract_numeric_from_measurement(measured_str):
            import re
            match = re.search('[-+]?\\d*\\.?\\d+', measured_str)
            if match:
                return float(match.group())
            return 0.0
        metrics_list = [('Mean Signal', f'{self.mean_signal:.0f} PV', f">{self.targets['mean_signal']['min']} PV", 'mean_signal', 'min'), ('Noise StdDev', f'{self.noise_std:.2f} PV', f"<{self.targets['noise_std']['max']} PV", 'noise_std', 'max'), ('SNR', f'{self.snr:.1f}', f">{self.targets['snr']['min']}", 'snr', 'min'), ('CNR', f'{self.cnr:.2f}', f">{self.targets['cnr']['min']}", 'cnr', 'min'), ('Field Non-Uniformity', f'{self.non_uniformity:.1f}%', f"<{self.targets['non_uniformity']['max']}%", 'non_uniformity', 'max'), ('Dose Indicator', f'{self.rdi:.0f}', f"{self.targets['rdi']['min']}-{self.targets['rdi']['max']}", 'rdi', 'range'), ('Dynamic Range', f'{self.dynamic_range_db:.1f} dB', f">{self.targets['dynamic_range_db']['min']} dB", 'dynamic_range_db', 'min'), ('Effective Bits', f'{self.bit_depth_effective:.1f} bits', f">{self.targets['bit_depth_effective']['min']} bits", 'bit_depth_effective', 'min'), ('Contrast Resolution', f'{self.contrast_resolution:.0f} steps', f">{self.targets['contrast_resolution']['min']} steps", 'contrast_resolution', 'min')]
        for row, (name, measured, target, metric_key, check_type) in enumerate(metrics_list, 1):
            lbl_name = QLabel(name)
            lbl_name.setStyleSheet('color: #DDD; font-size: 12px;')
            metrics_layout.addWidget(lbl_name, row, 0)
            lbl_measured = QLabel(measured)
            lbl_measured.setStyleSheet("font-family: 'Consolas'; font-size: 13px; color: #EEE;")
            metrics_layout.addWidget(lbl_measured, row, 1)
            lbl_target = QLabel(target)
            lbl_target.setStyleSheet('color: #AAA; font-size: 12px;')
            metrics_layout.addWidget(lbl_target, row, 2)
            target_info = self.targets[metric_key]
            measured_numeric = extract_numeric_from_measurement(measured)
            if check_type == 'min':
                target_val = target_info['min']
                status = 'PASS' if measured_numeric >= target_val else 'FAIL'
            elif check_type == 'max':
                target_val = target_info['max']
                status = 'PASS' if measured_numeric <= target_val else 'FAIL'
            else:
                min_val = target_info['min']
                max_val = target_info['max']
                status = 'PASS' if min_val <= measured_numeric <= max_val else 'FAIL'
            lbl_status = QLabel(status)
            status_color = THEME_COLORS['success'] if status == 'PASS' else THEME_COLORS['danger']
            lbl_status.setStyleSheet(f'\n                color: {status_color};\n                font-weight: bold;\n                font-size: 12px;\n                padding: 2px 8px;\n                border-radius: 3px;\n                background-color: {status_color}20;\n            ')
            metrics_layout.addWidget(lbl_status, row, 3)
            lbl_weight = QLabel(f"{target_info['weight']:.1f}")
            lbl_weight.setStyleSheet('color: #888; font-size: 12px; text-align: center;')
            metrics_layout.addWidget(lbl_weight, row, 4)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.figure.patch.set_facecolor(THEME_COLORS['bg'])
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        self.plot_qa_graphs()
        button_layout = QHBoxLayout()
        btn_export = QPushButton('📊 Export QA Report')
        btn_export.setStyleSheet(f"\n            QPushButton {{\n                background-color: {THEME_COLORS['teal']};\n                color: white;\n                padding: 8px 16px;\n                border-radius: 4px;\n                font-weight: bold;\n            }}\n            QPushButton:hover {{\n                background-color: {THEME_COLORS['teal']}DD;\n            }}\n        ")
        btn_export.clicked.connect(self.export_qa_report)
        btn_details = QPushButton('🔍 Show Details')
        btn_details.setStyleSheet('\n            QPushButton {\n                background-color: #444;\n                color: white;\n                padding: 8px 16px;\n                border-radius: 4px;\n            }\n        ')
        btn_details.clicked.connect(self.show_detailed_analysis)
        button_layout.addStretch()
        button_layout.addWidget(btn_details)
        button_layout.addWidget(btn_export)
        layout.addLayout(button_layout)

    def plot_qa_graphs(self):
        self.figure.clear()
        params = {'axes.facecolor': THEME_COLORS['bg'], 'axes.edgecolor': THEME_COLORS['border'], 'axes.labelcolor': THEME_COLORS['text_dim'], 'xtick.color': THEME_COLORS['text_dim'], 'ytick.color': THEME_COLORS['text_dim'], 'text.color': THEME_COLORS['text_main'], 'grid.color': '#333333', 'font.family': 'sans-serif'}
        with matplotlib.rc_context(params):
            ax1 = self.figure.add_subplot(221)
            vals = self.raw_data.ravel()
            ax1.hist(vals, bins=200, color='#3498DB', alpha=0.6, edgecolor='none')
            tissue_ranges = [(0, 800, 'Air/Fondo', '#95A5A6'), (800, 4000, 'Adipose', '#E67E22'), (4000, 8000, 'FGT', '#E74C3C'), (8000, 20000, 'Dense', '#C0392B')]
            for xmin, xmax, label, color in tissue_ranges:
                ax1.axvspan(xmin, xmax, alpha=0.2, color=color)
                ax1.text((xmin + xmax) / 2, ax1.get_ylim()[1] * 0.9, label, ha='center', va='top', color=color, fontsize=8)
            ax1.set_title('Pixel Value Distribution (Tissue Ranges)', fontsize=10, pad=10)
            ax1.set_xlabel('Pixel Value (PV)', fontsize=9)
            ax1.set_ylabel('Frequency', fontsize=9)
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.2, linestyle='--')
            ax2 = self.figure.add_subplot(222)
            labels = ['NW', 'NE', 'Center', 'SW', 'SE']
            positions = np.arange(len(labels))
            bars = ax2.bar(positions, self.uniformity_means, color=['#1ABC9C', '#2ECC71', '#3498DB', '#9B59B6', '#E74C3C'], edgecolor='white', linewidth=1)
            mean_val = np.mean(self.uniformity_means)
            ax2.axhline(mean_val, color='yellow', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.0f} PV')
            ax2.set_title('Field Uniformity (5-Point Analysis)', fontsize=10, pad=10)
            ax2.set_xlabel('Region', fontsize=9)
            ax2.set_ylabel('Mean PV', fontsize=9)
            ax2.set_xticks(positions)
            ax2.set_xticklabels(labels)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.2, linestyle='--')
            ax3 = self.figure.add_subplot(223)
            mid_y = self.raw_data.shape[0] // 2
            line_profile = self.raw_data[mid_y, :]
            ax3.plot(line_profile, color='#9B59B6', linewidth=1.5)
            ax3.axhline(np.mean(line_profile), color='yellow', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(line_profile):.0f}')
            ax3.fill_between(range(len(line_profile)), np.mean(line_profile) - self.noise_std, np.mean(line_profile) + self.noise_std, alpha=0.2, color='#9B59B6', label=f'±σ: {self.noise_std:.1f}')
            ax3.set_title(f'Horizontal Profile at Y={mid_y}', fontsize=10, pad=10)
            ax3.set_xlabel('Pixel Position (X)', fontsize=9)
            ax3.set_ylabel('Pixel Value', fontsize=9)
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.2, linestyle='--')
            ax4 = self.figure.add_subplot(224)
            metrics_names = ['SNR', 'CNR', 'Uniformity', 'Dose', 'Dynamic Range']
            metrics_values = [min(100, self.snr / 30 * 100), min(100, self.cnr / 2.0 * 100), min(100, (1 - self.non_uniformity / 15) * 100), min(100, abs(self.rdi - 2000) / 500 * 100), min(100, self.dynamic_range_db / 60 * 100)]
            angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False)
            values = metrics_values + [metrics_values[0]]
            angles = np.concatenate((angles, [angles[0]]))
            ax4.plot(angles, values, 'o-', linewidth=2, color='#3498DB')
            ax4.fill(angles, values, alpha=0.25, color='#3498DB')
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics_names)
            ax4.set_ylim(0, 100)
            ax4.set_yticks([0, 50, 100])
            ax4.set_yticklabels(['0%', '50%', '100%'])
            ax4.set_title('QA Metrics Radar (Normalized)', fontsize=10, pad=10)
            ax4.grid(True)
            self.figure.tight_layout()
            self.canvas.draw()

    def get_score_color(self, score):
        if score >= 90:
            return '#2ECC71'
        elif score >= 70:
            return '#F39C12'
        else:
            return '#E74C3C'

    def get_score_text(self, score):
        if score >= 90:
            return 'EXCELLENT - Clinical Grade'
        elif score >= 80:
            return 'GOOD - Minor improvements needed'
        elif score >= 70:
            return 'FAIR - Review recommended'
        elif score >= 60:
            return 'MARGINAL - Needs optimization'
        else:
            return 'POOR - Not clinically acceptable'

    def export_qa_report(self):
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'mammo_qa_report_{timestamp}.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['MAMMOGRAPHY QA ANALYSIS REPORT'])
            writer.writerow([f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
            writer.writerow([f'Overall QA Score: {self.qa_score:.1f}/100'])
            writer.writerow([])
            writer.writerow(['MAIN METRICS', 'Value', 'Target', 'Status'])
            writer.writerow(['-' * 40, '-' * 15, '-' * 15, '-' * 10])
            main_metrics = [['Mean Signal (PV)', f'{self.mean_signal:.0f}', f">{self.targets['mean_signal']['min']}", 'PASS' if self.mean_signal > self.targets['mean_signal']['min'] else 'FAIL'], ['Noise StdDev (PV)', f'{self.noise_std:.2f}', f"<{self.targets['noise_std']['max']}", 'PASS' if self.noise_std < self.targets['noise_std']['max'] else 'FAIL'], ['Signal-to-Noise Ratio', f'{self.snr:.1f}', f">{self.targets['snr']['min']}", 'PASS' if self.snr > self.targets['snr']['min'] else 'FAIL'], ['Contrast-to-Noise Ratio', f'{self.cnr:.2f}', f">{self.targets['cnr']['min']}", 'PASS' if self.cnr > self.targets['cnr']['min'] else 'FAIL'], ['Field Non-Uniformity (%)', f'{self.non_uniformity:.1f}', f"<{self.targets['non_uniformity']['max']}", 'PASS' if self.non_uniformity < self.targets['non_uniformity']['max'] else 'FAIL'], ['Dose Indicator', f'{self.rdi:.0f}', f"{self.targets['rdi']['min']}-{self.targets['rdi']['max']}", 'PASS' if self.targets['rdi']['min'] <= self.rdi <= self.targets['rdi']['max'] else 'FAIL'], ['Dynamic Range (dB)', f'{self.dynamic_range_db:.1f}', f">{self.targets['dynamic_range_db']['min']}", 'PASS' if self.dynamic_range_db > self.targets['dynamic_range_db']['min'] else 'FAIL'], ['Effective Bits', f'{self.bit_depth_effective:.1f}', f">{self.targets['bit_depth_effective']['min']}", 'PASS' if self.bit_depth_effective > self.targets['bit_depth_effective']['min'] else 'FAIL']]
            for row in main_metrics:
                writer.writerow(row)
            writer.writerow([])
            writer.writerow(['ADDITIONAL STATISTICS', 'Value'])
            writer.writerow(['-' * 40, '-' * 15])
            additional_stats = [['Image Dimensions', f'{self.raw_data.shape}'], ['Min Pixel Value', f'{np.min(self.raw_data):.0f}'], ['Max Pixel Value', f'{np.max(self.raw_data):.0f}'], ['Background Mean', f'{self.bg_mean:.1f}'], ['CNR - FGT Mean (μ_tejido)', f'{self.mu_fgt:.1f}'], ['CNR - Fat Mean (μ_fondo)', f'{self.mu_fat:.1f}'], ['CNR - Fat StdDev (σ_fondo)', f'{self.sigma_fat:.2f}'], ['Dynamic Range (levels)', f'{self.dynamic_range_levels:.0f}'], ['Bit Depth Usage (%)', f'{self.bit_depth_usage_pct:.1f}'], ['Contrast Resolution (steps)', f'{self.contrast_resolution:.0f}'], ['Uniformity Variation', f'±{np.std(self.uniformity_means):.1f} PV']]
            for row in additional_stats:
                writer.writerow(row)
            writer.writerow([])
            writer.writerow(['REGION OF INTEREST ANALYSIS'])
            writer.writerow(['Region', 'Mean PV', 'Std Dev', 'SNR'])
            for (cy, cx, label), mean_val in zip(self.roi_centers, self.uniformity_means):
                roi = self.get_roi_data(cy, cx)
                std_val = np.std(roi)
                snr_val = mean_val / std_val if std_val > 0 else 0
                writer.writerow([label, f'{mean_val:.1f}', f'{std_val:.1f}', f'{snr_val:.1f}'])
        print(f'✅ QA report exported to: {filename}')
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, 'Export Complete', f'QA report saved as:\n{filename}')

    def get_roi_data(self, center_y, center_x, size=50):
        h, w = self.raw_data.shape
        y_start = max(0, center_y - size // 2)
        y_end = min(h, center_y + size // 2)
        x_start = max(0, center_x - size // 2)
        x_end = min(w, center_x + size // 2)
        return self.raw_data[y_start:y_end, x_start:x_end]

    def show_detailed_analysis(self):
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit
        detail_dialog = QDialog(self)
        detail_dialog.setWindowTitle('Detailed QA Analysis')
        detail_dialog.resize(600, 800)
        layout = QVBoxLayout(detail_dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet(f"\n            background-color: {THEME_COLORS['bg']};\n            color: {THEME_COLORS['text_main']};\n            font-family: 'Consolas', monospace;\n            font-size: 11px;\n        ")
        report = self.generate_detailed_report()
        text_edit.setPlainText(report)
        layout.addWidget(text_edit)
        detail_dialog.exec()

    def generate_detailed_report(self):
        import datetime
        report = f"\n{'=' * 80}\nMAMMOGRAPHY DIGITAL QA ANALYSIS REPORT\n{'=' * 80}\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nOverall QA Score: {self.qa_score:.1f}/100 - {self.get_score_text(self.qa_score)}\n\n{'=' * 80}\n1. IMAGE CHARACTERISTICS\n{'=' * 80}\nDimensions: {self.raw_data.shape[1]} x {self.raw_data.shape[0]} pixels\nBit Depth: 16-bit (0-65535)\nPixel Range: {np.min(self.raw_data):.0f} - {np.max(self.raw_data):.0f} PV\nMean Value: {np.mean(self.raw_data):.1f} PV\nMedian Value: {np.median(self.raw_data):.1f} PV\n\n{'=' * 80}\n2. PHYSICS METRICS\n{'=' * 80}\nMean Signal:          {self.mean_signal:>8.0f} PV   [Target: >{self.targets['mean_signal']['min']} PV]\nNoise (Std Dev):      {self.noise_std:>8.2f} PV   [Target: <{self.targets['noise_std']['max']} PV]\nSignal-to-Noise:      {self.snr:>8.1f}       [Target: >{self.targets['snr']['min']}]\nContrast-to-Noise:    {self.cnr:>8.2f}       [Target: >{self.targets['cnr']['min']}]\nField Non-Uniformity: {self.non_uniformity:>8.1f} %    [Target: <{self.targets['non_uniformity']['max']} %]\nDose Indicator:       {self.rdi:>8.0f}       [Target: {self.targets['rdi']['min']}-{self.targets['rdi']['max']}]\nDynamic Range:        {self.dynamic_range_db:>8.1f} dB  [Target: >{self.targets['dynamic_range_db']['min']} dB]\nEffective Bits:       {self.bit_depth_effective:>8.1f} bits [Target: >{self.targets['bit_depth_effective']['min']} bits]\n\n{'=' * 80}\n3. TISSUE RANGES ANALYSIS\n{'=' * 80}\nAir/Background:       {np.sum((self.raw_data >= 0) & (self.raw_data < 800)):>8d} pixels\nAdipose Tissue:       {np.sum((self.raw_data >= 800) & (self.raw_data < 4000)):>8d} pixels\nFibroglandular:       {np.sum((self.raw_data >= 4000) & (self.raw_data < 8000)):>8d} pixels\nDense Tissue:         {np.sum(self.raw_data >= 8000):>8d} pixels\n\nPercentile Analysis:\n  5th percentile:     {np.percentile(self.raw_data, 5):.0f} PV\n 25th percentile:     {np.percentile(self.raw_data, 25):.0f} PV (Adipose reference)\n 50th percentile:     {np.percentile(self.raw_data, 50):.0f} PV (Median)\n 75th percentile:     {np.percentile(self.raw_data, 75):.0f} PV (FGT reference)\n 95th percentile:     {np.percentile(self.raw_data, 95):.0f} PV\n\n{'=' * 80}\n4. UNIFORMITY ANALYSIS (5-POINT)\n{'=' * 80}\nRegion    Mean PV    Std Dev     SNR      Variation\n------    -------    -------     ---      ---------\nNW        {self.uniformity_means[1]:>8.1f}   {np.std(self.get_roi_data(*self.roi_centers[1][:2])):>8.1f}   {(self.uniformity_means[1] / np.std(self.get_roi_data(*self.roi_centers[1][:2])) if np.std(self.get_roi_data(*self.roi_centers[1][:2])) > 0 else 0):>8.1f}   {(self.uniformity_means[1] - np.mean(self.uniformity_means)) / np.mean(self.uniformity_means) * 100:>8.1f}%\nNE        {self.uniformity_means[2]:>8.1f}   {np.std(self.get_roi_data(*self.roi_centers[2][:2])):>8.1f}   {(self.uniformity_means[2] / np.std(self.get_roi_data(*self.roi_centers[2][:2])) if np.std(self.get_roi_data(*self.roi_centers[2][:2])) > 0 else 0):>8.1f}   {(self.uniformity_means[2] - np.mean(self.uniformity_means)) / np.mean(self.uniformity_means) * 100:>8.1f}%\nCenter    {self.uniformity_means[0]:>8.1f}   {np.std(self.get_roi_data(*self.roi_centers[0][:2])):>8.1f}   {(self.uniformity_means[0] / np.std(self.get_roi_data(*self.roi_centers[0][:2])) if np.std(self.get_roi_data(*self.roi_centers[0][:2])) > 0 else 0):>8.1f}   {(self.uniformity_means[0] - np.mean(self.uniformity_means)) / np.mean(self.uniformity_means) * 100:>8.1f}%\nSW        {self.uniformity_means[3]:>8.1f}   {np.std(self.get_roi_data(*self.roi_centers[3][:2])):>8.1f}   {(self.uniformity_means[3] / np.std(self.get_roi_data(*self.roi_centers[3][:2])) if np.std(self.get_roi_data(*self.roi_centers[3][:2])) > 0 else 0):>8.1f}   {(self.uniformity_means[3] - np.mean(self.uniformity_means)) / np.mean(self.uniformity_means) * 100:>8.1f}%\nSE        {self.uniformity_means[4]:>8.1f}   {np.std(self.get_roi_data(*self.roi_centers[4][:2])):>8.1f}   {(self.uniformity_means[4] / np.std(self.get_roi_data(*self.roi_centers[4][:2])) if np.std(self.get_roi_data(*self.roi_centers[4][:2])) > 0 else 0):>8.1f}   {(self.uniformity_means[4] - np.mean(self.uniformity_means)) / np.mean(self.uniformity_means) * 100:>8.1f}%\n\nMaximum Variation: {max(self.uniformity_means) - min(self.uniformity_means):.1f} PV ({self.non_uniformity:.1f}%)\nAcceptance: {('PASS' if self.non_uniformity < self.targets['non_uniformity']['max'] else 'FAIL')} (<{self.targets['non_uniformity']['max']}%)\n\n{'=' * 80}\n5. RECOMMENDATIONS\n{'=' * 80}\n"
        recommendations = []
        if self.snr < self.targets['snr']['min']:
            recommendations.append('• Increase exposure (flux_photons) to improve SNR')
        if self.cnr < self.targets['cnr']['min']:
            recommendations.append('• Check tissue attenuation coefficients (μ values)')
        if self.non_uniformity > self.targets['non_uniformity']['max']:
            recommendations.append('• Review compression uniformity and beam profile')
        if self.rdi < self.targets['rdi']['min']:
            recommendations.append('• Image may be underexposed - consider increasing dose proxy')
        elif self.rdi > self.targets['rdi']['max']:
            recommendations.append('• Image may be overexposed - consider decreasing dose proxy')
        if self.qa_score < 70:
            recommendations.append('• Overall image quality below clinical standards')
        elif self.qa_score >= 90:
            recommendations.append('• Image quality meets clinical excellence standards')
        for rec in recommendations:
            report += rec + '\n'
        if not recommendations:
            report += '• All metrics within acceptable clinical ranges\n'
        report += f"\n{'=' * 80}\nEND OF REPORT\n{'=' * 80}"
        return report