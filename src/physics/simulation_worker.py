import numpy as np
from PySide6.QtCore import QObject, Signal
from src.physics.physics_engine import PhysicsEngine
from src.utils.geometry_utils import transform_meshes_logic
import cv2

class SimulationWorker(QObject):
    finished = Signal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    progress = Signal(str)
    progress_int = Signal(int)
    error = Signal(str)

    def __init__(self, tissues, is_right, z_offset, chest_y, mlo_angle, chest_rot, mlo_rot_y=45, mlo_rot_x=25, gravity_strength=1.0):
        super().__init__()
        self.tissues = tissues
        self.is_right = is_right
        self.z_offset = z_offset
        self.chest_y = chest_y
        self.mlo_angle = mlo_angle
        self.chest_rot = chest_rot
        self.mlo_rot_y = mlo_rot_y
        self.mlo_rot_x = mlo_rot_x
        self.gravity_strength = gravity_strength

    def run(self):
        try:
            self.total_steps = len(self.tissues) * 2
            self.current_step = 0
            self.progress_int.emit(0)
            self.progress.emit(f'Iniciando Vista CC (0%)...')
            cc_disp, cc_raw = self.simulate_projection(is_mlo=False, view_name='CC')
            old_chest_rot = self.chest_rot
            old_z_offset = self.z_offset
            old_mlo_angle = self.mlo_angle
            old_chest_y = self.chest_y
            self.chest_rot = 45
            self.z_offset = 40
            self.mlo_angle = 145
            self.chest_y = self.chest_y - 60
            self.progress.emit(f'Iniciando Vista MLO (50%)...')
            mlo_disp, mlo_raw = self.simulate_projection(is_mlo=True, view_name='MLO')
            self.chest_rot = old_chest_rot
            self.z_offset = old_z_offset
            self.mlo_angle = old_mlo_angle
            self.chest_y = old_chest_y
            self.progress_int.emit(100)
            self.finished.emit(cc_disp, cc_raw, mlo_disp, mlo_raw)
        except Exception as e:
            self.error.emit(str(e))
            import traceback
            traceback.print_exc()

    def simulate_projection(self, is_mlo, view_name):
        meshes = transform_meshes_logic(self.tissues, self.is_right, self.z_offset, is_mlo=False, rotation_angle=0)
        if self.gravity_strength > 0:
            if is_mlo:
                mlo_rad = np.radians(self.mlo_rot_y)
                gravity_dir = np.array([0.0, np.sin(mlo_rad) * 0.3, -np.cos(mlo_rad)])
            else:
                gravity_dir = np.array([0.0, 0.0, -1.0])
            meshes = PhysicsEngine.apply_gravity_to_assembly(meshes, gravity_direction=gravity_dir, anchor_axis=1, anchor_limit=self.chest_y, gravity_strength=self.gravity_strength)
            print(f'   🌍 Gravedad aplicada ({view_name}): strength={self.gravity_strength:.2f}')
        voxel_spacing = 0.5
        phys_size = (240, 300)
        res_x, res_y = (1000, 1400)
        attenuation_map = np.zeros((res_y, res_x), dtype=np.float64)
        print(f'\n🔬 [DIAGNÓSTICO {view_name}]')
        print(f'   Attenuation map: min={np.min(attenuation_map):.3f}, max={np.max(attenuation_map):.3f}, mean={np.mean(attenuation_map):.3f}')
        print(f'   Porcentaje > 5: {np.sum(attenuation_map > 5) / attenuation_map.size * 100:.1f}%')
        print(f'   Porcentaje > 10: {np.sum(attenuation_map > 10) / attenuation_map.size * 100:.1f}%')
        if np.max(attenuation_map) > 3.0:
            print(f'   ⚠️  ¡¡¡ATENUACIÓN DEMASIADO ALTA!!! Escalando...')
            scale_factor = 1.5 / np.max(attenuation_map)
            attenuation_map = attenuation_map * scale_factor
            print(f'   Factor de escala: {scale_factor:.3f}')
            print(f'   Nuevo max: {np.max(attenuation_map):.3f}')
        y_limit = self.chest_y
        force_val = 180
        elasticity = 1.5
        steps = 3
        current_force = force_val / steps
        if is_mlo:
            rot_y = np.radians(self.mlo_rot_y)
            rot_x = np.radians(self.mlo_rot_x)
            base_axis = np.array([np.sin(rot_y), 0.0, np.cos(rot_y)])
            cos_x, sin_x = (np.cos(rot_x), np.sin(rot_x))
            rot_matrix_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
            compression_axis = rot_matrix_x @ base_axis
            compression_axis = compression_axis / np.linalg.norm(compression_axis)
            chest_plane_normal = np.array([0.0, 1.0, 0.0])
            chest_plane_point = np.array([0.0, y_limit, 0.0])
            plate_min, plate_max = PhysicsEngine.get_compressible_bounds_rotated(list(meshes.values()), compression_axis, chest_plane_point, chest_plane_normal)
            if plate_min >= plate_max:
                plate_min, plate_max = (-50, 50)
            self._compression_axis = compression_axis
            self._plate_min = plate_min
            self._plate_max = plate_max
            self._final_plate_max = plate_max
            self._chest_plane_normal = chest_plane_normal
            self._chest_plane_point = chest_plane_point
        else:
            z_min, z_max_init = PhysicsEngine.get_compressible_bounds(list(meshes.values()), y_limit)
            if z_min >= z_max_init:
                z_min, z_max_init = (-50, 50)
            final_top_z = z_max_init
        deformed_data = {name: m.copy() for name, m in meshes.items() if m.n_points > 0}
        all_tissue_pts = []
        for m in deformed_data.values():
            if m.n_points > 0:
                all_tissue_pts.append(m.points)
        if all_tissue_pts:
            combined_pts = np.vstack(all_tissue_pts)
            global_center_xy = [np.mean(combined_pts[:, 0]), np.mean(combined_pts[:, 1])]
            global_center_3d = np.mean(combined_pts, axis=0)
        else:
            global_center_xy = [0.0, 0.0]
            global_center_3d = np.array([0.0, 0.0, 0.0])
        for step in range(steps):
            current_f = force_val / steps * (step + 1)
            for name, m in deformed_data.items():
                if is_mlo:
                    deformed, _final_max = PhysicsEngine.apply_compression_rotated(name, m, current_f, elasticity, self._compression_axis, self._plate_min, self._plate_max, global_center=global_center_3d)
                    if step == steps - 1:
                        self._final_plate_max = _final_max
                else:
                    deformed, _final_z = PhysicsEngine.apply_compression(name, m, current_f, elasticity, z_min, z_max_init, global_center_xy=global_center_xy)
                    if step == steps - 1:
                        final_top_z = _final_z
                deformed_data[name] = deformed
        all_pts = []
        for name, m in list(deformed_data.items()):
            if is_mlo:
                m = m.clip(normal=self._chest_plane_normal.tolist(), origin=self._chest_plane_point.tolist(), invert=False)
                m = m.clip(normal=self._compression_axis.tolist(), origin=self._compression_axis * self._plate_min, invert=False)
                m = m.clip(normal=self._compression_axis.tolist(), origin=self._compression_axis * self._final_plate_max, invert=True)
            else:
                m = m.clip(normal=[0, 1, 0], origin=(0, y_limit, 0), invert=False)
                m = m.clip(normal='z', origin=(0, 0, z_min), invert=False)
                m = m.clip(normal='z', origin=(0, 0, final_top_z), invert=True)
            if m.n_points > 0:
                deformed_data[name] = m
                all_pts.append(m.points)
            else:
                del deformed_data[name]
        print(f'\n🔧 [FILTRADO PIEL - {view_name}]')
        if 'Skin' in deformed_data and deformed_data['Skin'].n_points > 0:
            skin_mesh = deformed_data['Skin']
            for name in list(deformed_data.keys()):
                if name == 'Skin':
                    continue
                m = deformed_data[name]
                if m.n_points == 0:
                    continue
                try:
                    result = m.compute_implicit_distance(skin_mesh)
                    distances = result.point_data['implicit_distance']
                    inside_mask = distances <= 1.0
                    if np.any(inside_mask):
                        m_filtered = m.extract_points(inside_mask, adjacent_cells=True)
                        if m_filtered.n_points > 0:
                            removed = m.n_points - m_filtered.n_points
                            if removed > 0:
                                print(f'   {name}: {m.n_points} -> {m_filtered.n_points} (quitados {removed} fuera)')
                            deformed_data[name] = m_filtered
                        else:
                            del deformed_data[name]
                    else:
                        del deformed_data[name]
                        print(f'   {name}: eliminado (todo fuera de piel)')
                except Exception as e:
                    print(f'   {name}: error ({e})')
        all_pts = []
        for m in deformed_data.values():
            if m.n_points > 0:
                all_pts.append(m.points)
        if all_pts:
            combined = np.vstack(all_pts)
            if is_mlo:
                comp_axis = self._compression_axis
                up = np.array([0.0, 1.0, 0.0])
                img_x = np.cross(up, comp_axis)
                if np.linalg.norm(img_x) < 0.001:
                    img_x = np.array([1.0, 0.0, 0.0])
                img_x = img_x / np.linalg.norm(img_x)
                img_y = np.cross(comp_axis, img_x)
                img_y = img_y / np.linalg.norm(img_y)
                proj_x = np.dot(combined, img_x)
                proj_y = np.dot(combined, img_y)
                center_x = np.mean(proj_x)
                center_y = np.mean(proj_y)
                half_w = max(np.max(proj_x) - center_x, center_x - np.min(proj_x))
                half_h = max(np.max(proj_y) - center_y, center_y - np.min(proj_y))
                x_min, x_max = (center_x - half_w, center_x + half_w)
                y_min, y_max = (center_y - half_h, center_y + half_h)
            else:
                x_min, x_max = (np.min(combined[:, 0]), np.max(combined[:, 0]))
                y_min, y_max = (np.min(combined[:, 1]), np.max(combined[:, 1]))
            w, h = (x_max - x_min, y_max - y_min)
            if w < 20:
                x_min -= (20 - w) / 2
                x_max += (20 - w) / 2
                w = 20
            if h < 20:
                y_min -= (20 - h) / 2
                y_max += (20 - h) / 2
                h = 20
            pad_x = w * 0.2
            pad_y = h * 0.2
            x_min, x_max = (x_min - pad_x, x_max + pad_x)
            y_min, y_max = (y_min - pad_y, y_max + pad_y)
            w, h = (x_max - x_min, y_max - y_min)
            aspect_ratio = res_x / res_y
            center_x_final = (x_min + x_max) / 2
            center_y_final = (y_min + y_max) / 2
            if w / h > aspect_ratio:
                new_h = w / aspect_ratio
                y_min = center_y_final - new_h / 2
                y_max = center_y_final + new_h / 2
            else:
                new_w = h * aspect_ratio
                x_min = center_x_final - new_w / 2
                x_max = center_x_final + new_w / 2
            custom_bounds = (x_min, x_max, y_min, y_max)
        else:
            custom_bounds = None
        print(f'\n📐 [PROYECCIÓN - {view_name}]')
        if is_mlo:
            ray_dir = -self._compression_axis
        else:
            ray_dir = np.array([0.0, 0.0, -1.0])
        for name, deformed in deformed_data.items():
            if deformed.n_points == 0:
                print(f'   [SKIP] {name}: mesh vacío')
                continue
            self.progress.emit(f'Proyectando {name} ({view_name})...')
            if is_mlo:
                L_i = PhysicsEngine.calculate_thickness_map_oblique(deformed, phys_size, res_x, res_y, ray_dir=ray_dir, custom_bounds=None)
            else:
                L_i = PhysicsEngine.calculate_thickness_map_fast(deformed, phys_size, res_x, res_y, custom_bounds=custom_bounds)
            print(f'   {name}: L_i max={np.max(L_i):.2f} mm')
            if np.max(L_i) > 0:
                mu = PhysicsEngine.get_mu(name)
                print(f'      mu={mu:.4f} cm⁻¹')
                contrib = mu * L_i / 10.0
                attenuation_map += contrib
                print(f'      contrib max={np.max(contrib):.3f}')
            self.current_step += 1
            if self.total_steps > 0:
                percentage = int(self.current_step / self.total_steps * 100)
                self.progress_int.emit(min(percentage, 100))
        print(f'\n🎯 Atenuation_map final antes de física:')
        print(f'   min={np.min(attenuation_map):.6f}, max={np.max(attenuation_map):.6f}')
        print(f'   mean={np.mean(attenuation_map):.6f}')
        self.progress.emit(f'Aplicando Física de Rayos X ({view_name})...')
        self.progress.emit(f'Aplicando Física de Rayos X ({view_name})...')
        print(f'\n⚛️ [FÍSICA CORREGIDA V3 - {view_name}]')
        flux_photons = 250000.0
        grid_factor = 0.8
        transmission = np.exp(-attenuation_map) * grid_factor
        I_trans = flux_photons * transmission
        noisy_signal = np.random.poisson(np.maximum(I_trans, 0.1)).astype(np.float64)
        I_0_ref = flux_photons * grid_factor
        raw_log = -np.log(np.maximum(noisy_signal, 1.0) / (I_0_ref + 1.0))
        pv_min = 1000
        pv_scale = 2800
        pv_float = pv_min + raw_log * pv_scale
        pv_float = 1000 + 55000 * (1 - np.exp(-(pv_float - 1000) / 15000))
        raw_16bit = np.clip(pv_float, 0, 65535).astype(np.uint16)
        print(f'   Post-procesamiento refinado...')
        img_8bit = (raw_16bit // 256).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_8bit)
        denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=70, sigmaSpace=70)
        final_16bit = (denoised.astype(np.uint32) * 256).astype(np.uint16)
        p05 = np.percentile(final_16bit, 0.5)
        p995 = np.percentile(final_16bit, 99.5)
        raw_16bit = np.clip((final_16bit - p05) * (58000 / (p995 - p05 + 1)) + 1000, 800, 65535).astype(np.uint16)
        print(f'   PV Final: min={np.min(raw_16bit)}, max={np.max(raw_16bit)}, mean={np.mean(raw_16bit):.0f}')

        def crop_to_breast_region(image, min_air_threshold=550, max_tissue_threshold=800, debug=False):
            h, w = image.shape
            print(f'\n[CROP] Imagen original: {w}x{h} px')
            p95 = np.percentile(image, 95)
            p05 = np.percentile(image, 5)
            rango_dinamico = p95 - p05
            threshold = p05 + rango_dinamico * 0.2
            mask = image > threshold
            print(f'[CROP] Rango dinámico: {p05:.0f} - {p95:.0f}, umbral: {threshold:.0f}')
            min_pixels_row = max(1, min(5, int(w * 0.005)))
            min_pixels_col = max(1, min(5, int(h * 0.005)))
            print(f'[CROP] Mínimo píxeles: fila>{min_pixels_row}, col>{min_pixels_col}')
            rows = np.sum(mask, axis=1) > min_pixels_row
            cols = np.sum(mask, axis=0) > min_pixels_col
            if not np.any(rows) or not np.any(cols):
                print('[CROP] ⚠️ Primer intento falló, usando umbral más bajo...')
                threshold_fallback = p05 + rango_dinamico * 0.1
                mask = image > threshold_fallback
                rows = np.sum(mask, axis=1) > 0
                cols = np.sum(mask, axis=0) > 0
                if not np.any(rows) or not np.any(cols):
                    print('[CROP] ⚠️ No se detectó tejido. Devolviendo imagen original.')
                    return (image, (0, w, 0, h))
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            y_min, y_max = (y_indices[0], y_indices[-1])
            x_min, x_max = (x_indices[0], x_indices[-1])
            detected_h = y_max - y_min
            detected_w = x_max - x_min
            print(f'[CROP] Región detectada: {detected_w}x{detected_h} px')
            margin_x = max(3, min(10, int(detected_w * 0.03)))
            margin_y = max(3, min(10, int(detected_h * 0.03)))
            print(f'[CROP] Márgenes adaptativos: x={margin_x}px, y={margin_y}px')
            y_min = max(0, y_min - margin_y)
            y_max = min(h, y_max + margin_y + 1)
            x_min = max(0, x_min - margin_x)
            x_max = min(w, x_max + margin_x + 1)
            crop_h = y_max - y_min
            crop_w = x_max - x_min
            min_size_h = max(20, int(h * 0.05))
            min_size_w = max(20, int(w * 0.05))
            if crop_h < min_size_h or crop_w < min_size_w:
                print(f'[CROP] ⚠️ Crop muy pequeño ({crop_w}x{crop_h}), expandiendo...')
                if crop_h < min_size_h:
                    expand_y = (min_size_h - crop_h) // 2
                    y_min = max(0, y_min - expand_y)
                    y_max = min(h, y_max + expand_y)
                if crop_w < min_size_w:
                    expand_x = (min_size_w - crop_w) // 2
                    x_min = max(0, x_min - expand_x)
                    x_max = min(w, x_max + expand_x)
            cropped = image[y_min:y_max, x_min:x_max]
            ratio = cropped.shape[1] / max(1, cropped.shape[0])
            print(f'[CROP] Resultado: {image.shape} → {cropped.shape} (ratio: {ratio:.2f})')
            print(f'[CROP] Bounding box: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]')
            tissue_pct = np.sum(cropped > 800) / max(1, cropped.size) * 100
            print(f'[CROP] Tejido en resultado: {tissue_pct:.1f}%')
            return (cropped, (x_min, x_max, y_min, y_max))
        raw_16bit_cropped, crop_coords = crop_to_breast_region(raw_16bit, min_air_threshold=550, max_tissue_threshold=800, debug=False)
        if is_mlo:
            raw_16bit_cropped = np.rot90(raw_16bit_cropped, k=1)
            h, w = raw_16bit_cropped.shape
            center = (w // 2, h // 2)
            mlo_rotation_angle = -45
            rotation_matrix = cv2.getRotationMatrix2D(center, mlo_rotation_angle, 1.0)
            cos_val = np.abs(rotation_matrix[0, 0])
            sin_val = np.abs(rotation_matrix[0, 1])
            new_w = int(h * sin_val + w * cos_val)
            new_h = int(h * cos_val + w * sin_val)
            rotation_matrix[0, 2] += new_w / 2 - center[0]
            rotation_matrix[1, 2] += new_h / 2 - center[1]
            bg_val = int(np.min(raw_16bit_cropped))
            raw_16bit_cropped = cv2.warpAffine(raw_16bit_cropped, rotation_matrix, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_val)
        disp_8bit = (raw_16bit_cropped // 256).astype(np.uint8)
        disp_rgb = np.stack((disp_8bit,) * 3, axis=-1)
        return (disp_rgb, raw_16bit_cropped)