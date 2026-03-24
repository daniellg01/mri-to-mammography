import os
import time
import numpy as np
import pyvista as pv
from PySide6.QtCore import QObject, Signal
NIST_PHYSICS = {'FGT': {'mu_rho': 0.751, 'rho': 1.04, 'color': 'yellow', 'opacity': 0.6, 'is_rigid': False}, 'Fat': {'mu_rho': 0.463, 'rho': 0.93, 'color': 'bisque', 'opacity': 0.15, 'is_rigid': False}, 'Muscle': {'mu_rho': 0.797, 'rho': 1.06, 'color': 'red', 'opacity': 0.8, 'is_rigid': True}, 'Skin': {'mu_rho': 0.756, 'rho': 1.1, 'color': 'pink', 'opacity': 0.2, 'is_rigid': False}, 'Vessels': {'mu_rho': 0.779, 'rho': 1.06, 'color': 'blue', 'opacity': 0.8, 'is_rigid': False}}

def transform_meshes_logic(tissues, is_right, z_offset, is_mlo=False):
    transformed = {}
    offset = [0, 0, z_offset]
    for name, mesh in tissues.items():
        m = mesh.copy()
        if is_right:
            m = m.clip(normal='x', origin=(0, 0, 0), invert=False)
        else:
            m = m.clip(normal='x', origin=(0, 0, 0), invert=True)
        m.translate(offset)
        if is_mlo:
            center = m.center
            m.rotate_z(45, point=(0, 0, 0))
            m.translate(center)
        transformed[name] = m
    return transformed

class PhysicsEngine:

    @staticmethod
    def get_mu(name):
        data = NIST_PHYSICS[name]
        return data['mu_rho'] * data['rho']

    @staticmethod
    def get_compressible_bounds(meshes, chest_y_limit):
        protruding_pts = []
        for mesh in meshes:
            pts = mesh.points
            mask = pts[:, 1] > chest_y_limit
            if np.any(mask):
                protruding_pts.append(pts[mask])
        if not protruding_pts:
            return (0, 50)
        combined = np.vstack(protruding_pts)
        return (np.min(combined[:, 2]), np.max(combined[:, 2]))

    @staticmethod
    def apply_compression(name, mesh, force, elasticity, z_min, z_max, global_center_xy=None):
        deformed = mesh.copy()
        if NIST_PHYSICS[name]['is_rigid']:
            return (deformed, z_max)
        pts = deformed.points
        orig_h = z_max - z_min
        deformation = force / elasticity
        final_top_z = max(z_min + 15, z_max - deformation)
        scale_z = (final_top_z - z_min) / max(orig_h, 1)
        pts[:, 2] = np.maximum(z_min, z_min + (pts[:, 2] - z_min) * scale_z)
        lat_exp = 1.0 / np.sqrt(max(scale_z, 0.01))
        for i in [0, 1]:
            if global_center_xy is not None:
                mid = global_center_xy[i]
            else:
                mid = np.mean(pts[:, i])
            pts[:, i] = mid + (pts[:, i] - mid) * lat_exp
        return (deformed, final_top_z)

class AssetLoaderWorker(QObject):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, base_path, patient_id):
        super().__init__()
        self.base_path = base_path
        self.pid = patient_id
        self.required_files = [f'{self.pid}_{name}.stl' for name in NIST_PHYSICS.keys()]

    def run(self):
        try:
            if not os.path.exists(self.base_path):
                os.makedirs(self.base_path)
            missing_files = [f for f in self.required_files if not os.path.exists(os.path.join(self.base_path, f))]
            if missing_files:
                self.status.emit(f'Warning: Missing {len(missing_files)} STL files.')
                time.sleep(1)
            else:
                self.status.emit('Verifying local assets...')
                time.sleep(0.5)
            self.progress.emit(100)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class SimulationWorker(QObject):
    finished = Signal(np.ndarray, np.ndarray)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, tissues, is_right, z_offset, chest_y):
        super().__init__()
        self.tissues = tissues
        self.is_right = is_right
        self.z_offset = z_offset
        self.chest_y = chest_y

    def run(self):
        try:
            self.progress.emit('Generating CC View (Ray-Tracing)...')
            cc_img = self.simulate_projection(is_mlo=False)
            self.progress.emit('Generating MLO View (Ray-Tracing)...')
            mlo_img = self.simulate_projection(is_mlo=True)
            self.finished.emit(cc_img, mlo_img)
        except Exception as e:
            self.error.emit(str(e))

    def simulate_projection(self, is_mlo):
        meshes = transform_meshes_logic(self.tissues, self.is_right, self.z_offset, is_mlo)
        y_limit = self.chest_y
        z_min, z_max_init = PhysicsEngine.get_compressible_bounds(list(meshes.values()), y_limit)
        img_plotter = pv.Plotter(off_screen=True, window_size=[1000, 1400])
        img_plotter.set_background('black')
        for name, m in meshes.items():
            if name == 'Muscle' and (not is_mlo):
                continue
            deformed, _ = PhysicsEngine.apply_compression(name, m, 130, 10, z_min, z_max_init)
            clip_box = [-200, 200, y_limit, y_limit + 400, -1000, 1000]
            deformed = deformed.clip_box(bounds=clip_box, invert=False)
            mu = PhysicsEngine.get_mu(name)
            vis_opacity = np.clip(mu * 0.15, 0.05, 1.0)
            img_plotter.add_mesh(deformed, color='white', opacity=vis_opacity, lighting=False)
        if is_mlo:
            img_plotter.camera_position = [(300, -200, 150), (0, 0, 50), (0, 0, 1)]
        else:
            img_plotter.view_xy()
        img_plotter.camera.parallel_projection = True
        img_plotter.reset_camera()
        img_plotter.camera.zoom(1.3)
        img_array = img_plotter.screenshot()
        img_plotter.close()
        noise = np.random.normal(0, 15, img_array.shape).astype(np.int16)
        return np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)