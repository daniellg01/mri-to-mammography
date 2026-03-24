import numpy as np
import pyvista as pv
from src.physics.physics_config import NIST_PHYSICS

class PhysicsEngine:

    @staticmethod
    def get_mu(name):
        data = NIST_PHYSICS[name]
        return data['mu_rho'] * data['rho']

    @staticmethod
    def calculate_thickness_map_voxelized(mesh, spacing, physical_size):
        volume = PhysicsEngine.mesh_to_voxels(mesh, spacing, physical_size)
        voxels_count = np.sum(volume, axis=0)
        thickness_mm = voxels_count * spacing
        return thickness_mm

    @staticmethod
    def calculate_thickness_map_oblique(mesh, physical_size, res_x, res_y, ray_dir, custom_bounds=None):
        pts = mesh.points.astype(np.float64)
        if pts.shape[0] == 0:
            return np.zeros((res_y, res_x), dtype=np.float32)
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        up = np.array([0, 1, 0], dtype=np.float64)
        if abs(np.dot(up, ray_dir)) > 0.9:
            up = np.array([1, 0, 0], dtype=np.float64)
        u = np.cross(up, ray_dir)
        u /= np.linalg.norm(u)
        v = np.cross(ray_dir, u)
        x_d = pts @ u
        y_d = pts @ v
        s = pts @ ray_dir
        if custom_bounds is None:
            center_x = np.mean(x_d)
            center_y = np.mean(y_d)
            half_w = max(np.max(x_d) - center_x, center_x - np.min(x_d))
            half_h = max(np.max(y_d) - center_y, center_y - np.min(y_d))
            half_w *= 1.2
            half_h *= 1.2
            x_min = center_x - half_w
            x_max = center_x + half_w
            y_min = center_y - half_h
            y_max = center_y + half_h
            pw = x_max - x_min
            ph = y_max - y_min
            aspect_ratio = res_x / res_y
            if pw / ph > aspect_ratio:
                new_ph = pw / aspect_ratio
                y_min = center_y - new_ph / 2
                y_max = center_y + new_ph / 2
                ph = new_ph
            else:
                new_pw = ph * aspect_ratio
                x_min = center_x - new_pw / 2
                x_max = center_x + new_pw / 2
                pw = new_pw
        else:
            x_min, x_max, y_min, y_max = custom_bounds
            pw = x_max - x_min if x_max > x_min else 1.0
            ph = y_max - y_min if y_max > y_min else 1.0
        ix = ((x_d - x_min) / pw * (res_x - 1)).astype(np.int32)
        iy = ((y_d - y_min) / ph * (res_y - 1)).astype(np.int32)
        mask = (ix >= 0) & (ix < res_x) & (iy >= 0) & (iy < res_y)
        ix, iy, s = (ix[mask], iy[mask], s[mask])
        s_max = np.full((res_y, res_x), -1000000.0, dtype=np.float32)
        s_min = np.full((res_y, res_x), 1000000.0, dtype=np.float32)
        if len(s) > 0:
            np.maximum.at(s_max, (iy, ix), s)
            np.minimum.at(s_min, (iy, ix), s)
        thickness = s_max - s_min
        thickness[thickness < 0] = 0
        thickness[s_max == -1000000.0] = 0
        return thickness

    @staticmethod
    def calculate_thickness_map_fast(mesh, physical_size, res_x, res_y, custom_bounds=None):
        pts = mesh.points
        if custom_bounds is not None:
            x_min, x_max, y_min, y_max = custom_bounds
            pw = x_max - x_min if x_max > x_min else 1.0
            ph = y_max - y_min if y_max > y_min else 1.0
            ix = ((pts[:, 0] - x_min) / pw * (res_x - 1)).astype(int)
            iy = ((pts[:, 1] - y_min) / ph * (res_y - 1)).astype(int)
        else:
            pw, ph = physical_size
            ix = ((pts[:, 0] + pw / 2) / pw * (res_x - 1)).astype(int)
            iy = ((pts[:, 1] + ph / 2) / ph * (res_y - 1)).astype(int)
        iz = pts[:, 2]
        mask = (ix >= 0) & (ix < res_x) & (iy >= 0) & (iy < res_y)
        ix, iy, iz = (ix[mask], iy[mask], iz[mask])
        z_max_map = np.full((res_y, res_x), -1000000.0, dtype=np.float32)
        z_min_map = np.full((res_y, res_x), 1000000.0, dtype=np.float32)
        if len(iz) > 0:
            np.maximum.at(z_max_map, (iy, ix), iz)
            np.minimum.at(z_min_map, (iy, ix), iz)
        thickness = z_max_map - z_min_map
        thickness[thickness < 0] = 0
        thickness[z_max_map == -1000000.0] = 0
        return thickness

    @staticmethod
    def mesh_to_voxels(mesh, spacing, physical_size):
        pw, ph = physical_size
        bounds = mesh.bounds
        z_min, z_max = (bounds[4], bounds[5])
        depth = z_max - z_min
        nx = int(pw / spacing)
        ny = int(ph / spacing)
        nz = int(depth / spacing) if depth > 0 else 1
        grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(spacing, spacing, spacing), origin=(-pw / 2, -ph / 2, z_min))
        voxels = grid.select_enclosed_points(mesh, tolerance=0.0, check_surface=False)
        inside = voxels.point_data['SelectedPoints'].view(np.uint8)
        return inside.reshape((nz, ny, nx), order='F')

    @staticmethod
    def get_compressible_bounds(meshes, chest_y_limit):
        protruding_pts = []
        all_pts = []
        for mesh in meshes:
            if mesh is None:
                continue
            pts = mesh.points
            all_pts.append(pts)
            mask = pts[:, 1] > chest_y_limit
            if np.any(mask):
                protruding_pts.append(pts[mask])
        if not protruding_pts:
            if not all_pts:
                return (0, 50)
            combined_all = np.vstack(all_pts)
            return (np.min(combined_all[:, 2]), np.max(combined_all[:, 2]))
        combined = np.vstack(protruding_pts)
        return (np.min(combined[:, 2]), np.max(combined[:, 2]))

    @staticmethod
    def apply_compression(name, mesh, force, elasticity, z_min, z_max, global_center_xy=None):
        deformed = mesh.copy()
        if NIST_PHYSICS[name].get('is_rigid', False):
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

    @staticmethod
    def apply_compression_rotated(name, mesh, force, elasticity, compression_axis, plate_min, plate_max, global_center=None):
        deformed = mesh.copy()
        if NIST_PHYSICS[name].get('is_rigid', False):
            return (deformed, plate_max)
        pts = deformed.points
        compression_axis = compression_axis / np.linalg.norm(compression_axis)
        projections = pts @ compression_axis
        orig_h = plate_max - plate_min
        if orig_h <= 0:
            return (deformed, plate_max)
        deformation = force / elasticity
        final_plate_max = max(plate_min + 15, plate_max - deformation)
        scale = (final_plate_max - plate_min) / max(orig_h, 1)
        new_projections = plate_min + (projections - plate_min) * scale
        new_projections = np.maximum(plate_min, new_projections)
        delta = new_projections - projections
        pts += np.outer(delta, compression_axis)
        lat_exp = 1.0 / np.sqrt(max(scale, 0.01))
        if global_center is not None:
            center = np.array(global_center)
        else:
            center = np.mean(pts, axis=0)
        to_pts = pts - center
        along_axis = np.outer(to_pts @ compression_axis, compression_axis)
        perp = to_pts - along_axis
        pts[:] = center + along_axis + perp * lat_exp
        return (deformed, final_plate_max)

    @staticmethod
    def get_compressible_bounds_rotated(meshes, compression_axis, chest_plane_point, chest_plane_normal):
        compression_axis = compression_axis / np.linalg.norm(compression_axis)
        chest_plane_normal = chest_plane_normal / np.linalg.norm(chest_plane_normal)
        all_projections = []
        for mesh in meshes:
            if mesh is None or mesh.n_points == 0:
                continue
            pts = mesh.points
            to_pts = pts - chest_plane_point
            distances = to_pts @ chest_plane_normal
            mask = distances > 0
            if np.any(mask):
                filtered_pts = pts[mask]
                projections = filtered_pts @ compression_axis
                all_projections.extend(projections)
        if not all_projections:
            for mesh in meshes:
                if mesh is None or mesh.n_points == 0:
                    continue
                projections = mesh.points @ compression_axis
                all_projections.extend(projections)
        if not all_projections:
            return (0, 50)
        return (np.min(all_projections), np.max(all_projections))
    GRAVITY_DEFLECTION = {'Fat': 25.0, 'FGT': 15.0, 'Skin': 20.0, 'Muscle': 0.0, 'Vessels': 12.0, 'Ligaments': 2.0, 'Lymphatics': 10.0, 'Nipple': 22.0, 'Bone': 0.0}

    @staticmethod
    def apply_gravity_deformation(name, mesh, gravity_direction, anchor_point, anchor_axis=1, gravity_strength=1.0):
        from src.physics.physics_config import NIST_PHYSICS
        deformed = mesh.copy()
        delta_base = PhysicsEngine.GRAVITY_DEFLECTION.get(name, 10.0)
        if delta_base <= 0 or NIST_PHYSICS.get(name, {}).get('is_rigid', False):
            return deformed
        pts = deformed.points
        if len(pts) == 0:
            return deformed
        g_dir = np.array(gravity_direction, dtype=np.float64)
        g_norm = np.linalg.norm(g_dir)
        if g_norm < 1e-06:
            return deformed
        g_dir = g_dir / g_norm
        if np.isscalar(anchor_point):
            anchor_val = anchor_point
        else:
            anchor_val = anchor_point[anchor_axis]
        dist_from_anchor = pts[:, anchor_axis] - anchor_val
        dist_from_anchor = np.maximum(dist_from_anchor, 0)
        L_max = np.max(dist_from_anchor)
        if L_max < 5.0:
            return deformed
        s_norm = dist_from_anchor / L_max
        deflection_profile = s_norm ** 1.5
        delta_max = delta_base * gravity_strength
        displacement = np.outer(deflection_profile * delta_max, g_dir)
        pts += displacement
        return deformed

    @staticmethod
    def apply_gravity_to_assembly(meshes_dict, gravity_direction, anchor_axis, anchor_limit, gravity_strength=1.0):
        result = {}
        for name, mesh in meshes_dict.items():
            if mesh is None or mesh.n_points == 0:
                result[name] = mesh
                continue
            result[name] = PhysicsEngine.apply_gravity_deformation(name, mesh, gravity_direction, anchor_limit, anchor_axis, gravity_strength)
        return result