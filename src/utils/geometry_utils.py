import numpy as np

def transform_meshes_logic(tissues, is_right, z_offset, is_mlo=False, rotation_angle=45):
    transformed = {}
    offset = [0, 0, z_offset]
    for name, mesh in tissues.items():
        m = mesh.copy()
        m.points = m.points.astype(np.float32)
        if is_right:
            m = m.clip(normal='x', origin=(0, 0, 0), invert=False)
        else:
            m = m.clip(normal='x', origin=(0, 0, 0), invert=True)
        if is_mlo:
            pass
        m.translate(offset)
        transformed[name] = m
    return transformed