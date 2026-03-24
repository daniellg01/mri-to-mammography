"""
Script de prueba para verificar que las transformaciones MLO y CC funcionan correctamente.
"""

import sys
import os
# Agregar el directorio raíz al path para importar módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pyvista as pv
from src.utils.geometry_utils import transform_meshes_logic
from src.physics.physics_config import NIST_PHYSICS

def test_transformations():
    """
    Prueba las transformaciones de geometría para CC y MLO
    """
    print("=" * 80)
    print("INICIANDO PRUEBA DE TRANSFORMACIONES MLO vs CC")
    print("=" * 80)
    
    # Cargar datos
    path = "./export_models"
    if not os.path.exists(path):
        print(f"Error: Ruta no encontrada: {path}")
        return
    
    tissues = {}
    files = os.listdir(path)
    
    detected_pid = None
    for f in files:
        if f.endswith("_FGT.stl"):
            detected_pid = f.replace("_FGT.stl", "")
            break
    
    if not detected_pid:
        for f in files:
            if f.endswith("_Grasa.stl"):
                detected_pid = f.replace("_Grasa.stl", "")
                break
    
    print(f"\nID de Paciente Detectado: {detected_pid}")
    print(f"Cargando modelos desde: {path}\n")
    
    # Cargar meshes
    for name in NIST_PHYSICS.keys():
        filename = f"{detected_pid}_{name}.stl" if detected_pid else f"BreastDx-01-0002_{name}.stl"
        f_path = os.path.join(path, filename)
        if os.path.exists(f_path):
            try:
                m = pv.read(f_path)
                m.translate(-np.array(m.center))
                tissues[name] = m
                print(f"  ✓ Cargado: {name:20s} | Puntos: {m.n_points:7d} | Centro: ({m.center[0]:7.2f}, {m.center[1]:7.2f}, {m.center[2]:7.2f})")
            except Exception as e:
                print(f"  ✗ Error cargando {name}: {e}")
        else:
            print(f"  - No encontrado: {filename}")
    
    if not tissues:
        print("\nError: No se cargaron meshes. Verifica la ruta de modelos.")
        return
    
    # Parámetros de prueba
    is_right = True
    z_offset = 40
    
    print("\n" + "=" * 80)
    print(f"PRUEBA 1: Transformación CC (is_mlo=False)")
    print(f"  Parámetros: is_right={is_right}, z_offset={z_offset}")
    print("=" * 80 + "\n")
    
    meshes_cc = transform_meshes_logic(tissues, is_right, z_offset, is_mlo=False)
    
    print("\nResultados CC:")
    for name, m in meshes_cc.items():
        if m is not None and m.n_points > 0:
            print(f"  {name:20s} | Puntos: {m.n_points:7d} | Centro: ({m.center[0]:7.2f}, {m.center[1]:7.2f}, {m.center[2]:7.2f})")
            bounds = m.bounds
            print(f"    Límites: X=[{bounds[0]:7.2f}, {bounds[1]:7.2f}], Y=[{bounds[2]:7.2f}, {bounds[3]:7.2f}], Z=[{bounds[4]:7.2f}, {bounds[5]:7.2f}]")
        else:
            print(f"  {name:20s} | VACÍO DESPUÉS DE TRANSFORMACIÓN")
    
    print("\n" + "=" * 80)
    print(f"PRUEBA 2: Transformación MLO (is_mlo=True)")
    print(f"  Parámetros: is_right={is_right}, z_offset={z_offset}")
    print("=" * 80 + "\n")
    
    meshes_mlo = transform_meshes_logic(tissues, is_right, z_offset, is_mlo=True)
    
    print("\nResultados MLO:")
    for name, m in meshes_mlo.items():
        if m is not None and m.n_points > 0:
            print(f"  {name:20s} | Puntos: {m.n_points:7d} | Centro: ({m.center[0]:7.2f}, {m.center[1]:7.2f}, {m.center[2]:7.2f})")
            bounds = m.bounds
            print(f"    Límites: X=[{bounds[0]:7.2f}, {bounds[1]:7.2f}], Y=[{bounds[2]:7.2f}, {bounds[3]:7.2f}], Z=[{bounds[4]:7.2f}, {bounds[5]:7.2f}]")
        else:
            print(f"  {name:20s} | VACÍO DESPUÉS DE TRANSFORMACIÓN")
    
    # Comparar cambios
    print("\n" + "=" * 80)
    print("COMPARATIVA: Diferencias entre CC y MLO")
    print("=" * 80)
    
    for name in tissues.keys():
        m_cc = meshes_cc.get(name)
        m_mlo = meshes_mlo.get(name)
        
        if m_cc and m_mlo and m_cc.n_points > 0 and m_mlo.n_points > 0:
            print(f"\n{name}:")
            print(f"  CC center:  ({m_cc.center[0]:7.2f}, {m_cc.center[1]:7.2f}, {m_cc.center[2]:7.2f})")
            print(f"  MLO center: ({m_mlo.center[0]:7.2f}, {m_mlo.center[1]:7.2f}, {m_mlo.center[2]:7.2f})")
            
            center_diff = m_mlo.center - m_cc.center
            print(f"  Diferencia: ({center_diff[0]:7.2f}, {center_diff[1]:7.2f}, {center_diff[2]:7.2f})")
            
            # Calcular promedio de Z
            z_cc = np.mean(m_cc.points[:, 2])
            z_mlo = np.mean(m_mlo.points[:, 2])
            print(f"  Z promedio CC:  {z_cc:7.2f}")
            print(f"  Z promedio MLO: {z_mlo:7.2f}")
            print(f"  Diferencia Z:   {z_mlo - z_cc:7.2f}")
    
    print("\n" + "=" * 80)
    print("PRUEBA COMPLETADA")
    print("=" * 80)

if __name__ == "__main__":
    test_transformations()
