import sys
import os
# Agregar el directorio raíz al path para importar módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PySide6.QtWidgets import QApplication
from src.ui.ui_simulation import MammographyProSim

def test_simulation():
    app = QApplication(sys.argv)
    sim = MammographyProSim()
    
    # Test 1: Generate Right MLO
    print("Testing Right MLO generation...")
    sim.btn_right.setChecked(True)
    img_right = sim.simulate_projection(is_mlo=True)
    
    if img_right is None:
        print("FAIL: Image is None")
        return
        
    print(f"Success: Image shape {img_right.shape}")
    print(f"Stats: Mean={np.mean(img_right)}, Std={np.std(img_right)}")
    
    # Check for noise (std deviation should be significant even in 'flat' areas, but simplest check is global std > 0)
    if np.std(img_right) < 1.0:
         print("WARNING: Image seems very flat, noise might not be working or image is empty.")
    
    # Test 2: Generate Left CC (Asymmetry check)
    print("\nTesting Left CC generation...")
    sim.btn_left.setChecked(True)
    img_left = sim.simulate_projection(is_mlo=False)
    
    # Basic check that it runs
    if img_left is not None:
        print("Success: Left CC generated.")
    
    print("Verification Script Completed.")

if __name__ == "__main__":
    test_simulation()
