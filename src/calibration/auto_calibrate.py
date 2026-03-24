import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.physics.physics_config import NIST_PHYSICS

class AutoCalibrator:

    def __init__(self, target_ranges):
        self.targets = target_ranges
        self.mu_fat = NIST_PHYSICS['Fat']['mu_rho'] * NIST_PHYSICS['Fat']['rho'] / 10.0
        self.mu_fgt = NIST_PHYSICS['FGT']['mu_rho'] * NIST_PHYSICS['FGT']['rho'] / 10.0
        self.thickness_air = 0.0
        self.thickness_fat = 40.0
        self.thickness_fgt = 40.0
        self.flux_init = 15000.0
        self.a_init = 1100.0
        self.b_init = 500.0

    def simulate_tissue_pv(self, mu, thickness, flux, a, b, n_samples=1000):
        I0 = flux
        I_trans = I0 * np.exp(-mu * thickness)
        noisy_vals = np.random.poisson(I_trans, size=n_samples)
        noisy_vals = np.maximum(noisy_vals, 1.0)
        raw_log = -np.log(noisy_vals / I0)
        pv_vals = a * raw_log + b
        return {'mean': np.mean(pv_vals), 'std': np.std(pv_vals), 'min': np.min(pv_vals), 'max': np.max(pv_vals), 'snr': np.mean(pv_vals) / np.std(pv_vals) if np.std(pv_vals) > 0 else 0}

    def cost_function(self, params):
        a, b, flux = params
        air = self.simulate_tissue_pv(0, self.thickness_air, flux, a, b)
        fat = self.simulate_tissue_pv(self.mu_fat, self.thickness_fat, flux, a, b)
        fgt = self.simulate_tissue_pv(self.mu_fgt, self.thickness_fgt, flux, a, b)
        cost = 0.0
        air_target = (self.targets['air'][0] + self.targets['air'][1]) / 2
        cost += ((air['mean'] - air_target) / 100) ** 2
        if fat['mean'] < self.targets['fat'][0]:
            cost += ((self.targets['fat'][0] - fat['mean']) / 100) ** 2
        elif fat['mean'] > self.targets['fat'][1]:
            cost += ((fat['mean'] - self.targets['fat'][1]) / 100) ** 2
        if fgt['mean'] < self.targets['fgt'][0]:
            cost += ((self.targets['fgt'][0] - fgt['mean']) / 100) ** 2
        elif fgt['mean'] > self.targets['fgt'][1]:
            cost += ((fgt['mean'] - self.targets['fgt'][1]) / 100) ** 2
        if fat['snr'] < 30:
            cost += (30 - fat['snr']) ** 2
        if a < 800 or a > 2000:
            cost += (abs(a - 1400) / 100) ** 2
        return cost

    def calibrate(self):
        print('=' * 60)
        print('CALIBRACIÓN AUTOMÁTICA DE PARÁMETROS FÍSICOS')
        print('=' * 60)
        x0 = [self.a_init, self.b_init, self.flux_init]
        bounds = [(800, 2000), (200, 800), (5000, 50000)]
        result = minimize(self.cost_function, x0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 100, 'disp': True})
        a_opt, b_opt, flux_opt = result.x
        print('\n' + '=' * 60)
        print('RESULTADOS DE CALIBRACIÓN')
        print('=' * 60)
        print(f'\nParámetros óptimos encontrados:')
        print(f'  a (ganancia)    = {a_opt:.1f}')
        print(f'  b (offset/aire) = {b_opt:.1f}')
        print(f'  flux_photons    = {flux_opt:.0f}')
        print(f'\nValidación con tejidos:')
        tissues = [('Aire', 0, self.thickness_air, 'air'), ('Grasa', self.mu_fat, self.thickness_fat, 'fat'), ('FGT', self.mu_fgt, self.thickness_fgt, 'fgt')]
        for name, mu, thick, key in tissues:
            sim = self.simulate_tissue_pv(mu, thick, flux_opt, a_opt, b_opt, n_samples=5000)
            target = self.targets[key]
            in_range = target[0] <= sim['mean'] <= target[1]
            status = '✅' if in_range else '⚠️'
            print(f'\n{status} {name}:')
            print(f"    PV: {sim['mean']:.0f} ± {sim['std']:.1f} (target: {target[0]}-{target[1]})")
            print(f"    Rango: [{sim['min']:.0f}, {sim['max']:.0f}]")
            print(f"    SNR: {sim['snr']:.1f}")
        print(f'\nCosto final: {result.fun:.4f}')
        print('=' * 60)
        return {'a': a_opt, 'b': b_opt, 'flux': flux_opt, 'success': result.success}

def run_calibration():
    TARGET_RANGES = {'air': (450, 550), 'fat': (1500, 3500), 'fgt': (3000, 7000)}
    print('Objetivos de calibración:')
    for tissue, (min_val, max_val) in TARGET_RANGES.items():
        print(f'  {tissue}: {min_val} - {max_val} PV')
    calibrator = AutoCalibrator(TARGET_RANGES)
    results = calibrator.calibrate()
    if results['success']:
        print('\n' + '=' * 60)
        print('CÓDIGO PARA ACTUALIZAR TU SIMULACIÓN')
        print('=' * 60)
        print('\nEn tu función simulate_projection(), reemplaza:')
        print(f"1. flux_photons = {results['flux']:.0f}")
        print(f"2. b = {results['b']:.1f}    # Offset para aire")
        print(f"3. a = {results['a']:.1f}    # Ganancia")
        plot_validation(calibrator, results)
    return results

def plot_validation(calibrator, results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    n_samples = 10000
    tissues = [('Aire', 0, calibrator.thickness_air, 'air', 'skyblue'), ('Grasa', calibrator.mu_fat, calibrator.thickness_fat, 'fat', 'orange'), ('FGT', calibrator.mu_fgt, calibrator.thickness_fgt, 'fgt', 'red')]
    for idx, (name, mu, thick, key, color) in enumerate(tissues):
        ax = axes[idx]
        I0 = results['flux']
        I_trans = I0 * np.exp(-mu * thick)
        noisy_vals = np.random.poisson(I_trans, size=n_samples)
        noisy_vals = np.maximum(noisy_vals, 1.0)
        raw_log = -np.log(noisy_vals / I0)
        pv_vals = results['a'] * raw_log + results['b']
        ax.hist(pv_vals, bins=50, alpha=0.7, color=color, edgecolor='black')
        ax.axvline(calibrator.targets[key][0], color='red', linestyle='--', label='Límite inferior')
        ax.axvline(calibrator.targets[key][1], color='red', linestyle='--', label='Límite superior')
        mean_val = np.mean(pv_vals)
        std_val = np.std(pv_vals)
        ax.axvline(mean_val, color='blue', linewidth=2, label=f'Media: {mean_val:.0f}')
        ax.fill_betweenx([0, n_samples / 50], mean_val - std_val, mean_val + std_val, alpha=0.3, color='blue', label=f'±1σ: {std_val:.1f}')
        ax.set_xlabel('Pixel Value (PV)')
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'Distribución de PV: {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('calibration_results.png', dpi=150, bbox_inches='tight')
    print('\nGráfico guardado como: calibration_results.png')
    plt.show()
if __name__ == '__main__':
    results = run_calibration()