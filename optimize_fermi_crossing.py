#!/usr/bin/env python3
"""
Optimize tU and tch_U to achieve target Fermi surface crossing
===============================================================
Find parameter values so that the Fermi contour at ky=-2π/b 
crosses at kx = ±0.29π/a
"""

import numpy as np
from scipy.optimize import minimize
from UTe2_fixed import H_full, a, b, c, set_parameters
import matplotlib.pyplot as plt

# Global parameters that we'll modify
current_params = {}

def H_full_with_params(kx, ky, kz, tU, tch_U, muU=-0.4):
    """
    Modified H_full that uses custom tU, tch_U, muU values.
    
    This is a simplified version that modifies only the U orbital block.
    """
    # Start with default parameters
    set_parameters('odd_parity_paper')
    
    # Get the standard Hamiltonian structure but we'll override specific terms
    # We need to rebuild the U block with new parameters
    
    # Band parameters for U orbitals (modified)
    DeltaU = 0.38  # Keep fixed
    tpU = 0.08     # Keep fixed
    tpch_U = 0.01  # Keep fixed
    tz_U = -0.0375 # Keep fixed
    
    # Te orbital parameters (keep default)
    muTe = -2.25
    DeltaTe = -1.4
    tTe = -1.5
    tch_Te = 0
    tz_Te = -0.05
    
    # Hybridization
    delta = 0.13
    
    # Construct Hamiltonian
    H = np.zeros((4, 4), dtype=complex)
    
    # U orbital block (2x2, upper left)
    # Diagonal terms
    H[0, 0] = muU - 2*tU*np.cos(kx*a) - 2*tpU*np.cos(2*kx*a) - 2*tch_U*np.cos(ky*b) - 2*tpch_U*np.cos(2*ky*b) - 2*tz_U*np.cos(kz*c/2)
    H[1, 1] = muU - 2*tU*np.cos(kx*a) - 2*tpU*np.cos(2*kx*a) - 2*tch_U*np.cos(ky*b) - 2*tpch_U*np.cos(2*ky*b) - 2*tz_U*np.cos(kz*c/2)
    
    # Off-diagonal in U block
    H[0, 1] = DeltaU
    H[1, 0] = DeltaU
    
    # Te orbital block (2x2, lower right)
    H[2, 2] = muTe - 2*tTe*np.cos(ky*b) - 2*tch_Te*np.cos(kx*a) - 2*tz_Te*np.cos(kz*c/2)
    H[3, 3] = muTe - 2*tTe*np.cos(ky*b) - 2*tch_Te*np.cos(kx*a) - 2*tz_Te*np.cos(kz*c/2)
    
    # Off-diagonal in Te block
    H[2, 3] = DeltaTe
    H[3, 2] = DeltaTe
    
    # Hybridization (U-Te coupling)
    H[0, 2] = delta
    H[2, 0] = delta
    H[1, 3] = delta
    H[3, 1] = delta
    
    return H


def find_fermi_crossing_at_ky(tU, tch_U, muU, target_ky_pi, kz=0.0, resolution=200):
    """
    Find where the Fermi surface crosses at a specific ky value.
    
    Parameters:
    - tU, tch_U, muU: parameter values to test
    - target_ky_pi: target ky in units of π/b
    - kz: out-of-plane momentum
    - resolution: number of kx points to sample
    
    Returns:
    - List of kx crossing values in units of π/a
    """
    # Convert to absolute units
    ky = target_ky_pi * np.pi / b
    
    # Create kx range
    kx_vals = np.linspace(-np.pi/a, np.pi/a, resolution)
    
    # Find crossings for each band
    crossings = []
    
    for band_idx in range(4):  # 4 bands
        energies = []
        
        for kx in kx_vals:
            H = H_full_with_params(kx, ky, kz, tU, tch_U, muU)
            eigenvals = np.linalg.eigvalsh(H)
            energies.append(eigenvals[band_idx])
        
        energies = np.array(energies)
        
        # Find zero crossings
        sign_changes = np.diff(np.sign(energies))
        crossing_indices = np.where(sign_changes != 0)[0]
        
        for idx in crossing_indices:
            # Linear interpolation for more accurate crossing
            kx1, kx2 = kx_vals[idx], kx_vals[idx + 1]
            E1, E2 = energies[idx], energies[idx + 1]
            
            if E2 != E1:
                kx_cross = kx1 - E1 * (kx2 - kx1) / (E2 - E1)
                # Convert to π/a units
                kx_cross_pi = kx_cross / (np.pi / a)
                crossings.append(kx_cross_pi)
    
    return sorted(crossings)


def objective_function(params, target_kx_pi, target_ky_pi, muU):
    """
    Objective function to minimize: difference between actual and target crossing.
    
    Parameters:
    - params: [tU, tch_U]
    - target_kx_pi: target kx crossing in π/a units
    - target_ky_pi: ky value in π/b units
    - muU: chemical potential (fixed)
    
    Returns:
    - Error (squared difference from target)
    """
    tU, tch_U = params
    
    # Get crossings at this ky
    crossings = find_fermi_crossing_at_ky(tU, tch_U, muU, target_ky_pi)
    
    if len(crossings) == 0:
        # No crossing found - bad parameters
        return 1e6
    
    # Find the crossing closest to our target (in absolute value)
    # We want crossings at ±target_kx_pi
    abs_crossings = [abs(c) for c in crossings]
    
    # Find crossing closest to target
    closest_idx = np.argmin([abs(c - target_kx_pi) for c in abs_crossings])
    closest_crossing = abs_crossings[closest_idx]
    
    # Error is squared difference
    error = (closest_crossing - target_kx_pi)**2
    
    print(f"  tU={tU:.4f}, tch_U={tch_U:.5f} -> crossings at kx = {crossings} π/a, error={error:.6f}")
    
    return error


def optimize_parameters(target_kx_pi=0.29, target_ky_pi=-2.0, muU=-0.4, 
                       initial_tU=0.17, initial_tch_U=0.02):
    """
    Optimize tU and tch_U to achieve target Fermi crossing.
    
    Parameters:
    - target_kx_pi: target kx crossing in π/a units
    - target_ky_pi: ky slice in π/b units
    - muU: chemical potential (fixed)
    - initial_tU, initial_tch_U: starting values
    
    Returns:
    - Optimized (tU, tch_U) values
    """
    print("="*70)
    print("Optimizing tU and tch_U for Fermi surface crossing")
    print("="*70)
    print(f"Target: Fermi crossing at kx = ±{target_kx_pi} π/a")
    print(f"        at ky = {target_ky_pi} π/b")
    print(f"Fixed:  muU = {muU}")
    print(f"Initial: tU = {initial_tU}, tch_U = {initial_tch_U}")
    print()
    
    # Check current crossing first
    print("Checking initial parameters...")
    initial_crossings = find_fermi_crossing_at_ky(initial_tU, initial_tch_U, muU, target_ky_pi)
    print(f"  Initial crossings: {initial_crossings}")
    print()
    
    # Initial parameters
    x0 = [initial_tU, initial_tch_U]
    
    # Bounds for parameters (wider range for better search)
    bounds = [
        (0.01, 0.50),   # tU bounds (wider range)
        (0.0001, 0.15)  # tch_U bounds (wider range)
    ]
    
    print("Starting optimization...")
    print()
    
    # Optimize
    result = minimize(
        objective_function,
        x0,
        args=(target_kx_pi, target_ky_pi, muU),
        method='Nelder-Mead',
        options={'maxiter': 200, 'disp': True, 'xatol': 1e-5, 'fatol': 1e-6}
    )
    
    tU_opt, tch_U_opt = result.x
    
    print()
    print("="*70)
    print("Optimization complete!")
    print("="*70)
    print(f"Optimized parameters:")
    print(f"  tU = {tU_opt:.6f}")
    print(f"  tch_U = {tch_U_opt:.6f}")
    print(f"  muU = {muU} (fixed)")
    print()
    
    # Verify the result
    print("Verifying result...")
    crossings = find_fermi_crossing_at_ky(tU_opt, tch_U_opt, muU, target_ky_pi)
    print(f"  Fermi crossings at ky = {target_ky_pi} π/b:")
    for c in crossings:
        print(f"    kx = {c:+.4f} π/a")
    print()
    
    return tU_opt, tch_U_opt


def plot_dispersion_comparison(tU_initial, tch_U_initial, tU_opt, tch_U_opt, 
                               muU, target_ky_pi=-2.0, target_kx_pi=0.29):
    """
    Plot band structure comparison before and after optimization.
    """
    print("Creating comparison plot...")
    
    kx_vals = np.linspace(-1.2, 1.2, 400)  # In π/a units
    ky = target_ky_pi * np.pi / b
    kz = 0.0
    
    # Calculate bands for both parameter sets
    energies_initial = np.zeros((len(kx_vals), 4))
    energies_opt = np.zeros((len(kx_vals), 4))
    
    for i, kx_pi in enumerate(kx_vals):
        kx = kx_pi * np.pi / a
        
        H_initial = H_full_with_params(kx, ky, kz, tU_initial, tch_U_initial, muU)
        H_opt = H_full_with_params(kx, ky, kz, tU_opt, tch_U_opt, muU)
        
        energies_initial[i, :] = np.linalg.eigvalsh(H_initial)
        energies_opt[i, :] = np.linalg.eigvalsh(H_opt)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    
    # Plot initial parameters
    for band in range(4):
        ax1.plot(kx_vals, energies_initial[:, band], linewidth=2, 
                label=f'Band {band+1}')
    ax1.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(target_kx_pi, color='r', linestyle=':', linewidth=2, alpha=0.7, 
               label=f'Target kx = {target_kx_pi} π/a')
    ax1.axvline(-target_kx_pi, color='r', linestyle=':', linewidth=2, alpha=0.7)
    ax1.set_xlabel('kx (π/a)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Energy (eV)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Initial: tU={tU_initial:.4f}, tch_U={tch_U_initial:.5f}\n' + 
                 f'at ky={target_ky_pi}π/b, muU={muU}', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim(-0.15, 0.15)
    
    # Plot optimized parameters
    for band in range(4):
        ax2.plot(kx_vals, energies_opt[:, band], linewidth=2, 
                label=f'Band {band+1}')
    ax2.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(target_kx_pi, color='r', linestyle=':', linewidth=2, alpha=0.7,
               label=f'Target kx = {target_kx_pi} π/a')
    ax2.axvline(-target_kx_pi, color='r', linestyle=':', linewidth=2, alpha=0.7)
    ax2.set_xlabel('kx (π/a)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Energy (eV)', fontsize=14, fontweight='bold')
    ax2.set_title(f'Optimized: tU={tU_opt:.6f}, tch_U={tch_U_opt:.6f}\n' + 
                 f'at ky={target_ky_pi}π/b, muU={muU}', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim(-0.15, 0.15)
    
    plt.tight_layout()
    
    import os
    os.makedirs('outputs/week16', exist_ok=True)
    save_path = 'outputs/week16/parameter_optimization_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to: {save_path}")
    
    plt.show()


def main():
    """Main optimization routine."""
    
    # Target and initial parameters
    target_kx_pi = 0.29   # Target crossing at ±0.29 π/a
    target_ky_pi = -2.0   # At ky = -2π/b
    muU = -0.4            # Fixed chemical potential
    initial_tU = 0.17     # Starting tU
    initial_tch_U = 0.02  # Starting tch_U
    
    # Run optimization
    tU_opt, tch_U_opt = optimize_parameters(
        target_kx_pi=target_kx_pi,
        target_ky_pi=target_ky_pi,
        muU=muU,
        initial_tU=initial_tU,
        initial_tch_U=initial_tch_U
    )
    
    # Plot comparison
    plot_dispersion_comparison(
        initial_tU, initial_tch_U,
        tU_opt, tch_U_opt,
        muU, target_ky_pi, target_kx_pi
    )
    
    print("="*70)
    print("Done! Use these optimized parameters in your parameter set:")
    print(f"  muU = {muU}")
    print(f"  tU = {tU_opt:.6f}")
    print(f"  tch_U = {tch_U_opt:.6f}")
    print("="*70)


if __name__ == "__main__":
    main()
