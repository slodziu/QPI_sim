#!/usr/bin/env python3
"""
UTe2 Simple Fermi Contour Plotter
=================================
Minimal script for creating clean UTe2 Fermi surface contours
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Lattice constants (same for all parameter sets)
a, b, c = 0.41, 0.61, 1.39

# Parameter sets to test
parameter_sets = {
    'DFT': {
        'delta': 0.1,  # hybridization
        'muU': -0.35, 'DeltaU': 0.4, 'tU': 0.15, 'tpU': 0.08, 
        'tch_U': 0.01, 'tpch_U': 0, 'tz_U': -0.03,
        'muTe': -1.8, 'DeltaTe': -1.5, 'tTe': -1.5, 
        'tch_Te': 0, 'tz_Te': -0.05
    },
    'QuantumOscillation': {
        'delta': 0.1,  # using same hybridization as DFT 
        'muU': -0.17, 'DeltaU': 0.05, 'tU': 0.1, 'tpU': 0.08,
        'tch_U': 0.01, 'tpch_U': 0, 'tz_U': 0.04,
        'muTe': -1.8, 'DeltaTe': -1.5, 'tTe': -1.5,
        'tch_Te': -0.03, 'tz_Te': -0.5
    },
    'QPIFS': {
        'delta': 0.1,  # using same hybridization as DFT 
        'muU': -0.17, 'DeltaU': -0.05, 'tU': 0.1, 'tpU': 0.08,
        'tch_U': 0.01, 'tpch_U': 0, 'tz_U': 0.04,
        'muTe': -1.8, 'DeltaTe': -1.5, 'tTe': -1.5,
        'tch_Te': -0.03, 'tz_Te': -0.5
    },
    'odd_parity_paper': {
        'delta': 0.13,  
        'muU': -0.355, 'DeltaU': 0.38, 'tU': 0.17, 'tpU': 0.08,
        'tch_U': 0.015, 'tpch_U': 0.01, 'tz_U': -0.0375,
        'muTe': -2.25, 'DeltaTe': -1.4, 'tTe': -1.5,
        'tch_Te': 0, 'tz_Te': -0.05
    }
}

# Current parameter set (will be set by set_parameters function)
muU = DeltaU = tU = tpU = tch_U = tpch_U = tz_U = None
muTe = DeltaTe = tTe = tch_Te = tz_Te = delta = None

def set_parameters(param_set_name):
    """Set global parameters from the specified parameter set"""
    global muU, DeltaU, tU, tpU, tch_U, tpch_U, tz_U
    global muTe, DeltaTe, tTe, tch_Te, tz_Te, delta
    
    params = parameter_sets[param_set_name]
    muU = params['muU']
    DeltaU = params['DeltaU'] 
    tU = params['tU']
    tpU = params['tpU']
    tch_U = params['tch_U']
    tpch_U = params['tpch_U']
    tz_U = params['tz_U']
    muTe = params['muTe']
    DeltaTe = params['DeltaTe']
    tTe = params['tTe']
    tch_Te = params['tch_Te']
    tz_Te = params['tz_Te']
    delta = params['delta']
    
    print(f"Set parameters for: {param_set_name}")
    print(f"  U:  μ={muU:.2f}, Δ={DeltaU:.2f}, t={tU:.2f}, t'={tpU:.2f}, tch={tch_U:.3f}, tz={tz_U:.3f}")
    print(f"  Te: μ={muTe:.2f}, Δ={DeltaTe:.2f}, t={tTe:.2f}, tch={tch_Te:.3f}, tz={tz_Te:.2f}")
    print(f"  δ={delta:.2f}")
    print()

def H_full(kx, ky, kz):
    """Compute 4x4 UTe2 Hamiltonian"""
    # U block
    diag_U = muU - 2*tU*np.cos(kx*a) - 2*tch_U*np.cos(ky*b)
    off_U = -DeltaU - 2*tpU*np.cos(kx*a) - 2*tpch_U*np.cos(ky*b)
    complex_U = -4*tz_U*np.exp(-1j*kz*c/2)*np.cos(kx*a/2)*np.cos(ky*b/2)
    
    # Te block  
    diag_Te = muTe - 2*tch_Te*np.cos(kx*a) 
    off_Te = -DeltaTe - tTe*np.exp(-1j*ky*b) - 2*tz_Te*np.cos(kz*c/2)*np.cos(kx*a/2)*np.cos(ky*b/2)
    
    # Build 4x4 matrix
    H = np.zeros((4, 4), dtype=complex)
    H[0,0] = H[1,1] = diag_U
    H[0,1] = off_U + complex_U
    H[1,0] = off_U + np.conj(complex_U)
    H[2,2] = H[3,3] = diag_Te  
    H[2,3] = off_Te
    H[3,2] = np.conj(off_Te)
    H[0,2] = H[1,3] = H[2,0] = H[3,1] = delta
    return H

def create_fermi_contours(param_set_name, kz=0, resolution=300):
    """Create clean Fermi surface contours using UTe2_fixed.py method"""
    print(f"Computing Fermi surface at kz = {kz:.3f} for {param_set_name} parameters")
    
    # Set parameters for this run
    set_parameters(param_set_name)
    
    # Create momentum grid (extended ky range to capture full FS)
    kx_vals = np.linspace(-np.pi/a, np.pi/a, resolution)
    ky_vals = np.linspace(-3*np.pi/b, 3*np.pi/b, resolution) 
    
    # Compute band energies
    energies = np.zeros((resolution, resolution, 4))
    for i, kx in enumerate(kx_vals):
        if (i + 1) % (resolution // 10) == 0:
            print(f"  Progress: {(i+1)/resolution*100:.0f}%")
        for j, ky in enumerate(ky_vals):
            eigvals = np.linalg.eigvals(H_full(kx, ky, kz))
            energies[i, j, :] = np.sort(np.real(eigvals))
    
    # Plot Fermi contours
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    colors = ['red', 'blue']
    labels = ['Band 3', 'Band 4']
    
    contour_count = 0
    for band_idx, color, label in zip([2, 3], colors, labels):
        energy_data = energies[:, :, band_idx]
        print(f"  {label}: Energy range {energy_data.min():.4f} to {energy_data.max():.4f} eV")
        
        if energy_data.min() <= 0 <= energy_data.max():
            print(f"    ✓ {label} crosses Fermi level")
            # Extract contours at E=0 (Fermi level)
            temp_fig, temp_ax = plt.subplots()
            contour = temp_ax.contour(ky_vals, kx_vals, energy_data, levels=[0.0])
            plt.close(temp_fig)
            
            # Plot each contour segment
            first = True
            for path in contour.allsegs[0]:
                if len(path) > 0:
                    ky_plot = path[:, 0] / (np.pi/b)  # Convert to π/b units
                    kx_plot = path[:, 1] / (np.pi/a)  # Convert to π/a units
                    ax.plot(ky_plot, kx_plot, color=color, linewidth=2.5,
                           label=label if first else '', alpha=0.9)
                    first = False
                    contour_count += 1
        else:
            print(f"    ✗ {label} does not cross Fermi level")
    
    # Style the plot
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('ky (π/b units)', fontsize=14)
    ax.set_ylabel('kx (π/a units)', fontsize=14) 
    ax.set_title(f'UTe2 Fermi Surface Contours ({param_set_name})', fontsize=16)
    ax.grid(True, alpha=0.3)
    if contour_count > 0:
        ax.legend()
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    
    # Save and show
    os.makedirs('outputs/fermi_countours_clean', exist_ok=True)
    save_path = f'outputs/fermi_countours_clean/ute2_fermi_{param_set_name.lower()}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved to: {save_path}")
    print(f"Found {contour_count} contour segments\n")
    print("="*60)

if __name__ == "__main__":
    print("=" * 60)
    print("UTe2 Fermi Surface Comparison")
    print("Testing Multiple Parameter Sets")
    print("=" * 60)
    
    # Run all parameter sets
    for param_set in ['DFT', 'QuantumOscillation', 'QPIFS', 'odd_parity_paper']:
        print(f"\n{'='*20} {param_set} Parameters {'='*20}")
        create_fermi_contours(param_set, kz=0, resolution=512)
    
    print("\n" + "="*60)
    print("✓ All parameter sets completed!")
    print("Check outputs/ directory for all plots:")
    print("  - ute2_fermi_dft.png")
    print("  - ute2_fermi_quantumoscillation.png") 
    print("  - ute2_fermi_qpifs.png")
    print("  - ute2_fermi_odd_parity_paper.png")
    print("="*60)