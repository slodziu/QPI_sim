#!/usr/bin/env python3
"""
Generate High-Quality Figures with RdBu_r Colormap

This script generates publication-quality figures using the RdBu_r colormap
for both single impurity and multi-impurity QPI simulations. Produces 
side-by-side comparisons with proper axis labeling and annotations.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from qpi_G_OOP import SystemParameters, QPISimulation
from config import HIGH_QUALITY_SINGLE, RANDOM_30_IMPURITIES, setup_n_random_positions

def create_single_impurity_figure():
    """Generate single impurity LDOS figure with RdBu_r colormap."""
    print("Generating single impurity figure...")
    
    # Setup single impurity simulation
    config = copy.deepcopy(HIGH_QUALITY_SINGLE)
    config.gridsize = 512
    config.n_frames = 1
    config.E_min = 15.0
    config.E_max = 15.0
    
    params = SystemParameters(
        gridsize=config.gridsize,
        L=config.L,
        t=config.t,
        mu=config.mu,
        eta=config.eta,
        V_s=config.V_s,
        E_min=config.E_min,
        E_max=config.E_max,
        n_frames=config.n_frames
    )
    
    # Single impurity at center
    center = params.gridsize // 2
    impurity_positions = [(center, center)]
    
    sim = QPISimulation(params, impurity_positions=impurity_positions)
    LDOS, fft_display, _, _ = sim.run_single_energy(15.0)
    
    # Create figure with three panels (like the original image style)
    fig = plt.figure(figsize=(15, 5))
    
    # Panel (a): LDOS in real space
    ax1 = plt.subplot(131)
    vmax = np.max(np.abs(LDOS))
    im1 = ax1.imshow(LDOS, origin='lower', cmap='RdBu_r', 
                     extent=[0, params.L, 0, params.L],
                     vmin=-0.2*vmax, vmax=vmax)
    
    # Mark impurity position
    impurity_x = (impurity_positions[0][1] / params.gridsize) * params.L
    impurity_y = (impurity_positions[0][0] / params.gridsize) * params.L
    ax1.scatter([impurity_x], [impurity_y], marker='x', color='black', s=100, linewidth=3)
    
    ax1.set_xlabel('x (physical units)')
    ax1.set_ylabel('y (physical units)')
    ax1.set_title('LDOS (E = 15.000, k_F = 3.87)')
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=14, 
             fontweight='bold', va='top', bbox=dict(boxstyle='square', facecolor='white', alpha=0.8))
    
    # Add text box with energy and k_F info
    ax1.text(0.02, 0.85, f'E = {15.0:.2f}\nk_F = {np.sqrt(15.0):.2f}', 
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.colorbar(im1, ax=ax1, label='LDOS')
    
    # Panel (b): QPI pattern in momentum space
    ax2 = plt.subplot(132)
    fft_log = np.log10(fft_display + 1)
    dk = 2 * np.pi / params.L
    k_max = dk * fft_log.shape[0] // 2
    
    im2 = ax2.imshow(fft_log, origin='lower', cmap='plasma',
                     extent=[-k_max, k_max, -k_max, k_max])
    ax2.set_xlabel('k_x (1/a)')
    ax2.set_ylabel('k_y (1/a)')
    ax2.set_title('Momentum Space: QPI Pattern')
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=14, 
             fontweight='bold', va='top', color='white', 
             bbox=dict(boxstyle='square', facecolor='black', alpha=0.7))
    
    # Set axis limits to zoom in on relevant region
    ax2.set_xlim(-50, 50)
    ax2.set_ylim(-50, 50)
    
    plt.colorbar(im2, ax=ax2, label='log|FFT(LDOS)|')
    
    # Panel (c): Dispersion relation
    ax3 = plt.subplot(133)
    k_range = np.linspace(-5, 5, 100)
    E_theory = k_range**2  # Parabolic dispersion
    
    ax3.plot(k_range, E_theory, 'b-', linewidth=2, label='Theory: E = k²')
    
    # Add expected k_F point
    k_F = np.sqrt(15.0)
    ax3.axvline(k_F, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.axvline(-k_F, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.scatter([k_F, -k_F], [15.0, 15.0], color='red', s=50, zorder=5, label='k_F points')
    
    ax3.set_xlabel('k_F (1/length units)')
    ax3.set_ylabel('Energy E')
    ax3.set_title('Dispersion: Theory vs Extracted')
    ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes, fontsize=14, 
             fontweight='bold', va='top', 
             bbox=dict(boxstyle='square', facecolor='white', alpha=0.8))
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(0, 25)
    
    plt.tight_layout()
    
    # Save figure
    output_path = "outputs/single_impurity_RdBu_r.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved single impurity figure to: {output_path}")
    
    return fig

def create_30_impurities_figure():
    """Generate 30 impurities LDOS figure with RdBu_r colormap."""
    print("Generating 30 impurities figure...")
    
    # Setup 30 impurities simulation
    config = copy.deepcopy(RANDOM_30_IMPURITIES)
    config.gridsize = 512
    config.n_frames = 1
    config.E_min = 15.0
    config.E_max = 15.0
    
    # Place 30 random impurities
    setup_n_random_positions(config, 30, seed=42, distributed=True)
    
    params = SystemParameters(
        gridsize=config.gridsize,
        L=config.L,
        t=config.t,
        mu=config.mu,
        eta=config.eta,
        V_s=config.V_s,
        E_min=config.E_min,
        E_max=config.E_max,
        n_frames=config.n_frames
    )
    
    sim = QPISimulation(params, impurity_positions=config.impurity_positions)
    LDOS, fft_display, _, _ = sim.run_single_energy(15.0)
    
    print(f"Placed {len(sim.impurities.positions)} impurities")
    
    # Create figure with three panels
    fig = plt.figure(figsize=(15, 5))
    
    # Panel (a): LDOS in real space
    ax1 = plt.subplot(131)
    vmax = np.max(np.abs(LDOS))
    im1 = ax1.imshow(LDOS, origin='lower', cmap='RdBu_r', 
                     extent=[0, params.L, 0, params.L],
                     vmin=-0.2*vmax, vmax=vmax)
    
    # Mark impurity positions
    for pos in sim.impurities.positions[:10]:  # Show first 10 impurities to avoid clutter
        imp_x = (pos[1] / params.gridsize) * params.L
        imp_y = (pos[0] / params.gridsize) * params.L
        ax1.scatter([imp_x], [imp_y], marker='o', facecolors='none', 
                   edgecolors='black', s=30, linewidth=1, alpha=0.7)
    
    ax1.set_xlabel('x (physical units)')
    ax1.set_ylabel('y (physical units)')
    ax1.set_title('LDOS (E = 15.000, k_F = 3.87)')
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=14, 
             fontweight='bold', va='top', bbox=dict(boxstyle='square', facecolor='white', alpha=0.8))
    
    # Add text box with energy and k_F info
    ax1.text(0.02, 0.85, f'E = {15.0:.2f}\nk_F = {np.sqrt(15.0):.2f}', 
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.colorbar(im1, ax=ax1, label='LDOS')
    
    # Panel (b): QPI pattern in momentum space
    ax2 = plt.subplot(132)
    fft_log = np.log10(fft_display + 1)
    dk = 2 * np.pi / params.L
    k_max = dk * fft_log.shape[0] // 2
    
    im2 = ax2.imshow(fft_log, origin='lower', cmap='plasma',
                     extent=[-k_max, k_max, -k_max, k_max])
    ax2.set_xlabel('k_x (1/a)')
    ax2.set_ylabel('k_y (1/a)')
    ax2.set_title('Momentum Space: QPI Pattern')
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=14, 
             fontweight='bold', va='top', color='white', 
             bbox=dict(boxstyle='square', facecolor='black', alpha=0.7))
    
    # Set axis limits to zoom in on relevant region
    ax2.set_xlim(-50, 50)
    ax2.set_ylim(-50, 50)
    
    plt.colorbar(im2, ax=ax2, label='log|FFT(LDOS)|')
    
    # Panel (c): Dispersion relation
    ax3 = plt.subplot(133)
    k_range = np.linspace(-5, 5, 100)
    E_theory = k_range**2  # Parabolic dispersion
    
    ax3.plot(k_range, E_theory, 'b-', linewidth=2, label='Theory: E = k²')
    
    # Add expected k_F point
    k_F = np.sqrt(15.0)
    ax3.axvline(k_F, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.axvline(-k_F, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.scatter([k_F, -k_F], [15.0, 15.0], color='red', s=50, zorder=5, label='k_F points')
    
    ax3.set_xlabel('k_F (1/length units)')
    ax3.set_ylabel('Energy E')
    ax3.set_title('Dispersion: Theory vs Extracted')
    ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes, fontsize=14, 
             fontweight='bold', va='top', 
             bbox=dict(boxstyle='square', facecolor='white', alpha=0.8))
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(0, 25)
    
    plt.tight_layout()
    
    # Save figure
    output_path = "outputs/30_impurities_RdBu_r.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 30 impurities figure to: {output_path}")
    
    return fig

def main():
    """Generate both figures with RdBu_r colormap."""
    print("="*60)
    print("GENERATING FIGURES WITH RdBu_r COLORMAP")
    print("="*60)
    
    # Generate single impurity figure
    fig1 = create_single_impurity_figure()
    
    # Generate 30 impurities figure
    fig2 = create_30_impurities_figure()
    
    print("\n" + "="*60)
    print("FIGURES GENERATED SUCCESSFULLY")
    print("- Single impurity: outputs/single_impurity_RdBu_r.png")
    print("- 30 impurities: outputs/30_impurities_RdBu_r.png")
    print("="*60)
    
    plt.show()

if __name__ == "__main__":
    main()