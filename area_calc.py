from matplotlib.collections import LineCollection
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import RegularGridInterpolator
import scipy.fft as fft

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
def plot_all_parameter_sets(kz=0, resolution=512, plot_fermi_surface=True):
            for param_set in parameter_sets.keys():
                print(f"\n=== Plotting for parameter set: {param_set} ===")
                create_fermi_contours_for_set(param_set, kz=kz, resolution=resolution, plot_fermi_surface=plot_fermi_surface)

def create_fermi_contours_for_set(parameter_set, kz=0, resolution=512, plot_fermi_surface=True):
            set_parameters(parameter_set)
            extent=2.0
            kx_vals = np.linspace(-extent * np.pi/a, extent * np.pi/a, resolution)
            ky_vals = np.linspace(-extent * np.pi/b, extent * np.pi/b, resolution)
            energies = np.zeros((resolution, resolution, 4))
            for i, kx in enumerate(kx_vals):
                if (i + 1) % (resolution // 10) == 0:
                    print(f"  Progress: {(i+1)/resolution*100:.0f}%")
                for j, ky in enumerate(ky_vals):
                    H = H_full(kx, ky, kz)
                    evals, evecs = np.linalg.eigh(H)
                    sort_idx = np.argsort(np.real(evals))
                    energies[i, j, :] = np.real(evals[sort_idx])
            if plot_fermi_surface:
                fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
                band_colors = {2: "#9635E5", 3: "#FD0000"}
                band_labels = {2: "hole-like", 3: "electron-like"}
                from matplotlib.lines import Line2D
                legend_handles = []
                for band_idx in [2, 3]:
                    energy_data = energies[:, :, band_idx]
                    if energy_data.min() <= 0 <= energy_data.max():
                        contour = ax.contour(
                            ky_vals / (np.pi/b),
                            kx_vals / (np.pi/a),
                            energy_data,
                            levels=[0.0],
                            colors=[band_colors[band_idx]],
                            linewidths=2
                        )
                        legend_handles.append(Line2D([0], [0], color=band_colors[band_idx], lw=2, label=band_labels[band_idx]))
                        tol = 0.01
                        closed_pocket_count = 0
                        for i, seg in enumerate(contour.allsegs[0]):
                            v = seg
                            min_x, max_x = v[:,0].min(), v[:,0].max()
                            min_y, max_y = v[:,1].min(), v[:,1].max()
                            touches_left = np.any(np.isclose(v[:,0], -extent, atol=tol))
                            touches_right = np.any(np.isclose(v[:,0], extent, atol=tol))
                            touches_bottom = np.any(np.isclose(v[:,1], -extent, atol=tol))
                            touches_top = np.any(np.isclose(v[:,1], extent, atol=tol))
                            print(f"DEBUG: Path {i}: min/max x: {min_x:.3f}/{max_x:.3f}, y: {min_y:.3f}/{max_y:.3f}")
                            print(f"DEBUG: Path {i}: touches_left={touches_left}, touches_right={touches_right}, touches_bottom={touches_bottom}, touches_top={touches_top}")
                            if touches_left or touches_right or touches_bottom or touches_top:
                                print(f"DEBUG: Path {i}: Skipping open contour touching boundary.")
                                continue
                            area = 0.5 * np.abs(np.dot(v[:,0], np.roll(v[:,1], 1)) - np.dot(v[:,1], np.roll(v[:,0], 1)))
                            area *= (np.pi/a) * (np.pi/b)
                            centroid = v.mean(axis=0)
                            ax.text(centroid[0], centroid[1], f"{area:.3f}", color=band_colors[band_idx], fontsize=12, ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
                            hbar = 1.054571817e-34
                            e = 1.602176634e-19
                            area_m2 = area * 1e18
                            freq_T = hbar/(2*np.pi*e) * area_m2
                            print(f"Band {band_labels[band_idx]} pocket area: {area:.4f} nm^-2, frequency: {freq_T:.2f} T")
                            closed_pocket_count += 1
                        print(f"SUMMARY: Band {band_labels[band_idx]} closed pockets found: {closed_pocket_count}")
                ax.set_xlim(-extent, extent)
                ax.set_ylim(-extent, extent)
                ax.set_aspect(b/a, adjustable='box')
                ax.set_xlabel('ky (π/b units)', fontsize=14)
                ax.set_ylabel('kx (π/a units)', fontsize=14)
                ax.set_title(f'UTe2 Fermi Surface Contours (kz = {kz:.3f}, Parameter Set: {parameter_set})', fontsize=16)
                ax.grid(True, alpha=0.3)
                ax.axhline(0, color='k', linestyle='--', alpha=0.5)
                ax.axvline(0, color='k', linestyle='--', alpha=0.5)
                if legend_handles:
                    ax.legend(handles=legend_handles, loc='upper right', fontsize=12)
                os.makedirs('outputs/area_calc', exist_ok=True)
                save_path = f'outputs/area_calc/ute2_fermi_contours_kz_{kz:.3f}_{parameter_set}.png'
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved Fermi contour plot to: {save_path}")
                print("="*60)
def create_fermi_contours(kz=0, resolution=512, plot_fermi_surface=True):
    """Create Fermi surface contours using UTe2_fixed.py method"""
    parameter_set = 'odd_parity_paper'  # Default, but can be set by caller
    set_parameters(parameter_set)

    extent=2.0
    # Create momentum grid matching the paper's convention
    kx_vals = np.linspace(-extent * np.pi/a, extent * np.pi/a, resolution)
    ky_vals = np.linspace(-extent * np.pi/b, extent * np.pi/b, resolution)
    
    # Compute band energies and 5f orbital weights
    energies = np.zeros((resolution, resolution, 4))
    
    for i, kx in enumerate(kx_vals):
        if (i + 1) % (resolution // 10) == 0:
            print(f"  Progress: {(i+1)/resolution*100:.0f}%")
        for j, ky in enumerate(ky_vals):
            H = H_full(kx, ky, kz)
            evals, evecs = np.linalg.eigh(H)  # Get both eigenvalues and eigenvectors
            
            # Sort by eigenvalue and rearrange eigenvectors accordingly
            sort_idx = np.argsort(np.real(evals))
            energies[i, j, :] = np.real(evals[sort_idx])
            

    
    if plot_fermi_surface:
        # Plot Fermi surface contours for both bands with different colors
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
        band_colors = {2: "#9635E5", 3: "#FD0000"}
        band_labels = {2: "hole-like", 3: "electron-like"}
        from matplotlib.lines import Line2D
        legend_handles = []
        for band_idx in [2, 3]:
            energy_data = energies[:, :, band_idx]
            if energy_data.min() <= 0 <= energy_data.max():
                contour = ax.contour(
                    ky_vals / (np.pi/b),
                    kx_vals / (np.pi/a),
                    energy_data,
                    levels=[0.0],
                    colors=[band_colors[band_idx]],
                    linewidths=2
                )
                legend_handles.append(Line2D([0], [0], color=band_colors[band_idx], lw=2, label=band_labels[band_idx]))
                # Calculate and annotate area for each pocket
                tol = 0.01  # tolerance for edge detection
                closed_pocket_count = 0
                for i, seg in enumerate(contour.allsegs[0]):
                    v = seg
                    min_x, max_x = v[:,0].min(), v[:,0].max()
                    min_y, max_y = v[:,1].min(), v[:,1].max()
                    touches_left = np.any(np.isclose(v[:,0], -extent, atol=tol))
                    touches_right = np.any(np.isclose(v[:,0], extent, atol=tol))
                    touches_bottom = np.any(np.isclose(v[:,1], -extent, atol=tol))
                    touches_top = np.any(np.isclose(v[:,1], extent, atol=tol))
                    print(f"DEBUG: Path {i}: min/max x: {min_x:.3f}/{max_x:.3f}, y: {min_y:.3f}/{max_y:.3f}")
                    print(f"DEBUG: Path {i}: touches_left={touches_left}, touches_right={touches_right}, touches_bottom={touches_bottom}, touches_top={touches_top}")
                    if touches_left or touches_right or touches_bottom or touches_top:
                        print(f"DEBUG: Path {i}: Skipping open contour touching boundary.")
                        continue
                    area = 0.5 * np.abs(np.dot(v[:,0], np.roll(v[:,1], 1)) - np.dot(v[:,1], np.roll(v[:,0], 1)))
                    area *= (np.pi/a) * (np.pi/b)  # Scale area to physical units
                    centroid = v.mean(axis=0)
                    ax.text(centroid[0], centroid[1], f"{area:.3f}", color=band_colors[band_idx], fontsize=12, ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
                    # Calculate frequency in Tesla
                    hbar = 1.054571817e-34  # J*s
                    e = 1.602176634e-19     # C
                    area_m2 = area * 1e18   # Convert nm^-2 to m^-2
                    freq_T = hbar/(2*np.pi*e) * area_m2  # Tesla
                    print(f"Band {band_labels[band_idx]} pocket area: {area:.4f} nm^-2, frequency: {freq_T:.2f} T")
                    closed_pocket_count += 1
                print(f"SUMMARY: Band {band_labels[band_idx]} closed pockets found: {closed_pocket_count}")
        # Style plot
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect(b/a, adjustable='box')
        ax.set_xlabel('ky (π/b units)', fontsize=14)
        ax.set_ylabel('kx (π/a units)', fontsize=14)
        ax.set_title(f'UTe2 Fermi Surface Contours (kz = {kz:.3f}, Parameter Set: {parameter_set})', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', fontsize=12)

        # Move these functions to top-level scope

        os.makedirs('outputs/area_calc', exist_ok=True)
        save_path = f'outputs/area_calc/ute2_fermi_contours_kz_{kz:.3f}_{parameter_set}.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved Fermi contour plot to: {save_path}")
        print("="*60)

if __name__ == "__main__":
    plot_all_parameter_sets(kz=0.0, resolution=512, plot_fermi_surface=True)