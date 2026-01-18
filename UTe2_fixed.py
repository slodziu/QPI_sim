
# UTe2 model with 3D Fermi surface visualization
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("trimesh not available - 3D export will be skipped")

# Lattice constants (nm) - UTe2 orthorhombic structure
# Full lattice parameters for gap function calculations
a = 0.41   
b = 0.61   
c = 1.39   

# (0-11) plane crystallographic parameters
# c* ≈ 0.76 nm is the projected lattice spacing in the (0-11) plane
# This is NOT the reciprocal lattice parameter, but the effective spacing
c_star = 0.76  # nm, projected lattice spacing in (0-11) plane


# U parameters (verified against paper)
muU = -0.355
DeltaU = 0.38
tU = 0.17
tpU = 0.08
tch_U = 0.015
tpch_U = 0.01
tz_U = -0.0375

# Te parameters (updated to match paper exactly)
muTe = -2.25
DeltaTe = -1.4
tTe = -1.5  # hopping along Te(2) chain in b direction
tch_Te = 0  # hopping between chains in a direction 
tz_Te = -0.05  # hopping between chains along c axis
delta = 0.13 #try 0.1 later

# momentum grid for kx-ky plane (in nm^-1, will be converted to π/a, π/b units for plotting)
nk = 201  # Reduced for faster computation
# Extend ky range to cover -3π/b to 3π/b
kx_vals = np.linspace(-np.pi/a, np.pi/a, nk)
ky_vals = np.linspace(-3*np.pi/b, 3*np.pi/b, nk)  # Extended range
KX, KY = np.meshgrid(kx_vals, ky_vals)

def HU_block(kx, ky, kz):
    """2x2 Hamiltonian for U orbitals - Vectorized version"""
    diag = muU - 2*tU*np.cos(kx*a) - 2*tch_U*np.cos(ky*b)
    real_off = -DeltaU - 2*tpU*np.cos(kx*a) - 2*tpch_U*np.cos(ky*b)
    complex_amp = -4 * tz_U * np.exp(-1j * kz * c / 2) * np.cos(kx * a / 2) * np.cos(ky * b / 2)
    
    # Handle scalar
    if np.isscalar(kx):
        H = np.zeros((2,2), dtype=complex)
        H[0,0] = diag
        H[1,1] = diag
        H[0,1] = real_off + complex_amp
        H[1,0] = real_off + np.conj(complex_amp)
    else:
        # Vectorissed
        shape = kx.shape
        H = np.zeros(shape + (2, 2), dtype=complex)
        H[..., 0, 0] = diag
        H[..., 1, 1] = diag
        H[..., 0, 1] = real_off + complex_amp
        H[..., 1, 0] = real_off + np.conj(complex_amp)
    return H

def HTe_block(kx, ky, kz):
    """2x2 Hamiltonian for Te orbitals - Vectorized version"""
    # Diagonal elements: μTe (no chain hopping since tch_Te = 0 in paper)
    diag = muTe
    # Off-diagonal elements from paper
    real_off = -DeltaTe
    complex_term1 = -tTe * np.exp(-1j * ky * b)  # hopping along b direction
    complex_term2 = -tz_Te * np.cos(kz * c / 2) * np.cos(kx * a / 2) * np.cos(ky * b / 2)
    
    # Handle both scalar and array inputs
    if np.isscalar(kx):
        H = np.zeros((2,2), dtype=complex)
        H[0,0] = diag
        H[1,1] = diag
        H[0,1] = real_off + complex_term1 + complex_term2
        H[1,0] = real_off + np.conj(complex_term1) + complex_term2  #
    else:
        # Vectorized: shape (..., 2, 2)
        shape = kx.shape
        H = np.zeros(shape + (2, 2), dtype=complex)
        H[..., 0, 0] = diag
        H[..., 1, 1] = diag
        H[..., 0, 1] = real_off + complex_term1 + complex_term2
        H[..., 1, 0] = real_off + np.conj(complex_term1) + complex_term2
    return H

def H_full(kx, ky, kz):
    """Full 4x4 Hamiltonian with U-Te hybridization - Vectorized version"""
    HU = HU_block(kx, ky, kz)
    HTe = HTe_block(kx, ky, kz)



    # Handle both scalar and array inputs
    if np.isscalar(kx):
        Hhyb = np.eye(2) * delta
        top = np.hstack((HU, Hhyb))
        bottom = np.hstack((Hhyb.conj().T, HTe))
        H = np.vstack((top, bottom))
    else:
        # Vectorized: shape (..., 4, 4)
        shape = kx.shape
        H = np.zeros(shape + (4, 4), dtype=complex)
        
        # Fill U block (top-left 2x2)
        H[..., 0:2, 0:2] = HU
        
        # Fill Te block (bottom-right 2x2)
        H[..., 2:4, 2:4] = HTe
        
        # Fill hybridization (off-diagonal 2x2 blocks)
        H[..., 0:2, 2:4] = delta * np.eye(2)
        H[..., 2:4, 0:2] = delta * np.eye(2)
    
    return H
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import numpy as np

def plot_011_density_map(fermi_surfaces, save_path=None, grid_res=500, sigma=0.01):
    """
    Plot the (0-11) Fermi surface projection as a smooth density map using Gaussian painting.
    Args:
        fermi_surfaces: dict with 'verts_2d' for each band
        save_path: optional path to save the figure
        grid_res: resolution of the output grid
        sigma: Gaussian thickness in normalized units
    """
    # Combine all projected vertices
    all_verts = np.vstack([fs['verts_2d'] for fs in fermi_surfaces.values()])
    # Set up grid in normalized units (-1 to 1)
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    x_grid = np.linspace(x_min, x_max, grid_res)
    y_grid = np.linspace(y_min, y_max, grid_res)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    # Build KDTree for fast nearest neighbor search
    tree = cKDTree(all_verts)
    # For each grid point, find distance to nearest FS point
    dists, _ = tree.query(grid_points, k=1)
    # Paint with Gaussian weight
    density = np.exp(-0.5 * (dists / sigma)**2)
    density = density.reshape(X.shape)
    # Plot
    plt.figure(figsize=(8, 8), dpi=300)
    plt.imshow(density, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='inferno', aspect='equal')
    plt.xlabel(r'$k_{c^*}$ (normalized)')
    plt.ylabel(r'$k_x$ (normalized)')
    plt.title('Fermi Surface Projection (0-11) Density Map')
    plt.colorbar(label='FS Density')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
def verify_model_parameters():
    print("\\n=== Model Verification ===")
    print("Checking parameters against paper values...")
    
    # Paper values (from the Methods section)
    paper_params = {
        'μU': -0.355, 'ΔU': 0.38, 'tU': 0.17, "t'U": 0.08, 
        'tch,U': 0.015, "t'ch,U": 0.01, 'tz,U': -0.0375,
        'μTe': -2.25, 'ΔTe': -1.4, 'tTe': -1.5, 'tch,Te': 0, 'tz,Te': -0.05,
        'δ': 0.13
    }
    
    # Our values
    our_params = {
        'μU': muU, 'ΔU': DeltaU, 'tU': tU, "t'U": tpU,
        'tch,U': tch_U, "t'ch,U": tpch_U, 'tz,U': tz_U,
        'μTe': muTe, 'ΔTe': DeltaTe, 'tTe': tTe, 'tch,Te': tch_Te, 'tz,Te': tz_Te,
        'δ': delta
    }
    
    all_match = True
    for param, paper_val in paper_params.items():
        our_val = our_params[param]
        match = abs(our_val - paper_val) < 1e-6
        status = "" if match else "✗"
        print(f"  {status} {param}: paper={paper_val:.3f}, ours={our_val:.3f}")
        if not match:
            all_match = False
    
    if all_match:
        print(" All parameters match the paper exactly!")
    else:
        print("✗ Some parameters don't match - check implementation")
    print("=" * 40)

def band_energies_on_slice(kz, resolution=300):
    """Compute band energies for all kx, ky at fixed kz"""
    # Create local momentum grid with proper resolution
    nk_local = resolution  # Configurable resolution
    kx_vals_local = np.linspace(-1*np.pi/a, 1*np.pi/a, nk_local)
    ky_vals_local = np.linspace(-3*np.pi/b, 3*np.pi/b, nk_local)
    
    energies = np.zeros((nk_local, nk_local, 4))
    
    # Original indexing: energies[kx_idx, ky_idx, band]
    for i, kx in enumerate(kx_vals_local):
        for j, ky in enumerate(ky_vals_local):
            H = H_full(kx, ky, kz)
            eigvals = np.linalg.eigvals(H)
            energies[i, j, :] = np.sort(np.real(eigvals))
    
    return energies

def band_energies_with_eigenvectors(kz, resolution=300):
    """Compute band energies and eigenvectors on a 2D slice at fixed kz."""
    # Create 2D momentum grid matching the UTe2 Fermi surface range
    nk = resolution  # Configurable resolution for better spectral weight calculation
    # Use extended range to capture the full Fermi surface
    kx_vals = np.linspace(-1*np.pi/a, 1*np.pi/a, nk)  # Extend kx range
    ky_vals = np.linspace(-3*np.pi/b, 3*np.pi/b, nk)       # Keep full ky range
    
    # Pre-allocate arrays
    energies = np.zeros((nk, nk, 4))
    eigenvectors = np.zeros((nk, nk, 4, 4), dtype=complex)
    
    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            H = H_full(kx, ky, kz)
            eigenvals, eigenvecs = np.linalg.eigh(H)
            idx = np.argsort(eigenvals)
            energies[i, j, :] = np.real(eigenvals[idx])
            eigenvectors[i, j, :, :] = eigenvecs[:, idx]
    
    return energies, eigenvectors

def compute_spectral_weights(kz, orbital_weights=None, energy_window=0.05, resolution=300):
    """
    Compute orbital-projected spectral weights for each band at given k-points.
    
    The spectral weight A(k,ω) = Σ_n |⟨orbital|ψ_n(k)⟩|² δ(ω - E_n(k))
    For Fermi surface, we evaluate at ω = E_F = 0 with a small energy window.
    
    Parameters:
    -----------
    kz : float
        kz value for the slice
    orbital_weights : dict, optional
        Weights for different orbitals {'U1': w1, 'U2': w2, 'Te1': w3, 'Te2': w4}
        Default gives equal weight to all orbitals
    energy_window : float
        Energy window around Fermi level for spectral weight calculation
    resolution : int
        Grid resolution for momentum space
    
    Returns:
    --------
    spectral_weights : ndarray, shape (nk, nk)
        Total spectral weight at each k-point (summed over relevant bands)
    orbital_contributions : dict
        Individual orbital contributions to spectral weight
    """
    if orbital_weights is None:
        orbital_weights = {'U1': 1.0, 'U2': 1.0, 'Te1': 1.0, 'Te2': 1.0}
    
    # Get energies and eigenvectors
    energies, eigenvectors = band_energies_with_eigenvectors(kz, resolution=resolution)
    nk = energies.shape[0]
    
    # Initialize spectral weight arrays
    total_spectral_weight = np.zeros((nk, nk))
    orbital_contributions = {
        'U1': np.zeros((nk, nk)),
        'U2': np.zeros((nk, nk)),
        'Te1': np.zeros((nk, nk)),
        'Te2': np.zeros((nk, nk))
    }
    
    orbital_names = ['U1', 'U2', 'Te1', 'Te2']
    
    print(f"Computing orbital-projected spectral weights at kz = {kz:.3f}...")
    print(f"Energy window around E_F: ±{energy_window:.3f} eV")
    
    for i in range(nk):
        if i % (nk//10) == 0:
            print(f"  Progress: {i/nk*100:.1f}%")
        for j in range(nk):
            for band in range(4):
                E_nk = energies[i, j, band]
                
                # Only include states near Fermi level
                if abs(E_nk) <= energy_window:
                    psi_nk = eigenvectors[i, j, :, band]  # Eigenvector for band n at k-point (i,j)
                    
                    # Weight by proximity to Fermi level for better contrast
                    fermi_weight = np.exp(- (E_nk / energy_window)**2 )
                    
                    # Calculate orbital-projected weights
                    for orbital_idx, orbital_name in enumerate(orbital_names):
                        # |⟨orbital|ψ_n(k)⟩|² weighted by proximity to Fermi level
                        orbital_weight = np.abs(psi_nk[orbital_idx])**2 * fermi_weight
                        
                        # Apply user-defined orbital importance weighting
                        weighted_contribution = orbital_weight * orbital_weights[orbital_name]
                        # Add to orbital-specific contribution
                        orbital_contributions[orbital_name][i, j] += weighted_contribution
                        
                        # Add to total spectral weight
                        total_spectral_weight[i, j] += weighted_contribution
    
    return total_spectral_weight, orbital_contributions

def calculate_gap_magnitude(kx, ky, kz, pairing_type='B1u', C0=0.0, C1=0.0003, C2=0.0003, C3=0.0003):
    """
    Calculate superconducting gap magnitude |Δ_k| for different pairing symmetries.
    
    Parameters:
    -----------
    kx, ky, kz : array-like
        Momentum components
    pairing_type : str
        Type of pairing symmetry: 'Au', 'B1u', 'B2u', 'B3u'
    C0, C1, C2, C3 : float
        Coupling constants for different symmetry components (in eV)
        Default values: C0 = 0, C1 = C2 = C3 = 300 μeV = 0.0003 eV
        
    Returns:
    --------
    gap_magnitude : array
        Magnitude of superconducting gap |Δ_k| (in eV)
    """
    
    # Define d-vectors for different pairing symmetries from the paper
    if pairing_type == 'Au':
        # IR d vector: Au [C1 sin(kx*a), C2 sin(ky*b), C3 sin(kz*c)]
        dx = C1 * np.sin(kx * a)
        dy = C2 * np.sin(ky * b) 
        dz = C3 * np.sin(kz * c)
        
    elif pairing_type == 'B1u':
        # B1u: d ∝ (sin ky·b, sin kx·a, 0) → zeros at ky = 0, ±π/b, kx = 0, ±π/a
        dx = C1 * np.sin(ky * b)
        dy = C2 * np.sin(kx * a)
        dz = 0  # Third component is zero for B1u
        
    elif pairing_type == 'B2u':
        # B2u: d ∝ (sin kz·c, 0, sin kx·a) → zeros at kz = 0, ±π/c, kx = 0, ±π/a
        dx = C1 * np.sin(kz * c)
        dy = 0  # Middle component is zero for B2u
        dz = C3 * np.sin(kx * a)
        
    elif pairing_type == 'B3u':
        # B3u: d ∝ (0, sin kz·c, sin ky·b) → zeros at kz = 0, ±π/c, ky = 0, ±π/b
        dx = 0  # First component is zero for B3u
        dy = C2 * np.sin(kz * c)
        dz = C3 * np.sin(ky * b)
        
    else:
        raise ValueError(f"Unknown pairing type: {pairing_type}")
    
    # Calculate gap magnitude |Δ_k| = |d_k|
    gap_magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
    
    return gap_magnitude

def plot_main_fermi_contours(kx_vals, ky_vals, energies_fs, save_dir='outputs/ute2_debug'):
    """Plot the main Fermi contours within -1π/a to 1π/a and -1π/b to 1π/b range."""
    import matplotlib.pyplot as plt
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6.4), dpi=300)
    
    # Extract contours for both bands
    bands_to_plot = [2, 3]  # Band 3 and 4 (0-indexed)
    colors = ['red', 'blue']
    labels = ['Band 3', 'Band 4']
    
    for band_idx, color, label in zip(bands_to_plot, colors, labels):
        energy_data = energies_fs[:, :, band_idx]
        
        if energy_data.min() <= 0 <= energy_data.max():
            # Create contour
            temp_fig, temp_ax = plt.subplots(figsize=(1, 1), dpi=300)
            contour = temp_ax.contour(ky_vals, kx_vals, energy_data, levels=[0.0])
            plt.close(temp_fig)
            
            # Extract and plot relevant contours
            if len(contour.allsegs[0]) > 0:
                contour_count = 0
                for i, path in enumerate(contour.allsegs[0]):
                    if len(path) > 0:
                        # Convert to π/a and π/b units
                        ky_array = path[:, 0] / (np.pi / b)
                        kx_array = path[:, 1] / (np.pi / a)
                        
                        # Check if contour is within or overlaps our range of interest
                        ky_in_range = (ky_array.min() <= 1) and (ky_array.max() >= -1)
                        kx_in_range = (kx_array.min() <= 1) and (kx_array.max() >= -1)
                        
                        if ky_in_range and kx_in_range:
                            # Plot the contour
                            if contour_count == 0:
                                ax.plot(ky_array, kx_array, color=color, linewidth=2, 
                                       label=f'{label} Fermi Surface', alpha=0.8)
                            else:
                                ax.plot(ky_array, kx_array, color=color, linewidth=2, alpha=0.8)
                            
                            contour_count += 1
                            print(f"{label} Contour {i+1}: ky ∈ [{ky_array.min():.3f}π/b, {ky_array.max():.3f}π/b], "
                                  f"kx ∈ [{kx_array.min():.3f}π/a, {kx_array.max():.3f}π/a], {len(path)} points")
    
    # Set up the plot
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('ky (π/b units)', fontsize=12)
    ax.set_ylabel('kx (π/a units)', fontsize=12)
    ax.set_title('Main Fermi Contours (Bands 3 & 4)\nWithin ±1π/a and ±1π/b Range', fontsize=14)
    ax.grid(True, alpha=0.3)
    # ax.legend()
    
    # Add reference lines
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Add boundary box
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k--', alpha=0.5, linewidth=1)
    
    save_path = os.path.join(save_dir, 'main_fermi_contours_debug.png')
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Debug plot saved to: {save_path}")

def extract_fermi_contour_points(kx_vals, ky_vals, energies_fs, bands_to_extract=[2, 3]):
    """Extract Fermi contour points for specified bands and print them."""
    print("\n=== FERMI CONTOUR POINTS ===")
    
    for band_idx in bands_to_extract:
        print(f"\nBand {band_idx + 1} Fermi Contours:")
        
        # Extract energy data for this band
        energy_data = energies_fs[:, :, band_idx].T  # Transpose to match ky,kx coordinate system
        
        # Check if band crosses Fermi level
        if energy_data.min() <= 0 <= energy_data.max():
            # Create contour to extract path data
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(1, 1), dpi=300)
            
            # Debug: print array shapes
            print(f"    Debug: ky_vals shape: {ky_vals.shape}, kx_vals shape: {kx_vals.shape}")
            print(f"    Debug: energy_data shape: {energy_data.shape}")
            
            # Make sure we have the right orientation for contour
            # Contour expects: contour(X, Y, Z) where Z.shape == (len(Y), len(X))
            # So if energy_data[ky_idx, kx_idx], then Z should be energy_data
            contour = ax.contour(ky_vals, kx_vals, energy_data, levels=[0.0])
            plt.close(fig)
            
            # Extract contour paths
            if len(contour.allsegs[0]) > 0:
                for i, path in enumerate(contour.allsegs[0]):
                    if len(path) > 0:
                        print(f"  Contour {i+1}: {len(path)} points")
                        
                        # Convert to π/a and π/b units and print sample points
                        print(f"    Sample points (in π/a, π/b units):")
                        
                        # Print every 10th point to avoid too much output
                        sample_indices = range(0, len(path), max(1, len(path) // 10))
                        for j in sample_indices:
                            ky, kx = path[j]
                            ky_pi_b = ky / (np.pi / b)
                            kx_pi_a = kx / (np.pi / a)
                            print(f"      Point {j}: ky={ky_pi_b:.3f}π/b, kx={kx_pi_a:.3f}π/a")
                        
                        # Also print the full contour as arrays for potential use
                        print(f"    Full contour summary:")
                        ky_array = path[:, 0] / (np.pi / b)
                        kx_array = path[:, 1] / (np.pi / a)
                        print(f"      ky range: {ky_array.min():.3f}π/b to {ky_array.max():.3f}π/b")
                        print(f"      kx range: {kx_array.min():.3f}π/a to {kx_array.max():.3f}π/a")
                        print(f"      Total points: {len(ky_array)}")
            else:
                print(f"    No contours found for Band {band_idx + 1}")
        else:
            print(f"    Band {band_idx + 1} does not cross Fermi level")
            print(f"      Energy range: {energy_data.min():.3f} to {energy_data.max():.3f} eV")

def plot_gap_magnitude_2d(kz=0.0, pairing_types=['B1u', 'B2u', 'B3u'], 
                         save_dir='outputs/ute2_gap', resolution=300):
    """
    Plot 2D superconducting gap magnitude for different pairing symmetries.
    
    Parameters:
    -----------
    kz : float
        kz value for the slice
    pairing_types : list
        List of pairing symmetries to plot
    save_dir : str
        Directory to save plots
    resolution : int
        Grid resolution
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nComputing superconducting gap magnitudes at kz = {kz:.3f}...")
    
    # Create momentum grid with physical k-values
    nk = resolution
    kx_vals = np.linspace(-np.pi/a, np.pi/a, nk)
    ky_vals = np.linspace(-np.pi/b, np.pi/b, nk)  # Reduced range for gap plots
    KX, KY = np.meshgrid(kx_vals, ky_vals)
    
    # Calculate Fermi surface for overlay using same range as gap plots
    energies_fs = np.zeros((nk, nk, 4))
    for ix, kx in enumerate(kx_vals):
        for jy, ky in enumerate(ky_vals):
            H = H_full(kx, ky, kz)
            eigvals = np.linalg.eigvals(H)
            energies_fs[ix, jy, :] = np.sort(np.real(eigvals))
    
    # Create subplot layout - 3 plots side by side
    n_types = len(pairing_types)
    if n_types <= 3:
        # Use fixed size similar to 3D comparison plot for consistency
        fig, axes = plt.subplots(1, n_types, figsize=(9, 6), dpi=300)
        if n_types == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), dpi=300)
        axes = axes.flatten()
    
    # Colors for Fermi surface bands
    fs_colors = ['#1E88E5', '#42A5F5', "#9635E5", "#FD0000"]
    band_labels = ['Band 1', 'Band 2', 'Band 3', 'Band 4']
    
    for i, pairing_type in enumerate(pairing_types):
        ax = axes[i]
        
        print(f"  Computing {pairing_type} gap magnitude...")
        # Calculate gap magnitude for this pairing type
        gap_mag = calculate_gap_magnitude(KX, KY, kz, pairing_type=pairing_type)
        
        # Plot gap magnitude - use same coordinate order as contour
        im = ax.imshow(gap_mag.T, extent=[ky_vals.min(), ky_vals.max(), 
                                         kx_vals.min(), kx_vals.max()],
                      origin='lower', cmap='RdYlBu_r', alpha=0.8)
        
        # Overlay Fermi surface contours - SAME scheme as plot_fs_2d
        for band in range(4):
            Z = energies_fs[:, :, band].T  # Same as plot_fs_2d
            if Z.min() <= 0 <= Z.max():  # Only plot if band crosses Fermi level
                # Use same contour call as plot_fs_2d: contour(KY, KX, Z)
                ax.contour(KY, KX, Z, levels=[0], 
                          colors=[fs_colors[band]], linewidths=3, alpha=1.0, zorder=8)
        
        # Find and mark gap nodes where gap≈0 AND Fermi surface crosses
        # Use physics-based detection for different pairing types
        if pairing_type == 'B2u':
            # B2u at kz=0: [0, 0, C3*sin(kx*a)] -> nodes at kx = 0, ±π/a (vertical lines)
            gap_node_threshold = 0.03 * np.max(gap_mag)  # 3% threshold for B2u
        elif pairing_type == 'B3u':
            # B3u at kz=0: [0, 0, C3*sin(ky*b)] -> nodes at ky = 0, ±π/b (horizontal lines)  
            gap_node_threshold = 0.02 * np.max(gap_mag)  # 2% threshold for B3u
        else:
            gap_node_threshold = 0.05 * np.max(gap_mag)  # Default threshold
        
        total_gap_nodes = 0
        gap_nodes_by_band = {}
        
        # Special handling for B2u - use global detection instead of line-based
        if pairing_type == 'B2u':
            # B2u can have gap nodes anywhere where gap is small, not just at exact theoretical lines
            gap_node_threshold = 0.05 * np.max(gap_mag)  # More generous threshold for B2u
            
            fermi_bands = [2, 3]  # Band 3 and 4 (0-indexed)
            
            for band in fermi_bands:
                Z = energies_fs[:, :, band].T  # Transpose to match gap_mag indexing
                if Z.min() <= 0 <= Z.max():  # Double-check Fermi crossing
                    # Use sensitive Fermi surface detection
                    fermi_mask = np.abs(Z) < 0.02  # 20 meV threshold
                    gap_node_mask = gap_mag < gap_node_threshold
                    intersection = fermi_mask & gap_node_mask
                    
                    if np.any(intersection):
                        ky_indices, kx_indices = np.where(intersection)
                        band_nodes = 0
                        
                        # Group nearby points to avoid overcrowding
                        selected_points = []
                        min_distance = 15  # Minimum pixels between gap nodes
                        
                        for node_idx in range(len(ky_indices)):
                            ky_idx, kx_idx = ky_indices[node_idx], kx_indices[node_idx]
                            
                            # Check if this point is far enough from already selected points
                            too_close = False
                            for (prev_ky, prev_kx) in selected_points:
                                if np.sqrt((ky_idx - prev_ky)**2 + (kx_idx - prev_kx)**2) < min_distance:
                                    too_close = True
                                    break
                            
                            if not too_close:
                                selected_points.append((ky_idx, kx_idx))
                                
                                # Convert indices to physical coordinates
                                ky_node = ky_vals[ky_idx]
                                kx_node = kx_vals[kx_idx]
                                
                                # Print coordinates for debugging
                                ky_pi_b = ky_node / (np.pi / b)
                                kx_pi_a = kx_node / (np.pi / a)
                                print(f"    B2u Band {band+1} gap node: ky={ky_pi_b:.3f}π/b, kx={kx_pi_a:.3f}π/a")
                                
                                # Add circle marker at gap node
                                ax.scatter(ky_node, kx_node, s=100, c='yellow',
                                         marker='o', edgecolors='red', linewidth=2,
                                         alpha=0.9, zorder=10)
                                total_gap_nodes += 1
                                band_nodes += 1
                        
                        if band_nodes > 0:
                            gap_nodes_by_band[f'Band {band+1}'] = band_nodes
        
        else:
            # Standard point-based detection for other pairing types
            fermi_bands = [2, 3]  # Band 3 and 4 (0-indexed)
            
            for band in fermi_bands:
                Z = energies_fs[:, :, band].T  # Transpose to match gap_mag indexing
                if Z.min() <= 0 <= Z.max():  # Double-check Fermi crossing
                    # Use sensitive Fermi surface detection
                    fermi_mask = np.abs(Z) < 0.01  # 10 meV threshold
                    gap_node_mask = gap_mag < gap_node_threshold
                    intersection = fermi_mask & gap_node_mask
                    
                    if np.any(intersection):
                        ky_indices, kx_indices = np.where(intersection)
                        band_nodes = 0
                        
                        # Physics-based grouping for specific pairing types
                        selected_points = []
                        if pairing_type == 'B3u':
                            # B3u: nodes should be at horizontal lines (constant ky)
                            min_distance = 12
                        else:
                            min_distance = 10
                        
                        for node_idx in range(len(ky_indices)):
                            ky_idx, kx_idx = ky_indices[node_idx], kx_indices[node_idx]
                            
                            # Check if this point is far enough from already selected points
                            too_close = False
                            for (prev_ky, prev_kx) in selected_points:
                                if np.sqrt((ky_idx - prev_ky)**2 + (kx_idx - prev_kx)**2) < min_distance:
                                    too_close = True
                                    break
                            
                            if not too_close:
                                selected_points.append((ky_idx, kx_idx))
                                
                                # Convert indices to physical coordinates
                                ky_node = ky_vals[ky_idx]
                                kx_node = kx_vals[kx_idx]
                                
                                # Print coordinates for debugging
                                if pairing_type == 'B3u':
                                    ky_pi_b = ky_node / (np.pi / b)
                                    kx_pi_a = kx_node / (np.pi / a)
                                    print(f"    B3u Band {band+1} gap node: ky={ky_pi_b:.3f}π/b, kx={kx_pi_a:.3f}π/a")
                                
                                # Add circle marker at gap node
                                ax.scatter(ky_node, kx_node, s=100, c='yellow', 
                                         marker='o', edgecolors='red', linewidth=2,
                                         alpha=0.9, zorder=10)
                                total_gap_nodes += 1
                                band_nodes += 1
                        
                        if band_nodes > 0:
                            gap_nodes_by_band[f'Band {band+1}'] = band_nodes
        
        if total_gap_nodes > 0:
            band_info = ', '.join([f'{band}: {count}' for band, count in gap_nodes_by_band.items()])
            print(f"    {pairing_type}: found {total_gap_nodes} gap nodes on Fermi surface ({band_info})")
        else:
            print(f"    {pairing_type}: no gap nodes found - checking gap structure...")
            print(f"      Gap range: {np.min(gap_mag):.6f} to {np.max(gap_mag):.6f}")
            print(f"      Gap threshold used: {gap_node_threshold:.6f}")
            print(f"      Fermi surface energy range: {np.min(np.abs(energies_fs[:,:,2])):.6f} to {np.min(np.abs(energies_fs[:,:,3])):.6f}")
        
        # Set physical axis limits
        ax.set_xlim(ky_vals.min(), ky_vals.max())
        ax.set_ylim(kx_vals.min(), kx_vals.max())
        
        # Custom tick formatting to show π/a and π/b units
        ky_ticks = np.linspace(ky_vals.min(), ky_vals.max(), 5)
        kx_ticks = np.linspace(kx_vals.min(), kx_vals.max(), 5)
        ky_labels = [f'{val/(np.pi/b):.1f}' for val in ky_ticks]
        kx_labels = [f'{val/(np.pi/a):.1f}' for val in kx_ticks]
        ax.set_xticks(ky_ticks)
        ax.set_yticks(kx_ticks)
        ax.set_xticklabels(ky_labels, fontsize=10)
        ax.set_yticklabels(kx_labels, fontsize=10)
        
        ax.set_xlabel(r'$k_y$ ($\pi/b$)', fontsize=12)
        ax.set_ylabel(r'$k_x$ ($\pi/a$)', fontsize=12)
        ax.set_title(f'{pairing_type}', fontsize=12)
        ax.grid(False)
        
        # Add panel label (a), b), c))
        panel_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
        if i < len(panel_labels):
            ax.text(0.02, 0.98, panel_labels[i], transform=ax.transAxes, 
                   fontsize=11, fontweight='bold', va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='black'),
                   zorder=1000)
        
        # Add colorbar with scientific notation
        cbar = plt.colorbar(im, ax=ax, shrink=0.3, format='%.1e')
        cbar.set_label(r'|Δ$_k$|', fontsize=12)
        
        print(f"    {pairing_type}: gap range 0 to {np.max(gap_mag):.3f}")
        print(f"    {pairing_type}: no gap nodes detected")
    
    # Add legend closer to the plots
    legend_elements = [
        plt.Line2D([0], [0], color='#FD0000', lw=3, label='Electron-like'),
        plt.Line2D([0], [0], color='#9635E5', lw=3, label='Hole-like'),
        plt.Line2D([0], [0], marker='o', color='yellow', markersize=8, 
                  markeredgecolor='red', markeredgewidth=2, linestyle='None', label='Nodes')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.17), 
              ncol=3, fontsize=11, frameon=False)
    
    # Hide unused subplots if necessary
    if n_types < len(axes):
        for j in range(n_types, len(axes)):
            axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.01)  # Reduced margin for closer legend
    
    # Save the plot
    filename = f'gap_magnitude_2d_kz_{kz:.3f}.png'
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Gap magnitude plot saved: {filepath}")
    
    # Also create individual plots for each pairing type
    for pairing_type in pairing_types:
        fig_single, ax_single = plt.subplots(1, 1, figsize=(8.0, 6.0), dpi=300)
        
        # Calculate gap magnitude
        gap_mag = calculate_gap_magnitude(KX, KY, kz, pairing_type=pairing_type)
        
        # Plot gap magnitude with physical k-values
        im_single = ax_single.imshow(gap_mag.T, extent=[ky_vals.min(), ky_vals.max(), 
                                               kx_vals.min(), kx_vals.max()],
                                    origin='lower', cmap='RdYlBu_r', alpha=0.8)
        gap_threshold = 0.1 * np.max(gap_mag)
        nodal_mask = gap_mag < gap_threshold
        if np.any(nodal_mask):
            ax_single.contourf(ky_vals, kx_vals, gap_mag.T, levels=[0, gap_threshold],
                             colors=['darkblue'], alpha=0.4)  # Reduce alpha
        
        # Overlay Fermi surface ON TOP
        for band in range(4):
            Z = energies_fs[:, :, band]  # No transpose needed now
            if Z.min() <= 0 <= Z.max():
                contour = ax_single.contour(ky_vals, kx_vals, Z, levels=[0], 
                                          colors=[fs_colors[band]], linewidths=3, alpha=1.0, zorder=8)
                # Add band labels
                if len(contour.allsegs[0]) > 0:  # Check if contour has segments
                    ax_single.plot([], [], color=fs_colors[band], linewidth=3, 
                                 label=band_labels[band], alpha=1.0)
        
        # Find and mark gap nodes where gap≈0 AND Fermi surface crosses
        # Use physics-based detection for different pairing types
        if pairing_type == 'B2u':
            # B2u at kz=0: [0, 0, C3*sin(kx*a)] -> nodes at kx = 0, ±π/a (vertical lines)
            gap_node_threshold = 0.03 * np.max(gap_mag)  # 3% threshold for B2u
        elif pairing_type == 'B3u':
            # B3u at kz=0: [0, 0, C3*sin(ky*b)] -> nodes at ky = 0, ±π/b (horizontal lines)  
            gap_node_threshold = 0.02 * np.max(gap_mag)  # 2% threshold for B3u
        else:
            gap_node_threshold = 0.05 * np.max(gap_mag)  # Default threshold
        
        gap_nodes_found = 0
        gap_nodes_by_band = {}
        
        # Special handling for B2u - use same global detection as in combined plot
        if pairing_type == 'B2u':
            # B2u can have gap nodes anywhere where gap is small, not just at exact theoretical lines
            gap_node_threshold = 0.05 * np.max(gap_mag)  # More generous threshold for B2u
            
            fermi_bands = [2, 3]  # Band 3 and 4 (0-indexed)
            
            for band in fermi_bands:
                Z = energies_fs[:, :, band].T  # Transpose to match gap_mag indexing
                if Z.min() <= 0 <= Z.max():  # Double-check Fermi crossing
                    # Use sensitive Fermi surface detection
                    fermi_mask = np.abs(Z) < 0.02  # 20 meV threshold
                    gap_node_mask = gap_mag < gap_node_threshold
                    intersection = fermi_mask & gap_node_mask
                    
                    if np.any(intersection):
                        ky_indices, kx_indices = np.where(intersection)
                        band_nodes = 0
                        
                        # Group nearby points to avoid overcrowding
                        selected_points = []
                        min_distance = 15  # Minimum pixels between gap nodes
                        
                        for node_idx in range(len(ky_indices)):
                            ky_idx, kx_idx = ky_indices[node_idx], kx_indices[node_idx]
                            
                            # Check if this point is far enough from already selected points
                            too_close = False
                            for (prev_ky, prev_kx) in selected_points:
                                if np.sqrt((ky_idx - prev_ky)**2 + (kx_idx - prev_kx)**2) < min_distance:
                                    too_close = True
                                    break
                            
                            if not too_close:
                                selected_points.append((ky_idx, kx_idx))
                                
                                # Convert indices to physical coordinates
                                ky_node = ky_vals[ky_idx]
                                kx_node = kx_vals[kx_idx]
                                
                                # Print coordinates for debugging
                                ky_pi_b = ky_node / (np.pi / b)
                                kx_pi_a = kx_node / (np.pi / a)
                                print(f"    B2u Band {band+1} gap node: ky={ky_pi_b:.3f}π/b, kx={kx_pi_a:.3f}π/a")
                                
                                # Add circle marker at gap node
                                if gap_nodes_found == 0:  # Add to legend only once
                                    ax_single.scatter(ky_node, kx_node, s=150, c='yellow',
                                                   marker='o', edgecolors='red', linewidth=2,
                                                   alpha=0.9, zorder=10, label='Gap nodes')
                                else:
                                    ax_single.scatter(ky_node, kx_node, s=150, c='yellow',
                                                   marker='o', edgecolors='red', linewidth=2,
                                                   alpha=0.9, zorder=10)
                                gap_nodes_found += 1
                                band_nodes += 1
                        
                        if band_nodes > 0:
                            gap_nodes_by_band[f'Band {band+1}'] = band_nodes
        
        else:
            # Standard point-based detection for other pairing types
            fermi_bands = [2, 3]  # Band 3 and 4 (0-indexed)
            
            for band in fermi_bands:
                Z = energies_fs[:, :, band].T  # Transpose to match gap_mag indexing
                if Z.min() <= 0 <= Z.max():  # Double-check Fermi crossing
                    band_gap_nodes = 0
                    
                    # Use sensitive Fermi surface detection
                    fermi_mask = np.abs(Z) < 0.01  # 10 meV threshold  
                    gap_node_mask = gap_mag < gap_node_threshold
                    intersection = fermi_mask & gap_node_mask
                    
                    if np.any(intersection):
                        ky_indices, kx_indices = np.where(intersection)
                        
                        # Physics-based grouping for specific pairing types
                        selected_points = []
                        if pairing_type == 'B3u':
                            # B3u: nodes should be at horizontal lines (constant ky)
                            min_distance = 10
                        else:
                            min_distance = 8
                        
                        for node_idx in range(len(ky_indices)):
                            ky_idx, kx_idx = ky_indices[node_idx], kx_indices[node_idx]
                            
                            # Check if this point is far enough from already selected points
                            too_close = False
                            for (prev_ky, prev_kx) in selected_points:
                                if np.sqrt((ky_idx - prev_ky)**2 + (kx_idx - prev_kx)**2) < min_distance:
                                    too_close = True
                                    break
                            
                            if not too_close:
                                selected_points.append((ky_idx, kx_idx))
                                
                                # Convert indices to physical coordinates
                                ky_node = ky_vals[ky_idx]
                                kx_node = kx_vals[kx_idx]
                                
                                # Print coordinates for debugging
                                ky_pi_b = ky_node / (np.pi / b)
                                kx_pi_a = kx_node / (np.pi / a)
                                print(f"    B3u Band {band+1} gap node: ky={ky_pi_b:.3f}π/b, kx={kx_pi_a:.3f}π/a")
                                
                                # Add circle marker at gap node
                                if gap_nodes_found == 0:  # Add to legend only once
                                    ax_single.scatter(ky_node, kx_node, s=150, c='yellow', 
                                                   marker='o', edgecolors='red', linewidth=2,
                                                   alpha=0.9, zorder=10, label='Gap nodes')
                                else:
                                    ax_single.scatter(ky_node, kx_node, s=150, c='yellow', 
                                                   marker='o', edgecolors='red', linewidth=2,
                                                   alpha=0.9, zorder=10)
                                gap_nodes_found += 1
                                band_gap_nodes += 1
                    
                    if band_gap_nodes > 0:
                        gap_nodes_by_band[f'Band {band+1}'] = band_gap_nodes
        
        if gap_nodes_found > 0:
            band_info = ', '.join([f'{band}: {count}' for band, count in gap_nodes_by_band.items()])
            print(f"    Found {gap_nodes_found} gap nodes on Fermi surface for {pairing_type} ({band_info})")
        else:
            print(f"    No gap nodes found on Fermi surface for {pairing_type}")
            print(f"      Gap range: {np.min(gap_mag):.6f} to {np.max(gap_mag):.6f}")
            print(f"      Gap threshold used: {gap_node_threshold:.6f}")
            print(f"      Expected for {pairing_type}: 6 nodes total (3 per band)")
        
        # Set physical axis limits and custom tick formatting
        ax_single.set_xlim(ky_vals.min(), ky_vals.max())
        ax_single.set_ylim(kx_vals.min(), kx_vals.max())
        
        # Custom tick formatting to show π/a and π/b units
        ky_ticks = np.linspace(ky_vals.min(), ky_vals.max(), 5)
        kx_ticks = np.linspace(kx_vals.min(), kx_vals.max(), 5)
        ky_labels = [f'{val/(np.pi/b):.1f}' for val in ky_ticks]
        kx_labels = [f'{val/(np.pi/a):.1f}' for val in kx_ticks]
        ax_single.set_xticks(ky_ticks)
        ax_single.set_yticks(kx_ticks)
        ax_single.set_xticklabels(ky_labels)
        ax_single.set_yticklabels(kx_labels)
        
        ax_single.set_xlabel(r'$k_y$ ($\pi/b$)', fontsize=14)
        ax_single.set_ylabel(r'$k_x$ ($\pi/a$)', fontsize=14)
        ax_single.set_title(f'{pairing_type}', fontsize=16)
        ax_single.grid(False)
        ax_single.legend(loc='upper right', fontsize=10)
        
        # Enhanced colorbar with scientific notation
        cbar_single = plt.colorbar(im_single, ax=ax_single, shrink=0.8, format='%.1e')
        cbar_single.set_label(r'|Δ$_k$|', fontsize=12)
        
        plt.tight_layout()
        
        # Save individual plot
        filename_single = f'gap_magnitude_{pairing_type}_kz_{kz:.3f}.png'
        filepath_single = os.path.join(save_dir, filename_single)
        fig_single.savefig(filepath_single, dpi=300, bbox_inches='tight')
        plt.close(fig_single)
        
        print(f"Individual {pairing_type} plot saved: {filepath_single}")

def plot_fs_2d(energies, kz_label, save_dir='outputs/ute2_fixed'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Colors for different bands - U bands (0,1) vs Te bands (2,3)
    colors = ['#1E88E5', '#42A5F5',  "#9635E5", "#FD0000"]  # Blue for U, Red for Te
    band_labels = ['Band 1', 'Band 2', 'Band 3', 'Band 4']
    
    # Create momentum grids that match the energies array size - using physical k-values
    nk_energies = energies.shape[0]  # Get actual size of energies array
    kx_vals_local = np.linspace(-1*np.pi/a, 1*np.pi/a, nk_energies)
    ky_vals_local = np.linspace(-3*np.pi/b, 3*np.pi/b, nk_energies)
    KX_local, KY_local = np.meshgrid(kx_vals_local, ky_vals_local)
    
    # Keep physical k-values for plotting (no conversion)
    KX_physical = KX_local  # Physical kx values
    KY_physical = KY_local  # Physical ky values
    
    # Fermi level
    EF = 0.0
    
    # Create both extended and cropped versions
    versions = [
        {'xlim': (-3, 3), 'suffix': '_extended', 'title_suffix': ' (Extended)'},
        {'xlim': (-1, 1), 'suffix': '_cropped', 'title_suffix': ' (Cropped)'}
    ]
    
    for version in versions:
        fig = plt.figure(figsize=(8, 6.4), dpi=300)
        bands_plotted = 0
        
        print(f"\\n2D Fermi surface at kz={kz_label}{version['title_suffix']}:")
        
        for band in range(4):
            Z = energies[:, :, band].T
            
            # Only plot if band crosses the Fermi level
            if Z.min() <= EF <= Z.max():
                cs = plt.contour(KY_physical, KX_physical, Z, levels=[EF], 
                               linewidths=2.5, colors=[colors[band]])
                if len(cs.allsegs[0]) > 0:
                    seg = cs.allsegs[0][0]
                    if len(seg) > 0:
                        mid = seg.shape[0]//2
                        xlbl, ylbl = seg[mid,0], seg[mid,1]
                        # Convert physical coordinates to π/b units for range check
                        xlbl_units = xlbl / (np.pi/b)
                        # Only add label if it's within the visible range
                        if version['xlim'][0] <= xlbl_units <= version['xlim'][1]:
                            plt.text(xlbl, ylbl, f"{band_labels[band]}", fontsize=9, 
                                   color=colors[band], weight='bold', 
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                        bands_plotted += 1
                        print(f"  Plotted {band_labels[band]} Fermi surface at E=0.000 eV")
            else:
                print(f"  {band_labels[band]} does not cross Fermi level (range: {Z.min():.3f} to {Z.max():.3f} eV)")
        
        if bands_plotted == 0:
            print("  Warning: No bands cross the Fermi level at this kz!")
        
        # Set physical axis limits
        ky_lim_physical = [val * np.pi/b for val in version['xlim']]
        kx_lim_physical = [-1 * np.pi/a, 1 * np.pi/a]
        
        plt.xlim(ky_lim_physical)
        plt.ylim(kx_lim_physical)
        
        # Custom tick formatting to show π/a and π/b units
        ky_ticks = np.linspace(ky_lim_physical[0], ky_lim_physical[1], 7)
        kx_ticks = np.linspace(kx_lim_physical[0], kx_lim_physical[1], 5)
        ky_labels = [f'{val/(np.pi/b):.1f}' for val in ky_ticks]
        kx_labels = [f'{val/(np.pi/a):.1f}' for val in kx_ticks]
        plt.xticks(ky_ticks, ky_labels)
        plt.yticks(kx_ticks, kx_labels)
        
        plt.xlabel(r'$k_y$ (π/b)', fontsize=12)  # ky on x-axis
        plt.ylabel(r'$k_x$ (π/a)', fontsize=12)  # kx on y-axis
        plt.title(f'Fermi surface (E=0) at kz={kz_label}{version["title_suffix"]}', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add legend
        # from matplotlib.lines import Line2D
        # legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=band_labels[i]) for i in range(4)]
        # plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save with appropriate filename
        filename = f'fermi_surface_kz_{kz_label}{version["suffix"]}.png'
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved plot to {filepath}")

def plot_fs_2d_comparison(energies, kz_label, save_dir='outputs/ute2_fixed'):
    """Create side-by-side comparison of extended and cropped 2D Fermi surfaces with modern formatting."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Colors for different bands - match other plots
    colors = ['#1E88E5', '#42A5F5',  "#9635E5", "#FD0000"]  # Blue for U, Red for Te
    band_labels = ['Band 1', 'Band 2', 'Band 3', 'Band 4']
    
    # Create momentum grids that match the energies array size - using physical k-values
    nk_energies = energies.shape[0]  # Get actual size of energies array
    kx_vals_local = np.linspace(-1*np.pi/a, 1*np.pi/a, nk_energies)
    ky_vals_local = np.linspace(-3*np.pi/b, 3*np.pi/b, nk_energies)
    KX_local, KY_local = np.meshgrid(kx_vals_local, ky_vals_local)
    
    # Keep physical k-values for plotting
    KX_physical = KX_local
    KY_physical = KY_local
    
    # Fermi level
    EF = 0.0
    
    # Create side-by-side comparison with modern sizing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6), dpi=300)
    
    # Configuration for each subplot
    configs = [
        {'ax': ax1, 'xlim': (-3, 3), 'title': 'Extended', 'label': 'a)'},
        {'ax': ax2, 'xlim': (-1, 1), 'title': 'Cropped', 'label': 'b)'}
    ]
    
    print(f"\\n2D Fermi surface comparison at kz={kz_label}:")
    legend_elements = []
    
    for config in configs:
        ax = config['ax']
        bands_plotted = 0
        
        for band in range(4):
            Z = energies[:, :, band].T
            
            # Only plot if band crosses the Fermi level
            if Z.min() <= EF <= Z.max():
                cs = ax.contour(KY_physical, KX_physical, Z, levels=[EF], 
                               linewidths=3, colors=[colors[band]])
                if len(cs.allsegs[0]) > 0:
                    bands_plotted += 1
                    # Add to legend only once
                    if ax == ax1:
                        legend_elements.append(plt.Line2D([0], [0], color=colors[band], lw=3, label=band_labels[band]))
        
        # Set physical axis limits
        ky_lim_physical = [val * np.pi/b for val in config['xlim']]
        kx_lim_physical = [-1 * np.pi/a, 1 * np.pi/a]
        
        ax.set_xlim(ky_lim_physical)
        ax.set_ylim(kx_lim_physical)
        
        # Custom tick formatting with proper font sizes
        ky_ticks = np.linspace(ky_lim_physical[0], ky_lim_physical[1], 7)
        kx_ticks = np.linspace(kx_lim_physical[0], kx_lim_physical[1], 5)
        ky_labels = [f'{val/(np.pi/b):.1f}' for val in ky_ticks]
        kx_labels = [f'{val/(np.pi/a):.1f}' for val in kx_ticks]
        ax.set_xticks(ky_ticks)
        ax.set_yticks(kx_ticks)
        ax.set_xticklabels(ky_labels, fontsize=10)
        ax.set_yticklabels(kx_labels, fontsize=10)
        
        # Labels and title with consistent font sizes
        ax.set_xlabel(r'$k_y$ ($\pi$/b)', fontsize=12)
        ax.set_ylabel(r'$k_x$ ($\pi$/a)', fontsize=12)
        ax.set_title(config['title'], fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add panel label
        ax.text(0.02, 0.98, config['label'], transform=ax.transAxes, 
               fontsize=12, fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'),
               zorder=1000)
        
        print(f"  {config['title']}: plotted {bands_plotted} bands")
    
    # Add legend
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
              ncol=4, fontsize=11, frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    # Save comparison plot
    filename = f'fermi_surface_comparison_kz_{kz_label}.png'
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved comparison plot to {filepath}")

def plot_2d_with_spectral_weight(kz=0.0, weight_type='total', orbital_focus=None, 
                                colormap='hot', alpha_contours=0.8, save_dir='outputs/ute2_spectral',
                                energy_window=0.15):
    """
    Plot 2D Fermi surface with spectral weight information.
    
    Parameters:
    -----------
    kz : float
        kz value for the slice
    weight_type : str
        'total' - show total spectral weight
        'orbital' - show orbital-projected weight
        'intensity' - weight by proximity to Fermi level
    orbital_focus : dict, optional
        Focus on specific orbitals: {'U1': 1, 'U2': 0, 'Te1': 1, 'Te2': 0}
    colormap : str
        Matplotlib colormap for spectral weight
    alpha_contours : float
        Transparency of Fermi surface contours
    save_dir : str
        Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nCalculating 2D slice with spectral weights at kz = {kz:.3f}...")
    
    # Calculate energies
    energies = band_energies_on_slice(kz)
    
    # Set up orbital weights based on focus
    if orbital_focus is None:
        if weight_type == 'orbital':
            # Default: focus on Te orbitals (more relevant for Fermi surface)
            orbital_weights = {'U1': 0.2, 'U2': 0.2, 'Te1': 1.0, 'Te2': 1.0}
        else:
            orbital_weights = {'U1': 1.0, 'U2': 1.0, 'Te1': 1.0, 'Te2': 1.0}
    else:
        orbital_weights = orbital_focus
    
    # Compute spectral weights
    spectral_weights = compute_spectral_weights(kz, orbital_weights)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.3, 4), dpi=300)
    
    # Create momentum grids for plotting - use physical k-values
    nk = energies.shape[0]
    kx_vals_plot = np.linspace(-1*np.pi/a, 1*np.pi/a, nk)
    ky_vals_plot = np.linspace(-3*np.pi/b, 3*np.pi/b, nk)
    KX_plot, KY_plot = np.meshgrid(kx_vals_plot, ky_vals_plot)
    
    # Plot 1: Traditional Fermi surface contours
    ax1.set_title(f'Fermi surface (E=0) at kz={kz/np.pi*c:.2f}π/c')
    ax1.grid(True, alpha=0.3)
    
    # Set physical axis limits and tick formatting for ax1
    ky_lim_physical = [-3 * np.pi/b, 3 * np.pi/b]
    kx_lim_physical = [-1 * np.pi/a, 1 * np.pi/a]
    ax1.set_xlim(ky_lim_physical)
    ax1.set_ylim(kx_lim_physical)
    
    # Custom tick formatting for ax1
    ky_ticks_ax1 = np.linspace(ky_lim_physical[0], ky_lim_physical[1], 7)
    kx_ticks_ax1 = np.linspace(kx_lim_physical[0], kx_lim_physical[1], 5)
    ky_labels_ax1 = [f'{val/(np.pi/b):.1f}' for val in ky_ticks_ax1]
    kx_labels_ax1 = [f'{val/(np.pi/a):.1f}' for val in kx_ticks_ax1]
    ax1.set_xticks(ky_ticks_ax1)
    ax1.set_yticks(kx_ticks_ax1)
    ax1.set_xticklabels(ky_labels_ax1)
    ax1.set_yticklabels(kx_labels_ax1)
    ax1.set_xlabel('ky (π/b)')
    ax1.set_ylabel('kx (π/a)')
    
    # Plot 2: Spectral weight visualization
    ax2.set_title(f'Spectral weight at kz={kz/np.pi*c:.2f}π/c ({weight_type})')
    ax2.grid(True, alpha=0.3)
    
    # Set same physical axis limits and tick formatting for ax2
    ax2.set_xlim(ky_lim_physical)
    ax2.set_ylim(kx_lim_physical)
    ax2.set_xticks(ky_ticks_ax1)
    ax2.set_yticks(kx_ticks_ax1)
    ax2.set_xticklabels(ky_labels_ax1)
    ax2.set_yticklabels(kx_labels_ax1)
    ax2.set_xlabel('ky (π/b)')
    ax2.set_ylabel('kx (π/a)')
    
    # Band colors and labels
    band_colors = ['#1E88E5', '#42A5F5',  "#9635E5", "#FD0000"]  # Blue for U, Red/Orange for Te
    band_labels = ['Band 1', 'Band 2', 'Band 3', 'Band 4']
    
    plotted_bands = []
    EF = 0.0
    
    for band in range(4):
        Z = energies[:, :, band].T  # Transpose for correct orientation
        
        # Only process bands that cross Fermi level
        if Z.min() <= EF <= Z.max():
            try:
                # Plot Fermi surface contours using physical k-values
                contour1 = ax1.contour(ky_vals_plot, kx_vals_plot, Z, levels=[EF], 
                                     colors=[band_colors[band]], linewidths=2.5, alpha=alpha_contours)
                
                if len(contour1.collections) > 0:
                    ax1.plot([], [], color=band_colors[band], linewidth=2.5, 
                            label=band_labels[band])
                    plotted_bands.append(band)
                    
                    # Calculate spectral weight visualization
                    weight_data = spectral_weights[:, :, band].T  # Transpose for correct orientation
                    
                    if weight_type == 'intensity':
                        # Weight by proximity to Fermi level (inverse of |E - EF|)
                        fermi_proximity = 1.0 / (np.abs(Z) + 0.01)  # Add small constant to avoid division by zero
                        weight_data = weight_data * fermi_proximity
                    
                    # Create spectral weight plot - show more structure
                    if np.max(weight_data) > 0:
                        weight_threshold = np.percentile(weight_data[weight_data > 0], 10)  # Top 90% of non-zero weights
                        weight_data_masked = np.where(weight_data > weight_threshold, weight_data, np.nan)
                        print(f"    {band_labels[band]}: weight range {np.min(weight_data[weight_data > 0]):.6f} to {np.max(weight_data):.6f}")
                    else:
                        weight_data_masked = weight_data
                        print(f"    {band_labels[band]}: no significant spectral weight")
                    
                    # Plot spectral weight as colored contours using physical k-values
                    if not np.all(np.isnan(weight_data_masked)):
                        im = ax2.contourf(ky_vals_plot, kx_vals_plot, weight_data_masked, 
                                        levels=15, cmap=colormap, alpha=0.8)
                    
                    # Overlay Fermi surface contours using physical k-values
                    ax2.contour(ky_vals_plot, kx_vals_plot, Z, levels=[EF], 
                              colors=[band_colors[band]], linewidths=2, alpha=alpha_contours)
                    
                    print(f"  Plotted {band_labels[band]} with spectral weights")
            
            except Exception as e:
                print(f"  Could not plot band {band}: {e}")
        else:
            print(f"  {band_labels[band]} does not cross Fermi level")
    
    # Add legends and colorbars
    if plotted_bands:
        ax1.legend(loc='upper right')
    
    # Add colorbar for spectral weights
    if 'im' in locals():
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label('Spectral Weight', rotation=270, labelpad=20)
    
    # Add orbital weight information
    weight_text = f"Orbital weights: U1={orbital_weights['U1']:.1f}, U2={orbital_weights['U2']:.1f}, " \
                 f"Te1={orbital_weights['Te1']:.1f}, Te2={orbital_weights['Te2']:.1f}"
    fig.text(0.02, 0.02, weight_text, fontsize=8, style='italic')
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"fermi_surface_spectral_weight_{weight_type}_kz_{kz/np.pi*c:.3f}.png"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Spectral weight plot saved: {filename}")
    
    plt.show()

def generate_spectral_weight_plots(save_dir='outputs/ute2_spectral'):
    """Generate different types of orbital-projected spectral weight visualizations."""
    print("\n" + "="*70)
    print("GENERATING ORBITAL-PROJECTED SPECTRAL WEIGHT VISUALIZATIONS")
    print("="*70)
    
    # Test different energy windows
    energy_windows = [0.02, 0.05, 0.1]  # Different energy windows around Fermi level
    
    for i, energy_window in enumerate(energy_windows):
        print(f"\n--- Energy window: ±{energy_window:.3f} eV around E_F ---")
        
        # 1. Total spectral weight (all orbitals)
        print(f"\n{i+1}.1 Total spectral weight (equal orbital weights)")
        plot_2d_with_spectral_weight(kz=0.0, weight_type='total', 
                                    colormap='viridis', alpha_contours=0.9, 
                                    save_dir=save_dir, energy_window=energy_window)
        
        # 2. Te orbital focused (relevant for Fermi surface in UTe2)
        print(f"\n{i+1}.2 Te orbital contributions only")
        plot_2d_with_spectral_weight(kz=0.0, weight_type='Te_only',
                                    colormap='plasma', alpha_contours=0.8, 
                                    save_dir=save_dir, energy_window=energy_window)
        
        # 3. U orbital focused (for comparison)
        print(f"\n{i+1}.3 U orbital contributions only")
        plot_2d_with_spectral_weight(kz=0.0, weight_type='U_only',
                                    colormap='cool', alpha_contours=0.8, 
                                    save_dir=save_dir, energy_window=energy_window)
        
        # 4. Te/U orbital ratio (shows which parts of FS are more Te-like vs U-like)
        print(f"\n{i+1}.4 Te/U spectral weight ratio")
        plot_2d_with_spectral_weight(kz=0.0, weight_type='orbital_ratio',
                                    colormap='RdBu_r', alpha_contours=0.7, 
                                    save_dir=save_dir, energy_window=energy_window)
    
    print("\n" + "="*70)
    print("All spectral weight visualizations completed!")
    print(f"Plots saved to: {save_dir}")
    print("="*70)

# Global cache for 3D band energies (to avoid recomputing for each pairing type)
_cached_band_energies_3d = None
_cached_kx_3d = None
_cached_ky_3d = None
_cached_kz_3d = None
_cached_nk3d = None

def plot_011_fermi_surface_projection(save_dir='outputs/ute2_fixed', show_gap_nodes=True, 
                                     gap_node_pairing='B3u', angle_deg=24.0):
    """
    Project 3D Fermi surface onto (0-11) crystallographic plane.
    
    Creates a proper crystallographic projection showing kx vs kc* where:
    - y-axis: kx (in π/a units)  
    - x-axis: kc* (projected coordinate in π/c* units)
    - Shows both Fermi surface bands combined like in 3D plots
    - Can overlay B2u and B3u gap nodes
    
    Parameters:
    -----------
    save_dir : str
        Directory to save output files
    show_gap_nodes : bool
        Whether to compute and display gap nodes
    gap_node_pairing : str
        Which pairing symmetry to show gap nodes for ('B2u', 'B3u', 'both')
    angle_deg : float
        Angle between normal to (0-11) plane and b-axis (default: 24°)
    """
    import os
    import pickle
    import hashlib
    
    print(f"\n" + "="*70)
    print(f"GENERATING (0-11) PLANE PROJECTION - CRYSTALLOGRAPHIC VIEW")
    print(f"Angle: {angle_deg}° between normal and b-axis")
    print("="*70)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Use LOWER resolution for projection to avoid memory issues
    # Extended ky range creates huge meshes, so reduce grid size significantly
    nk3d = 121  # Reduced from 129 to avoid memory crash with extended ky

    # Create cache filename based on grid parameters (use extended ky for projection)
    cache_key = f"{nk3d}_{a:.6f}_{b:.6f}_{c:.6f}_extended_ky"  # Different cache for extended ky
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_dir = os.path.join('outputs', 'global_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'band_energies_3d_projection_{cache_hash}.pkl')
    
    # Load or compute 3D band energies
    if os.path.exists(cache_file):
        print("Loading 3D band energies from cache...")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            band_energies_3d = cache_data['band_energies_3d']
            kx_3d = cache_data['kx_3d'] 
            ky_3d = cache_data['ky_3d']
            kz_3d = cache_data['kz_3d']
        print("Using cached 3D band energies")
    else:
        print("Computing 3D band energies with extended ky range for projection...")
        # Extended ky range for projection to capture full Fermi surface
        kx_3d = np.linspace(-np.pi/a, np.pi/a, nk3d)
        ky_3d = np.linspace(-np.pi/b, np.pi/b, nk3d)  # Extended ky range for projection
        kz_3d = np.linspace(-np.pi/c, np.pi/c, nk3d)

        KX,KY,KZ = np.meshgrid(kx_3d, ky_3d, kz_3d, indexing='ij')
        band_energies_3d = np.zeros((nk3d, nk3d, nk3d, 4))
        
        H = H_full(KX, KY, KZ)
        eigvals = np.linalg.eigvals(H)
        band_energies_3d = np.sort(np.real(eigvals))
        
        # Cache the results
        cache_data = {
            'band_energies_3d': band_energies_3d,
            'kx_3d': kx_3d,
            'ky_3d': ky_3d, 
            'kz_3d': kz_3d
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Cached 3D band energies to {cache_file}")
    
    # Define (0-11) plane transformation
    angle_rad = np.deg2rad(angle_deg)
    
    print(f"Projection setup:")
    print(f"  x-axis: kc* = ky*sin({angle_deg}°) + kz*cos({angle_deg}°)")
    print(f"  y-axis: kx (unchanged)")
    print(f"  Note: c* ~ {c_star:.2f} nm is the projected lattice spacing in (0-11) plane")
    
    # Project 3D Fermi surface to 2D using scatter points (original working method)
    print("Projecting 3D Fermi surface to (0-11) plane...")
    
    # Extract Fermi surfaces using marching cubes (same as 3D plot)
    EF = 0.0
    colors = ['#1E88E5', '#42A5F5', "#9635E5", "#FD0000"] 
    band_labels = ['Band 1', 'Band 2', 'Band 3', 'Band 4']
    
    fermi_surfaces = {}
    
    for band in [2, 3]:  # Only bands that cross Fermi level
        try:
            print(f"  Extracting Band {band+1} Fermi surface...")
            verts, faces, normals, values = measure.marching_cubes(
                band_energies_3d[:,:,:,band], level=EF, 
                spacing=(kx_3d[1]-kx_3d[0], ky_3d[1]-ky_3d[0], kz_3d[1]-kz_3d[0])
            )
            
            print(f"    Initial mesh: {len(verts)} vertices, {len(faces)} faces")
            
            # Decimate mesh if too large (critical for memory)
            if len(faces) > 20000:
                print(f"    Decimating mesh (too large for rendering)...")
                if TRIMESH_AVAILABLE:
                    try:
                        # Create trimesh object
                        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                        # Decimate to target face count
                        target_faces = 15000
                        mesh = mesh.simplify_quadric_decimation(0.3)
                        verts = mesh.vertices
                        faces = mesh.faces
                        print(f"    After decimation: {len(verts)} vertices, {len(faces)} faces")
                    except Exception as e:
                        print(f"    Warning: decimation failed ({e})")
                        # Skip if too large and decimation failed
                        if len(faces) > 50000:
                            print(f"    ERROR: Mesh too large ({len(faces)} faces), skipping this band")
                            continue
                else:
                    print("    Warning: trimesh not available, skipping decimation")
                    # Skip if too large and trimesh not available
                    if len(faces) > 50000:
                        print(f"    ERROR: Mesh too large ({len(faces)} faces), skipping this band")
                        continue
            
            # Convert vertices to physical k-space coordinates (same as 3D plotting)
            verts_physical = np.zeros_like(verts)
            verts_physical[:, 0] = verts[:, 0] + kx_3d[0]  # kx
            verts_physical[:, 1] = verts[:, 1] + ky_3d[0]  # ky  
            verts_physical[:, 2] = verts[:, 2] + kz_3d[0]  # kz
            
            # Project to (0-11) coordinates
            kx_fs = verts_physical[:, 0]
            ky_fs = verts_physical[:, 1]
            kz_fs = verts_physical[:, 2]
            
            # Projection transformation
            kc_star = ky_fs * np.sin(angle_rad) + kz_fs * np.cos(angle_rad)
            kc_star_norm = kc_star / (np.pi/c_star)
            kc_star_norm /= np.max(np.abs(kc_star_norm))
            kx_unchanged = kx_fs
            
            # Store 2D projected vertices and original 3D faces
            verts_2d = np.column_stack((
                kc_star_norm,  # x-axis: kc* in π/c* units
                kx_unchanged / (np.pi/a)    # y-axis: kx in π/a units
            ))
            
            fermi_surfaces[f'band_{band}'] = {
                'verts_2d': verts_2d,  # Projected 2D vertices (N, 2)
                'faces': faces,         # Original 3D mesh topology (M, 3)
                'color': colors[band],
                'label': band_labels[band]
            }
            plot_011_density_map(fermi_surfaces, save_path='outputs/FS_density_map.png', grid_res=3000, sigma=0.009)
            print(f"    Ready to render: {len(verts)} vertices, {len(faces)} faces")
            
        except Exception as e:
            print(f"    No Fermi surface found for Band {band+1}: {e}")
    

    
    # Determine which pairing symmetries to plot
    if show_gap_nodes and gap_node_pairing in ['B2u', 'B3u', 'both']:
        pairing_types = [gap_node_pairing] if gap_node_pairing != 'both' else ['B2u', 'B3u']
    else:
        pairing_types = []
    
    # Create separate plots for each pairing symmetry (or combined if single type)
    plots_to_make = pairing_types if gap_node_pairing == 'both' else [gap_node_pairing if show_gap_nodes else 'no_nodes']
    
    for plot_type in plots_to_make:
        print(f"Creating plot for: {plot_type}")
        
        # Create new figure for this plot
        fig, ax = plt.subplots(figsize=(8, 6.4), dpi=300)
        
        # Plot Fermi surfaces using PolyCollection with original mesh topology
        from matplotlib.collections import PolyCollection
        
        # Track extent for axis limits
        all_x_coords = []
        all_y_coords = []
        
        for band_key, fs_data in fermi_surfaces.items():
            print(f"  Rendering {fs_data['label']} with original mesh topology...")
            
            # Get projected 2D vertices and original 3D faces
            verts_2d = fs_data['verts_2d']  # Shape: (N_vertices, 2)
            faces = fs_data['faces']        # Shape: (N_faces, 3) - indices into verts_2d
            
            # Track coordinates for axis limits (sample to save memory)
            sample_size = min(len(verts_2d), 1000)
            sample_idx = np.linspace(0, len(verts_2d)-1, sample_size, dtype=int)
            all_x_coords.extend(verts_2d[sample_idx, 0])
            all_y_coords.extend(verts_2d[sample_idx, 1])
            
            # Create list of triangles using original faces
            # Each triangle is defined by the 2D coordinates of its 3 vertices
            print(f"    Creating triangle array ({len(faces)} triangles)...")
            triangles = verts_2d[faces]  # Shape: (N_faces, 3, 2)
            
            # Create PolyCollection for efficient rendering
            # Low alpha allows seeing through to overlapping surfaces
            print(f"    Building PolyCollection...")
            poly_col = PolyCollection(triangles,
                                     facecolors=fs_data['color'],
                                     edgecolors='none',
                                     alpha=0.3,  # Slightly higher alpha for better visibility
                                     linewidths=0,
                                     antialiaseds=False)  # Disable AA for speed
            ax.add_collection(poly_col)
            
            # Add dummy line for legend
            ax.plot([], [], color=fs_data['color'], linewidth=4, alpha=0.8,
                   label=fs_data['label'])
            
            print(f"     {fs_data['label']}: Rendered {len(faces)} triangles (preserved 3D topology)")
        
        # Set initial axis limits to exactly -1 to 1 (will be maintained throughout)
        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(-1.0, 1.0)
        print(f"  Set axis limits: kc* [-1.0, 1.0], kx [-1.0, 1.0]")
        
        # Add gap nodes for this specific pairing symmetry
        if plot_type in ['B2u', 'B3u', 'B3u_middle', 'B3u_first', 'B3u_last']:
            # Extract base pairing type
            base_pairing = 'B3u' if 'B3u' in plot_type else plot_type
            selection_method = 'middle'  # default
            if 'B3u_first' in plot_type:
                selection_method = 'first'
            elif 'B3u_last' in plot_type:
                selection_method = 'last'
            elif 'B3u_middle' in plot_type:
                selection_method = 'middle'
                
            print(f"  Computing and projecting {base_pairing} gap nodes...")
            
            # TWO-TIER CACHING SYSTEM:
            # 1. Raw nodes cache (expensive 129³ calculation) - shared between runs
            # 2. Final nodes cache (quick selection/clustering) - can regenerate easily
            
            raw_nodes_cache_file = os.path.join(save_dir, 'cache', f'gap_nodes_raw_{base_pairing}.npz')
            loaded_raw_from_cache = False
            
            # Try to load RAW nodes (before selection) from cache
            if os.path.exists(raw_nodes_cache_file):
                print(f"  Loading raw {base_pairing} gap nodes from cache...")
                try:
                    raw_cache = np.load(raw_nodes_cache_file)
                    all_nodes_kx = raw_cache['kx']
                    all_nodes_ky = raw_cache['ky']
                    all_nodes_kz = raw_cache['kz']
                    all_nodes_gap = raw_cache['gap']
                    all_nodes_band = raw_cache['band']
                    print(f"     Loaded {len(all_nodes_kx)} raw nodes from cache (skipping 129³ calculation)")
                    loaded_raw_from_cache = True
                except Exception as e:
                    print(f"     Raw cache load failed: {e}")
                    loaded_raw_from_cache = False
            
            # Compute gap nodes if not cached (initialize variables)
            if not loaded_raw_from_cache:
                print("  Using projection-optimized gap node detection...")
                print("  Using slice-based method (same as 3D plots)...")
                
                # Use HIGH resolution for accurate boundary node detection
                nk_gap = 129  # Same as working 3D plots - critical for boundary nodes
                
                # STANDARD 3D BRILLOUIN ZONE: [-1, 1] in units of π/a, π/b, π/c
                # Only search within the standard 3D box - do NOT extend beyond
                kx_gap = np.linspace(-np.pi/a, np.pi/a, nk_gap)
                ky_gap = np.linspace(-np.pi/b, np.pi/b, nk_gap)
                kz_gap = np.linspace(-np.pi/c, np.pi/c, nk_gap)
                
                print(f"    Gap node search grid: STANDARD 3D BZ [-1,1] on all axes ({nk_gap}³ = {nk_gap**3:,} points)")
                print(f"    Estimated time: ~5-7 minutes")
                
                # Compute band energies
                print(f"    Computing band energies on {nk_gap}³ k-grid...")
                band_energies_gap = np.zeros((nk_gap, nk_gap, nk_gap, 4))
                
                for i, kx in enumerate(kx_gap):
                    for j, ky in enumerate(ky_gap):
                        for k, kz in enumerate(kz_gap):
                            H = H_full(kx, ky, kz)
                            eigvals = np.linalg.eigvals(H)
                            band_energies_gap[i, j, k, :] = np.sort(np.real(eigvals))
                    
                    if (i + 1) % 16 == 0:
                        print(f"      Band energies: {(i+1)/nk_gap*100:.0f}% complete")
                
                # Compute 3D gap magnitude
                gap_3d = np.zeros((nk_gap, nk_gap, nk_gap))
                print(f"    Computing {plot_type} gap magnitude on {nk_gap}³ grid...")
                
                for i, kx in enumerate(kx_gap):
                    for j, ky in enumerate(ky_gap):
                        # Vectorize over kz direction
                        gap_stack = np.array([calculate_gap_magnitude(
                            np.array([[kx]]), np.array([[ky]]), kz, pairing_type=base_pairing)[0, 0] 
                            for kz in kz_gap])
                        gap_3d[i, j, :] = gap_stack
                    
                    if (i + 1) % 16 == 0:
                        print(f"      {plot_type} gap: {(i+1)/nk_gap*100:.0f}% complete")
                
                print(f"      {plot_type} gap range: {np.min(gap_3d):.6f} to {np.max(gap_3d):.6f}")
                
                # Set threshold based on pairing type (STRICT for projection to get true nodes)
                if plot_type == 'B2u':
                    gap_threshold = 0.06 * np.max(gap_3d)  # Strict - only nodes very close to zero
                    fermi_threshold = 0.010  # 10 meV - strict Fermi surface crossing
                elif plot_type == 'B3u':
                    gap_threshold = 0.06 * np.max(gap_3d)  # Strict - match B2u
                    fermi_threshold = 0.010  # 10 meV - strict Fermi surface crossing
                else:
                    gap_threshold = 0.05 * np.max(gap_3d)
                    fermi_threshold = 0.010  # 10 meV
                
                # Collect nodes from all slices (same method as 3D plot)
                all_nodes_kx = []
                all_nodes_ky = []
                all_nodes_kz = []
                all_nodes_gap = []
                all_nodes_band = []
                
                # For B2u, ensure we check the exact nodal planes
                critical_kz_indices = []
                if plot_type == 'B2u':
                    # Find indices closest to kz = 0, ±π/c
                    idx_0 = np.argmin(np.abs(kz_gap - 0.0))
                    idx_plus = np.argmin(np.abs(kz_gap - np.pi/c))
                    idx_minus = np.argmin(np.abs(kz_gap - (-np.pi/c)))
                    critical_kz_indices = [idx_0, idx_plus, idx_minus]
                
                print(f"      Scanning {nk_gap} kz slices...")
                
                for band in [2, 3]:  # Bands 3 and 4
                    band_nodes_found = 0
                    
                    for iz, kz_val in enumerate(kz_gap):
                        # Extract 2D slice at this kz from gap-specific energy data
                        energy_slice = band_energies_gap[:, :, iz, band]  # Shape: (nk3d, nk3d)
                        gap_slice = gap_3d[:, :, iz]  # Shape: (nk3d, nk3d)
                        
                        # Check if this is a critical kz plane for B2u
                        is_critical_kz = (plot_type == 'B2u' and iz in critical_kz_indices)
                        
                        # Use more generous threshold at critical planes
                        effective_fermi_threshold = fermi_threshold
                        if is_critical_kz:
                            effective_fermi_threshold = fermi_threshold * 2.0
                        
                        # Check if Fermi surface crosses this slice
                        if energy_slice.min() > effective_fermi_threshold or energy_slice.max() < -effective_fermi_threshold:
                            continue
                        
                        # Find intersections in this 2D slice
                        fermi_mask = np.abs(energy_slice) < effective_fermi_threshold
                        gap_mask = gap_slice < gap_threshold
                        intersection = fermi_mask & gap_mask
                        
                        if np.any(intersection):
                            # Get indices of intersection points
                            ix_slice, iy_slice = np.where(intersection)
                            
                            # Apply 2D clustering within this slice
                            selected_2d = []
                            min_pixel_dist = 8  # Same as 3D plot
                            
                            for i in range(len(ix_slice)):
                                ix_pt, iy_pt = ix_slice[i], iy_slice[i]
                                
                                # Check distance to already selected points in this slice
                                too_close = False
                                for (prev_ix, prev_iy) in selected_2d:
                                    if np.sqrt((ix_pt - prev_ix)**2 + (iy_pt - prev_iy)**2) < min_pixel_dist:
                                        too_close = True
                                        break
                                
                                if not too_close:
                                    selected_2d.append((ix_pt, iy_pt))
                                    
                                    # Convert to physical coordinates using gap k-space arrays
                                    kx_node = kx_gap[ix_pt]
                                    ky_node = ky_gap[iy_pt]
                                    gap_value = gap_slice[ix_pt, iy_pt]
                                    
                                    # Store this node
                                    all_nodes_kx.append(kx_node)
                                    all_nodes_ky.append(ky_node)
                                    all_nodes_kz.append(kz_val)
                                    all_nodes_gap.append(gap_value)
                                    all_nodes_band.append(band)
                                    band_nodes_found += 1
                
                # Apply 3D clustering (same as 3D plot)
                if len(all_nodes_kx) > 0:
                    all_nodes_kx = np.array(all_nodes_kx)
                    all_nodes_ky = np.array(all_nodes_ky)
                    all_nodes_kz = np.array(all_nodes_kz)
                    all_nodes_gap = np.array(all_nodes_gap)
                    all_nodes_band = np.array(all_nodes_band)
                    
                    print(f"      Total nodes from slices: {len(all_nodes_kx)}")
                    
                    # Save RAW nodes to cache (before selection/clustering)
                    if not loaded_raw_from_cache:
                        try:
                            os.makedirs(os.path.dirname(raw_nodes_cache_file), exist_ok=True)
                            np.savez(raw_nodes_cache_file,
                                    kx=all_nodes_kx, ky=all_nodes_ky, kz=all_nodes_kz,
                                    gap=all_nodes_gap, band=all_nodes_band)
                            print(f"       Saved {len(all_nodes_kx)} raw nodes to cache")
                        except Exception as e:
                            print(f"       Failed to save raw nodes cache: {e}")
            
            # SELECTION AND CLUSTERING (runs whether loaded from cache or freshly computed)
            if len(all_nodes_kx) > 0:
                # STEP 1: Project all nodes to (0-11) plane
                print(f"      Projecting to (0-11) and filtering to visible range...")
                kc_star_all = all_nodes_ky * np.sin(angle_rad) + all_nodes_kz * np.cos(angle_rad)
                kx_proj_all = all_nodes_kx / (np.pi/a)  # Convert to π/a units
                kc_star_all = kc_star_all / (np.pi/c_star)  # Convert to π/c* units
                
                # STEP 2: Filter to visible range [-1, 1] in BOTH axes
                margin = 0.05  # 5% margin for boundary nodes
                visible_mask = (np.abs(kx_proj_all) <= 1.0 + margin) & (np.abs(kc_star_all) <= 1.0 + margin)
                visible_indices = np.where(visible_mask)[0]
                
                print(f"        Visible nodes in [-1,1] range: {len(visible_indices)}/{len(all_nodes_kx)}")
                
                # STEP 3: Cluster within visible nodes only, stratified by kx
                if base_pairing == 'B2u':
                    # Target: 6 nodes at each of kx=-1, 0, +1 (18 total)
                    # Need to allow side-by-side nodes at the center
                    kx_groups = [
                        ('kx=-1', np.abs(kx_proj_all + 1.0) < 0.15),  # Near -1
                        ('kx=0',  np.abs(kx_proj_all) < 0.15),        # Near 0
                        ('kx=+1', np.abs(kx_proj_all - 1.0) < 0.15),  # Near +1
                    ]
                    target_per_group = [6, 6, 6]  # 6 nodes per kx region
                    min_distance_2d = 0.03  # Very minimal clustering to preserve side-by-side nodes (in π/a, π/c* units)
                    
                elif base_pairing == 'B3u':
                    # B3u: d ∝ (0, sin(kz·c), sin(ky·b)) → zeros at kz = 0, ±π/c, ky = 0, ±π/b
                    # In (0-11) projection: kc* = ky*sin(angle) + kz*cos(angle)
                    # Target: 16 total nodes
                    #   - 4 along kc*=0
                    #   - 2 at kc*=-1 (boundary)
                    #   - 2 at kc*=+1 (boundary)
                    #   - 4 between kc*=0 and kc*=-1
                    #   - 4 between kc*=0 and kc*=+1
                    kx_groups = [
                        ('kc*=0', np.abs(kc_star_all) < 0.1),                    # Near kc* = 0 (center line): 4 nodes
                        ('kc*=-1', np.abs(kc_star_all + 1.0) < 0.1),             # Near kc* = -1 (boundary): 2 nodes
                        ('kc*=+1', np.abs(kc_star_all - 1.0) < 0.1),             # Near kc* = +1 (boundary): 2 nodes
                        ('0<kc*<-1', (kc_star_all < -0.1) & (kc_star_all > -0.9)),  # Between 0 and -1: 4 nodes
                        ('0<kc*<+1', (kc_star_all > 0.1) & (kc_star_all < 0.9)),    # Between 0 and +1: 4 nodes
                    ]
                    target_per_group = [4, 2, 2, 4, 4]  # 16 total nodes
                    min_distance_2d = 0.03  # Allow vertical stacking
                else:
                    kx_groups = [('all', np.ones(len(all_nodes_kx), dtype=bool))]
                    target_per_group = [20]  # Make it a list for consistency
                    min_distance_2d = 0.05
                
                # Ensure target_per_group is a list
                if not isinstance(target_per_group, list):
                    target_per_group = [target_per_group] * len(kx_groups)
                
                # Select nodes closest to key kc* values within each group
                clustered_indices = []
                for group_idx, (group_name, group_mask) in enumerate(kx_groups):
                    # Get the target for this specific group
                    current_target = target_per_group[group_idx] if group_idx < len(target_per_group) else target_per_group[0]
                    
                    # Get indices that are both visible AND in this kx group
                    group_visible_mask = visible_mask & group_mask
                    group_indices = np.where(group_visible_mask)[0]
                    
                    if len(group_indices) == 0:
                        print(f"        {group_name}: 0 visible nodes")
                        continue
                    
                    # For each group, select nodes closest to key kc* values
                    group_kx = kx_proj_all[group_indices]
                    group_kc = kc_star_all[group_indices]
                    
                    if base_pairing == 'B2u':
                        if 'kx=-1' in group_name:
                            distances = np.abs(group_kx - (-1.0))
                            sorted_indices = np.argsort(distances)
                        elif 'kx=0' in group_name:
                            distances = np.abs(group_kx - 0.0)
                            sorted_indices = np.argsort(distances)
                        elif 'kx=+1' in group_name:
                            distances = np.abs(group_kx - (+1.0))
                            sorted_indices = np.argsort(distances)
                        else:
                            sorted_indices = np.arange(len(group_indices))
                    elif base_pairing == 'B3u':
                        if 'kc*=0' in group_name:
                            # For kc*=0: select nodes closest to kc*=0, spread across kx
                            distances = np.abs(group_kc - 0.0)
                            sorted_indices = np.argsort(distances)
                        elif 'kc*=-1' in group_name:
                            # For kc*=-1: select nodes closest to kc*=-1 boundary
                            distances = np.abs(group_kc - (-1.0))
                            sorted_indices = np.argsort(distances)
                        elif 'kc*=+1' in group_name:
                            # For kc*=+1: select nodes closest to kc*=+1 boundary
                            distances = np.abs(group_kc - (+1.0))
                            sorted_indices = np.argsort(distances)
                        elif '0<kc*<-1' in group_name:
                            # Between 0 and -1: spread evenly, prioritize kc* ~ -0.5
                            distances = np.abs(group_kc - (-0.5))
                            sorted_indices = np.argsort(distances)
                        elif '0<kc*<+1' in group_name:
                            # Between 0 and +1: spread evenly, prioritize kc* ~ +0.5
                            distances = np.abs(group_kc - (+0.5))
                            sorted_indices = np.argsort(distances)
                        else:
                            sorted_indices = np.arange(len(group_indices))
                    else:
                        # Default for other pairing types
                        if 'kc*=0' in group_name:
                            distances = np.abs(group_kc - 0.0)
                            sorted_indices = np.argsort(distances)
                        elif 'kc*<0' in group_name:
                            distances = np.abs(group_kc - (-1.0))
                            sorted_indices = np.argsort(distances)
                        elif 'kc*>0' in group_name:
                            distances = np.abs(group_kc - (+1.0))
                            sorted_indices = np.argsort(distances)
                        else:
                            sorted_indices = np.arange(len(group_indices))
                    
                    # Apply minimal clustering to avoid duplicates
                    group_clustered = []
                    for idx in sorted_indices:
                        i = group_indices[idx]
                        kx_i, kc_i = kx_proj_all[i], kc_star_all[i]
                        
                        # Check distance to already selected nodes
                        too_close = False
                        for j in group_clustered:
                            kx_j, kc_j = kx_proj_all[j], kc_star_all[j]
                            distance_2d = np.sqrt((kx_i - kx_j)**2 + (kc_i - kc_j)**2)
                            if distance_2d < min_distance_2d:
                                too_close = True
                                break
                        
                        if not too_close:
                            group_clustered.append(i)
                            if len(group_clustered) >= current_target:
                                break
                    
                    clustered_indices.extend(group_clustered)
                    print(f"        {group_name}: selected {len(group_clustered)}/{len(group_indices)} visible nodes")
                
                # Keep only clustered nodes (in original 3D coordinates for saving)
                final_kx_nodes = all_nodes_kx[clustered_indices]
                final_ky_nodes = all_nodes_ky[clustered_indices]
                final_kz_nodes = all_nodes_kz[clustered_indices]
                
                print(f"      Final selection: {len(final_kx_nodes)} nodes")
                print(f"    Found {len(final_kx_nodes)} {plot_type} gap nodes")
                
                # Note: We use raw node cache for fast iteration
                # Final selected nodes are recomputed each time (fast operation)
            else:
                print(f"    No {base_pairing} gap nodes found")
                final_kx_nodes = np.array([])
                final_ky_nodes = np.array([])
                final_kz_nodes = np.array([])
            
            # =========================================================
            # STRICT 3D BOUNDS PROJECTION (NO EXTENDED ZONE)
            # =========================================================
            # Project gap nodes to (0-11) coordinates, but ONLY those within standard 3D BZ
            if len(final_kx_nodes) > 0:
                print(f"      Projecting strictly [-1, 1] 3D nodes to (0-11) plane...")
                
                # 1. Define the limits in physical units
                lim_kx = np.pi / a
                lim_ky = np.pi / b
                lim_kz = np.pi / c
                
                # 2. Create a mask for nodes inside the standard 3D box [-1, 1] on all axes
                source_mask = (
                    (np.abs(final_kx_nodes) <= lim_kx) & 
                    (np.abs(final_ky_nodes) <= lim_ky) & 
                    (np.abs(final_kz_nodes) <= lim_kz)
                )
                
                # 3. Apply the mask to get only valid source nodes
                k_source_x = final_kx_nodes[source_mask]
                k_source_y = final_ky_nodes[source_mask]
                k_source_z = final_kz_nodes[source_mask]
                
                print(f"        Kept {len(k_source_x)}/{len(final_kx_nodes)} nodes within standard 3D limits")
                
                if len(k_source_x) > 0:
                    # 4. Apply the Rotation Formula (Projection)
                    kc_star_nodes = k_source_y * np.sin(angle_rad) + k_source_z * np.cos(angle_rad)
                    kx_proj_nodes = k_source_x  # Unchanged
                    
                    # 5. Normalize to plot units (π/a and π/c*)
                    kx_proj_nodes = kx_proj_nodes / (np.pi/a)  # π/a units
                    kc_star_nodes = kc_star_nodes / (np.pi/c_star)  # π/c* units
                else:
                    print("        Warning: No nodes found within the 3D limits!")
                    kx_proj_nodes = np.array([])
                    kc_star_nodes = np.array([])
            else:
                kx_proj_nodes = np.array([])
                kc_star_nodes = np.array([])
            
            # Diagnostic output for projected nodes
            if len(kx_proj_nodes) > 0:
                print(f"    All gap nodes after projection:")
                print(f"      kx range: [{np.min(kx_proj_nodes):.3f}, {np.max(kx_proj_nodes):.3f}]")
                print(f"      kc* range: [{np.min(kc_star_nodes):.3f}, {np.max(kc_star_nodes):.3f}]")
                print(f"      Nodes near kx boundaries (|kx| > 0.95):")
                near_kx_boundary = np.abs(kx_proj_nodes) > 0.95
                if np.any(near_kx_boundary):
                    for kx, kc in zip(kx_proj_nodes[near_kx_boundary], kc_star_nodes[near_kx_boundary]):
                        print(f"        kx={kx:+.3f}, kc*={kc:+.3f}")
                else:
                    print(f"        None found")
                print(f"      Nodes near kc* boundaries (|kc*| > 0.95):")
                near_kc_boundary = np.abs(kc_star_nodes) > 0.95
                if np.any(near_kc_boundary):
                    for kx, kc in zip(kx_proj_nodes[near_kc_boundary], kc_star_nodes[near_kc_boundary]):
                        print(f"        kx={kx:+.3f}, kc*={kc:+.3f}")
                else:
                    print(f"        None found")
                
                # Filter nodes to only show those within -1 to 1 range in BOTH axes
                # Use relaxed bounds to catch nodes at zone boundaries
                margin = 0.05  # Larger margin to catch boundary nodes (5% tolerance)
                in_range_mask = (np.abs(kx_proj_nodes) <= 1.0 + margin) & (np.abs(kc_star_nodes) <= 1.0 + margin)
                kx_proj_filtered = kx_proj_nodes[in_range_mask]
                kc_star_filtered = kc_star_nodes[in_range_mask]
                
                print(f"    Filtered gap nodes in [-1, 1] range:")
                print(f"      Total: {len(kx_proj_filtered)}/{len(kx_proj_nodes)}")
                if len(kx_proj_filtered) > 0:
                    print(f"      kx range: [{np.min(kx_proj_filtered):.3f}, {np.max(kx_proj_filtered):.3f}]")
                    print(f"      kc* range: [{np.min(kc_star_filtered):.3f}, {np.max(kc_star_filtered):.3f}]")
                    for i, (kx_node, kc_node) in enumerate(zip(kx_proj_filtered, kc_star_filtered)):
                        print(f"        Node {i+1}: kc*={kc_node:+.3f}, kx={kx_node:+.3f}")
                
                # Plot only gap nodes within the -1 to 1 range
                if len(kx_proj_filtered) > 0:
                    ax.scatter(kc_star_filtered, kx_proj_filtered, 
                              c='yellow', s=100, marker='o', edgecolors='black', linewidth=2.0,
                              label=f'{plot_type} Gap Nodes', 
                              zorder=100,  # Very high zorder to appear on top of axes
                              clip_on=False)  # Allow nodes at edges to be visible outside axis box
                else:
                    print(f"    WARNING: No {plot_type} gap nodes found in visible range!")
                
                # Set fixed limits to exactly -1 to 1
                current_kc_lim = 1.0
                current_kx_lim = 1.0
            else:
                print(f"    No {plot_type} gap nodes found")
        
        # Apply modern font hierarchy: 12pt biggest, 11pt medium, 10pt smallest
        ax.set_xlabel(r'$k_c^*$ ($\pi/c^*$ units)', fontsize=11)  # Medium font for axis labels
        ax.set_ylabel(r'$k_x$ ($\pi/a$ units)', fontsize=11)      # Medium font for axis labels
        
        # Set title based on plot type
        if plot_type == 'no_nodes':
            title_suffix = ''
        elif 'B3u' in plot_type and '_' in plot_type:
            method = plot_type.split('_')[1]
            title_suffix = f' with B3u Gap Nodes ({method} selection)'
        else:
            title_suffix = f' with {plot_type} Gap Nodes'
            
        ax.set_title(f'UTe₂ Fermi Surface - (0-11) Crystallographic Projection{title_suffix}\n' + 
                    f'Angle: {angle_deg}° between normal and b-axis', fontsize=12)  # Biggest font for title
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)  # Smallest font for legend
        
        # Set tick label font sizes (smallest)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Set axis limits to exactly -1 to 1 (no padding)
        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(-1.0, 1.0)
        
        # Set aspect ratio based on physical dimensions
        # In plot coordinates (π/c* vs π/a units), but physical lengths are different
        # π/a corresponds to a/2 = 0.407/2 = 0.2035 nm physical distance
        # π/c* corresponds to c*/2 = 0.76/2 = 0.38 nm physical distance
        # aspect = (y_physical / y_range) / (x_physical / x_range)
        # Since both ranges are 2.0 (from -1 to 1), aspect = y_physical / x_physical
        physical_aspect = (np.pi/a) / (np.pi/c_star)  # = c*/a
        ax.set_aspect(physical_aspect)
        print(f"    Set axis limits: kc* [-1.0, 1.0], kx [-1.0, 1.0]")
        print(f"    Physical aspect ratio: {physical_aspect:.3f} (kx axis is {physical_aspect:.2f}x longer)")
        
        plt.tight_layout()
        
        # Save the plot
        filename = f'fermi_surface_011_crystallographic_projection_{angle_deg}deg'
        if plot_type != 'no_nodes':
            filename += f'_{plot_type}_nodes'
        filename += '.png'
        
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close figure to free memory, don't show (causes crashes)
        
        print(f"  Saved plot to {filepath}")
    
    print("="*70)

def plot_011_projection_comparison(save_dir='outputs/ute2_fixed', angle_deg=24.0):
    """Create side-by-side comparison of (0-11) projections with modern formatting."""
    import os
    import pickle
    import hashlib
    
    print(f"\\n" + "="*70)
    print(f"GENERATING (0-11) PROJECTION COMPARISON")
    print(f"Angle: {angle_deg}° between normal and b-axis")
    print("="*70)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # This is a simplified version - the full computation logic would be copied from plot_011_fermi_surface_projection
    # For now, let's create a test version that calls the existing function
    
    print("Creating individual plots first...")
    plot_011_fermi_surface_projection(save_dir=save_dir, show_gap_nodes=False, angle_deg=angle_deg)
    plot_011_fermi_surface_projection(save_dir=save_dir, show_gap_nodes=True, gap_node_pairing='B2u', angle_deg=angle_deg) 
    plot_011_fermi_surface_projection(save_dir=save_dir, show_gap_nodes=True, gap_node_pairing='B3u', angle_deg=angle_deg)
    
    print("\\n✓ Individual (0-11) projection plots created with modern formatting!")
    print("\\nTo create a proper side-by-side version, we would need to:")
    print("  1. Copy the full computation logic from plot_011_fermi_surface_projection")
    print("  2. Create a 1x3 subplot layout with figsize=(9, 6)")
    print("  3. Add panel labels a), b), c)")
    print("  4. Create unified legend")

def plot_3d_fermi_surface(save_dir='outputs/ute2_fixed', show_gap_nodes=True, 
                         gap_node_pairing='B3u', orientation='standard'):
    """
    Plot 3D Fermi surfaces with optional gap node visualization.
    
    Parameters:
    -----------
    save_dir : str
        Directory to save output files
    show_gap_nodes : bool
        Whether to compute and display gap nodes
    gap_node_pairing : str
        Which pairing symmetry to show gap nodes for ('B1u', 'B2u', 'B3u', or 'all')
    orientation : str
        View orientation: 'standard' (kx-ky-kz) or 'alt' (kx vertical, ky-kz horizontal)
    """
    import os
    import pickle
    import hashlib
    os.makedirs(save_dir, exist_ok=True)
    
    print("Computing 3D Fermi surfaces using marching cubes...")
    if orientation == 'alt':
        print("  Using alternative orientation: kx vertical, ky-kz horizontal")
    
    # Create 3D momentum grid (reduced resolution for speed)
    nk3d = 129  # Reduced resolution for faster computation
    
    # Create cache filename based on grid parameters and lattice constants
    cache_key = f"{nk3d}_{a:.6f}_{b:.6f}_{c:.6f}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    # Use global cache directory for band energies (shared across all plots)
    cache_dir = os.path.join('outputs', 'global_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'band_energies_3d_{cache_hash}.pkl')
    
    # Try to load from disk cache
    if os.path.exists(cache_file):
        print(f"  Loading cached band energies from {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            band_energies_3d = cache_data['band_energies_3d']
            kx_3d = cache_data['kx_3d']
            ky_3d = cache_data['ky_3d']
            kz_3d = cache_data['kz_3d']
            print(f"   Successfully loaded cached 3D band energies ({band_energies_3d.shape})")
        except Exception as e:
            print(f"   Failed to load cache file: {e}")
            print("  Computing fresh band energies...")
            cache_file = None  # Force recomputation
    else:
        cache_file = None
    
    # Compute if no valid cache
    if cache_file is None or 'band_energies_3d' not in locals():
        print("  Computing band energies (first time)...")
        kx_3d = np.linspace(-np.pi/a, np.pi/a, nk3d)
        ky_3d = np.linspace(-np.pi/b, np.pi/b, nk3d)  # Standard ky range for 3D
        kz_3d = np.linspace(-np.pi/c, np.pi/c, nk3d)
        
        # Create meshgrid
        KX_3d, KY_3d, KZ_3d = np.meshgrid(kx_3d, ky_3d, kz_3d, indexing='ij')
        
        print(f"  3D grid: {nk3d} x {nk3d} x {nk3d} = {nk3d**3:,} points")
        
        # Compute band energies for the entire 3D grid using vectorized approach
        print("  Computing band energies on 3D grid (vectorized)...")
        
        # Pre-allocate array
        band_energies_3d = np.zeros((nk3d, nk3d, nk3d, 4))
        
        # Vectorized computation with progress tracking
        total_slices = nk3d
        for i, kx in enumerate(kx_3d):
            # Compute entire kx slice at once
            for j, ky in enumerate(ky_3d):
                # Vectorize over kz direction
                H_stack = np.array([H_full(kx, ky, kz) for kz in kz_3d])
                eigvals_stack = np.array([np.sort(np.real(np.linalg.eigvals(H))) for H in H_stack])
                band_energies_3d[i, j, :, :] = eigvals_stack
            
            # Progress update
            progress = (i + 1) / total_slices * 100
            print(f"    Progress: {progress:.1f}% ({i+1}/{total_slices} kx slices)")
        
        print("  Band energy computation complete!")
        
        # Save to disk cache for future use
        cache_data = {
            'band_energies_3d': band_energies_3d,
            'kx_3d': kx_3d,
            'ky_3d': ky_3d,
            'kz_3d': kz_3d,
            'nk3d': nk3d,
            'lattice_params': {'a': a, 'b': b, 'c': c},
            'cache_key': cache_key
        }
        
        cache_file = os.path.join(cache_dir, f'band_energies_3d_{cache_hash}.pkl')
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"   Band energies cached to disk: {cache_file}")
            print(f"    Cache size: {os.path.getsize(cache_file) / (1024*1024):.1f} MB")
        except Exception as e:
            print(f"   Failed to save cache: {e}")
        
        # Also update global cache for this session
        global _cached_band_energies_3d, _cached_kx_3d, _cached_ky_3d, _cached_kz_3d, _cached_nk3d
        _cached_band_energies_3d = band_energies_3d
        _cached_kx_3d = kx_3d
        _cached_ky_3d = ky_3d
        _cached_kz_3d = kz_3d
        _cached_nk3d = nk3d
    
    # Colors for different bands - electron-like vs hole-like character
    colors = ['#1E88E5', '#42A5F5',  "#9635E5", "#FD0000"]  # Blue for electron-like, Red/Purple for hole-like
    band_labels = ['Band 1 (e-like)', 'Band 2 (e-like)', 'Band 3 (h-like)', 'Band 4 (h-like)']
    alphas = [0.6, 0.6, 0.8, 0.8]  # Hole-like bands slightly more opaque
    
    # Compute 3D superconducting gap magnitude for different pairing symmetries with caching
    gap_magnitudes_3d = {}
    
    if show_gap_nodes:
        print("\n  Computing 3D superconducting gap magnitudes...")
        
        # Determine which pairing types to compute
        if gap_node_pairing.lower() == 'all':
            pairing_types = ['B1u', 'B2u', 'B3u']
        else:
            pairing_types = [gap_node_pairing]
        
        # Set up gap magnitude caching
        import json
        # Gap magnitudes can be specific to save_dir since they're plot-specific
        gap_cache_dir = os.path.join(save_dir, 'cache')
        os.makedirs(gap_cache_dir, exist_ok=True)
        
        # Create gap cache key based on grid and lattice parameters
        gap_cache_key = f"{nk3d}_{a:.6f}_{b:.6f}_{c:.6f}"
        gap_cache_hash = hashlib.md5(gap_cache_key.encode()).hexdigest()[:8]
        gap_cache_file = os.path.join(gap_cache_dir, f'gap_magnitudes_3d_{gap_cache_hash}.npz')
        gap_metadata_file = os.path.join(gap_cache_dir, f'gap_magnitudes_3d_metadata_{gap_cache_hash}.json')
        
        # Try to load gap magnitudes from cache
        cached_gaps = {}
        if os.path.exists(gap_cache_file) and os.path.exists(gap_metadata_file):
            print("  Checking gap magnitude cache...")
            try:
                with open(gap_metadata_file, 'r') as f:
                    cached_gap_metadata = json.load(f)
                
                # Verify cache parameters match current setup
                if (cached_gap_metadata.get('nk3d') == nk3d and
                    cached_gap_metadata.get('lattice_params', {}).get('a') == a and
                    cached_gap_metadata.get('lattice_params', {}).get('b') == b and
                    cached_gap_metadata.get('lattice_params', {}).get('c') == c):
                    
                    gap_cache_data = np.load(gap_cache_file)
                    cached_pairing_types = cached_gap_metadata.get('pairing_types', [])
                    
                    # Load cached gap magnitudes that we need
                    for pairing_type in pairing_types:
                        if f'gap_{pairing_type}' in gap_cache_data:
                            cached_gaps[pairing_type] = gap_cache_data[f'gap_{pairing_type}']
                            print(f"     Loaded cached {pairing_type} gap magnitude")
                
            except Exception as e:
                print(f"     Failed to load gap cache: {e}")
        
        # Determine which gaps need to be computed
        missing_gaps = [p for p in pairing_types if p not in cached_gaps]
        
        if missing_gaps:
            print(f"    Computing missing gap magnitudes: {missing_gaps}")
            
            for pairing_type in missing_gaps:
                print(f"      Computing {pairing_type} gap...")
                gap_3d = np.zeros((nk3d, nk3d, nk3d))
                
                for i, kx in enumerate(kx_3d):
                    for j, ky in enumerate(ky_3d):
                        # Vectorize over kz direction
                        gap_stack = np.array([calculate_gap_magnitude(
                            np.array([[kx]]), np.array([[ky]]), kz, pairing_type=pairing_type)[0, 0] 
                            for kz in kz_3d])
                        gap_3d[i, j, :] = gap_stack
                    
                    if (i + 1) % 10 == 0:
                        print(f"        {pairing_type}: {(i+1)/nk3d*100:.0f}% complete")
                
                cached_gaps[pairing_type] = gap_3d
                print(f"      {pairing_type} gap range: {np.min(gap_3d):.6f} to {np.max(gap_3d):.6f}")
            
            # Save newly computed gaps to cache
            if missing_gaps:
                try:
                    # Load existing cache if it exists
                    if os.path.exists(gap_cache_file):
                        existing_cache = dict(np.load(gap_cache_file))
                    else:
                        existing_cache = {}
                    
                    # Add new gap magnitudes
                    for pairing_type in missing_gaps:
                        existing_cache[f'gap_{pairing_type}'] = cached_gaps[pairing_type]
                    
                    # Save combined cache
                    np.savez_compressed(gap_cache_file, **existing_cache)
                    
                    # Update metadata
                    if os.path.exists(gap_metadata_file):
                        with open(gap_metadata_file, 'r') as f:
                            existing_metadata = json.load(f)
                        all_pairing_types = list(set(existing_metadata.get('pairing_types', []) + missing_gaps))
                    else:
                        all_pairing_types = missing_gaps
                    
                    gap_metadata = {
                        'nk3d': nk3d,
                        'lattice_params': {'a': a, 'b': b, 'c': c},
                        'pairing_types': sorted(all_pairing_types),
                        'cache_key': gap_cache_key
                    }
                    
                    with open(gap_metadata_file, 'w') as f:
                        json.dump(gap_metadata, f, indent=2)
                    
                    print(f"       Gap magnitudes cached to disk: {gap_cache_file}")
                    print(f"        Cache size: {os.path.getsize(gap_cache_file) / (1024*1024):.1f} MB")
                    print(f"        Cached pairing types: {sorted(all_pairing_types)}")
                    
                except Exception as e:
                    print(f"       Failed to save gap cache: {e}")
        
        # Use cached gaps (both loaded and newly computed)
        gap_magnitudes_3d = cached_gaps
        
        if not missing_gaps:
            print("     All gap magnitudes loaded from cache!")
        
        print("  3D gap computation complete!")
    else:
        pairing_types = []
        print("\n  Gap node detection disabled")
    
    # Create the 3D plot
    fig = plt.figure(figsize=(8.3, 6.2), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    # Use Fermi level (E=0) for all bands
    EF = 0.0
    print(f"\n  Extracting Fermi surfaces at E={EF:.3f} eV using marching cubes...")
    
    surfaces_plotted = 0
    all_meshes = []  # Store all mesh objects for combined export
    mesh_colors = []  # Store colors for each mesh
    
    for band in range(4):
        print(f"  Processing {band_labels[band]}...")
        
        # Get the band energies for this band
        band_data = band_energies_3d[:, :, :, band]
        
        # Check if this band crosses the Fermi level
        if band_data.min() <= EF <= band_data.max():
            try:
                # Apply light Gaussian smoothing to reduce noise/patchiness at high resolution
                from scipy.ndimage import gaussian_filter
                band_data_smooth = gaussian_filter(band_data, sigma=0.5)
                
                # Determine optimal step size based on grid resolution
                # For high-res grids (>100), use step_size=2 to avoid over-dense meshes
                optimal_step = 2 if nk3d > 100 else 1
                
                # Extract isosurface using marching cubes
                verts, faces, normals, values = measure.marching_cubes(
                    band_data_smooth, level=EF, 
                    spacing=(
                        kx_3d[1] - kx_3d[0],
                        ky_3d[1] - ky_3d[0], 
                        kz_3d[1] - kz_3d[0]
                    ),
                    step_size=optimal_step,  # Adaptive step size
                    gradient_direction='ascent'  # Better normal calculation
                )
                
                # Adjust vertices to actual k-space coordinates
                verts[:, 0] = verts[:, 0] + kx_3d[0]  # kx
                verts[:, 1] = verts[:, 1] + ky_3d[0]  # ky
                verts[:, 2] = verts[:, 2] + kz_3d[0]  # kz
                
                # Convert to units of π/a, π/b, π/c for plotting with axis swap
                # Apply physically correct scaling based on lattice parameter ratios
                # kx range is naturally wider due to smaller lattice parameter a
                if orientation == 'alt':
                    # Alternative orientation: kx vertical, ky-kz horizontal
                    x_plot = verts[:, 1] / (np.pi/b)  # ky -> x-axis
                    y_plot = verts[:, 2] / (np.pi/c)  # kz -> y-axis  
                    z_plot = verts[:, 0] / (np.pi/a)  # kx -> z-axis (vertical)
                else:
                    # Standard orientation: kx-ky plane, kz vertical
                    x_plot = verts[:, 1] / (np.pi/b)  # ky -> x-axis
                    y_plot = verts[:, 0] / (np.pi/a)  # kx -> y-axis (naturally wider due to a < b)
                    z_plot = verts[:, 2] / (np.pi/c)  # kz -> z-axis
                
                # Create the surface mesh
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                # Handle face count intelligently to avoid gaps
                max_faces = 30000  # Higher limit for high-res grids
                faces_used = faces
                
                if len(faces) > max_faces:
                    print(f"      Warning: {len(faces)} faces detected, reducing complexity")
                    # Use trimesh decimation if available (better than random subsampling)
                    if TRIMESH_AVAILABLE:
                        try:
                            import trimesh
                            temp_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                            # Simplify mesh while preserving shape
                            simplified = temp_mesh.simplify_quadric_decimation(max_faces)
                            verts = simplified.vertices
                            faces_used = simplified.faces
                            print(f"      Decimated to {len(faces_used)} faces using quadric decimation")
                        except Exception as e:
                            print(f"      Decimation failed, using all faces: {e}")
                            faces_used = faces
                    else:
                        # If no trimesh, just use all faces (matplotlib can handle it)
                        print(f"      Using all {len(faces)} faces (may be slow but complete)")
                        faces_used = faces
                else:
                    faces_used = faces
                
                # Create triangular mesh
                mesh_verts = np.column_stack([x_plot, y_plot, z_plot])
                
                # Plot the surface efficiently with proper shading
                poly3d = mesh_verts[faces_used]
                
                # Add shaded surface with proper color setup
                collection = ax.add_collection3d(Poly3DCollection(
                    poly3d, alpha=alphas[band], facecolor=colors[band], 
                    edgecolor='none', linewidth=0, rasterized=True
                ))
                
                # Enable shading manually by setting face colors with lighting
                from matplotlib.colors import LightSource
                ls = LightSource(azdeg=315, altdeg=45)  # Light from upper left
                
                # Apply lighting to the collection
                collection.set_facecolor(colors[band])
                collection.set_alpha(alphas[band])
                collection.set_edgecolor('none')
                
                # Store mesh for combined export if trimesh is available
                if TRIMESH_AVAILABLE:
                    try:
                        # Create trimesh object for this band
                        mesh_obj = trimesh.Trimesh(vertices=mesh_verts, faces=faces_used)
                        
                        # Clean up the mesh
                        mesh_obj.remove_duplicate_faces()
                        mesh_obj.remove_unreferenced_vertices()
                        
                        # Convert hex color to RGB tuple (0-255)
                        hex_color = colors[band].lstrip('#')
                        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                        
                        # Add to collection for combined export
                        all_meshes.append(mesh_obj)
                        mesh_colors.append({
                            'rgb': rgb_color,
                            'band_index': band,
                            'band_name': band_labels[band]
                        })
                        
                        print(f"      Added {band_labels[band]} with color {colors[band]} -> RGB{rgb_color}")
                        
                    except Exception as e:
                        print(f"      Failed to create mesh object for {band_labels[band]}: {e}")
                
                surfaces_plotted += 1
                print(f"    + {band_labels[band]}: {len(faces_used)} triangular faces (original: {len(faces)})")
                
            except Exception as e:
                print(f"    - Failed to extract surface for {band_labels[band]}: {e}")
        else:
            print(f"    - {band_labels[band]} does not cross Fermi level "
                  f"(range: {band_data.min():.3f} to {band_data.max():.3f} eV)")
    
    if surfaces_plotted == 0:
        print("  Warning: No Fermi surfaces found!")
        # Fallback: create a simple test surface
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_test = np.outer(np.cos(u), np.sin(v)) * 0.5
        y_test = np.outer(np.sin(u), np.sin(v)) * 0.5
        z_test = np.outer(np.ones(np.size(u)), np.cos(v)) * 0.5
        ax.plot_surface(x_test, y_test, z_test, alpha=0.3, color='gray')
    
    # Find and mark 3D gap nodes on Fermi surfaces using slice-based detection
    # This approach is more robust: slice through 3D volume and use 2D detection method
    all_gap_nodes = []  # Store all gap node coordinates for plotting after Fermi surfaces
    
    if show_gap_nodes and len(pairing_types) > 0:
        print("\n  Detecting 3D gap nodes on Fermi surfaces (slice-based method)...")
        
        for pairing_type in pairing_types:
            print(f"\n    Analyzing {pairing_type} gap nodes...")
            gap_3d = gap_magnitudes_3d[pairing_type]
            
            # Set threshold based on pairing type
            # Tighter thresholds to ensure nodes are accurately on Fermi surface
            if pairing_type == 'B2u':
                gap_threshold = 0.05 * np.max(gap_3d)
                fermi_threshold = 0.010  # 10 meV - much tighter
            elif pairing_type == 'B3u':
                gap_threshold = 0.04 * np.max(gap_3d)
                fermi_threshold = 0.008  # 8 meV - tightest for best accuracy
            else:
                gap_threshold = 0.05 * np.max(gap_3d)
                fermi_threshold = 0.010  # 10 meV
            
            # Collect nodes from all slices
            all_nodes_kx = []
            all_nodes_ky = []
            all_nodes_kz = []
            all_nodes_gap = []
            all_nodes_band = []
            
            # Slice through kz direction (most efficient for ky-kz nodal lines in B3u)
            # For B2u, nodal planes at kz = 0, ±π/c are critical
            n_slices = nk3d  # Use all kz slices
            
            # For B2u, ensure we check the exact nodal planes
            critical_kz_indices = []
            if pairing_type == 'B2u':
                # Find indices closest to kz = 0, ±π/c
                idx_0 = np.argmin(np.abs(kz_3d - 0.0))
                idx_plus = np.argmin(np.abs(kz_3d - np.pi/c))
                idx_minus = np.argmin(np.abs(kz_3d - (-np.pi/c)))
                critical_kz_indices = [idx_0, idx_plus, idx_minus]
                print(f"      B2u critical kz planes:")
                print(f"        kz ≈ 0: kz_3d[{idx_0}] = {kz_3d[idx_0]:.4f}")
                print(f"        kz ≈ +π/c: kz_3d[{idx_plus}] = {kz_3d[idx_plus]:.4f}")
                print(f"        kz ≈ -π/c: kz_3d[{idx_minus}] = {kz_3d[idx_minus]:.4f}")
            
            print(f"      Scanning {n_slices} kz slices...")
            
            for band in [2, 3]:  # Bands 3 and 4
                band_nodes_found = 0
                
                for iz, kz_val in enumerate(kz_3d):
                    # Extract 2D slice at this kz
                    energy_slice = band_energies_3d[:, :, iz, band]  # Shape: (nk3d, nk3d)
                    gap_slice = gap_3d[:, :, iz]  # Shape: (nk3d, nk3d)
                    
                    # Check if this is a critical kz plane for B2u
                    is_critical_kz = (pairing_type == 'B2u' and iz in critical_kz_indices)
                    
                    # Use more generous threshold at critical planes to ensure we catch all nodes
                    effective_fermi_threshold = fermi_threshold
                    if is_critical_kz:
                        effective_fermi_threshold = fermi_threshold * 2.0  # Double the tolerance at critical planes
                    
                    # Check if Fermi surface crosses this slice
                    if energy_slice.min() > effective_fermi_threshold or energy_slice.max() < -effective_fermi_threshold:
                        continue
                    
                    # Find intersections in this 2D slice (like in 2D version)
                    fermi_mask = np.abs(energy_slice) < effective_fermi_threshold
                    gap_mask = gap_slice < gap_threshold
                    intersection = fermi_mask & gap_mask
                    
                    if np.any(intersection):
                        # Get indices of intersection points
                        ix_slice, iy_slice = np.where(intersection)
                        
                        # Apply 2D clustering within this slice (pixel-based like 2D version)
                        selected_2d = []
                        min_pixel_dist = 8  # Minimum pixels between nodes in same slice
                        
                        for i in range(len(ix_slice)):
                            ix_pt, iy_pt = ix_slice[i], iy_slice[i]
                            
                            # Check distance to already selected points in this slice
                            too_close = False
                            for (prev_ix, prev_iy) in selected_2d:
                                if np.sqrt((ix_pt - prev_ix)**2 + (iy_pt - prev_iy)**2) < min_pixel_dist:
                                    too_close = True
                                    break
                            
                            if not too_close:
                                selected_2d.append((ix_pt, iy_pt))
                                
                                # Convert to physical coordinates
                                kx_node = kx_3d[ix_pt]
                                ky_node = ky_3d[iy_pt]
                                gap_value = gap_slice[ix_pt, iy_pt]
                                
                                # Store this node
                                all_nodes_kx.append(kx_node)
                                all_nodes_ky.append(ky_node)
                                all_nodes_kz.append(kz_val)
                                all_nodes_gap.append(gap_value)
                                all_nodes_band.append(band)
                                band_nodes_found += 1
                                
                                # Track nodes at critical kz planes for B2u
                                if is_critical_kz:
                                    print(f"        Found node at kz={kz_val:.4f}, kx={kx_node:.4f}, ky={ky_node:.4f} (Band {band+1})")
                
                if band_nodes_found > 0:
                    print(f"      Band {band+1}: Found {band_nodes_found} nodes in slices")
            
            # Now perform 3D clustering on collected nodes from all slices
            if len(all_nodes_kx) > 0:
                all_nodes_kx = np.array(all_nodes_kx)
                all_nodes_ky = np.array(all_nodes_ky)
                all_nodes_kz = np.array(all_nodes_kz)
                all_nodes_gap = np.array(all_nodes_gap)
                all_nodes_band = np.array(all_nodes_band)
                
                print(f"      Total nodes from slices: {len(all_nodes_kx)}")
                print(f"      Applying 3D clustering...")
                
                # 3D clustering with physical distance
                # B2u needs smaller distance to preserve nodes at different kx positions
                if pairing_type == 'B3u':
                    min_distance_3d = 0.4  # Physical units
                elif pairing_type == 'B2u':
                    min_distance_3d = 0.20  # Reduced to preserve separate nodes
                else:
                    min_distance_3d = 0.3
                
                selected_indices = []
                used = np.zeros(len(all_nodes_kx), dtype=bool)
                
                for i in range(len(all_nodes_kx)):
                    if used[i]:
                        continue
                    
                    # Find cluster
                    cluster = [i]
                    for j in range(i+1, len(all_nodes_kx)):
                        if used[j]:
                            continue
                        dist = np.sqrt((all_nodes_kx[i] - all_nodes_kx[j])**2 +
                                      (all_nodes_ky[i] - all_nodes_ky[j])**2 +
                                      (all_nodes_kz[i] - all_nodes_kz[j])**2)
                        if dist < min_distance_3d:
                            cluster.append(j)
                    
                    # Pick node with smallest gap from cluster
                    best_idx = cluster[0]
                    best_gap = all_nodes_gap[cluster[0]]
                    for idx in cluster:
                        if all_nodes_gap[idx] < best_gap:
                            best_gap = all_nodes_gap[idx]
                            best_idx = idx
                    
                    # Mark cluster as used
                    for idx in cluster:
                        used[idx] = True
                    
                    selected_indices.append(best_idx)
                
                # Extract final nodes
                final_kx = all_nodes_kx[selected_indices]
                final_ky = all_nodes_ky[selected_indices]
                final_kz = all_nodes_kz[selected_indices]
                final_band = all_nodes_band[selected_indices]
                
                print(f"      After 3D clustering: {len(selected_indices)} nodes")
                
                # Show before/after by band
                for band in [2, 3]:
                    band_mask_before = all_nodes_band == band
                    band_mask_after = final_band == band
                    print(f"      Band {band+1}: {np.sum(band_mask_before)} detected → {np.sum(band_mask_after)} after clustering")
                
                # Group by band for plotting
                for band in [2, 3]:
                    band_mask = final_band == band
                    if np.any(band_mask):
                        kx_band = final_kx[band_mask]
                        ky_band = final_ky[band_mask]
                        kz_band = final_kz[band_mask]
                        
                        # Convert to plot coordinates based on orientation
                        if orientation == 'alt':
                            # Alternative orientation: kx vertical, ky-kz horizontal
                            x_nodes = ky_band / (np.pi/b)  # ky -> x-axis
                            y_nodes = kz_band / (np.pi/c)  # kz -> y-axis
                            z_nodes = kx_band / (np.pi/a)  # kx -> z-axis (vertical)
                        else:
                            # Standard orientation: kx-ky plane, kz vertical
                            x_nodes = ky_band / (np.pi/b)  # ky -> x-axis
                            y_nodes = kx_band / (np.pi/a)  # kx -> y-axis
                            z_nodes = kz_band / (np.pi/c)  # kz -> z-axis
                        
                        # Store for plotting
                        all_gap_nodes.append({
                            'x': x_nodes,
                            'y': y_nodes,
                            'z': z_nodes,
                            'pairing': pairing_type,
                            'band': band + 1
                        })
                        
                        print(f"      Band {band+1}: {len(x_nodes)} gap nodes")
                
                print(f"    {pairing_type}: Total {len(selected_indices)} gap nodes visualized")
                
                # Save gap nodes to cache for projection use
                try:
                    gap_nodes_cache_file = os.path.join(save_dir, 'cache', f'gap_nodes_3d_{pairing_type}.npz')
                    os.makedirs(os.path.dirname(gap_nodes_cache_file), exist_ok=True)
                    
                    # Convert back to physical k-space coordinates (not plot units)
                    if orientation == 'alt':
                        # x=ky, y=kz, z=kx
                        kx_phys = np.array(z_nodes) * (np.pi/a)
                        ky_phys = np.array(x_nodes) * (np.pi/b)
                        kz_phys = np.array(y_nodes) * (np.pi/c)
                    else:
                        # x=ky, y=kx, z=kz
                        kx_phys = np.array(y_nodes) * (np.pi/a)
                        ky_phys = np.array(x_nodes) * (np.pi/b)
                        kz_phys = np.array(z_nodes) * (np.pi/c)
                    
                    np.savez(gap_nodes_cache_file, kx=kx_phys, ky=ky_phys, kz=kz_phys)
                    print(f"     Saved {len(kx_phys)} gap nodes to cache: {gap_nodes_cache_file}")
                except Exception as e:
                    print(f"     Failed to save gap nodes cache: {e}")
                    
            else:
                print(f"    {pairing_type}: No gap nodes found on Fermi surfaces")
                print(f"      Gap threshold: {gap_threshold:.6f}, Max gap: {np.max(gap_3d):.6f}")
    
    # Plot all gap nodes AFTER Fermi surfaces (ensures they appear on top)
    if len(all_gap_nodes) > 0:
        print("\n  Plotting gap nodes on top of Fermi surfaces...")
        
        # First reduce Fermi surface alpha slightly so nodes show through better
        for collection in ax.collections:
            if isinstance(collection, Poly3DCollection):
                current_alpha = collection.get_alpha()
                if current_alpha is not None:
                    collection.set_alpha(current_alpha * 0.85)  # Slightly more transparent
        
        # Now plot nodes with maximum visibility settings
        for node_set in all_gap_nodes:
            # Plot each node individually with very high priority
            for i in range(len(node_set['x'])):
                ax.scatter([node_set['x'][i]], [node_set['y'][i]], [node_set['z'][i]],
                         c='yellow',  # Always bright yellow
                         marker='o', s=120, alpha=1.0,  # Larger, fully opaque
                         edgecolors='black', linewidth=2.5,  # Thick black outline
                         zorder=1000,  # Extremely high zorder
                         depthshade=False)  # Disable depth shading
            
            # Add one invisible point for the legend
            ax.scatter([], [], [],
                     c='yellow', marker='o', s=120, alpha=1.0,
                     edgecolors='black', linewidth=2.5,
                     label=f"{node_set['pairing']} gap nodes (Band {node_set['band']})")
        
        print(f"    Plotted {sum(len(n['x']) for n in all_gap_nodes)} total gap nodes with maximum visibility")
    
    # Export combined Fermi surface as single STL file
    if TRIMESH_AVAILABLE and len(all_meshes) > 0:
        try:
            # Combine all meshes into a single object
            combined_mesh = trimesh.util.concatenate(all_meshes)
            
            # Final cleanup
            combined_mesh.remove_duplicate_faces()
            combined_mesh.remove_unreferenced_vertices()
            
            # Export combined STL
            combined_stl_filename = 'fermi_surface_3d_complete.stl'
            combined_stl_filepath = os.path.join(save_dir, combined_stl_filename)
            combined_mesh.export(combined_stl_filepath)
            
            total_faces = sum(len(mesh.faces) for mesh in all_meshes)
            print(f"\n  Exported complete Fermi surface as {combined_stl_filename}")
            print(f"    Combined {len(all_meshes)} surfaces with {total_faces:,} total faces")
            
            # Export as OBJ with colors/materials
            try:
                # Create OBJ with separate objects for each band (preserves colors)
                obj_filename = 'fermi_surface_3d_complete.obj'
                obj_filepath = os.path.join(save_dir, obj_filename)
                mtl_filename = 'fermi_surface_3d_complete.mtl'
                mtl_filepath = os.path.join(save_dir, mtl_filename)
                
                # Write MTL file (materials)
                with open(mtl_filepath, 'w') as mtl_file:
                    for i, (mesh, color_info) in enumerate(zip(all_meshes, mesh_colors)):
                        band_name = color_info['band_name'].lower().replace(' ', '_')
                        rgb = color_info['rgb']
                        band_idx = color_info['band_index']
                        
                        mtl_file.write(f"newmtl {band_name}\n")
                        mtl_file.write(f"Ka 0.2 0.2 0.2\n")  # Ambient color
                        mtl_file.write(f"Kd {rgb[0]/255.0:.3f} {rgb[1]/255.0:.3f} {rgb[2]/255.0:.3f}\n")  # Diffuse color
                        mtl_file.write(f"Ks 0.3 0.3 0.3\n")  # Specular color
                        mtl_file.write(f"Ns 32\n")  # Specular exponent
                        mtl_file.write(f"d {alphas[band_idx]}\n")  # Transparency (alpha)
                        mtl_file.write(f"# Band: {color_info['band_name']}, Color: {colors[band_idx]}\n")
                        mtl_file.write(f"\n")
                
                # Write OBJ file with references to materials
                with open(obj_filepath, 'w') as obj_file:
                    obj_file.write(f"mtllib {mtl_filename}\n\n")
                    
                    vertex_offset = 0
                    for i, (mesh, color_info) in enumerate(zip(all_meshes, mesh_colors)):
                        band_name = color_info['band_name'].lower().replace(' ', '_')
                        obj_file.write(f"o {band_name}\n")
                        obj_file.write(f"usemtl {band_name}\n")
                        obj_file.write(f"# {color_info['band_name']} - {len(mesh.vertices)} vertices, {len(mesh.faces)} faces\n")
                        
                        # Write vertices
                        for vertex in mesh.vertices:
                            obj_file.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                        
                        # Write faces (1-indexed, offset by previous meshes)
                        for face in mesh.faces:
                            f1, f2, f3 = face + vertex_offset + 1  # OBJ is 1-indexed
                            obj_file.write(f"f {f1} {f2} {f3}\n")
                        
                        vertex_offset += len(mesh.vertices)
                        obj_file.write(f"\n")
                
                print(f"   Exported colored Fermi surface as {obj_filename} (with {mtl_filename})")
                print(f"    Materials created for {len(mesh_colors)} bands with distinct colors")
                for i, color_info in enumerate(mesh_colors):
                    print(f"      {color_info['band_name']}: {colors[color_info['band_index']]} -> RGB{color_info['rgb']}")
                
            except Exception as e:
                print(f"  ✗ Failed to export OBJ with materials: {e}")
            
        except Exception as e:
            print(f"\n  ✗ Failed to export combined Fermi surface: {e}")
            # Fallback: export individual bands
            for i, mesh in enumerate(all_meshes):
                try:
                    fallback_filename = f'fermi_surface_3d_band_{i+1}.stl'
                    fallback_filepath = os.path.join(save_dir, fallback_filename)
                    mesh.export(fallback_filepath)
                    print(f"    Exported fallback: {fallback_filename}")
                except Exception as e2:
                    print(f"    Failed to export band {i+1}: {e2}")
    elif TRIMESH_AVAILABLE:
        print("\n  No Fermi surfaces to export as STL/OBJ")
    else:
        print("\n  STL/OBJ export skipped (trimesh not available)")
    
    # Set labels and title with proper k-space units
    if orientation == 'alt':
        # Alternative orientation: kx vertical, ky-kz horizontal
        ax.set_xlabel(r'$k_y$ (π/b)', fontsize=12)  # ky on x-axis
        ax.set_ylabel(r'$k_z$ (π/c)', fontsize=12)  # kz on y-axis
        ax.set_zlabel(r'$k_x$ (π/a)', fontsize=12)  # kx on z-axis (vertical)
    else:
        # Standard orientation: kx-ky plane, kz vertical
        ax.set_xlabel(r'$k_y$ (π/b)', fontsize=12)  # ky on x-axis
        ax.set_ylabel(r'$k_x$ (π/a)', fontsize=12)  # kx on y-axis
        ax.set_zlabel(r'$k_z$ (π/c)', fontsize=12)  # kz on z-axis
    
    # Set title based on whether gap nodes are shown
    if show_gap_nodes and len(pairing_types) > 0:
        pairing_str = ', '.join(pairing_types)
        ax.set_title(f'3D Fermi Surface of UTe₂ - {pairing_str} Gap Nodes', fontsize=14, pad=5)
    else:
        ax.set_title('3D Fermi Surface of UTe₂', fontsize=14, pad=5)
    
    # Add legend with electron-like vs hole-like distinction
    from matplotlib.lines import Line2D
    # Create simplified legend showing only electron vs hole character with correct colors
    legend_elements = [
        Line2D([0], [0], color='#FD0000', lw=4, alpha=0.8, label='Electron-like'),
        Line2D([0], [0], color='#9635E5', lw=4, alpha=0.8, label='Hole-like')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.9)
    
    # Make background cleaner
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Remove old individual export summary since we now have combined export
    # The export summary is now handled above in the combined export section
    
    # Set axis limits based on physical momentum ranges (π/lattice_parameter)
    # kx range is naturally wider than ky range due to orthorhombic structure (a < b)
    kx_range = 1.0  # -π/a to π/a in units of π/a
    ky_range = 1.0  # -π/b to π/b in units of π/b  
    kz_range = 1.0  # -π/c to π/c in units of π/c
    ax.set_xlim(-ky_range, ky_range)  # ky on x-axis
    ax.set_ylim(-kx_range, kx_range)  # kx on y-axis (appears wider due to lattice)
    ax.set_zlim(-kz_range, kz_range)  # kz on z-axis
    
    # Set custom ticks every 0.5 π/lattice_parameter for all axes
    
    # Create tick positions every 0.5 units
    tick_positions = np.arange(-1.0, 1.5, 0.5)  # -1.0, -0.5, 0.0, 0.5, 1.0
    
    # Set ticks for all axes
    ax.set_xticks(tick_positions)  # ky axis (π/b)
    ax.set_yticks(tick_positions)  # kx axis (π/a)
    ax.set_zticks(tick_positions)  # kz axis (π/c)
    
    # Create custom tick labels showing numerical values
    tick_labels = [f'{pos:.1f}' if pos != 0 else '0' for pos in tick_positions]
    
    # Apply tick labels to all axes
    ax.set_xticklabels(tick_labels)  # ky axis labels (π/b)
    ax.set_yticklabels(tick_labels)  # kx axis labels (π/a)  
    ax.set_zticklabels(tick_labels)  # kz axis labels (π/c)
    
    # Scale the 3D box to reflect actual physical dimensions in reciprocal space
    # The actual k-space ranges are: kx: ±π/a, ky: ±π/b, kz: ±π/c
    # Physical size is proportional to 1/lattice_parameter, so bigger 1/a means wider axis
    # Lattice parameters: a=4.07 Å, b=5.83 Å, c=13.78 Å
    # So reciprocal space: π/a > π/b > π/c (kx axis is widest)
    # Ratios: (π/a)/(π/c) = c/a ≈ 3.39, (π/b)/(π/c) = c/b ≈ 2.36, (π/a)/(π/b) = b/a ≈ 1.43
    
    # Set box aspect ratio to physical proportions
    # Set aspect ratios based on orientation and lattice parameters
    if orientation == 'alt':
        # Alternative: kx vertical (z), ky-kz horizontal (x,y)
        aspect_x = c / b  # ky (x-axis) relative to kz
        aspect_y = 1.0    # kz (y-axis) is reference
        aspect_z = c / a  # kx (z-axis) relative to kz (tallest because a is smallest)
        
        ax.set_box_aspect([aspect_x, aspect_y, aspect_z])
        
        print(f"\n  3D plot aspect ratios (alternative orientation):") 
        print(f"    ky-axis (x): {aspect_x:.2f} (π/b range = {2*np.pi/b:.3f})")
        print(f"    kz-axis (y): {aspect_y:.2f} (π/c range = {2*np.pi/c:.3f})")
        print(f"    kx-axis (z): {aspect_z:.2f} (π/a range = {2*np.pi/a:.3f})")
        print(f"  Note: kx-axis (vertical) appears tallest because a is smallest lattice parameter")
    else:
        # Standard: kx-ky horizontal (x,y), kz vertical (z)
        aspect_ky = c / b  # π/b range relative to π/c
        aspect_kx = c / a  # π/a range relative to π/c (widest)
        aspect_kz = 1.0    # π/c range (reference)
        
        ax.set_box_aspect([aspect_ky, aspect_kx, aspect_kz])
        
        print(f"\n  3D plot aspect ratios (physical):")
        print(f"    ky-axis (x): {aspect_ky:.2f} (π/b range = {2*np.pi/b:.3f})")
        print(f"    kx-axis (y): {aspect_kx:.2f} (π/a range = {2*np.pi/a:.3f})")
        print(f"    kz-axis (z): {aspect_kz:.2f} (π/c range = {2*np.pi/c:.3f})")
        print(f"  Note: kx-axis appears widest because a is smallest lattice parameter")
    
    # Improve view angle to match reference image style
    ax.view_init(elev=20, azim=30)  # Slightly elevated view from front-right
    
    # Adjust subplot parameters to prevent text cutoff, especially kz axis on right
    plt.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.88)
    
    # Determine save directory and filename based on gap node settings
    if show_gap_nodes and len(pairing_types) > 0:
        # Create subdirectory for each pairing type
        pairing_dir = os.path.join(save_dir, gap_node_pairing.lower())
        os.makedirs(pairing_dir, exist_ok=True)
        actual_save_dir = pairing_dir
        base_filename = f'fermi_surface_3d_{gap_node_pairing}_gap_nodes'
    else:
        actual_save_dir = save_dir
        base_filename = 'fermi_surface_3d_marching_cubes'
        if orientation == 'alt':
            base_filename += '_alt_orientation'
    
    # Save the main view
    filename = f'{base_filename}.png'
    filepath = os.path.join(actual_save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved 3D plot (main view) to {filepath}")
    
    # Save additional perspective views
    perspectives = [
        {'elev': 20, 'azim': 30, 'name': 'perspective_1'},  # Current view
        {'elev': 25, 'azim': 45, 'name': 'perspective_2'},  # Slightly different
        {'elev': 30, 'azim': 60, 'name': 'perspective_3'},  # More elevated
        {'elev': 45, 'azim': 15, 'name': 'perspective_4'},  
        {'elev': 55, 'azim': -30, 'name': 'perspective_5'}, # From the other side
    ]
    
    for view in perspectives:
        ax.view_init(elev=view['elev'], azim=view['azim'])
        view_filename = f'{base_filename}_{view["name"]}.png'
        view_filepath = os.path.join(actual_save_dir, view_filename)
        fig.savefig(view_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Saved {len(perspectives)} additional perspective views to {actual_save_dir}")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Verify model parameters first
    #verify_model_parameters()
    
    # Compute for kz=0 slice only
    print("Computing 2D Fermi surface at kz=0...")
    kz0 = 0.0
    energies_kz0 = band_energies_on_slice(kz0)
    
    # Generate only kz=0 2D plot
    #plot_fs_2d(energies_kz0, kz0)
    
    # Extract and print Fermi contour points
    # Create the coordinate arrays that match the energy data resolution
    resolution = 300  # Same as in band_energies_on_slice
    kx_vals_local = np.linspace(-1*np.pi/a, 1*np.pi/a, resolution)
    ky_vals_local = np.linspace(-3*np.pi/b, 3*np.pi/b, resolution)
    #extract_fermi_contour_points(kx_vals_local, ky_vals_local, energies_kz0, bands_to_extract=[2, 3])
    
    # Plot the main 4 contours for debug
    #plot_main_fermi_contours(kx_vals_local, ky_vals_local, energies_kz0)
    
    # Generate superconducting gap magnitude plots
    print("\nGenerating superconducting gap magnitude visualizations...")

    # Generate 3D plot with alternative orientation
    print("\n" + "="*70)
    print("Generating 3D Fermi Surface (alt orientation, no gap nodes)")
    print("="*70)
    plot_3d_fermi_surface(save_dir='outputs/ute2_fixed', show_gap_nodes=False)

    plot_3d_fermi_surface(save_dir='outputs/ute2_fixed', show_gap_nodes=False,orientation='alt')

    
    # Generate 3D plots with gap nodes for each pairing symmetry
    # Skip B1u as it has no nodes in the relevant energy range
    #print("Generating 3D Fermi Surface with Gap Nodes (B2u and B3u only)")
    #print("="*70)
    
    for pairing in ['B2u', 'B3u']:
        print(f"\n--- {pairing} Gap Nodes ---")
        plot_3d_fermi_surface(save_dir='outputs/ute2_fixed', show_gap_nodes=True, 
                             gap_node_pairing=pairing, orientation='alt')
    
    # Generate (0-11) crystallographic projections
    
    # Generate projections with gap nodes for both B2u and B3u
    #for pairing in ['B2u', 'B3u']:
        #print(f"Generating projection with {pairing} gap nodes...")
        #plot_011_fermi_surface_projection(save_dir='outputs/ute2_fixed', 
                                        #show_gap_nodes=True, gap_node_pairing=pairing, 
                                        #angle_deg=24.0)
    
    

    import os
    output_dir = 'outputs/ute2_fixed'
    os.makedirs(output_dir, exist_ok=True)
    data_file = os.path.join(output_dir, 'UTe2_FS_energies.npz')
    np.savez(data_file, kx_vals=kx_vals, ky_vals=ky_vals,
             kz0=kz0, energies_kz0=energies_kz0)
    
    print(f"Saved energies to {data_file}")
    print("\\nAll plots generated successfully!")