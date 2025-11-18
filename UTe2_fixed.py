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
a = 0.407   # Lattice constant along a direction: 4.07 Å
b = 0.583   # Lattice constant along b direction: 5.83 Å
c = 1.378   # Lattice constant along c direction: 13.78 Å


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
tch_Te = 0  # hopping between chains in a direction (paper shows 0)
tz_Te = -0.05  # hopping between chains along c axis
delta = 0.13

# momentum grid for kx-ky plane (in nm^-1, will be converted to π/a, π/b units for plotting)
nk = 201  # Reduced for faster computation
# Extend ky range to cover -3π/b to 3π/b
kx_vals = np.linspace(-np.pi/a, np.pi/a, nk)
ky_vals = np.linspace(-3*np.pi/b, 3*np.pi/b, nk)  # Extended range
KX, KY = np.meshgrid(kx_vals, ky_vals)

def HU_block(kx, ky, kz):
    """2x2 Hamiltonian for U orbitals"""
    diag = muU - 2*tU*np.cos(kx*a) - 2*tch_U*np.cos(ky*b)
    real_off = -DeltaU - 2*tpU*np.cos(kx*a) - 2*tpch_U*np.cos(ky*b)
    complex_amp = -4 * tz_U * np.exp(-1j * kz * c / 2) * np.cos(kx * a / 2) * np.cos(ky * b / 2)
    H = np.zeros((2,2), dtype=complex)
    H[0,0] = diag
    H[1,1] = diag
    H[0,1] = real_off + complex_amp
    H[1,0] = real_off + np.conj(complex_amp)
    return H

def HTe_block(kx, ky, kz):
    """2x2 Hamiltonian for Te orbitals"""
    # Diagonal elements: μTe (no chain hopping since tch_Te = 0 in paper)
    diag = muTe
    # Off-diagonal elements from paper:
    # -ΔTe - tTe*exp(-iky*b) - tz_Te*cos(kz*c/2)*cos(kx*a/2)*cos(ky*b/2)
    real_off = -DeltaTe
    complex_term1 = -tTe * np.exp(-1j * ky * b)  # hopping along b direction
    complex_term2 = -tz_Te * np.cos(kz * c / 2) * np.cos(kx * a / 2) * np.cos(ky * b / 2)
    
    H = np.zeros((2,2), dtype=complex)
    H[0,0] = diag
    H[1,1] = diag
    H[0,1] = real_off + complex_term1 + complex_term2
    H[1,0] = real_off + np.conj(complex_term1) + complex_term2  # ensures Hermiticity
    return H

def H_full(kx, ky, kz):
    """Full 4x4 Hamiltonian with U-Te hybridization"""
    HU = HU_block(kx, ky, kz)
    HTe = HTe_block(kx, ky, kz)
    Hhyb = np.eye(2) * delta
    top = np.hstack((HU, Hhyb))
    bottom = np.hstack((Hhyb.conj().T, HTe))
    H = np.vstack((top, bottom))
    return H

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
        status = "✓" if match else "✗"
        print(f"  {status} {param}: paper={paper_val:.3f}, ours={our_val:.3f}")
        if not match:
            all_match = False
    
    if all_match:
        print("✓ All parameters match the paper exactly!")
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

def compute_spectral_weights(kz, orbital_weights=None, energy_window=0.2, resolution=300):
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
                    fermi_weight = 1.0 / (abs(E_nk) + 0.001)  # Inverse distance to EF
                    
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
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract contours for both bands
    bands_to_plot = [2, 3]  # Band 3 and 4 (0-indexed)
    colors = ['red', 'blue']
    labels = ['Band 3', 'Band 4']
    
    for band_idx, color, label in zip(bands_to_plot, colors, labels):
        energy_data = energies_fs[:, :, band_idx]
        
        if energy_data.min() <= 0 <= energy_data.max():
            # Create contour
            temp_fig, temp_ax = plt.subplots(figsize=(1, 1))
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
    ax.legend()
    
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
            fig, ax = plt.subplots(figsize=(1, 1))
            
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
    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            H = H_full(kx, ky, kz)
            eigvals = np.linalg.eigvals(H)
            energies_fs[i, j, :] = np.sort(np.real(eigvals))
    
    # Create subplot layout
    n_types = len(pairing_types)
    if n_types <= 2:
        fig, axes = plt.subplots(1, n_types, figsize=(8*n_types, 7))
        if n_types == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
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
        
        # Highlight nodal regions FIRST (where gap is very small)
        gap_threshold = 0.1 * np.max(gap_mag)  # 10% of maximum gap
        nodal_mask = gap_mag < gap_threshold
        if np.any(nodal_mask):
            ax.contourf(KY, KX, gap_mag.T, levels=[0, gap_threshold],
                       colors=['darkblue'], alpha=0.4)
            ax.contour(KY, KX, gap_mag.T, levels=[gap_threshold],
                      colors=['darkblue'], linewidths=2, alpha=0.8, linestyles='--')
        
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
                        
                        for i in range(len(ky_indices)):
                            ky_idx, kx_idx = ky_indices[i], kx_indices[i]
                            
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
                        
                        for i in range(len(ky_indices)):
                            ky_idx, kx_idx = ky_indices[i], kx_indices[i]
                            
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
        ax.set_xticklabels(ky_labels)
        ax.set_yticklabels(kx_labels)
        
        ax.set_xlabel('ky (π/b)', fontsize=12)
        ax.set_ylabel('kx (π/a)', fontsize=12)
        ax.set_title(f'{pairing_type} Gap Magnitude |Δ_k|', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('|Δ_k|', fontsize=12)
        
        print(f"    {pairing_type}: gap range 0 to {np.max(gap_mag):.3f}")
        if np.any(nodal_mask):
            print(f"    {pairing_type}: nodal regions marked in dark blue")
        else:
            print(f"    {pairing_type}: no gap nodes detected")
    
    # Hide unused subplots if necessary
    if n_types < len(axes):
        for j in range(n_types, len(axes)):
            axes[j].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    filename = f'gap_magnitude_2d_kz_{kz:.3f}.png'
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Gap magnitude plot saved: {filepath}")
    
    # Also create individual plots for each pairing type
    for pairing_type in pairing_types:
        fig_single, ax_single = plt.subplots(1, 1, figsize=(10, 8))
        
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
                        
                        for i in range(len(ky_indices)):
                            ky_idx, kx_idx = ky_indices[i], kx_indices[i]
                            
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
                        
                        for i in range(len(ky_indices)):
                            ky_idx, kx_idx = ky_indices[i], kx_indices[i]
                            
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
        
        ax_single.set_xlabel('ky (π/b)', fontsize=14)
        ax_single.set_ylabel('kx (π/a)', fontsize=14)
        ax_single.set_title(f'{pairing_type} Superconducting Gap |Δ_k| at kz = {kz:.1f}', fontsize=16)
        ax_single.grid(True, alpha=0.3)
        ax_single.legend(loc='upper right', fontsize=10)
        
        # Enhanced colorbar
        cbar_single = plt.colorbar(im_single, ax=ax_single, shrink=0.8)
        cbar_single.set_label('Gap Magnitude |Δ_k|', fontsize=12)
        
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
        fig = plt.figure(figsize=(10, 8))
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
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=band_labels[i]) for i in range(4)]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save with appropriate filename
        filename = f'fermi_surface_kz_{kz_label}{version["suffix"]}.png'
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved plot to {filepath}")

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
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

def plot_3d_fermi_surface(save_dir='outputs/ute2_fixed', show_gap_nodes=True, 
                         gap_node_pairing='B3u'):
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
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("Computing 3D Fermi surfaces using marching cubes...")
    
    # Use global cache for band energies (computed once, reused for all pairing types)
    global _cached_band_energies_3d, _cached_kx_3d, _cached_ky_3d, _cached_kz_3d, _cached_nk3d
    
    # Create 3D momentum grid (reduced resolution for speed)
    nk3d = 129  # Reduced resolution for faster computation
    
    # Check if we have cached data
    if _cached_band_energies_3d is not None and _cached_nk3d == nk3d:
        print("  Using cached band energies (Fermi surface already computed)")
        band_energies_3d = _cached_band_energies_3d
        kx_3d = _cached_kx_3d
        ky_3d = _cached_ky_3d
        kz_3d = _cached_kz_3d
    else:
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
        
        # Cache for future calls
        _cached_band_energies_3d = band_energies_3d
        _cached_kx_3d = kx_3d
        _cached_ky_3d = ky_3d
        _cached_kz_3d = kz_3d
        _cached_nk3d = nk3d
        print("  Band energies cached for subsequent plots")
    
    # Colors for different bands - U bands (0,1) vs Te bands (2,3)
    colors = ['#1E88E5', '#42A5F5',  "#9635E5", "#FD0000"]  # Blue for U, Red for Te
    band_labels = ['Band 1', 'Band 2', 'Band 3', 'Band 4']
    alphas = [0.6, 0.6, 0.8, 0.8]  # Te bands slightly more opaque
    
    # Compute 3D superconducting gap magnitude for different pairing symmetries
    gap_magnitudes_3d = {}
    
    if show_gap_nodes:
        print("\n  Computing 3D superconducting gap magnitudes...")
        
        # Determine which pairing types to compute
        if gap_node_pairing.lower() == 'all':
            pairing_types = ['B1u', 'B2u', 'B3u']
        else:
            pairing_types = [gap_node_pairing]
        
        for pairing_type in pairing_types:
            print(f"    Computing {pairing_type} gap...")
            gap_3d = np.zeros((nk3d, nk3d, nk3d))
            
            for i, kx in enumerate(kx_3d):
                for j, ky in enumerate(ky_3d):
                    # Vectorize over kz direction
                    gap_stack = np.array([calculate_gap_magnitude(
                        np.array([[kx]]), np.array([[ky]]), kz, pairing_type=pairing_type)[0, 0] 
                        for kz in kz_3d])
                    gap_3d[i, j, :] = gap_stack
                
                if (i + 1) % 10 == 0:
                    print(f"      {pairing_type}: {(i+1)/nk3d*100:.0f}% complete")
            
            gap_magnitudes_3d[pairing_type] = gap_3d
            print(f"    {pairing_type} gap range: {np.min(gap_3d):.6f} to {np.max(gap_3d):.6f}")
        
        print("  3D gap computation complete!")
    else:
        pairing_types = []
        print("\n  Gap node detection disabled")
    
    # Create the 3D plot
    fig = plt.figure(figsize=(14, 10))
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
                        
                        # Convert to plot coordinates
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
                
                print(f"  ✓ Exported colored Fermi surface as {obj_filename} (with {mtl_filename})")
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
    ax.set_xlabel(r'$k_y$ (π/b)', fontsize=12)  # ky on x-axis
    ax.set_ylabel(r'$k_x$ (π/a)', fontsize=12)  # kx on y-axis
    ax.set_zlabel(r'$k_z$ (π/c)', fontsize=12)
    
    # Set title based on whether gap nodes are shown
    if show_gap_nodes and len(pairing_types) > 0:
        pairing_str = ', '.join(pairing_types)
        ax.set_title(f'3D Fermi Surface of UTe₂ - {pairing_str} Gap Nodes', fontsize=14, pad=20)
    else:
        ax.set_title('3D Fermi Surface of UTe₂ (Physical Proportions)', fontsize=14, pad=20)
    
    # Add legend with U/Te distinction
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[i], lw=3, alpha=alphas[i], 
                             label=band_labels[i]) for i in range(4)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
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
    
    # Scale the 3D box to reflect actual physical dimensions in reciprocal space
    # The actual k-space ranges are: kx: ±π/a, ky: ±π/b, kz: ±π/c
    # Physical size is proportional to 1/lattice_parameter, so bigger 1/a means wider axis
    # Lattice parameters: a=4.07 Å, b=5.83 Å, c=13.78 Å
    # So reciprocal space: π/a > π/b > π/c (kx axis is widest)
    # Ratios: (π/a)/(π/c) = c/a ≈ 3.39, (π/b)/(π/c) = c/b ≈ 2.36, (π/a)/(π/b) = b/a ≈ 1.43
    
    # Set box aspect ratio to physical proportions
    # Order: [x-axis (ky), y-axis (kx), z-axis (kz)]
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
    
    # Add multiple saved views for different perspectives
    plt.tight_layout()
    
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
    verify_model_parameters()
    
    # Compute for kz=0 slice only
    print("Computing 2D Fermi surface at kz=0...")
    kz0 = 0.0
    energies_kz0 = band_energies_on_slice(kz0)
    
    # Generate only kz=0 2D plot
    plot_fs_2d(energies_kz0, kz0)
    
    # Extract and print Fermi contour points
    # Create the coordinate arrays that match the energy data resolution
    resolution = 300  # Same as in band_energies_on_slice
    kx_vals_local = np.linspace(-1*np.pi/a, 1*np.pi/a, resolution)
    ky_vals_local = np.linspace(-3*np.pi/b, 3*np.pi/b, resolution)
    extract_fermi_contour_points(kx_vals_local, ky_vals_local, energies_kz0, bands_to_extract=[2, 3])
    
    # Plot the main 4 contours for debug
    plot_main_fermi_contours(kx_vals_local, ky_vals_local, energies_kz0)
    
    # Generate superconducting gap magnitude plots
    print("\nGenerating superconducting gap magnitude visualizations...")
    # Skip B1u as it has no nodes at kz=0
    plot_gap_magnitude_2d(kz=kz0, pairing_types=['B2u', 'B3u'], resolution=200)
    
    # Generate 3D plot without gap nodes
    print("\n" + "="*70)
    print("Generating 3D Fermi Surface (no gap nodes)")
    print("="*70)
    plot_3d_fermi_surface(save_dir='outputs/ute2_fixed', show_gap_nodes=False)
    
    # Generate 3D plots with gap nodes for each pairing symmetry
    # Skip B1u as it has no nodes in the relevant energy range
    print("\n" + "="*70)
    print("Generating 3D Fermi Surface with Gap Nodes (B2u and B3u only)")
    print("="*70)
    
    for pairing in ['B2u', 'B3u']:
        print(f"\n--- {pairing} Gap Nodes ---")
        plot_3d_fermi_surface(save_dir='outputs/ute2_gap', show_gap_nodes=True, 
                             gap_node_pairing=pairing)
    
    # Save energies to file
    import os
    output_dir = 'outputs/ute2_fixed'
    os.makedirs(output_dir, exist_ok=True)
    data_file = os.path.join(output_dir, 'UTe2_FS_energies.npz')
    np.savez(data_file, kx_vals=kx_vals, ky_vals=ky_vals,
             kz0=kz0, energies_kz0=energies_kz0)
    
    print(f"Saved energies to {data_file}")
    print("\\nAll plots generated successfully!")