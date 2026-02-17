#!/usr/bin/env python3
"""
HAEM Signal vs Energy for UTe2 - Clean Implementation
Plots ρ⁻(E) = δN(q,+E) - δN(q,-E) for specific q-vectors
Based on Sharma et al. npj Quantum Materials (2021)

Key features:
- Weak impurity scattering (V_imp ~ 30 meV) with full T-matrix
- Integration over disk regions around q-vectors (simple grid method)
- Energy range: 50 µeV - 8 meV to capture full gap structure
- Specific vectors: p2, p5, p6 from experimental data
- Increased broadening (50 µeV) for smoother signals
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import os
import time
from UTe2_fixed import H_full, a, b, c, set_parameters

# =============================================================================
# BRILLOUIN ZONE FOLDING
# =============================================================================

def fold_to_first_bz(kx, ky, kz):
    """
    Fold k-points back to first Brillouin zone using periodic boundary conditions
    
    Important for umklapp processes where k+q goes outside [-π/a, π/a] × [-π/b, π/b]
    
    Args:
        kx, ky, kz: Momentum components (can be outside first BZ)
        
    Returns:
        kx_folded, ky_folded, kz_folded: Folded back to first BZ
    """
    # BZ boundaries
    kx_max = np.pi / a
    ky_max = np.pi / b
    kz_max = np.pi / c
    
    # Fold using modulo arithmetic: k_folded = ((k + k_max) % (2*k_max)) - k_max
    # This maps any k to [-k_max, k_max]
    kx_folded = ((kx + kx_max) % (2 * kx_max)) - kx_max
    ky_folded = ((ky + ky_max) % (2 * ky_max)) - ky_max
    kz_folded = ((kz + kz_max) % (2 * kz_max)) - kz_max
    
    return kx_folded, ky_folded, kz_folded

# =============================================================================
# CORE PHYSICS: d-vector and BdG Hamiltonian
# =============================================================================

def d_vector(kx, ky, kz, pairing_type, C1=0.0003, C2=0.0003, C3=0.0003):
    """Vectorized triplet d-vector for UTe2 pairing symmetries"""
    kx, ky, kz = np.atleast_1d(kx), np.atleast_1d(ky), np.atleast_1d(kz)
    kx, ky, kz = np.broadcast_arrays(kx, ky, kz)
    shape = kx.shape
    
    dx = np.zeros(shape)
    dy = np.zeros(shape) 
    dz = np.zeros(shape)
    
    if pairing_type == 'B2u':
        dx = C1 * np.sin(kz * c)
        dz = C3 * np.sin(kx * a)
    elif pairing_type == 'B3u':
        dy = C2 * np.sin(kz * c)
        dz = C3 * np.sin(ky * b)
    else:
        raise ValueError(f"Unknown pairing type: {pairing_type}")
    
    dx = np.broadcast_to(dx, shape)
    dy = np.broadcast_to(dy, shape) 
    dz = np.broadcast_to(dz, shape)
    
    return np.stack([dx, dy, dz], axis=-1)


def construct_BdG_hamiltonian_vectorized(kx_array, ky_array, kz, pairing_type):
    """Vectorized 8x8 BdG Hamiltonian construction for UTe2 (4 bands)"""
    kx_flat = kx_array.flatten()
    ky_flat = ky_array.flatten()
    n_points = len(kx_flat)
    
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_matrices = np.array([sigma_x, sigma_y, sigma_z])
    
    # Vectorized d-vectors
    d_vecs = d_vector(kx_flat, ky_flat, kz, pairing_type)  # (n_points, 3)
    
    # Pre-allocate Hamiltonian array - 8x8 for 4-band system
    H_BdG_array = np.zeros((n_points, 8, 8), dtype=complex)
    
    for i, (kx, ky) in enumerate(zip(kx_flat, ky_flat)):
        # Normal state Hamiltonian (4 bands)
        H_normal = H_full(kx, ky, kz)
        
        # Gap matrix: Δ = d·σ iσ_y
        d = d_vecs[i]
        Delta_spin = (d[0] * sigma_x + d[1] * sigma_y + d[2] * sigma_z) @ (1j * sigma_y)
        
        # Expand to 4-band: Δ_4x4 = I_2 ⊗ Δ_spin
        Delta = np.kron(np.eye(2), Delta_spin)
        
        # BdG Hamiltonian
        H_BdG_array[i, :4, :4] = H_normal
        H_BdG_array[i, 4:, 4:] = -np.conj(H_full(-kx, -ky, -kz))
        H_BdG_array[i, :4, 4:] = Delta
        H_BdG_array[i, 4:, :4] = np.conj(Delta.T)
    
    return H_BdG_array.reshape(kx_array.shape + (8, 8))


def compute_green_function_vectorized(kx_array, ky_array, kz, energy, pairing_type, eta=1e-5):
    """Vectorized 8x8 BdG Green's function G(k,E)"""
    H_BdG_array = construct_BdG_hamiltonian_vectorized(kx_array, ky_array, kz, pairing_type)
    shape = H_BdG_array.shape
    n_points = shape[0] * shape[1]
    
    G_k_array = np.zeros_like(H_BdG_array)
    H_flat = H_BdG_array.reshape(n_points, 8, 8)
    G_flat = G_k_array.reshape(n_points, 8, 8)
    
    identity = np.eye(8, dtype=complex)
    for i in range(n_points):
        try:
            G_flat[i] = linalg.inv((energy + 1j*eta) * identity - H_flat[i])
        except linalg.LinAlgError:
            G_flat[i] = np.zeros((8, 8), dtype=complex)
    
    return G_k_array


def compute_local_green_function(energy, pairing_type, eta=1e-5, nk=70, kz=0):
    """Vectorized local Green's function G_loc(E) = ∑_k G(k,E)"""
    dk = 2*np.pi / nk
    kx_range = np.linspace(-np.pi/a, np.pi/a, nk)
    ky_range = np.linspace(-np.pi/b, np.pi/b, nk)
    
    KX, KY = np.meshgrid(kx_range, ky_range, indexing='ij')
    G_k_array = compute_green_function_vectorized(KX, KY, kz, energy, pairing_type, eta)
    
    # Sum over k-points with proper normalization
    normalization = (dk/(2*np.pi))**2
    G_loc = np.sum(G_k_array, axis=(0, 1)) * normalization
    
    return G_loc


def compute_t_matrix(energy, pairing_type, V_imp=0.15, eta=1e-5, nk=70, kz=0):
    """T-matrix: T(E) = [I - V_imp G_loc(E)]^-1 V_imp"""
    G_loc = compute_local_green_function(energy, pairing_type, eta, nk, kz)
    V_matrix = V_imp * np.eye(8)
    
    try:
        T_matrix = linalg.inv(np.eye(8) - V_matrix @ G_loc) @ V_matrix
    except linalg.LinAlgError:
        T_matrix = V_matrix
    
    return T_matrix


def fermi_surface_weight_vectorized(kx_array, ky_array, kz, threshold=0.001):
    """Vectorized Fermi surface weighting function"""
    kx_flat = kx_array.flatten()
    ky_flat = ky_array.flatten() 
    weights = np.zeros_like(kx_flat)
    
    for i, (kx, ky) in enumerate(zip(kx_flat, ky_flat)):
        H = H_full(kx, ky, kz)
        eigenvals = np.linalg.eigvalsh(H)
        weights[i] = np.exp(-np.min(np.abs(eigenvals))**2 / threshold**2)
    
    return weights.reshape(kx_array.shape)

# =============================================================================
# HAEM SIGNAL COMPUTATION
# =============================================================================

def compute_delta_N_element(qx, qy, qz, energy, pairing_type, T_matrix, eta=1e-5, nk=60):
    """Vectorized δN(q,E) = (1/π)Im∑_k[G₀(k,E)T(E)G₀(k+q,E)]₁₁"""
    dk = 2*np.pi / nk
    kx_range = np.linspace(-np.pi/a + qx/2, np.pi/a - qx/2, nk)
    ky_range = np.linspace(-np.pi/b + qy/2, np.pi/b - qy/2, nk)
    
    KX, KY = np.meshgrid(kx_range, ky_range, indexing='ij')
    
    # Vectorized Fermi surface weights
    fs_weights = fermi_surface_weight_vectorized(KX, KY, qz)
    valid_mask = fs_weights > 1e-6
    
    if not np.any(valid_mask):
        return 0.0
    
    # Vectorized Green's functions
    G_k_array = compute_green_function_vectorized(KX, KY, qz, energy, pairing_type, eta)
    
    # CRITICAL: Fold k+q back to first BZ for umklapp processes (e.g., q with Δky = 2π/b)
    KX_plus_q, KY_plus_q, qz_folded = fold_to_first_bz(KX + qx, KY + qy, qz)
    G_k_plus_q_array = compute_green_function_vectorized(KX_plus_q, KY_plus_q, qz_folded, energy, pairing_type, eta)
    
    # Vectorized matrix multiplication: G(k) @ T @ G(k+q)
    n_points = np.sum(valid_mask)
    G_k_flat = G_k_array[valid_mask]
    G_kq_flat = G_k_plus_q_array[valid_mask]
    weights_flat = fs_weights[valid_mask]
    
    matrix_elements = np.zeros(n_points, dtype=complex)
    for i in range(n_points):
        product = G_k_flat[i] @ T_matrix @ G_kq_flat[i] 
        # Upper-left 4x4 electron block, first orbital spin-up
        electron_block = product[:4, :4]
        matrix_elements[i] = electron_block[0, 0]
    
    # Weighted sum
    normalization = (dk/(2*np.pi))**2
    delta_N_sum = np.sum(matrix_elements * weights_flat) * normalization
    total_weight = np.sum(weights_flat) * normalization
    
    if total_weight > 0:
        delta_N_sum /= total_weight
    
    return np.imag(delta_N_sum) / np.pi


def compute_delta_N_element_integrated(qx, qy, qz, energy, pairing_type, T_matrix, eta=1e-5, nk=60, 
                                       q_radius=0.05, n_grid=5):
    """
    Integrate δN(q,E) over a disk around q-vector using simple grid sampling
    
    This matches experimental approach: finite instrument resolution means
    the signal is naturally averaged over a small area in q-space.
    
    Physical motivation: 
        - Experiments have finite momentum resolution δq
        - Signal is effectively integrated over |q' - q| < δq
        - This is a disk in 2D q-space
    
    Integration scheme (Cartesian grid):
        - Create grid of points around (qx, qy)
        - Keep only points within radius: |q' - q| < R
        - Compute δN at each point
        - Simple average: ⟨δN⟩ = (1/N) Σᵢ δN(qᵢ)
    
    Args:
        qx, qy: Center q-vector in absolute units
        qz: Out-of-plane momentum
        energy: Energy in eV
        pairing_type: 'B2u' or 'B3u'
        T_matrix: Pre-computed T-matrix
        eta: Broadening
        nk: k-space sampling
        q_radius: Integration radius in units of 2π/a (default: 0.05)
        n_grid: Grid size (n_grid x n_grid points sampled, default: 5)
        
    Returns:
        Average δN over points within the disk
    """
    # Convert radius to absolute units
    q_radius_x = q_radius * 2*np.pi / a
    q_radius_y = q_radius * 2*np.pi / b
    
    # Create grid around q-vector
    qx_range = np.linspace(qx - q_radius_x, qx + q_radius_x, n_grid)
    qy_range = np.linspace(qy - q_radius_y, qy + q_radius_y, n_grid)
    
    delta_N_values = []
    
    # Sample all points in grid that fall within the circle
    for qx_sample in qx_range:
        for qy_sample in qy_range:
            # Check if point is within radius
            dx = (qx_sample - qx) / q_radius_x
            dy = (qy_sample - qy) / q_radius_y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance <= 1.0:  # Within unit circle
                delta_N = compute_delta_N_element(qx_sample, qy_sample, qz, energy, pairing_type, T_matrix, eta, nk)
                delta_N_values.append(delta_N)
    
    # Return simple average
    if len(delta_N_values) > 0:
        return np.mean(delta_N_values)
    else:
        # Fallback to center point
        return compute_delta_N_element(qx, qy, qz, energy, pairing_type, T_matrix, eta, nk)


def compute_haem_signal(qx, qy, qz, energy, pairing_type, V_imp=0.15, eta=1e-5, nk_t=150, nk_ldos=150):
    """Compute HAEM signal ρ⁻(q,E) = δN(q,+E) - δN(q,-E)"""
    T_matrix_pos = compute_t_matrix(energy, pairing_type, V_imp, eta, nk_t, qz)
    T_matrix_neg = compute_t_matrix(-energy, pairing_type, V_imp, eta, nk_t, qz)
    
    delta_N_pos = compute_delta_N_element(qx, qy, qz, energy, pairing_type, T_matrix_pos, eta, nk_ldos)
    delta_N_neg = compute_delta_N_element(qx, qy, qz, -energy, pairing_type, T_matrix_neg, eta, nk_ldos)
    
    return delta_N_pos - delta_N_neg

# =============================================================================
# ENERGY SCAN FOR SPECIFIC VECTORS
# =============================================================================

def compute_haem_energy_scan_specific_vectors(vectors_dict, pairing_types=['B2u', 'B3u'], 
                                             energy_range=(10e-6, 500e-6), n_energies=50,
                                             kz=0, V_imp=0.15, eta=1e-5, q_radius=0.05):
    """
    Compute HAEM signal as a function of energy for specific q-vectors
    
    Integrates signal over small circles around each q-vector to match
    experimental approach and improve signal quality.
    
    Args:
        vectors_dict: Dictionary {label: (qx_2pi, qy_2pi)} where qx_2pi is in units of 2π/a
        pairing_types: List of pairing symmetries
        energy_range: (E_min, E_max) in eV
        n_energies: Number of energy points
        kz: Out-of-plane momentum
        V_imp: Impurity potential strength
        eta: Broadening parameter
        q_radius: Radius for q-space integration (in units of 2π/a)
        
    Returns:
        Dictionary with results for each pairing type and vector
    """
    print(f"🔬 Computing HAEM energy scan for specific vectors")
    print(f"   Energy range: {energy_range[0]*1e6:.0f}-{energy_range[1]*1e6:.0f} µeV")
    print(f"   Number of energies: {n_energies}")
    print(f"   Vectors: {list(vectors_dict.keys())}")
    print(f"   q-space integration radius: {q_radius:.3f} × 2π/a")
    print(f"   kz = {kz*1.39/np.pi:.2f}π/c")
    print(f"   V_imp = {V_imp:.3f} eV")
    
    energies = np.linspace(energy_range[0], energy_range[1], n_energies)
    
    results = {}
    
    for pairing_type in pairing_types:
        print(f"\n📊 Computing {pairing_type} signal...")
        start_time = time.time()
        
        # Pre-compute T-matrices for all energies
        print(f"   Pre-computing T-matrices...")
        T_matrices_pos = []
        T_matrices_neg = []
        nk_t = 70  # T-matrix k-space resolution
        nk_ldos = 60  # LDOS k-space resolution
        for i, energy in enumerate(energies):
            if (i + 1) % 5 == 0:
                print(f"      T-matrix progress: {i+1}/{n_energies}")
            T_pos = compute_t_matrix(energy, pairing_type, V_imp, eta, nk_t, kz)
            T_neg = compute_t_matrix(-energy, pairing_type, V_imp, eta, nk_t, kz)
            T_matrices_pos.append(T_pos)
            T_matrices_neg.append(T_neg)
        
        # Compute HAEM for each vector
        vector_results = {}
        
        for vector_label, (qx_2pi, qy_2pi) in vectors_dict.items():
            print(f"   Computing vector {vector_label}: ({qx_2pi:.2f}, {qy_2pi:.2f}) × 2π/(a,b)...")
            
            # Convert to absolute units
            qx = qx_2pi * 2*np.pi / a
            qy = qy_2pi * 2*np.pi / b
            qz_val = kz
            
            haem_values = []
            
            for i, energy in enumerate(energies):
                # Skip exactly zero energy (causes numerical issues)
                if abs(energy) < 1e-8:
                    haem_values.append(0.0)
                    continue
                    
                # Integrate over disk around q-vector (matches experimental finite resolution)
                delta_N_pos = compute_delta_N_element_integrated(qx, qy, qz_val, energy, pairing_type, 
                                                                 T_matrices_pos[i], eta, nk_ldos, 
                                                                 q_radius=q_radius, n_grid=5)
                delta_N_neg = compute_delta_N_element_integrated(qx, qy, qz_val, -energy, pairing_type, 
                                                                 T_matrices_neg[i], eta, nk_ldos,
                                                                 q_radius=q_radius, n_grid=5)
                haem_values.append(delta_N_pos - delta_N_neg)
                
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    progress = (list(vectors_dict.keys()).index(vector_label) + (i+1)/n_energies) / len(vectors_dict)
                    remaining = elapsed / progress - elapsed if progress > 0 else 0
                    # Show actual signal values for debugging
                    recent_haem = haem_values[-1] if haem_values else 0.0
                    print(f"      Progress: {i+1}/{n_energies} ({100*(i+1)/n_energies:.1f}%) "
                          f"- ETA: {remaining:.1f}s - Last ρ⁻: {recent_haem:.3e}")
            
            vector_results[vector_label] = np.array(haem_values)
        
        results[pairing_type] = {
            'energies': energies * 1e6,  # Convert to µeV
            'vectors': vector_results
        }
        
        total_time = time.time() - start_time
        print(f"   ✓ {pairing_type} completed in {total_time:.1f}s")
    
    return results

# =============================================================================
# DATA SAVING AND LOADING
# =============================================================================

def save_haem_results(results, vectors_dict, save_dir='raw_data_out/haem_ute2', filename='haem_results.npz'):
    """
    Save HAEM results to .npz file for later analysis
    
    Args:
        results: Dictionary with HAEM results
        vectors_dict: Dictionary of q-vectors
        save_dir: Directory to save results
        filename: Name of the output file
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    
    # Prepare data for saving
    save_dict = {}
    
    for pairing_type in results.keys():
        energies = results[pairing_type]['energies']
        save_dict[f'{pairing_type}_energies'] = energies
        
        for vector_label, vector_data in results[pairing_type]['vectors'].items():
            key = f'{pairing_type}_{vector_label}'
            save_dict[key] = vector_data
    
    # Save vector coordinates
    for label, (qx, qy) in vectors_dict.items():
        save_dict[f'qvec_{label}_x'] = qx
        save_dict[f'qvec_{label}_y'] = qy
    
    # Save metadata
    save_dict['vector_labels'] = np.array(list(vectors_dict.keys()), dtype=object)
    save_dict['pairing_types'] = np.array(list(results.keys()), dtype=object)
    
    np.savez(filepath, **save_dict)
    print(f"💾 Results saved to: {filepath}")
    
    return filepath


def load_haem_results(filepath='raw_data_out/haem_ute2/haem_results.npz'):
    """
    Load previously saved HAEM results
    
    Args:
        filepath: Path to the .npz file
        
    Returns:
        results: Dictionary with HAEM results
        vectors_dict: Dictionary of q-vectors
    """
    data = np.load(filepath, allow_pickle=True)
    
    vector_labels = data['vector_labels']
    pairing_types = data['pairing_types']
    
    # Reconstruct vectors_dict
    vectors_dict = {}
    for label in vector_labels:
        qx = float(data[f'qvec_{label}_x'])
        qy = float(data[f'qvec_{label}_y'])
        vectors_dict[label] = (qx, qy)
    
    # Reconstruct results
    results = {}
    for pairing_type in pairing_types:
        energies = data[f'{pairing_type}_energies']
        vectors = {}
        for label in vector_labels:
            key = f'{pairing_type}_{label}'
            if key in data:
                vectors[label] = data[key]
        
        results[pairing_type] = {
            'energies': energies,
            'vectors': vectors
        }
    
    print(f"📂 Results loaded from: {filepath}")
    return results, vectors_dict


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_haem_vs_energy_paper_style(results, vectors_dict, save_dir='outputs/haem_ute2'):
    """
    Plot HAEM signal vs energy in the style of Sharma et al. paper
    
    Creates a figure similar to Fig. 1d in the paper showing ρ⁻(E) vs E
    for different pairing symmetries.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    pairing_types = list(results.keys())
    vector_labels = list(vectors_dict.keys())
    energies = results[pairing_types[0]]['energies']
    
    # Create figure with subplots for each vector
    n_vectors = len(vector_labels)
    fig, axes = plt.subplots(1, n_vectors, figsize=(6*n_vectors, 5))
    
    if n_vectors == 1:
        axes = [axes]
    
    # Colors matching the paper style
    colors = {
        'B2u': '#E91E63',  # Magenta/Pink for B2u
        'B3u': '#000000'   # Black for B3u
    }
    
    linestyles = {
        'B2u': '-',
        'B3u': '-'
    }
    
    labels = {
        'B2u': r'$\rho^-_{s_+}$',
        'B3u': r'$\rho^-_{s_-}$'
    }
    
    for idx, vector_label in enumerate(vector_labels):
        ax = axes[idx]
        
        qx_2pi, qy_2pi = vectors_dict[vector_label]
        
        # Plot each pairing type
        for pairing_type in pairing_types:
            vector_data = results[pairing_type]['vectors'][vector_label]
            
            # Normalize to make comparison clear
            if np.max(np.abs(vector_data)) > 0:
                vector_data_norm = vector_data / np.max(np.abs(vector_data))
            else:
                vector_data_norm = vector_data
            
            ax.plot(energies, vector_data_norm, 
                   color=colors[pairing_type], 
                   linestyle=linestyles[pairing_type], 
                   linewidth=2.5, 
                   label=labels[pairing_type],
                   alpha=0.9)
        
        # Formatting to match paper style
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
        ax.set_xlabel('Energy (meV)', fontsize=14, fontweight='bold')
        ax.set_ylabel(r'$\rho^-$(E) (normalized)', fontsize=14, fontweight='bold')
        ax.set_title(f'{vector_label}: q = ({qx_2pi:.2f}, {qy_2pi:.2f}) × 2π/(a,b)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=12, frameon=True, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=12)
        
        # Convert µeV to meV for x-axis
        ax.set_xlim(energies[0]/1000, energies[-1]/1000)
        
        # Adjust tick labels
        xticks = ax.get_xticks()
        ax.set_xticklabels([f'{x:.1f}' for x in xticks])
    
    plt.tight_layout()
    
    # Save figure
    plot_file = os.path.join(save_dir, 'haem_vs_energy_paper_style.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Paper-style plot saved: {plot_file}")
    
    plt.show()
    
    return fig, axes


def plot_haem_combined(results, vectors_dict, save_dir='outputs/haem_ute2'):
    """
    Plot all vectors in a single panel for comparison
    """
    os.makedirs(save_dir, exist_ok=True)
    
    pairing_types = list(results.keys())
    vector_labels = list(vectors_dict.keys())
    energies = results[pairing_types[0]]['energies']
    
    fig, axes = plt.subplots(1, len(pairing_types), figsize=(8*len(pairing_types), 6))
    
    if len(pairing_types) == 1:
        axes = [axes]
    
    # Different colors for different vectors
    vector_colors = {
        'p2': '#FF6347',    # Tomato
    }
    
    for idx, pairing_type in enumerate(pairing_types):
        ax = axes[idx]
        
        for vector_label in vector_labels:
            vector_data = results[pairing_type]['vectors'][vector_label]
            qx_2pi, qy_2pi = vectors_dict[vector_label]
            
            color = vector_colors.get(vector_label, '#1f77b4')
            
            ax.plot(energies/1000, vector_data, 
                   color=color, 
                   linewidth=2.5, 
                   label=f'{vector_label}: ({qx_2pi:.2f}, {qy_2pi:.2f})',
                   alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        ax.set_xlabel('Energy (meV)', fontsize=14, fontweight='bold')
        ax.set_ylabel(r'$\rho^-$(E) (arb. units)', fontsize=14, fontweight='bold')
        ax.set_title(f'{pairing_type} Pairing', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, frameon=True)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    plot_file = os.path.join(save_dir, 'haem_vs_energy_combined.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Combined plot saved: {plot_file}")
    
    plt.show()
    
    return fig, axes


def plot_haem_simple_overlay(results, vectors_dict, save_dir='outputs/haem_ute2'):
    """
    Simple overlay plot: B2u and B3u on same axes for direct comparison
    No normalization - shows absolute signal strength
    """
    os.makedirs(save_dir, exist_ok=True)
    
    pairing_types = list(results.keys())
    vector_labels = list(vectors_dict.keys())
    energies = results[pairing_types[0]]['energies']
    
    # One subplot per vector
    n_vectors = len(vector_labels)
    fig, axes = plt.subplots(1, n_vectors, figsize=(8*n_vectors, 6))
    
    if n_vectors == 1:
        axes = [axes]
    
    # Colors for pairing types
    colors = {
        'B2u': '#E91E63',  # Magenta
        'B3u': '#2196F3'   # Blue
    }
    
    for idx, vector_label in enumerate(vector_labels):
        ax = axes[idx]
        qx_2pi, qy_2pi = vectors_dict[vector_label]
        
        # Plot both pairing types on same axes
        for pairing_type in pairing_types:
            vector_data = results[pairing_type]['vectors'][vector_label]
            
            ax.plot(energies/1000, vector_data, 
                   color=colors[pairing_type], 
                   linewidth=2.5, 
                   label=f'{pairing_type}',
                   alpha=0.9)
        
        # Formatting
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.4, linewidth=1)
        ax.set_xlabel('Energy (meV)', fontsize=14, fontweight='bold')
        ax.set_ylabel(r'$\rho^-$(E) (arb. units)', fontsize=14, fontweight='bold')
        ax.set_title(f'{vector_label}: q = ({qx_2pi:.2f}, {qy_2pi:.2f}) × 2π/(a,b)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=13, frameon=True, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        
        # Use scientific notation for y-axis if needed
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
    
    plt.tight_layout()
    
    plot_file = os.path.join(save_dir, 'haem_overlay.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Overlay plot saved: {plot_file}")
    
    plt.show()
    
    return fig, axes

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution: Compute and plot HAEM vs energy for specific vectors
    
    To load previously saved results:
        results, vectors_dict = load_haem_results('outputs/haem_ute2/haem_results.npz')
        plot_haem_simple_overlay(results, vectors_dict)
    """
    print("🚀 HAEM Signal vs Energy for UTe₂")
    print("=" * 60)
    
    # Set UTe2 parameters
    set_parameters('odd_parity_paper')
    print("✓ Using 'odd_parity_paper' parameter set")
    
    # Define specific vectors from the paper/experiment
    vectors_dict = {
        'p2': (0.374, 1.000),
        'p5': (-0.244, 1.000),
        'p6': (0.619, 0.000)
    }
    
    print(f"\n📍 Vectors to analyze:")
    for label, (qx, qy) in vectors_dict.items():
        print(f"   {label}: ({qx:+.2f}, {qy:+.2f}) × 2π/(a,b)")
    
    # Compute HAEM energy scan
    print("\n⚡ Performance settings:")
    print("   T-matrix k-resolution: 70×70")
    print("   LDOS k-resolution: 60×60")
    print("   q-integration grid: 5×5 (~20 points per disk)")
    print("   Expected time: ~8-12 min per pairing type\n")
    
    results = compute_haem_energy_scan_specific_vectors(
        vectors_dict=vectors_dict,
        pairing_types=['B2u', 'B3u'],
        energy_range=(1e-6, 3e-4), 
        n_energies=20,
        kz=0,
        V_imp=0.03,  # Weak impurity potential (30 meV) as per HAEM paper
        eta=5e-5,  # Increased broadening for smoother signal (50 µeV)
        q_radius=0.08  # Integration radius around q-vectors (4% of BZ)
    )
    
    # Save results for later analysis
    save_haem_results(results, vectors_dict)
    
    # Create plots
    print("\n📊 Creating plots...")
    plot_haem_simple_overlay(results, vectors_dict)
    
    # Print summary
    print("\n📈 Summary Statistics:")
    print("=" * 60)
    for pairing_type in ['B2u', 'B3u']:
        print(f"\n{pairing_type}:")
        for vector_label in vectors_dict.keys():
            data = results[pairing_type]['vectors'][vector_label]
            print(f"  {vector_label}:")
            print(f"    max|ρ⁻| = {np.max(np.abs(data)):.3e}")
            print(f"    mean|ρ⁻| = {np.mean(np.abs(data)):.3e}")
    
    print("\n✅ Analysis completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
