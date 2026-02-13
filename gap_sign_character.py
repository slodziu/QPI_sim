#!/usr/bin/env python3
"""
HAEM Antisymmetrized QPI Signal for UTe2
Computes ρ⁻(E) = δN(q,+E) - δN(q,-E) for triplet pairing
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import os
import time
from UTe2_fixed import H_full, a, b, c, set_parameters

def d_vector(kx, ky, kz, pairing_type, C1=0.0003, C2=0.0003, C3=0.0003):
    """Vectorized triplet d-vector for UTe2 pairing symmetries"""
    kx, ky, kz = np.atleast_1d(kx), np.atleast_1d(ky), np.atleast_1d(kz)
    # Use broadcast_arrays to ensure consistent shapes
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
        raise ValueError(f"Unsupported pairing type: {pairing_type}")
    
    # Ensure all arrays have the same shape before stacking
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
        H_k = H_full(kx, ky, kz)  # 4x4 matrix
        d_k = d_vecs[i]
        
        # Gap matrix: Δ = i(d·σ)σ_y - operates on 2x2 spin space
        gap_matrix = 1j * np.sum(d_k[:, np.newaxis, np.newaxis] * sigma_matrices, axis=0) @ sigma_y
        # Extend to 4x4 for orbital space: Δ_4x4 = I_orbital ⊗ Δ_spin
        Delta_4x4 = np.kron(np.eye(2), gap_matrix)  # 4x4 gap matrix
        
        # Construct 8x8 BdG Hamiltonian
        H_BdG_array[i, :4, :4] = H_k
        H_BdG_array[i, 4:, 4:] = -H_k.conj()
        H_BdG_array[i, :4, 4:] = Delta_4x4
        H_BdG_array[i, 4:, :4] = Delta_4x4.conj().T
    
    return H_BdG_array.reshape(kx_array.shape + (8, 8))

def construct_BdG_hamiltonian(kx, ky, kz, pairing_type):
    """Single k-point wrapper for backward compatibility"""
    H_BdG_array = construct_BdG_hamiltonian_vectorized(np.array([[kx]]), np.array([[ky]]), kz, pairing_type)
    return H_BdG_array[0, 0]

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
        matrix = energy * identity - H_flat[i] + 1j * eta * identity
        try:
            G_flat[i] = linalg.inv(matrix)
        except linalg.LinAlgError:
            G_flat[i] = linalg.pinv(matrix)
    
    return G_k_array

def compute_green_function(kx, ky, kz, energy, pairing_type, eta=1e-5):
    """Single k-point wrapper"""
    G_array = compute_green_function_vectorized(np.array([[kx]]), np.array([[ky]]), kz, energy, pairing_type, eta)
    return G_array[0, 0]

def compute_local_green_function(energy, pairing_type, eta=1e-5, nk=150, kz=0):
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

def compute_t_matrix(energy, pairing_type, V_imp=0.15, eta=1e-5, nk=150, kz=0):
    """T-matrix: T(E) = [I - V_imp G_loc(E)]^-1 V_imp"""
    G_loc = compute_local_green_function(energy, pairing_type, eta, nk, kz)
    V_matrix = V_imp * np.eye(8)  # 8x8 for 4-band BdG system
    
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
        H_k = H_full(kx, ky, kz)
        eigenvals = linalg.eigvals(H_k)
        min_gap = np.min(np.abs(eigenvals))
        weights[i] = np.exp(-min_gap**2 / threshold**2)  # eV units
    
    return weights.reshape(kx_array.shape)

def fermi_surface_weight(kx, ky, kz, threshold=0.001):
    """Single k-point wrapper"""
    return fermi_surface_weight_vectorized(np.array([[kx]]), np.array([[ky]]), kz, threshold)[0, 0]

def compute_delta_N_element_no_weights(qx, qy, qz, energy, pairing_type, T_matrix, eta=1e-5, nk=150):
    """δN(q,E) without Fermi surface weighting - pure theoretical formula"""
    dk = 2*np.pi / nk
    kx_range = np.linspace(-np.pi/a + qx/2, np.pi/a - qx/2, nk)
    ky_range = np.linspace(-np.pi/b + qy/2, np.pi/b - qy/2, nk)
    
    KX, KY = np.meshgrid(kx_range, ky_range, indexing='ij')
    
    # No fermi surface weighting - use all k-points equally
    G_k_array = compute_green_function_vectorized(KX, KY, qz, energy, pairing_type, eta)
    G_k_plus_q_array = compute_green_function_vectorized(KX + qx, KY + qy, qz, energy, pairing_type, eta)
    
    # Vectorized matrix multiplication: G(k) @ T @ G(k+q)
    shape = G_k_array.shape
    n_points_total = shape[0] * shape[1]
    
    matrix_elements = np.zeros(n_points_total, dtype=complex)
    G_k_flat = G_k_array.reshape(n_points_total, 8, 8)
    G_kq_flat = G_k_plus_q_array.reshape(n_points_total, 8, 8)
    
    for i in range(n_points_total):
        product = G_k_flat[i] @ T_matrix @ G_kq_flat[i] 
        # Upper-left 4x4 electron block, first orbital spin-up
        matrix_elements[i] = product[0, 0]
    
    # Simple k-space average
    normalization = (dk/(2*np.pi))**2
    delta_N_sum = np.mean(matrix_elements) * (2*np.pi)**2 / (dk)**2 * normalization
    
    return np.imag(delta_N_sum) / np.pi

def compute_haem_signal_comparison(qx, qy, qz, energy, pairing_type, V_imp=0.15, eta=1e-5, nk=50):
    """Compare HAEM with and without Fermi weights"""
    T_matrix_pos = compute_t_matrix(energy, pairing_type, V_imp, eta, 30, qz)
    T_matrix_neg = compute_t_matrix(-energy, pairing_type, V_imp, eta, 30, qz)
    
    # With fermi weights
    delta_N_pos_weighted = compute_delta_N_element(qx, qy, qz, energy, pairing_type, T_matrix_pos, eta, nk)
    delta_N_neg_weighted = compute_delta_N_element(qx, qy, qz, -energy, pairing_type, T_matrix_neg, eta, nk)
    haem_weighted = delta_N_pos_weighted - delta_N_neg_weighted
    
    # Without fermi weights
    delta_N_pos_pure = compute_delta_N_element_no_weights(qx, qy, qz, energy, pairing_type, T_matrix_pos, eta, nk)
    delta_N_neg_pure = compute_delta_N_element_no_weights(qx, qy, qz, -energy, pairing_type, T_matrix_neg, eta, nk)
    haem_pure = delta_N_pos_pure - delta_N_neg_pure
    
    return haem_weighted, haem_pure

def compute_delta_N_element(qx, qy, qz, energy, pairing_type, T_matrix, eta=1e-5, nk=150):
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
    G_k_plus_q_array = compute_green_function_vectorized(KX + qx, KY + qy, qz, energy, pairing_type, eta)
    
    # Vectorized matrix multiplication: G(k) @ T @ G(k+q)
    n_points = np.sum(valid_mask)
    G_k_flat = G_k_array[valid_mask]
    G_kq_flat = G_k_plus_q_array[valid_mask]
    weights_flat = fs_weights[valid_mask]
    
    matrix_elements = np.zeros(n_points, dtype=complex)
    for i in range(n_points):
        product = G_k_flat[i] @ T_matrix @ G_kq_flat[i] 
        # Upper-left 4x4 electron block
        electron_block = product[:4, :4]
        # Take first orbital, spin-up component only
        matrix_elements[i] = electron_block[0, 0]
    
    # Weighted sum
    normalization = (dk/(2*np.pi))**2
    delta_N_sum = np.sum(matrix_elements * weights_flat) * normalization
    total_weight = np.sum(weights_flat) * normalization
    
    if total_weight > 0:
        delta_N_sum /= total_weight
    
    return np.imag(delta_N_sum) / np.pi

def compute_haem_signal(qx, qy, qz, energy, pairing_type, V_imp=0.15, eta=1e-5, nk_t=150, nk_ldos=150):
    """Compute HAEM signal ρ⁻(q,E) = δN(q,+E) - δN(q,-E)"""
    T_matrix_pos = compute_t_matrix(energy, pairing_type, V_imp, eta, nk_t, qz)
    T_matrix_neg = compute_t_matrix(-energy, pairing_type, V_imp, eta, nk_t, qz)
    
    delta_N_pos = compute_delta_N_element(qx, qy, qz, energy, pairing_type, T_matrix_pos, eta, nk_ldos)
    delta_N_neg = compute_delta_N_element(qx, qy, qz, -energy, pairing_type, T_matrix_neg, eta, nk_ldos)
    
    return delta_N_pos - delta_N_neg

def generate_q_vectors(n_patches=8):
    """Generate representative q-vectors from different Fermi surface patches"""
    q_vectors = []
    vector_labels = []
    
    # Primary vectors from experimental data
    base_vectors = [
        (0.290, 0.000, 'p1'),   # Γ-X vector
        (0.43, 1.000, 'p2'),   # Diagonal vector  
        (-0.14, 1.000, 'p5'),  # Y-boundary vector
        (0.57, 0.00, 'p6'),   # Γ-Y vector
    ]
    
    for qx_2pi, qy_2pi, label in base_vectors:
        qx = qx_2pi * 2*np.pi/a
        qy = qy_2pi * 2*np.pi/b
        q_vectors.append((qx, qy))
        vector_labels.append(label)
    
    # Additional patch sampling
    for i in range(n_patches - len(base_vectors)):
        theta = 2*np.pi * i / n_patches
        qx = 0.3 * np.pi/a * np.cos(theta)
        qy = 0.3 * np.pi/b * np.sin(theta)
        q_vectors.append((qx, qy))
        vector_labels.append(f'patch_{i+1}')
    
    return q_vectors, vector_labels

def compute_energy_scan_vectorized(pairing_types=['B2u', 'B3u'], energy_range=(10e-6, 300e-6), n_energies=30,
                                   kz=0, V_imp=0.15, eta=1e-5, use_q_average=True, n_patches=6):
    """Vectorized energy scan of HAEM signal ρ⁻(E)"""
    
    print(f"🔬 Computing HAEM energy scan for {pairing_types} (vectorized)")
    print(f"   Energy range: {energy_range[0]*1e6:.0f}-{energy_range[1]*1e6:.0f} µeV")
    print(f"   kz = {kz*1.39/np.pi:.2f}π/c")
    print(f"   V_imp = {V_imp:.2f} eV")
    
    energies = np.linspace(energy_range[0], energy_range[1], n_energies)
    q_vectors, q_labels = generate_q_vectors(n_patches)
    
    results = {}
    
    for pairing_type in pairing_types:
        print(f"\n📊 Computing {pairing_type} signal (vectorized)...")
        start_time = time.time()
        
        # Pre-compute T-matrices for all energies
        T_matrices_pos = []
        T_matrices_neg = []
        
        print("   Pre-computing T-matrices...")
        for energy in energies:
            T_pos = compute_t_matrix(energy, pairing_type, V_imp, eta, 30, kz)
            T_neg = compute_t_matrix(-energy, pairing_type, V_imp, eta, 30, kz)
            T_matrices_pos.append(T_pos)
            T_matrices_neg.append(T_neg)
        
        # Store individual vector results
        vector_results = {}
        rho_minus_energies = []
        
        print("   Computing HAEM signals...")
        for i, energy in enumerate(energies):
            haem_values = np.zeros(len(q_vectors))
            for j, (qx, qy) in enumerate(q_vectors):
                qz_val = kz
                delta_N_pos = compute_delta_N_element(qx, qy, qz_val, energy, pairing_type, 
                                                    T_matrices_pos[i], eta, 25)
                delta_N_neg = compute_delta_N_element(qx, qy, qz_val, -energy, pairing_type, 
                                                    T_matrices_neg[i], eta, 25)
                haem_values[j] = delta_N_pos - delta_N_neg
                
                # Store individual vector results
                vector_label = q_labels[j]
                if vector_label not in vector_results:
                    vector_results[vector_label] = []
                vector_results[vector_label].append(haem_values[j])
            
            # Average for overall signal
            if use_q_average:
                rho_minus = np.mean(haem_values)
            else:
                rho_minus = haem_values[0]
            
            rho_minus_energies.append(rho_minus)
            
            if (i + 1) % 5 == 0:
                elapsed = time.time() - start_time
                remaining = elapsed * (n_energies - i - 1) / (i + 1)
                print(f"     Progress: {i+1}/{n_energies} ({100*(i+1)/n_energies:.1f}%) "
                      f"- ETA: {remaining:.1f}s")
        
        results[pairing_type] = {
            'energies': energies * 1e6,  # Convert to µeV
            'rho_minus': np.array(rho_minus_energies),
            'q_vectors': q_vectors,
            'q_labels': q_labels,
            'vector_results': {label: np.array(values) for label, values in vector_results.items()}
        }
        
        total_time = time.time() - start_time
        print(f"   ✓ {pairing_type} completed in {total_time:.1f}s")
    
    return results

# Backward compatibility wrapper
compute_energy_scan = compute_energy_scan_vectorized

def compute_haem_along_vector(qx, qy, qz, energy, pairing_type, n_points=20, V_imp=0.15, eta=1e-5, use_fermi_weights=True):
    """Compute HAEM signal along a q-vector from origin to (qx,qy) at fixed energy"""
    t_values = np.linspace(0.05, 1.0, n_points)  # Avoid t=0 (no scattering)
    haem_values = np.zeros(n_points)
    
    # Pre-compute T-matrices (same for all points along vector)
    T_matrix_pos = compute_t_matrix(energy, pairing_type, V_imp, eta, 30, qz)
    T_matrix_neg = compute_t_matrix(-energy, pairing_type, V_imp, eta, 30, qz)
    
    for i, t in enumerate(t_values):
        qx_t = t * qx
        qy_t = t * qy
        
        if use_fermi_weights:
            delta_N_pos = compute_delta_N_element(qx_t, qy_t, qz, energy, pairing_type, T_matrix_pos, eta, 25)
            delta_N_neg = compute_delta_N_element(qx_t, qy_t, qz, -energy, pairing_type, T_matrix_neg, eta, 25)
        else:
            delta_N_pos = compute_delta_N_element_no_weights(qx_t, qy_t, qz, energy, pairing_type, T_matrix_pos, eta, 25)
            delta_N_neg = compute_delta_N_element_no_weights(qx_t, qy_t, qz, -energy, pairing_type, T_matrix_neg, eta, 25)
        
        haem_values[i] = delta_N_pos - delta_N_neg
    
    return t_values, haem_values

def plot_diagnostics(save_dir='outputs/phase_character'):
    """Plot diagnostic information to understand HAEM signal issues"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("🔧 Running diagnostic plots...")
    
    # 1. Fermi surface weights
    kx_range = np.linspace(-np.pi/a, np.pi/a, 100)
    ky_range = np.linspace(-np.pi/b, np.pi/b, 100)
    KX, KY = np.meshgrid(kx_range, ky_range, indexing='ij')
    
    fs_weights = fermi_surface_weight_vectorized(KX, KY, 0)
    
    # 2. Gap magnitudes for both pairing types
    d_b2u = d_vector(KX, KY, 0, 'B2u')
    d_b3u = d_vector(KX, KY, 0, 'B3u')
    
    gap_mag_b2u = np.linalg.norm(d_b2u, axis=-1)
    gap_mag_b3u = np.linalg.norm(d_b3u, axis=-1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Fermi surface weights
    im1 = axes[0,0].contourf(KX*a/(2*np.pi), KY*b/(2*np.pi), fs_weights, levels=50, cmap='viridis')
    axes[0,0].set_title('Fermi Surface Weights', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('kx (2π/a)')
    axes[0,0].set_ylabel('ky (2π/b)')
    plt.colorbar(im1, ax=axes[0,0])
    
    # B2u gap magnitude
    im2 = axes[0,1].contourf(KX*a/(2*np.pi), KY*b/(2*np.pi), gap_mag_b2u*1e6, levels=50, cmap='Reds')
    axes[0,1].set_title('B2u Gap Magnitude (µeV)', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('kx (2π/a)')
    axes[0,1].set_ylabel('ky (2π/b)')
    plt.colorbar(im2, ax=axes[0,1])
    
    # B3u gap magnitude  
    im3 = axes[0,2].contourf(KX*a/(2*np.pi), KY*b/(2*np.pi), gap_mag_b3u*1e6, levels=50, cmap='Blues')
    axes[0,2].set_title('B3u Gap Magnitude (µeV)', fontsize=14, fontweight='bold')
    axes[0,2].set_xlabel('kx (2π/a)')
    axes[0,2].set_ylabel('ky (2π/b)')
    plt.colorbar(im3, ax=axes[0,2])
    
    # Local density of states at different energies
    energies_test = [50e-6, 100e-6, 200e-6]  
    for i, energy in enumerate(energies_test):
        G_loc_b2u = compute_local_green_function(energy, 'B2u', eta=1e-4, nk=50)
        ldos_b2u = -np.imag(G_loc_b2u[0,0]) / np.pi  # First orbital, spin-up
        
        G_loc_b3u = compute_local_green_function(energy, 'B3u', eta=1e-4, nk=50)
        ldos_b3u = -np.imag(G_loc_b3u[0,0]) / np.pi
        
        if i < 3:  # Only plot first 3
            axes[1,i].bar(['B2u', 'B3u'], [ldos_b2u, ldos_b3u], color=['red', 'blue'], alpha=0.7)
            axes[1,i].set_title(f'LDOS at E = {energy*1e6:.0f} µeV', fontsize=12, fontweight='bold')
            axes[1,i].set_ylabel('LDOS (states/eV)')
            axes[1,i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save diagnostics
    diag_file = os.path.join(save_dir, 'haem_diagnostics.png')
    plt.savefig(diag_file, dpi=300, bbox_inches='tight')
    print(f"🔧 Diagnostic plots saved: {diag_file}")
    
    plt.show()
    
    # Print statistics
    print(f"\n🔍 Diagnostic Summary:")
    print(f"   Max Fermi weight: {np.max(fs_weights):.2e}")
    print(f"   Effective FS area: {np.sum(fs_weights > 1e-6) / fs_weights.size:.1%}")
    print(f"   B2u gap range: {np.min(gap_mag_b2u)*1e6:.1f} - {np.max(gap_mag_b2u)*1e6:.1f} µeV")
    print(f"   B3u gap range: {np.min(gap_mag_b3u)*1e6:.1f} - {np.max(gap_mag_b3u)*1e6:.1f} µeV")

def investigate_haem_spikes(qx, qy, qz, energy, pairing_type, V_imp=0.15, eta=1e-5):
    """Investigate why HAEM signal has spikes - diagnose the matrix elements"""
    print(f"\n🔍 Investigating HAEM spikes for q=({qx*a/(2*np.pi):.3f}, {qy*b/(2*np.pi):.3f})×2π/(a,b)")
    
    # Test along the vector
    t_values = np.linspace(0.1, 1.0, 10)
    
    T_matrix_pos = compute_t_matrix(energy, pairing_type, V_imp, eta, 30, qz)
    T_matrix_neg = compute_t_matrix(-energy, pairing_type, V_imp, eta, 30, qz)
    
    results = []
    for t in t_values:
        # Test both methods
        qx_t, qy_t = t * qx, t * qy
        
        haem_weighted, haem_pure = compute_haem_signal_comparison(qx_t, qy_t, qz, energy, pairing_type, V_imp, eta, 25)
        
        # Also look at individual matrix elements 
        # Sample a few k-points to see what's happening
        kx_test, ky_test = 0.2*np.pi/a, 0.1*np.pi/b
        G_k = compute_green_function(kx_test, ky_test, qz, energy, pairing_type, eta)
        G_kq = compute_green_function(kx_test + qx_t, ky_test + qy_t, qz, energy, pairing_type, eta)
        
        product = G_k @ T_matrix_pos @ G_kq
        
        results.append({
            't': t,
            'haem_weighted': haem_weighted,
            'haem_pure': haem_pure,
            'matrix_00': product[0,0], # electron, orbital 1, spin up
            'matrix_11': product[1,1], # electron, orbital 1, spin down  
            'matrix_22': product[2,2], # electron, orbital 2, spin up
            'G_k_00': G_k[0,0],
            'G_kq_00': G_kq[0,0],
            'T_00': T_matrix_pos[0,0]
        })
    
    # Print results
    print("     t    HAEM(wt)  HAEM(pure)  Matrix[0,0]     G(k)[0,0]    G(k+q)[0,0]")
    print("   " + "="*70)
    for r in results:
        print(f"   {r['t']:.1f}  {r['haem_weighted']:8.2e}  {r['haem_pure']:8.2e}  "
              f"{r['matrix_00']:10.2e}  {r['G_k_00']:10.2e}  {r['G_kq_00']:10.2e}")
    
    return results

def plot_haem_along_vectors(pairing_types=['B2u', 'B3u'], fixed_energy=100e-6, save_dir='outputs/phase_character', use_fermi_weights=True):
    """Plot HAEM signal along each q-vector at fixed energy"""
    os.makedirs(save_dir, exist_ok=True)
    
    q_vectors, q_labels = generate_q_vectors(8)
    
    weight_str = "with Fermi weights" if use_fermi_weights else "without Fermi weights"
    print(f"🎯 Computing HAEM along vectors at E = {fixed_energy*1e6:.0f} µeV ({weight_str})")
    
    n_vectors = len(q_vectors)
    n_cols = 2
    n_rows = (n_vectors + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    colors = {'B2u': '#1f77b4', 'B3u': '#ff7f0e'}
    linestyles = {'B2u': '-', 'B3u': '--'}
    
    for i, ((qx, qy), vector_label) in enumerate(zip(q_vectors, q_labels)):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        qx_2pi = qx * a / (2*np.pi)
        qy_2pi = qy * b / (2*np.pi)
        
        for pairing_type in pairing_types:
            print(f"   Computing {pairing_type} for {vector_label}...")
            t_values, haem_values = compute_haem_along_vector(qx, qy, 0, fixed_energy, pairing_type, 
                                                            use_fermi_weights=use_fermi_weights)
            
            ax.plot(t_values, haem_values, color=colors[pairing_type], 
                   linestyle=linestyles[pairing_type], linewidth=2.5, 
                   label=f'{pairing_type}', alpha=0.8, marker='o', markersize=4)
            ax.fill_between(t_values, 0, haem_values, color=colors[pairing_type], alpha=0.2)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        ax.set_xlabel('Position along q-vector (t)', fontsize=11)
        ax.set_ylabel(f'ρ⁻(t×q, {fixed_energy*1e6:.0f}µeV)', fontsize=11)
        ax.set_title(f'{vector_label}: q = ({qx_2pi:.3f}, {qy_2pi:.3f}) × 2π/(a,b)', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
    
    # Hide unused subplots
    for i in range(n_vectors, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    weight_suffix = "weighted" if use_fermi_weights else "pure"
    vector_path_file = os.path.join(save_dir, f'haem_along_vectors_E{fixed_energy*1e6:.0f}ueV_{weight_suffix}.png')
    plt.savefig(vector_path_file, dpi=300, bbox_inches='tight')
    print(f"🎯 Vector path plots saved: {vector_path_file}")
    
    plt.show()

def plot_individual_vectors(results, save_dir='outputs/phase_character'):
    """Plot HAEM signal ρ⁻(E) for each individual q-vector"""
    os.makedirs(save_dir, exist_ok=True)
    
    pairing_types = list(results.keys())
    vector_labels = results[pairing_types[0]]['q_labels']
    energies = results[pairing_types[0]]['energies']
    
    # Color schemes for different vectors
    vector_colors = {
        'p1': '#2E8B57',    # Sea green
        'p2': '#FF6347',    # Tomato  
        'p5': '#9370DB',    # Medium purple
        'p6': '#FF8C00',    # Dark orange
        'patch_1': '#CD853F', 'patch_2': '#808080',
        'patch_3': '#20B2AA', 'patch_4': '#DC143C'
    }
    
    n_vectors = len(vector_labels)
    n_cols = 2
    n_rows = (n_vectors + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, vector_label in enumerate(vector_labels):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        # Get vector info
        qx, qy = results[pairing_types[0]]['q_vectors'][i]
        qx_2pi = qx * a / (2*np.pi)
        qy_2pi = qy * b / (2*np.pi)
        
        # Plot each pairing type for this vector
        for pairing_type in pairing_types:
            vector_data = results[pairing_type]['vector_results'][vector_label]
            color = '#1f77b4' if pairing_type == 'B2u' else '#ff7f0e'
            linestyle = '-' if pairing_type == 'B2u' else '--'
            
            ax.plot(energies, vector_data, color=color, linestyle=linestyle, 
                   linewidth=2, label=f'{pairing_type}', alpha=0.8)
            ax.fill_between(energies, 0, vector_data, color=color, alpha=0.2)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        ax.set_xlabel('Energy (µeV)', fontsize=11)
        ax.set_ylabel('ρ⁻(E)', fontsize=11)
        ax.set_title(f'{vector_label}: q = ({qx_2pi:.3f}, {qy_2pi:.3f}) × 2π/(a,b)', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_vectors, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save individual vector plots
    vector_plot_file = os.path.join(save_dir, 'haem_individual_vectors.png')
    plt.savefig(vector_plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Individual vector plots saved: {vector_plot_file}")
    
    plt.show()

def plot_haem_comparison(results, save_dir='outputs/phase_character'):
    """Plot B2u vs B3u HAEM signals ρ⁻(E)"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    colors = {'B2u': '#1f77b4', 'B3u': '#ff7f0e'}
    linestyles = {'B2u': '-', 'B3u': '--'}
    
    # Individual signals (averaged over q-vectors)
    for pairing_type in results.keys():
        data = results[pairing_type]
        energies = data['energies']
        rho_minus = data['rho_minus']
        
        ax1.plot(energies, rho_minus, color=colors[pairing_type], 
                linestyle=linestyles[pairing_type], linewidth=2.5, 
                label=f'{pairing_type} ρ⁻(E) (averaged)', alpha=0.8)
        ax1.fill_between(energies, 0, rho_minus, color=colors[pairing_type], 
                        alpha=0.2)
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    ax1.set_xlabel('Energy (µeV)', fontsize=13)
    ax1.set_ylabel('ρ⁻(E) (arb. units)', fontsize=13)
    ax1.set_title('HAEM Antisymmetrized QPI Signal for UTe₂ (q-averaged)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Difference signal
    if 'B2u' in results and 'B3u' in results:
        energies = results['B2u']['energies']
        diff_signal = results['B2u']['rho_minus'] - results['B3u']['rho_minus']
        
        ax2.plot(energies, diff_signal, color='purple', linewidth=2.5, 
                label='ρ⁻(B2u) - ρ⁻(B3u)', alpha=0.8)
        ax2.fill_between(energies, 0, diff_signal, color='purple', alpha=0.2)
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        ax2.set_xlabel('Energy (µeV)', fontsize=13)
        ax2.set_ylabel('Δρ⁻(E) (arb. units)', fontsize=13)
        ax2.set_title('B2u - B3u Difference Signal', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(save_dir, 'haem_energy_scan_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📈 Averaged comparison plot saved: {plot_file}")
    
    plt.show()
    
    # Summary statistics
    print("\n📊 HAEM Signal Summary:")
    for pairing_type in results.keys():
        data = results[pairing_type]
        rho_max = np.max(np.abs(data['rho_minus']))
        rho_mean = np.mean(np.abs(data['rho_minus']))
        print(f"   {pairing_type}: max|ρ⁻| = {rho_max:.2e}, mean|ρ⁻| = {rho_mean:.2e}")

def main():
    """Main execution function"""
    print("🚀 HAEM Antisymmetrized QPI for UTe₂")
    print("Computing ρ⁻(E) = δN(q,+E) - δN(q,-E) with triplet pairing")
    
    # Set UTe2 parameters
    set_parameters('odd_parity_paper')
    
    # Compute energy scan (vectorized for speed) - adjusted parameters
    print("\n📊 Computing energy-resolved HAEM signals...")
    results = compute_energy_scan_vectorized(
        pairing_types=['B2u', 'B3u'],
        energy_range=(50e-6, 500e-6),   # Increased to probe different energy scales
        n_energies=20,   
        kz=0,  
        V_imp=0.2,       # Increased impurity strength for clearer signal
        eta=5e-5,        # Increased broadening for smoother curves
        use_q_average=False,
        n_patches=8
    )
    
    # Plot both averaged comparison and individual vectors
    plot_haem_comparison(results)
    plot_individual_vectors(results)
    
    # NEW: Plot HAEM signal along each q-vector at fixed energy (both methods)
    print("\n🎯 Computing HAEM signals along q-vectors...")
    plot_haem_along_vectors(
        pairing_types=['B2u', 'B3u'], 
        fixed_energy=200e-6,  # Energy near gap scale
        use_fermi_weights=True
    )
    
    print("\n🎯 Computing HAEM signals along q-vectors (without Fermi weights)...")
    plot_haem_along_vectors(
        pairing_types=['B2u', 'B3u'], 
        fixed_energy=200e-6,
        use_fermi_weights=False  
    )
    
    # Plot diagnostics to understand the system
    print("\n🔧 Generating diagnostic plots...")
    plot_diagnostics()
    
    # Diagnostic: Investigate the spiky behavior
    print("\n🔬 Investigating spiky HAEM behavior...")
    qx_test, qy_test = 0.290 * 2*np.pi/a, 0.000 * 2*np.pi/b  # p1 vector
    investigate_haem_spikes(qx_test, qy_test, 0, 200e-6, 'B2u', V_imp=0.2, eta=5e-5)
    
    # Compare with and without fermi weights
    print("\n📊 Comparing HAEM with vs without Fermi weights:")
    haem_weighted, haem_pure = compute_haem_signal_comparison(qx_test, qy_test, 0, 200e-6, 'B2u', V_imp=0.2, eta=5e-5, nk=30)
    print(f"   With Fermi weights: {haem_weighted:.3e}")
    print(f"   Without weights:    {haem_pure:.3e}")
    print(f"   Ratio (pure/weighted): {haem_pure/haem_weighted if haem_weighted != 0 else 'inf':.2f}")
    
    # Diagnostic: Check a single point calculation
    print("\n🔬 Diagnostic calculation:")
    qx, qy = 0.290 * 2*np.pi/a, 0.000 * 2*np.pi/b  # p1 vector
    energy = 200e-6
    haem_b2u = compute_haem_signal(qx, qy, 0, energy, 'B2u', V_imp=0.2, eta=5e-5, nk_t=50, nk_ldos=50)
    haem_b3u = compute_haem_signal(qx, qy, 0, energy, 'B3u', V_imp=0.2, eta=5e-5, nk_t=50, nk_ldos=50)
    
    print(f"   Single point (q=p1, E={energy*1e6:.0f}µeV):")
    print(f"   B2u: ρ⁻ = {haem_b2u:.3e}")
    print(f"   B3u: ρ⁻ = {haem_b3u:.3e}")
    print(f"   Ratio: {abs(haem_b2u/haem_b3u) if haem_b3u != 0 else 'inf':.2f}")
    
    print("\n✅ HAEM calculation completed successfully!")

if __name__ == "__main__":
    main()
