#!/usr/bin/env python3
"""
HAEM Signal for Specific Vectors (B2u/B3u)
==========================================
Plots HAEM signal as a function of energy for vectors p2, p5, p6,
including d-vector structure for B2u and B3u pairing states.
Integrates HAEM intensity in a small circle around each vector.

Constants:
- C0 = 0
- C1 = C2 = C3 = 300e-6 (μeV → eV)

Vectors:
p2: (0.86, 1.00) π/a, π/b
p5: (-0.28, 1.00) π/a, π/b
p6: (1.14, 0.00) π/a, π/b

"""
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# Constants
C0 = 0.0
C1 = 300e-6  # eV
C2 = 300e-6  # eV
C3 = 300e-6  # eV

# Vectors in (π/a, π/b) units
VECTORS = {
    'p2': (0.86, 2.00),
    'p5': (-0.28, 2.00),
    'p6': (1.14, 0.00)
}

# d-vector definitions

def d_vector_B2u(kx, ky, kz, a, b, c):
    # Vectorized: kx, ky can be arrays
    kx = np.asarray(kx)
    ky = np.asarray(ky)
    kz = np.asarray(kz)
    x = C1 * np.sin(kz * c)
    y = C0 * np.sin(kx * a) * np.sin(ky * b) * np.sin(kz * c)
    z = C3 * np.sin(kx * a)
    # Broadcast to same shape
    x = np.broadcast_to(x, kx.shape)
    y = np.broadcast_to(y, kx.shape)
    z = np.broadcast_to(z, kx.shape)
    return np.stack([x, y, z], axis=0)

def d_vector_B3u(kx, ky, kz, a, b, c):
    # Vectorized: kx, ky can be arrays
    kx = np.asarray(kx)
    ky = np.asarray(ky)
    kz = np.asarray(kz)
    x = C0 * np.sin(kx * a) * np.sin(ky * b) * np.sin(kz * c)
    y = C2 * np.sin(kz * c)
    z = C3 * np.sin(ky * b)
    x = np.broadcast_to(x, kx.shape)
    y = np.broadcast_to(y, kx.shape)
    z = np.broadcast_to(z, kx.shape)
    return np.stack([x, y, z], axis=0)

# Spectral function calculation

def spectral_and_jdos_for_energy(energies, weights_5f, kx_vals, ky_vals, E, energy_window=0.001):
    # Calculate spectral function for a given energy (harmonized with week16_JDOS_clean.py)
    energy_threshold = 5e-5  # Use same threshold as week16_JDOS_clean.py
    fermi_mask = np.abs(energies - E) < energy_threshold  # shape (nx, ny, nb)
    spectral_function = np.sum(weights_5f * fermi_mask, axis=2)  # shape (nx, ny)
    # Normalize spectral function
    if np.max(spectral_function) > 0:
        spectral_function /= np.max(spectral_function)
    # Apply Hanning window to reduce FFT edge artifacts
    hanning_x = np.hanning(len(kx_vals))
    hanning_y = np.hanning(len(ky_vals))
    window_2d = np.outer(hanning_x, hanning_y)
    windowed_spectral = spectral_function * window_2d
    # JDOS via FFT autocorrelation
    A_fft = np.fft.fft2(windowed_spectral)
    J_grid = np.fft.ifft2(np.abs(A_fft)**2)
    JDOS = np.real(np.fft.fftshift(J_grid))
    return spectral_function, JDOS

# HAEM signal integration for a vector

def haem_signal_for_vector(qx, qy, energies, energies_arr, weights_5f, kx_vals, ky_vals, kz, a, b, c, pairing, radius=2):
    # Direct, d-vector-weighted, antisymmetrized HAEM calculation (no FFT/JDOS)
    from scipy.interpolate import RegularGridInterpolator
    kx_mesh, ky_mesh = np.meshgrid(kx_vals, ky_vals, indexing='ij')
    haem_vs_E = []
    # Precompute d-vectors for all k
    if pairing == 'B2u':
        d_k = d_vector_B2u(kx_mesh, ky_mesh, kz, a, b, c)  # shape (3, nx, ny)
    else:
        d_k = d_vector_B3u(kx_mesh, ky_mesh, kz, a, b, c)
    d_k = np.moveaxis(d_k, 0, -1)  # shape (nx, ny, 3)
    # Interpolator for spectral function
    for idx, E in enumerate(energies):
        # Spectral function at +E
        energy_threshold = 5e-5
        fermi_mask_pos = np.abs(energies_arr - E) < energy_threshold
        A_k_pos = np.sum(weights_5f * fermi_mask_pos, axis=2)
        if np.max(A_k_pos) > 0:
            A_k_pos = A_k_pos / np.max(A_k_pos)
        # Spectral function at -E
        fermi_mask_neg = np.abs(energies_arr + E) < energy_threshold
        A_k_neg = np.sum(weights_5f * fermi_mask_neg, axis=2)
        if np.max(A_k_neg) > 0:
            A_k_neg = A_k_neg / np.max(A_k_neg)
        # Interpolators for A(k+q)
        interp_A_pos = RegularGridInterpolator((kx_vals, ky_vals), A_k_pos, bounds_error=False, fill_value=0)
        interp_A_neg = RegularGridInterpolator((kx_vals, ky_vals), A_k_neg, bounds_error=False, fill_value=0)
        # Interpolators for d(k+q)
        interp_d_pos = [RegularGridInterpolator((kx_vals, ky_vals), d_k[..., i], bounds_error=False, fill_value=0) for i in range(3)]
        # For each k, compute k+q
        kxq = kx_mesh + qx
        kyq = ky_mesh + qy
        pts = np.stack([kxq.ravel(), kyq.ravel()], axis=-1)
        A_kq_pos = interp_A_pos(pts).reshape(kx_mesh.shape)
        A_kq_neg = interp_A_neg(pts).reshape(kx_mesh.shape)
        d_kq = np.stack([interp_d_pos[i](pts).reshape(kx_mesh.shape) for i in range(3)], axis=-1)
        # d-vector weight
        dot = np.sum(d_k * d_kq, axis=-1)
        norm = np.linalg.norm(d_k, axis=-1) * np.linalg.norm(d_kq, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            weight = np.where(norm > 0, (dot / norm) ** 2, 0.0)
        # HAEM intensity at +E and -E
        I_pos = np.sum(weight * A_k_pos * A_kq_pos)
        I_neg = np.sum(weight * A_k_neg * A_kq_neg)
        haem_val = I_pos - I_neg
        haem_vs_E.append(haem_val)
    return np.array(haem_vs_E)

# Main pipeline

def main():
    # User-editable parameters for resolution
    energy_points = 50  # Number of energy points
    k_points = 513      # Number of k-space points (per axis)

    # Use UTe2_fixed for all parameters and Hamiltonian
    import UTe2_fixed as ute2
    ute2.set_parameters('odd_parity_paper_2')
    a, b, c = ute2.a, ute2.b, ute2.c
    kz = 0.0
    # Build k-space grid with user-specified points
    kx_vals = np.linspace(-np.pi/a, np.pi/a, k_points)
    ky_vals = np.linspace(-3*np.pi/b, 3*np.pi/b, k_points)
    n_kx = len(kx_vals)
    n_ky = len(ky_vals)
    KX, KY = np.meshgrid(kx_vals, ky_vals, indexing='ij')
    H_grid = ute2.H_full(KX, KY, kz)
    energies_arr = np.zeros((n_kx, n_ky, 4))
    weights_5f = np.zeros((n_kx, n_ky, 4))
    for i in range(n_kx):
        for j in range(n_ky):
            evals, evecs = np.linalg.eigh(H_grid[i, j])
            energies_arr[i, j, :] = evals.real
            weights_5f[i, j, :] = np.sum(np.abs(evecs[:2, :])**2, axis=0)
    metadata = {'source': 'UTe2_fixed', 'resolution': n_kx}
    # Energy grid: from small value near zero to max gap (300 μeV = 0.0003 eV)
    min_gap = 1e-5  # 10 μeV, avoids zero
    max_gap = 300e-6  # 300 μeV
    energy_vals = np.linspace(min_gap, max_gap, energy_points)
    # Convert vectors from π/a, π/b units to absolute units (1/nm)
    qx_pi_a = np.array([VECTORS[label][0] for label in VECTORS])
    qy_pi_b = np.array([VECTORS[label][1] for label in VECTORS])
    # qx_abs = qx_pi_a * (π/a) [1/nm], qy_abs = qy_pi_b * (π/b) [1/nm]
    qx_arr = qx_pi_a * (np.pi / a)
    qy_arr = qy_pi_b * (np.pi / b)
    labels = list(VECTORS.keys())
    # Calculate HAEM for each vector and pairing
    results = {pairing: {} for pairing in ['B2u', 'B3u']}
    for pairing in ['B2u', 'B3u']:
        for idx, label in enumerate(labels):
            haem = haem_signal_for_vector(qx_arr[idx], qy_arr[idx], energy_vals, energies_arr, weights_5f, kx_vals, ky_vals, kz, a, b, c, pairing)
            results[pairing][label] = haem
    # Plot
    plt.figure(figsize=(10, 6))
    for pairing in ['B2u', 'B3u']:
        for label in labels:
            plt.plot(energy_vals*1e3, results[pairing][label], label=f'{pairing} {label}')
    plt.xlabel('Energy (meV)')
    plt.ylabel('HAEM Signal (arb. units)')
    plt.title('HAEM Signal vs Energy for Selected Vectors (Antisymmetrized)')
    plt.legend()
    plt.grid(True)
    os.makedirs('outputs/week16', exist_ok=True)
    plt.savefig('outputs/week16/haem_signal_vectors.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
