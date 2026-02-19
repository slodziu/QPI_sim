#!/usr/bin/env python3
"""
3D Visualization of Energy-Dependent Gap and Phase for UTe2 (kz=0)
==================================================================
Visualizes the evolution of the spectral function A(k, E) as stacked isosurfaces in (kx, ky, E),
with phase/sign coloring for B2u and B3u d-vector symmetry. Overlays key q-vectors (p2, p5, p6).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import UTe2_fixed as ute2

# --- Parameters ---
k_points = 512
energy_points = 6
Emin, Emax = 1e-5, 0.0003  # eV (10 μeV to 300 μeV)
kz = 0.0
pairing = 'B3u'  # Only B3u as requested

# --- Lattice and vectors ---
ute2.set_parameters('odd_parity_paper_2')
a, b, c = ute2.a, ute2.b, ute2.c
pi_a = np.pi / a
# Set up axes for plotting and physics
pi_a = np.pi / a
pi_b = np.pi / b
kx_plot = np.linspace(-1, 1, k_points)  # in units of π/a
ky_plot = np.linspace(-3, 3, k_points)  # in units of π/b
KX_plot, KY_plot = np.meshgrid(kx_plot, ky_plot, indexing='ij')
# For physics, convert to 1/nm
kx_phys = KX_plot * pi_a
ky_phys = KY_plot * pi_b

# --- d-vector and gap magnitude ---
C0 = 0.0
C1 = 300e-6  # eV
C2 = 300e-6  # eV
C3 = 300e-6  # eV
def d_vector(kx, ky, kz, a, b, c, pairing):
    # kx, ky in 1/nm, kz is scalar
    if pairing == 'B2u':
        x = C1 * np.sin(kz * c)
        y = C0 * np.sin(kx * a) * np.sin(ky * b) * np.sin(kz * c)
        z = C3 * np.sin(kx * a)
    else:  # B3u
        x = C0 * np.sin(kx * a) * np.sin(ky * b) * np.sin(kz * c)
        y = C2 * np.sin(kz * c)
        z = C3 * np.sin(ky * b)
    x = np.broadcast_to(x, kx.shape)
    y = np.broadcast_to(y, kx.shape)
    z = np.broadcast_to(z, kx.shape)
    return np.stack([x, y, z], axis=-1)  # (..., 3)
# --- Spectral function calculation ---
def spectral_function(energies, weights_5f, E, threshold=5e-5):
    mask = np.abs(energies - E) < threshold
    A = np.sum(weights_5f * mask, axis=2)
    if np.max(A) > 0:
        A = A / np.max(A)
    return A

# --- Hamiltonian and weights ---
H_grid = ute2.H_full(kx_phys, ky_phys, kz)
nx, ny = kx_phys.shape
energies_arr = np.zeros((nx, ny, 4))
weights_5f = np.zeros((nx, ny, 4))
for i in range(nx):
    for j in range(ny):
        evals, evecs = np.linalg.eigh(H_grid[i, j])
        energies_arr[i, j, :] = evals.real
        weights_5f[i, j, :] = np.sum(np.abs(evecs[:2, :])**2, axis=0)

nx, ny = KX_plot.shape
# --- Energy grid ---
energy_vals = np.linspace(Emin, Emax, energy_points)

# --- Vectors in (π/a, π/b) units ---
VECTORS = {
    'p2': (0.86, 2.00),
    'p5': (-0.28, 2.00),
    'p6': (1.14, 0.00)
}
qx_arr = np.array([VECTORS[k][0] for k in VECTORS])
qy_arr = np.array([VECTORS[k][1] for k in VECTORS])
labels = list(VECTORS.keys())


# --- Plotting ---

fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(111, projection='3d')

# Scale energy to meV so the Z-axis units are roughly the same order as k-units
E_scale = 1e3

# Calculate the actual physical extent of your k-axes for the aspect ratio
kx_range = kx_plot[-1] - kx_plot[0]  # Total width (e.g., 2.0)
ky_range = ky_plot[-1] - ky_plot[0]  # Total depth (e.g., 6.0)

# 1. Choose a "visual height" for the energy stack.
visual_height = 2.0

# 2. Set the box aspect: [Width, Depth, Height]
ax.set_box_aspect([kx_range, ky_range, visual_height])

# --- Plot stacked isosurfaces ---
norm_gap = None
all_kx, all_ky, all_z, all_gap = [], [], [], []
for idx, E in enumerate(energy_vals):
    A = spectral_function(energies_arr, weights_5f, E)
    mask = A > 0.1
    if np.any(mask):
        kx_pts = KX_plot[mask]
        ky_pts = KY_plot[mask]
        dvec = d_vector(kx_phys, ky_phys, kz, a, b, c, 'B3u')
        dz_vals = dvec[..., 2][mask]
        z_pts = np.full_like(kx_pts, E * E_scale)
        all_kx.append(kx_pts)
        all_ky.append(ky_pts)
        all_z.append(z_pts)
        all_gap.append(dz_vals)
if all_kx:
    all_kx = np.concatenate(all_kx)
    all_ky = np.concatenate(all_ky)
    all_z = np.concatenate(all_z)
    all_gap = np.concatenate(all_gap)
    vabs = np.nanmax(np.abs(all_gap))
    norm_gap = plt.Normalize(vmin=-vabs, vmax=vabs)
    sc = ax.scatter(all_kx, all_ky, all_z, c=all_gap, cmap='seismic', s=7, alpha=0.8, norm=norm_gap)
    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('d_z(k) (eV)', fontsize=12)

# --- Plot Fermi surface E=0 contour ---

# --- Continuous Fermi surface contour lines at E=0 ---

from mpl_toolkits.mplot3d.art3d import Line3DCollection


# For each band, extract E=0 contour lines using plt.contour (as in spectral file)
for band in range(energies_arr.shape[2]):
    E_band = energies_arr[:, :, band]
    fig2, ax2 = plt.subplots()
    cs = ax2.contour(KX_plot, KY_plot, E_band, levels=[0.0])
    plt.close(fig2)
    for seg in cs.allsegs[0]:
        if len(seg) < 2:
            continue
        kx_seg = seg[:, 0]
        ky_seg = seg[:, 1]
        z_seg = np.full_like(kx_seg, Emin * E_scale)
        ax.plot(kx_seg, ky_seg, z_seg, color='black', linewidth=2, alpha=0.9)


# --- Overlay q-vectors ---
# Position them on the E=0 plane (Emin)
for i, label in enumerate(labels):
    # Starting point (adjust to a known FS point, e.g., (0.8, 0))
    ax.quiver(0.8, 0, Emin*E_scale, qx_arr[i], qy_arr[i], 0, 
              color='black', length=1.0, pivot='tail', linewidth=2)
    ax.text(0.8 + qx_arr[i], 0 + qy_arr[i], Emin*E_scale, label, fontsize=10, weight='bold')

ax.set_xlabel(r'$k_x$ ($\pi/a$)')
ax.set_ylabel(r'$k_y$ ($\pi/b$)')
ax.set_zlabel('Energy (meV)')
ax.set_zlim([Emin * E_scale, Emax * E_scale])

plt.title(f'UTe2 Gap Evolution: {pairing} Symmetry\nRed/Blue = Phase Sign')
plt.show()

# --- Fermi surface contour legend ---
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Fermi Surface (E=0)')]
ax.legend(handles=legend_elements, loc='upper right')

# --- Overlay q-vectors on E=0 plane ---
A0 = spectral_function(energies_arr, weights_5f, energy_vals[0])
A0_mask = A0 > 0.4
for i, label in enumerate(labels):
    idxs = np.argwhere(A0_mask)
    if len(idxs) > 0:
        kx0, ky0 = KX_plot[idxs[0][0], idxs[0][1]], KY_plot[idxs[0][0], idxs[0][1]]
        ax.quiver(kx0, ky0, energy_vals[0], qx_arr[i], qy_arr[i], 0, color='k', arrow_length_ratio=0.1, linewidth=2)
        ax.text(kx0 + qx_arr[i], ky0 + qy_arr[i], energy_vals[0], label, color='k', fontsize=12)

# --- Axes and labels ---
ax.set_xlabel(r'$k_x$ ($\pi/a$)')
ax.set_ylabel(r'$k_y$ ($\pi/b$)')
ax.set_zlabel('Energy (eV)')
ax.set_xlim([kx_plot[0], kx_plot[-1]])
ax.set_ylim([ky_plot[0], ky_plot[-1]])
# Set aspect ratio to equal in kx-ky
ax.set_box_aspect([1, (ky_plot[-1]-ky_plot[0])/(kx_plot[-1]-kx_plot[0]), (Emax-Emin)/(kx_plot[-1]-kx_plot[0])])
ax.set_zlim([Emin, Emax])
plt.tight_layout()
plt.show()
