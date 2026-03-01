import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

# ==========================================================
# FILE PATHS
# ==========================================================

topograph_file = "experimental_data/topograph.sxm"
spectrum_file  = "experimental_data/spectrum.3ds"

output_dir = "experiment_output/mahaem"
os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

ds_topo = xr.open_dataset(topograph_file, engine="nanonis")
ds_spec = xr.open_dataset(spectrum_file,  engine="nanonis")

topo_x = ds_topo.coords["x"].values
topo_y = ds_topo.coords["y"].values

spec_x = ds_spec.coords["x"].values
spec_y = ds_spec.coords["y"].values

dx = spec_x[1] - spec_x[0]
dy = spec_y[1] - spec_y[0]

Nx = len(spec_x)
Ny = len(spec_y)

# ==========================================================
# UNIT / CO-REGISTRATION CHECKS
# Topograph: 1024×1024 px, 50×50 nm, origin (0,0)
# Spectrum:   270×270 px, 66×66 nm, origin (0,0)
#
# Shared origin means topo pixel coords translate directly to
# physical meters that are valid positions within the spec FOV.
# Vacancy coords are stored as (x_pixel, y_pixel) where
# x_pixel indexes topo_x (fast-scan / horizontal axis) and
# y_pixel indexes topo_y (slow-scan / vertical axis).
# This matches standard Nanonis/WSxM image coordinate convention.
# ==========================================================

topo_fov_x = topo_x.max() - topo_x.min()
topo_fov_y = topo_y.max() - topo_y.min()
spec_fov_x = spec_x.max() - spec_x.min()
spec_fov_y = spec_y.max() - spec_y.min()

assert topo_x[0] == spec_x[0] and topo_y[0] == spec_y[0], \
    f"Topo and spec origins differ! topo=({topo_x[0]},{topo_y[0]}) spec=({spec_x[0]},{spec_y[0]})"
assert topo_fov_x <= spec_fov_x and topo_fov_y <= spec_fov_y, \
    f"Topo FOV ({topo_fov_x*1e9:.1f}×{topo_fov_y*1e9:.1f} nm) extends outside spec FOV ({spec_fov_x*1e9:.1f}×{spec_fov_y*1e9:.1f} nm)"

print(f"Co-registration OK: topo {topo_fov_x*1e9:.1f}×{topo_fov_y*1e9:.1f} nm "
      f"inside spec {spec_fov_x*1e9:.1f}×{spec_fov_y*1e9:.1f} nm, shared origin (0,0)")

# ==========================================================
# LATTICE CONSTANTS (PHYSICAL UNITS)
# ==========================================================

# Lattice constants — UTe2 orthorhombic structure (converted to metres)
a      = 0.41e-9   # a-axis
b      = 0.61e-9   # b-axis
c      = 1.39e-9   # c-axis
c_star = 0.76e-9  # intertellurium distance along c* (0-11) direction
scaling_factor = 0.5       # qy (π/b) → qc* (π/c*): c* ≈ b/2 for (0-11) projection

# ==========================================================
# SELECT dI/dV CHANNEL
# ==========================================================

di_channel = None
for var in ds_spec.data_vars:
    if "dI/dV" in var or "Input" in var:
        di_channel = var
        break

if di_channel is None:
    raise ValueError("dI/dV channel not found.")

bias_vals = ds_spec.coords["bias"].values

# ==========================================================
# VACANCY POSITIONS (PIXEL → METERS)
#
# Type 1: in-chain Te vacancies (27 defects)
# Type 2: between-chain Te vacancies (7 defects)
#
# Both are NON-MAGNETIC. Any non-zero ρ⁻ signal therefore
# originates purely from a TRS-breaking superconducting order
# parameter (chiral/topological scenario in UTe2).
#
# The two sublattices are related by a fractional translation τ,
# which introduces a q-dependent phase e^{iq·τ} between them.
# This means ρ⁻ can have opposite signs between types at certain
# q-vectors — this is PHYSICAL, not noise. Use both separately
# to cross-check: a feature appearing in both (even with a sign
# flip) is a genuine signal.
# ==========================================================

vacancy_coords_type1 = [
    (15, 29), (33, 439), (87, 703), (88, 907), (118, 153),
    (171, 337), (237, 81), (341, 680), (371, 84), (408, 472),
    (426, 780), (461, 850), (510, 834), (573, 12), (574, 500),
    (577, 1002), (660, 477), (663, 949), (729, 870),
    (627, 620), (797, 971), (844, 591), (847, 1000),
    (877, 560), (958, 239), (993, 147), (641, 138)
]

vacancy_coords_type2 = [
    (195, 669), (365, 555), (279, 366),
    (446, 105), (463, 120), (434, 902), (700, 508)
]

R_type1 = [(topo_x[x], topo_y[y]) for (x, y) in vacancy_coords_type1]
R_type2 = [(topo_x[x], topo_y[y]) for (x, y) in vacancy_coords_type2]

# Verify all vacancy positions lie within the spectrum scan area
for label, Rlist in [("type1", R_type1), ("type2", R_type2)]:
    for rx, ry in Rlist:
        assert spec_x.min() <= rx <= spec_x.max() and spec_y.min() <= ry <= spec_y.max(), \
            f"Vacancy {label} at ({rx*1e9:.2f},{ry*1e9:.2f}) nm is outside spectrum FOV!"
print(f"All {len(R_type1)} type-1 and {len(R_type2)} type-2 vacancy positions are within spectrum FOV.")

# ==========================================================
# BUILD Q GRID (rad/m)
# ==========================================================

qx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
qy = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)

qx = np.fft.fftshift(qx)
qy = np.fft.fftshift(qy)

QX, QY = np.meshgrid(qx, qy)

# ==========================================================
# DEFINE TARGET WAVEVECTORS
# ==========================================================

wavevectors_original = {
    'p2': (0.43, 0.5),
    'p4': (0, 1),
    'p5': (-0.14, 0.5),
    'p6': (0.57, 0)
}

# Convert to π/a , π/b
wavevectors_pi = {k: (v[0]*2, v[1]*2)
                  for k, v in wavevectors_original.items()}

wavevectors_projected = {}
for label, (qx_pi_a, qy_pi_b) in wavevectors_pi.items():
    qx_proj = qx_pi_a
    qc_proj = qy_pi_b * scaling_factor
    wavevectors_projected[label] = (qx_proj, qc_proj)

# Convert to physical rad/m
wavevectors_phys = {}
for label, (qx_pi_a, qc_pi_cstar) in wavevectors_projected.items():
    qx_phys = qx_pi_a * (np.pi / a)
    qy_phys = qc_pi_cstar * (np.pi / c_star)
    wavevectors_phys[label] = (qx_phys, qy_phys)

# ==========================================================
# PRECOMPUTE STRUCTURE FACTORS S(q) FOR BOTH VACANCY TYPES
#
# Sharma et al. eq 8+9:
#   ρ_MA(q,E) = Σ_i [ Re[g(q,+E)e^{iq·R_i}] - Re[g(q,-E)e^{iq·R_i}] ]
#             = Re[(G⁺-G⁻) · Σ_i e^{iq·R_i}]   (linearity of Re)
#             = Re[(G⁺-G⁻) · N·S(q)]
#
# We define S(q) = (1/N)Σ_i e^{iq·R_i}  so that ρ⁻ is per-vacancy,
# making the two vacancy types (27 vs 7) directly comparable in amplitude.
# The 2D map below is the direct implementation; the per-vector quantity
# additionally averages over a small circle in q for noise robustness.
# ==========================================================

R1 = np.array(R_type1)  # (N1, 2): columns = [x_m, y_m]
R2 = np.array(R_type2)  # (N2, 2)

# QX, QY already shape (Ny, Nx)
# Broadcast: (N_vac, Ny, Nx)
S_type1 = np.sum(
    np.exp(1j * (QX[np.newaxis] * R1[:, 0, np.newaxis, np.newaxis]
               + QY[np.newaxis] * R1[:, 1, np.newaxis, np.newaxis])),
    axis=0) / len(R_type1)   # (Ny, Nx)

S_type2 = np.sum(
    np.exp(1j * (QX[np.newaxis] * R2[:, 0, np.newaxis, np.newaxis]
               + QY[np.newaxis] * R2[:, 1, np.newaxis, np.newaxis])),
    axis=0) / len(R_type2)   # (Ny, Nx)

# ==========================================================
# STORAGE
# ==========================================================

rho_type1 = {label: [] for label in wavevectors_phys}
rho_type2 = {label: [] for label in wavevectors_phys}
energy_list = []

# Single-vacancy HAEM (no multi-atom sum) — for comparison on same plots.
# Uses the first vacancy of each type as the representative single defect.
# Change index to test a different vacancy.
R_single1 = np.array(R_type1[0])   # shape (2,)
R_single2 = np.array(R_type2[0])   # shape (2,)
rho_single1 = {label: [] for label in wavevectors_phys}
rho_single2 = {label: [] for label in wavevectors_phys}

# ==========================================================
# PREPROCESSING HELPER
# ==========================================================

def detrend_2d(arr):
    """Line-by-line linear detrend along rows then columns.
    Removes energy-asymmetric slow background that does NOT cancel in G⁺-G⁻
    and could otherwise leak into circle integrations near q=0.
    No Hann window applied: MAHAEM formula assumes G = FT[g] exactly;
    windowing introduces q-space convolution that shifts/smears target peaks.
    Edge discontinuities largely cancel in the g⁺ - g⁻ difference anyway.
    """
    arr = np.apply_along_axis(
        lambda r: r - np.polyval(np.polyfit(np.arange(len(r)), r, 1), np.arange(len(r))),
        axis=1, arr=arr)
    arr = np.apply_along_axis(
        lambda r: r - np.polyval(np.polyfit(np.arange(len(r)), r, 1), np.arange(len(r))),
        axis=0, arr=arr)
    return arr

# ==========================================================
# MAHAEM LOOP
# ==========================================================

for E in bias_vals:

    if E <= 0:
        continue

    g_plus  = detrend_2d(ds_spec[di_channel].sel(bias=E,  method="nearest").values.astype(float))
    g_minus = detrend_2d(ds_spec[di_channel].sel(bias=-E, method="nearest").values.astype(float))

    G_plus  = np.fft.fftshift(np.fft.fft2(g_plus))
    G_minus = np.fft.fftshift(np.fft.fft2(g_minus))

    energy_list.append(E)

    for label, (qx_target, qy_target) in wavevectors_phys.items():
        # Integrate over a small circle around (qx_target, qy_target)
        radius = 20  # pixels (adjust as needed)
        # Both qx and ogrid indices are in fftshifted space — consistent
        ix_center = np.abs(qx - qx_target).argmin()
        iy_center = np.abs(qy - qy_target).argmin()

        # Create a mask for the circle
        y_grid, x_grid = np.ogrid[:Ny, :Nx]
        mask = (x_grid - ix_center)**2 + (y_grid - iy_center)**2 <= radius**2

        # Extract q values and FFT difference at masked pixels (vectorized)
        rows, cols = np.where(mask)
        qx_m = qx[cols]                              # (n_mask,)
        qy_m = qy[rows]                              # (n_mask,)
        G_diff_m = G_plus[rows, cols] - G_minus[rows, cols]  # (n_mask,)
        n_mask = len(rows)

        # ----- Type 1 (vectorized over vacancies and mask pixels) -----
        # Sharma et al. eq: ρ⁻ = Re[(G⁺-G⁻)·S(q)] averaged over circle at q_target
        # phases1: (N1, n_mask) — e^{iq·R} evaluated at each mask pixel for each vacancy
        # R1/R2 here refer to the module-level precomputed arrays (not shadowed)
        phases1 = np.exp(1j * (R1[:, 0:1] * qx_m[np.newaxis, :]
                              + R1[:, 1:2] * qy_m[np.newaxis, :]))
        rho1 = np.sum(np.real(G_diff_m[np.newaxis, :] * phases1)) / (n_mask * len(R_type1))

        # ----- Type 2 (vectorized) -----
        phases2 = np.exp(1j * (R2[:, 0:1] * qx_m[np.newaxis, :]
                              + R2[:, 1:2] * qy_m[np.newaxis, :]))
        rho2 = np.sum(np.real(G_diff_m[np.newaxis, :] * phases2)) / (n_mask * len(R_type2))

        rho_type1[label].append(rho1)
        rho_type2[label].append(rho2)

        # Single-vacancy HAEM: Re[(G⁺-G⁻) · e^{iq·R}] averaged over circle
        # shape of phases: (n_mask,)
        phase_s1 = np.exp(1j * (R_single1[0] * qx_m + R_single1[1] * qy_m))
        phase_s2 = np.exp(1j * (R_single2[0] * qx_m + R_single2[1] * qy_m))
        rho_single1[label].append(np.sum(np.real(G_diff_m * phase_s1)) / n_mask)
        rho_single2[label].append(np.sum(np.real(G_diff_m * phase_s2)) / n_mask)

# ==========================================================
# PLOT ENERGY DEPENDENCE
# ==========================================================

# ==========================================================
# 2D ρ⁻(q) MAP AT FIRST POSITIVE ENERGY
# ==========================================================

first_pos_E = bias_vals[bias_vals > 0][0]
g_plus_2d  = detrend_2d(ds_spec[di_channel].sel(bias=first_pos_E,  method="nearest").values.astype(float))
g_minus_2d = detrend_2d(ds_spec[di_channel].sel(bias=-first_pos_E, method="nearest").values.astype(float))

G_plus_2d  = np.fft.fftshift(np.fft.fft2(g_plus_2d))
G_minus_2d = np.fft.fftshift(np.fft.fft2(g_minus_2d))
G_diff_2d  = G_plus_2d - G_minus_2d

rho_minus_2d_type1 = np.real(G_diff_2d * S_type1)
rho_minus_2d_type2 = np.real(G_diff_2d * S_type2)

# Convert q axes to π/a and π/c* for display
qx_plot = qx / (np.pi / a)          # π/a
qy_plot = qy / (np.pi / c_star)    # π/c*  (c_star already physical, no extra scaling)

extent_2d = [qy_plot.min(), qy_plot.max(), qx_plot.min(), qx_plot.max()]

for rho_2d, type_label, fname in [
    (rho_minus_2d_type1, "In-chain Te",      "rho_minus_2d_type1"),
    (rho_minus_2d_type2, "Between-chain Te", "rho_minus_2d_type2"),
]:
    vmax = np.percentile(np.abs(rho_2d), 99)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(rho_2d,
                   extent=extent_2d,
                   origin="lower", aspect="equal",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="ρ⁻(q,E)  [arb. units]")
    ax.set_xlabel(r"$q_{c^*}$ (π/c*)")
    ax.set_ylabel(r"$q_x$ (π/a)")
    ax.set_ylim(-1.4, 1.4)
    ax.set_title(f"ρ⁻(q, E={first_pos_E*1e3:.1f} meV) — {type_label} vacancies")
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fname}_E{first_pos_E*1e3:.1f}meV.png", dpi=300, bbox_inches="tight")
    plt.close()



for label in wavevectors_phys:

    plt.figure(figsize=(6,4))

    plt.plot(energy_list, rho_type1[label],    label="In-chain Te (MAHAEM)")
    plt.plot(energy_list, rho_type2[label],    label="Between-chain Te (MAHAEM)")
    plt.plot(energy_list, rho_single1[label],  label="In-chain Te (single, HAEM)",     ls='--', alpha=0.7)
    plt.plot(energy_list, rho_single2[label],  label="Between-chain Te (single, HAEM)", ls='--', alpha=0.7)

    plt.axhline(0, color='black', linewidth=1)

    plt.xlabel("Energy (V)")
    plt.ylabel("ρ⁻(q,E)  [per vacancy, per q-pixel]")
    plt.title(f"MAHAEM ρ⁻(E) at {label}\n(sign flip between types at some q is physical — structure factor)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{output_dir}/rho_minus_{label}.png", dpi=300)
    plt.close()

# ==========================================================
# PLOT: all vectors on one figure per vacancy type
# ==========================================================

fig, ax = plt.subplots(figsize=(8, 5))
for label in wavevectors_phys:
    l, = ax.plot(energy_list, rho_type1[label], label=f"{label} MAHAEM")
    ax.plot(energy_list, rho_single1[label], color=l.get_color(), ls='--', alpha=0.7, label=f"{label} HAEM (single)")
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel("Energy (V)")
ax.set_ylabel("ρ⁻(q,E)  [per vacancy, per q-pixel]")
ax.set_title("MAHAEM vs single-vacancy HAEM — In-chain Te, all wavevectors")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{output_dir}/rho_minus_type1_all_vectors.png", dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
for label in wavevectors_phys:
    l, = ax.plot(energy_list, rho_type2[label], label=f"{label} MAHAEM")
    ax.plot(energy_list, rho_single2[label], color=l.get_color(), ls='--', alpha=0.7, label=f"{label} HAEM (single)")
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel("Energy (V)")
ax.set_ylabel("ρ⁻(q,E)  [per vacancy, per q-pixel]")
ax.set_title("MAHAEM vs single-vacancy HAEM — Between-chain Te, all wavevectors")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{output_dir}/rho_minus_type2_all_vectors.png", dpi=300)
plt.close()

print("MAHAEM extraction complete.")
print(f"Results saved in: {output_dir}")