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

# ==========================================================
# BUILD PHYSICAL COORDINATE ARRAYS FROM FILE METADATA
#
# xarray-nanonis returns x/y starting from 0 for both files,
# ignoring the physical scan offset.  We must rebuild the axes
# from the actual center positions stored in the file headers:
#   .sxm  → SCAN_OFFSET  (center_x, center_y) and SCAN_RANGE (size_x, size_y)
#   .3ds  → Grid settings (center_x, center_y, size_x, size_y, angle)
#
# Both scans share SCAN_ANGLE = 4.7°.  We do NOT rotate here
# because both files use the same rotated frame, so relative
# positions are consistent.  Absolute q-vectors should be
# interpreted in the rotated frame as well.
# ==========================================================

def parse_scan_range_pair(s):
    """Parse a Nanonis two-value string like '5.0E-8  5.0E-8' → (float, float)."""
    parts = s.split()
    return float(parts[0]), float(parts[1])

# --- Topograph ---
topo_range_x, topo_range_y = parse_scan_range_pair(ds_topo.attrs["SCAN_RANGE"])
topo_ctr_x,   topo_ctr_y   = parse_scan_range_pair(ds_topo.attrs["SCAN_OFFSET"])
topo_Nx = len(ds_topo.coords["x"])
topo_Ny = len(ds_topo.coords["y"])
topo_x = np.linspace(topo_ctr_x - topo_range_x / 2,
                     topo_ctr_x + topo_range_x / 2, topo_Nx)
topo_y = np.linspace(topo_ctr_y - topo_range_y / 2,
                     topo_ctr_y + topo_range_y / 2, topo_Ny)

# --- Spectrum ---
# Grid settings field: (center_x, center_y, size_x, size_y, angle)
_gs = [float(v) for v in ds_spec.attrs["Grid settings"]]
spec_ctr_x, spec_ctr_y, spec_range_x, spec_range_y = _gs[0], _gs[1], _gs[2], _gs[3]
spec_Nx = len(ds_spec.coords["x"])
spec_Ny = len(ds_spec.coords["y"])
spec_x = np.linspace(spec_ctr_x - spec_range_x / 2,
                     spec_ctr_x + spec_range_x / 2, spec_Nx)
spec_y = np.linspace(spec_ctr_y - spec_range_y / 2,
                     spec_ctr_y + spec_range_y / 2, spec_Ny)

print(f"Topo physical range:  x [{topo_x[0]*1e9:.2f}, {topo_x[-1]*1e9:.2f}] nm  "
      f"y [{topo_y[0]*1e9:.2f}, {topo_y[-1]*1e9:.2f}] nm")
print(f"Spec physical range:  x [{spec_x[0]*1e9:.2f}, {spec_x[-1]*1e9:.2f}] nm  "
      f"y [{spec_y[0]*1e9:.2f}, {spec_y[-1]*1e9:.2f}] nm")

dx = spec_x[1] - spec_x[0]
dy = spec_y[1] - spec_y[0]

Nx = spec_Nx
Ny = spec_Ny

# ==========================================================
# CO-REGISTRATION CHECKS (physical coordinates)
# ==========================================================

assert spec_x[0] <= topo_x[0] and topo_x[-1] <= spec_x[-1], \
    f"Topo x [{topo_x[0]*1e9:.1f}, {topo_x[-1]*1e9:.1f}] nm extends outside " \
    f"spec x [{spec_x[0]*1e9:.1f}, {spec_x[-1]*1e9:.1f}] nm"
# y: topo bottom may extend slightly below spec (known offset ~3 nm) — handled by filtering below
if not (spec_y[0] <= topo_y[0]):
    print(f"WARNING: topo y_min ({topo_y[0]*1e9:.2f} nm) is below spec y_min ({spec_y[0]*1e9:.2f} nm) "
          f"by {(spec_y[0]-topo_y[0])*1e9:.2f} nm — vacancies in that strip will be dropped.")

print(f"Co-registration OK: topo center ({topo_ctr_x*1e9:.2f}, {topo_ctr_y*1e9:.2f}) nm  "
      f"spec center ({spec_ctr_x*1e9:.2f}, {spec_ctr_y*1e9:.2f}) nm  "
      f"offset = ({(topo_ctr_x-spec_ctr_x)*1e9:.2f}, {(topo_ctr_y-spec_ctr_y)*1e9:.2f}) nm")

# ==========================================================
# LATTICE CONSTANTS (PHYSICAL UNITS)
# ==========================================================

# Lattice constants — UTe2 orthorhombic structure (converted to metres)
a      = 0.41e-9   # a-axis
b      = 0.61e-9   # b-axis
c      = 1.39e-9   # c-axis
c_star = 0.76e-9  # intertellurium distance along c* (0-11) direction
# No scaling_factor needed: wavevectors below are defined directly in (π/a, π/c*) units,
# and the q axes are normalised by π/c_star directly — same convention as spectrum_read.py.

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

# Filter out vacancies outside the spectrum FOV (can happen at topo bottom edge)
def _in_spec(rx, ry):
    return spec_x[0] <= rx <= spec_x[-1] and spec_y[0] <= ry <= spec_y[-1]

R_type1_raw = [(topo_x[x], topo_y[y]) for (x, y) in vacancy_coords_type1]
R_type2_raw = [(topo_x[x], topo_y[y]) for (x, y) in vacancy_coords_type2]

R_type1 = [(rx, ry) for (rx, ry) in R_type1_raw if _in_spec(rx, ry)]
R_type2 = [(rx, ry) for (rx, ry) in R_type2_raw if _in_spec(rx, ry)]

if len(R_type1) < len(R_type1_raw):
    print(f"Type-1: dropped {len(R_type1_raw)-len(R_type1)} of {len(R_type1_raw)} vacancies outside spec FOV → {len(R_type1)} remaining")
if len(R_type2) < len(R_type2_raw):
    print(f"Type-2: dropped {len(R_type2_raw)-len(R_type2)} of {len(R_type2_raw)} vacancies outside spec FOV → {len(R_type2)} remaining")
print(f"Using {len(R_type1)} type-1 and {len(R_type2)} type-2 vacancy positions.")

# Express vacancy positions relative to the spectrum's first pixel (0,0 in FFT space).
# NumPy FFT assigns e^{-iq·r} with r=0 at the first array element, so R_i must use
# the same origin as the spectrum pixel grid, not absolute physical coordinates.
R_type1_fft = [(rx - spec_x[0], ry - spec_y[0]) for (rx, ry) in R_type1]
R_type2_fft = [(rx - spec_x[0], ry - spec_y[0]) for (rx, ry) in R_type2]

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

# Wavevectors defined directly in (π/a, π/c*) — matches spectrum_read.py wvecs.
# q_x component is the fast-scan direction (a-axis), q_c* is the c*-projected direction.
wavevectors_pi_cstar = {
    'p2': ( 0.84,  1.0),   # (q_x [π/a],  q_c* [π/c*])
    'p4': ( 0.00,  2.0),
    'p5': (-0.28,  1.0),
    'p6': ( 1.14,  0.0),
}

# Convert directly to physical rad/m
wavevectors_phys = {
    label: (qx_pi * (np.pi / a), qc_pi * (np.pi / c_star))
    for label, (qx_pi, qc_pi) in wavevectors_pi_cstar.items()
}

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

R1 = np.array(R_type1_fft)  # (N1, 2): positions relative to spec corner, in metres
R2 = np.array(R_type2_fft)  # (N2, 2)

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

# Real/imaginary decomposition of MAHAEM:
#   ρ⁻(q,E) = Re[ΔG · S]  =  Re[ΔG]·Re[S]  -  Im[ΔG]·Im[S]
#                          =  ρ_cos          +  ρ_sin
# ρ_cos: correlates Re[ΔG] with the cosine (symmetric) superposition of defect phases.
# ρ_sin: correlates Im[ΔG] with the sine   (antisymmetric) superposition.
# A genuine gap-sign signal should appear in BOTH; artifacts typically appear in only one.
rho_cos_type1 = {label: [] for label in wavevectors_phys}
rho_sin_type1 = {label: [] for label in wavevectors_phys}
rho_cos_type2 = {label: [] for label in wavevectors_phys}
rho_sin_type2 = {label: [] for label in wavevectors_phys}

# Single-vacancy HAEM averaged over ALL vacancies of each type,
# then also store per-vacancy arrays to show spread.
# MAHAEM ≈ mean-HAEM when phases add constructively (commensurate q),
# and MAHAEM < mean-|HAEM| when phases partially cancel (random positions).
# The ratio |MAHAEM| / mean-|HAEM| is a direct measure of phase coherence.
rho_mean_haem1 = {label: [] for label in wavevectors_phys}   # mean over all type-1 vacancies
rho_mean_haem2 = {label: [] for label in wavevectors_phys}   # mean over all type-2 vacancies
rho_std_haem1  = {label: [] for label in wavevectors_phys}   # std dev (spread)
rho_std_haem2  = {label: [] for label in wavevectors_phys}

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

        # --- Real/imaginary decomposition ---
        # Re[S] = (1/N)Σ cos(q·R),  Im[S] = (1/N)Σ sin(q·R)
        # ρ_cos = Re[ΔG]·Re[S],   ρ_sin = -Im[ΔG]·Im[S]
        ReG = np.real(G_diff_m)   # shape (n_mask,)
        ImG = np.imag(G_diff_m)

        ReS1 = np.sum(np.cos(R1[:, 0:1] * qx_m[np.newaxis, :]
                           + R1[:, 1:2] * qy_m[np.newaxis, :]), axis=0) / len(R_type1)
        ImS1 = np.sum(np.sin(R1[:, 0:1] * qx_m[np.newaxis, :]
                           + R1[:, 1:2] * qy_m[np.newaxis, :]), axis=0) / len(R_type1)
        rho_cos_type1[label].append(np.sum(ReG * ReS1) / n_mask)
        rho_sin_type1[label].append(np.sum(-ImG * ImS1) / n_mask)

        ReS2 = np.sum(np.cos(R2[:, 0:1] * qx_m[np.newaxis, :]
                           + R2[:, 1:2] * qy_m[np.newaxis, :]), axis=0) / len(R_type2)
        ImS2 = np.sum(np.sin(R2[:, 0:1] * qx_m[np.newaxis, :]
                           + R2[:, 1:2] * qy_m[np.newaxis, :]), axis=0) / len(R_type2)
        rho_cos_type2[label].append(np.sum(ReG * ReS2) / n_mask)
        rho_sin_type2[label].append(np.sum(-ImG * ImS2) / n_mask)

        # Mean single-HAEM over all vacancies + std dev
        # per_vac shape: (N_vac,)  — HAEM value for each individual vacancy
        per_vac1 = np.array([
            np.sum(np.real(G_diff_m * np.exp(1j * (ri[0]*qx_m + ri[1]*qy_m)))) / n_mask
            for ri in R1
        ])
        per_vac2 = np.array([
            np.sum(np.real(G_diff_m * np.exp(1j * (ri[0]*qx_m + ri[1]*qy_m)))) / n_mask
            for ri in R2
        ])
        rho_mean_haem1[label].append(per_vac1.mean())
        rho_std_haem1[label].append(per_vac1.std())
        rho_mean_haem2[label].append(per_vac2.mean())
        rho_std_haem2[label].append(per_vac2.std())
        # (kept as dummy for backward compat if needed)

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
                   origin="lower", aspect="auto",
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

    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    fig.subplots_adjust(hspace=0.08)
    E_arr = np.array(energy_list)

    # Top panel: MAHAEM vs mean single-HAEM with ±1σ spread band
    ax = axes[0]
    ax.plot(E_arr, rho_type1[label],     color='C0', label="In-chain MAHAEM")
    ax.plot(E_arr, rho_mean_haem1[label], color='C0', ls='--', label="In-chain mean HAEM")
    ax.plot(E_arr, rho_type2[label],     color='C1', label="B/w-chain MAHAEM")
    ax.plot(E_arr, rho_mean_haem2[label], color='C1', ls='--', label="B/w-chain mean HAEM")
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel("ρ⁻  [arb.]")
    ax.set_title(f"MAHAEM ρ⁻(E) at {label}  |  dashed=mean HAEM")
    ax.legend(fontsize=7)

    # Bottom panel: real/imaginary decomposition
    ax = axes[1]
    l1, = ax.plot(E_arr, rho_cos_type1[label], label="In-chain cos (Re[ΔG]·Re[S])")
    ax.plot(E_arr, rho_sin_type1[label],
            color=l1.get_color(), ls=':', label="In-chain sin (−Im[ΔG]·Im[S])")
    l2, = ax.plot(E_arr, rho_cos_type2[label], label="B/w-chain cos (Re[ΔG]·Re[S])")
    ax.plot(E_arr, rho_sin_type2[label],
            color=l2.get_color(), ls=':', label="B/w-chain sin (−Im[ΔG]·Im[S])")
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Energy (V)")
    ax.set_ylabel("ρ⁻  [arb.]")
    ax.set_title("Decomposition: solid=cos (Re[S]), dotted=sin (Im[S])")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/rho_minus_{label}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Standalone sin (Im[S]) component plot
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(E_arr, rho_sin_type1[label], color='C0', label="In-chain sin (−Im[ΔG]·Im[S])")
    ax.plot(E_arr, rho_sin_type2[label], color='C1', label="B/w-chain sin (−Im[ΔG]·Im[S])")
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Energy (V)")
    ax.set_ylabel("ρ⁻  [arb.]")
    ax.set_title(f"Sin component ρ⁻(E) at {label}  [−Im[ΔG]·Im[S]]")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rho_sin_{label}.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================================
# PLOT: all vectors on one figure per vacancy type
# ==========================================================

E_arr = np.array(energy_list)

fig, ax = plt.subplots(figsize=(8, 5))
for label in wavevectors_phys:
    l, = ax.plot(E_arr, rho_type1[label], label=f"{label} MAHAEM")
    ax.plot(E_arr, rho_mean_haem1[label], color=l.get_color(), ls='--', alpha=0.7, label=f"{label} mean HAEM")
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel("Energy (V)")
ax.set_ylabel("ρ⁻(q,E)  [arb.]")
ax.set_title("MAHAEM vs mean HAEM — In-chain Te, all wavevectors")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{output_dir}/rho_minus_type1_all_vectors.png", dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
for label in wavevectors_phys:
    l, = ax.plot(E_arr, rho_type2[label], label=f"{label} MAHAEM")
    ax.plot(E_arr, rho_mean_haem2[label], color=l.get_color(), ls='--', alpha=0.7, label=f"{label} mean HAEM")
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel("Energy (V)")
ax.set_ylabel("ρ⁻(q,E)  [arb.]")
ax.set_title("MAHAEM vs mean HAEM — Between-chain Te, all wavevectors")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{output_dir}/rho_minus_type2_all_vectors.png", dpi=300)
plt.close()

print("MAHAEM extraction complete.")
print(f"Results saved in: {output_dir}")