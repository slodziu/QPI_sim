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
# LATTICE CONSTANTS (PHYSICAL UNITS)
# ==========================================================

a = 0.38e-9
c_star = 0.76129292851e-9
scaling_factor = 0.5

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
    'p1': (0.29, 0),
    'p2': (0.44, 1),
    'p3': (0.29, 2),
    'p4': (0, 2),
    'p5': (-0.15, 1),
    'p6': (0.59, 0)
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
# STORAGE
# ==========================================================

rho_type1 = {label: [] for label in wavevectors_phys}
rho_type2 = {label: [] for label in wavevectors_phys}
energy_list = []

# ==========================================================
# MAHAEM LOOP
# ==========================================================

for E in bias_vals:

    if E <= 0:
        continue

    g_plus  = ds_spec[di_channel].sel(bias=E,  method="nearest").values
    g_minus = ds_spec[di_channel].sel(bias=-E, method="nearest").values

    G_plus  = np.fft.fftshift(np.fft.fft2(g_plus))
    G_minus = np.fft.fftshift(np.fft.fft2(g_minus))

    energy_list.append(E)

    for label, (qx_target, qy_target) in wavevectors_phys.items():
        # Integrate over a small circle around (qx_target, qy_target)
        radius = 5  # pixels (adjust as needed)
        ix_center = np.abs(qx - qx_target).argmin()
        iy_center = np.abs(qy - qy_target).argmin()

        # Create a mask for the circle
        y_grid, x_grid = np.ogrid[:Ny, :Nx]
        mask = (x_grid - ix_center)**2 + (y_grid - iy_center)**2 <= radius**2

        rho1 = 0.0
        rho2 = 0.0
        n_mask = np.sum(mask)

        # ----- Type 1 -----
        for (Rx, Ry) in R_type1:
            # Integrate over the circle
            sum_rho = 0.0
            for iy, ix in zip(*np.where(mask)):
                phase = np.exp(1j * (qx[ix]*Rx + qy[iy]*Ry))
                rho_i = np.real(G_plus[iy, ix] * phase) - np.real(G_minus[iy, ix] * phase)
                sum_rho += rho_i
            rho1 += sum_rho / n_mask  # average over circle

        # ----- Type 2 -----
        for (Rx, Ry) in R_type2:
            sum_rho = 0.0
            for iy, ix in zip(*np.where(mask)):
                phase = np.exp(1j * (qx[ix]*Rx + qy[iy]*Ry))
                rho_i = np.real(G_plus[iy, ix] * phase) - np.real(G_minus[iy, ix] * phase)
                sum_rho += rho_i
            rho2 += sum_rho / n_mask

        rho_type1[label].append(rho1)
        rho_type2[label].append(rho2)

# ==========================================================
# PLOT ENERGY DEPENDENCE
# ==========================================================

for label in wavevectors_phys:

    plt.figure(figsize=(6,4))

    plt.plot(energy_list, rho_type1[label], label="Type 1")
    plt.plot(energy_list, rho_type2[label], label="Type 2")

    plt.axhline(0, color='black', linewidth=1)

    plt.xlabel("Energy (V)")
    plt.ylabel("ρ⁻(q,E)")
    plt.title(f"MAHAEM ρ⁻(E) at {label}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{output_dir}/rho_minus_{label}.png", dpi=300)
    plt.close()

print("MAHAEM extraction complete.")
print(f"Results saved in: {output_dir}")