import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle

a = 0.41   # nm
b = 0.61   # nm
c = 1.39   # nm
c_star = 0.76  # nm, intertellurium distance
scaling_factor = 0.5  # for qc* axis

# --- Replicate Figure 3c: g(q,0) with vectors overlaid ---
spectrum_file = "experimental_data/spectrum.3ds"
output_dir = "experiment_output/mahaem"
os.makedirs(output_dir, exist_ok=True)

ds_spec = xr.open_dataset(spectrum_file, engine="nanonis")
spec_x = ds_spec.coords["x"].values
spec_y = ds_spec.coords["y"].values
dx = spec_x[1] - spec_x[0]
dy = spec_y[1] - spec_y[0]
Nx = len(spec_x)
Ny = len(spec_y)

# Find dI/dV channel
di_channel = None
for var in ds_spec.data_vars:
    if "dI/dV" in var or "Input" in var:
        di_channel = var
        break
if di_channel is None:
    raise ValueError("dI/dV channel not found.")

bias_vals = ds_spec.coords["bias"].values
E0 = bias_vals[np.argmin(np.abs(bias_vals))]
g0 = ds_spec[di_channel].sel(bias=E0, method="nearest").values

# FFT to get g(q,0)
G0 = np.fft.fftshift(np.fft.fft2(g0))

# FFT axes in π/a and π/b units
qx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
qy = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
qx = np.fft.fftshift(qx) / (np.pi / a)  # π/a
qy = np.fft.fftshift(qy) / (np.pi / b)  # π/b

# Project to (0-11) plane: qx stays, qc* = qy * 0.5
qx_011 = qx  # π/a units
qc_011_pi_cstar = qy * scaling_factor  # π/c* units

# Log scale for FFT magnitude
G0_log = np.log(np.abs(G0) + 1e-10)

# Mask out all but the top itensities
threshold = np.percentile(G0_log, 50)
G0_log_masked = np.where(G0_log >= threshold, G0_log, np.nan)

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
im = ax.imshow(G0_log_masked, extent=[qc_011_pi_cstar.min(), qc_011_pi_cstar.max(), qx_011.min(), qx_011.max()],
               origin='lower', cmap='viridis', aspect='equal')

ax.set_xlabel(r'$q_{c^*}$ ($\pi$/c*)', fontsize=12, fontweight='bold')
ax.set_ylabel(r'$q_x$ ($\pi$/a)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')

# Brillouin zone rectangle in π/a, π/c* units
bz_rect = Rectangle((-1, -1), 2, 2, linewidth=2, edgecolor='red', facecolor='none', linestyle='--', alpha=0.8)
ax.add_patch(bz_rect)
ax.set_xlim(qc_011_pi_cstar.min(), qc_011_pi_cstar.max())
ax.set_ylim(qx_011.min(), qx_011.max())
ax.set_aspect('equal')

# --- Overlay wavevectors (π/a, π/c* units, with qy scaled by 0.5) ---
wvecs = {
    'p1': (0.29*2, 0),
    'p2': (0.44*2, 1*2),
    'p3': (0.29*2, 2*2),
    'p4': (0, 2*2),
    'p5': (-0.15*2, 1*2),
    'p6': (0.59*2, 0)
}
vector_colors = {
    'p1': '#FF0000',  # Red
    'p2': '#0000FF',  # Blue
    'p3': "#9738737A",  
    'p4': '#00FF00', 
    'p5': '#FF8800',  # Orange
    'p6': '#FF00FF'   # Magenta
}
label_positions = {
    'p1': {'offset': (0.10, 0.08), 'ha': 'left', 'va': 'bottom'},
    'p2': {'offset': (0.10, 0.08), 'ha': 'left', 'va': 'bottom'},
    'p3': {'offset': (0.10, 0.08), 'ha': 'left', 'va': 'bottom'},
    'p4': {'offset': (-0.10, 0.08), 'ha': 'right', 'va': 'bottom'},
    'p5': {'offset': (-0.10, 0.08), 'ha': 'right', 'va': 'bottom'},
    'p6': {'offset': (0.10, 0.0), 'ha': 'left', 'va': 'center'}
}
origin = (0, 0)
for label, (qx_pi, qy_pi) in wvecs.items():
    qcstar_val = qy_pi * scaling_factor  # π/c*
    endpoint = (qcstar_val*1e9, qx_pi*1e9)
    ax.annotate('', xy=endpoint, xytext=(origin[0], origin[1]),
                arrowprops=dict(arrowstyle='->', color=vector_colors[label],
                                lw=2, alpha=0.9, shrinkA=0, shrinkB=0))
    ax.plot(endpoint[0], endpoint[1], 'o', color=vector_colors[label],
            markersize=5, markeredgecolor='black', markeredgewidth=1, zorder=10)
    pos_info = label_positions[label]
    label_qc = endpoint[0] + pos_info['offset'][0]
    label_qx = endpoint[1] + pos_info['offset'][1]
    ax.text(label_qc, label_qx, label,
            color='black', fontsize=6, fontweight='bold',
            horizontalalignment=pos_info['ha'],
            verticalalignment=pos_info['va'],
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', edgecolor='black', linewidth=1.5, alpha=0.95),
            zorder=11)

cbar = plt.colorbar(im, ax=ax, label='Log FFT Magnitude', orientation='vertical',
                   fraction=0.045, pad=0.04)
cbar.ax.tick_params(labelsize=11)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

save_path = f'{output_dir}/gq0_with_vectors_qcstar_projection.png'
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved g(q,0) plot with vectors to: {save_path}")
plt.show()