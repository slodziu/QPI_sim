import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

a = 0.41   # nm
b = 0.61   # nm
c = 1.39   # nm
c_star = 0.76  # nm, intertellurium distance
# Path to .3ds file
file_path = "experimental_data/spectrum.3ds"

# Output directory for plots
output_dir = "experiment_output/read_topography"
os.makedirs(output_dir, exist_ok=True)

# Load .3ds file using xarray-nanonis
ds = xr.open_dataset(file_path, engine="nanonis")

# Print dataset structure
print("Dimensions:", ds.dims)
print("Coordinates:", ds.coords)
print("Data variables:", ds.data_vars)

# Find dI/dV channel (handle spaces/parentheses)
di_channel = None
for var in ds.data_vars:
	if "dI/dV" in var or "Input 3" in var:
		di_channel = var
		break
if di_channel is None:
	raise ValueError("dI/dV channel not found.")

 # Get bias, spatial axes
if "bias" in ds.coords:
	bias = ds.coords["bias"].values
elif "Sweep Signal" in ds.coords:
	bias = ds.coords["Sweep Signal"].values
else:
	raise ValueError("Bias/energy coordinate not found.")
x = ds.coords["x"].values
y = ds.coords["y"].values
print("maximum x (m):", x.max())
print("maximum y (m):", y.max())
# Print energies (bias values)
print("Energies (Bias values in V):")
print(bias)

# Convert spatial axes to nm if needed
if np.max(x) < 1e-5:
	x_nm = x * 1e9
else:
	x_nm = x
if np.max(y) < 1e-5:
	y_nm = y * 1e9
else:
	y_nm = y

# Plot dI/dV spectrum at center pixel
center_x = x_nm[len(x_nm)//2]
center_y = y_nm[len(y_nm)//2]
di_data = ds[di_channel]
center_spec = di_data.sel(x=center_x, y=center_y, method="nearest")


# Plot log-scaled center pixel spectrum
plt.figure()
plt.plot(bias, np.log(np.abs(center_spec) + 1e-12))
plt.xlabel("Bias (V)")
plt.ylabel("log(|dI/dV|)")
plt.title("Log-Scaled Center Pixel dI/dV Spectrum")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "center_pixel_spectrum_log.png"), dpi=300)
plt.close()

# Plot dI/dV spatial map at selected bias value
selected_bias = 0.0  # Change as needed
idx = np.abs(bias - selected_bias).argmin()
nearest_bias = bias[idx]
spatial_map = di_data.sel(bias=nearest_bias, method="nearest")


# Plot log-scaled dI/dV spatial map
plt.figure()
plt.imshow(spatial_map, vmin=0.2, vmax=0.45, extent=[x_nm.min(), x_nm.max(), y_nm.min(), y_nm.max()],
		   origin="lower", aspect="equal", cmap="plasma")
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
plt.title(f"dI/dV Map at Bias={nearest_bias:.3f} V")
plt.colorbar(label="dI/dV")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"spatial_map_bias_{nearest_bias:.3f}V.png"), dpi=300)
plt.close()

# Plot and save FFT of dI/dV spatial map
fft_map = np.fft.fftshift(np.fft.fft2(spatial_map))
fft_mag = np.abs(fft_map)
scaling_factor = 0.5  # for qc* axis
# Calculate FFT axes in π/a and π/c* units
Nx, Ny = spatial_map.shape
qx = 2 * np.pi * np.fft.fftfreq(Nx, d=(x_nm[1] - x_nm[0]))
qy = 2 * np.pi * np.fft.fftfreq(Ny, d=(y_nm[1] - y_nm[0]))
qx = np.fft.fftshift(qx) / (np.pi / a)  # π/a
qy = np.fft.fftshift(qy) / (np.pi / b)  # π/b
qcstar = qy * scaling_factor  # π/c*

# Prepare meshgrid for plotting
qcstar_grid, qx_grid = np.meshgrid(qcstar, qx)

plt.figure()
plt.imshow(np.log(fft_mag + 1e-12), vmin=2.5, vmax=7,
           extent=[qcstar.min(), qcstar.max(), qx.min(), qx.max()],
           cmap="viridis", origin="lower", aspect="equal")
plt.colorbar(label="Log FFT Magnitude")
plt.title(f"FFT of dI/dV Map at Bias={nearest_bias:.3f} V")

# Overlay wavevectors
wvecs = {
    'p1': (0.29*2, 0),
    'p2': (0.44*2, 1),
    'p3': (0.29*2, 2),
    'p4': (0, 2),
    'p5': (-0.15*2, 1),
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
    endpoint = (qcstar_val, qx_pi)
    plt.annotate('', xy=endpoint, xytext=(origin[0], origin[1]),
                 arrowprops=dict(arrowstyle='->', color=vector_colors[label],
                                 lw=2, alpha=0.9, shrinkA=0, shrinkB=0))
    plt.plot(endpoint[0], endpoint[1], 'o', color=vector_colors[label],
             markersize=5, markeredgecolor='black', markeredgewidth=1, zorder=10)
    pos_info = label_positions[label]
    label_qc = endpoint[0] + pos_info['offset'][0]
    label_qx = endpoint[1] + pos_info['offset'][1]
    plt.text(label_qc, label_qx, label,
             color='black', fontsize=6, fontweight='bold',
             horizontalalignment=pos_info['ha'],
             verticalalignment=pos_info['va'],
             bbox=dict(boxstyle='round,pad=0.3',
                       facecolor='white', edgecolor='black', linewidth=1.5, alpha=0.95),
             zorder=11)

fft_plot_path = os.path.join(output_dir, f"spatial_map_bias_{nearest_bias:.3f}V_fft_with_vectors.png")
plt.tight_layout()
plt.savefig(fft_plot_path, bbox_inches="tight", dpi=300)
plt.close()
print("Plots saved to", output_dir)

