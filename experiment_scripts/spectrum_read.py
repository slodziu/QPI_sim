import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter

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
plt.imshow(spatial_map, vmin=0.01, vmax=2, extent=[x_nm.min(), x_nm.max(), y_nm.min(), y_nm.max()],
		   origin="lower", aspect="equal", cmap="plasma")
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
plt.title(f"dI/dV Map at Bias={nearest_bias:.3f} V")
plt.colorbar(label="dI/dV")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"spatial_map_bias_{nearest_bias:.3f}V.png"), dpi=300)
plt.close()
# --- Topograph embedded in the .3ds file ---
# The Z coordinate records the tip height at the start of each spectrum pixel,
# giving a full topographic map at no extra cost.
topo = ds.coords["Z"].values  # (y, x), metres
topo_pm = (topo - np.nanmean(topo)) * 1e12  # convert to pm, plane-subtract mean

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(
    topo_pm,
    origin="lower",
    aspect="equal",
    cmap="afmhot",
)
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Height (pm)")
ax.set_xlabel("x (pixel)")
ax.set_ylabel("y (pixel)")
ax.set_title("Topograph (Z channel from .3ds)")
topo_path = os.path.join(output_dir, "topograph_from_3ds.png")
plt.savefig(topo_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Topograph saved to", topo_path)

