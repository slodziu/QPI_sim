
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

# ── custom colormap matching crystal_cleave_figure.py ───────────────────────
cmap_bluegrey = mcolors.LinearSegmentedColormap.from_list(
    "black_bluegrey",
    [(0.00, "#000000"),
     (0.45, "#2a3a4a"),
     (0.75, "#7090a8"),
     (1.00, "#d8e8f0")],
)

# Load file
ds = xr.open_dataset("experimental_data/topograph.sxm", engine="nanonis")

# Output directory for plots
output_dir = "experiment_output/read_topography"
os.makedirs(output_dir, exist_ok=True)
print("Data variables:", ds.data_vars)
print(ds["Z"])


z = ds["Z"].sel(dir="forward")   # or .isel(dir=0)
z = z.squeeze()                  # remove any leftover singleton dims

print("Z dims:", z.dims)



plt.figure()
z.plot(cmap=cmap_bluegrey)
plt.gca().set_aspect("equal")
plt.title("Raw Topography")
raw_plot_path = os.path.join(output_dir, "raw_topography.png")
plt.savefig(raw_plot_path, bbox_inches="tight", dpi=300)
plt.close()

x = z.coords["x"].values
y = z.coords["y"].values
print("x range (m):", x.min(), "to", x.max())
print("y range (m):", y.min(), "to", y.max())
X, Y = np.meshgrid(x, y)

Zvals = z.values

# Fit plane ax + by + c
A = np.c_[X.ravel(), Y.ravel(), np.ones(X.size)]
C, _, _, _ = np.linalg.lstsq(A, Zvals.ravel(), rcond=None)

plane = (C[0]*X + C[1]*Y + C[2])

# Subtract plane
z_flat = Zvals - plane



z_pm = (z_flat - z_flat.min()) * 1e12
vmax_pm = float(np.percentile(z_pm, 98))

plt.figure()
plt.imshow(
    np.clip(z_pm, 0, vmax_pm),
    extent=[x.min()*1e9, x.max()*1e9, y.min()*1e9, y.max()*1e9],
    origin="lower",
    cmap=cmap_bluegrey,
    vmin=0, vmax=vmax_pm,
)
cbar = plt.colorbar()
cbar.set_label("Height (pm)")
cbar.set_ticks([0, vmax_pm/2, vmax_pm])
cbar.set_ticklabels(["0 pm", f"{vmax_pm/2:.0f} pm", f"{vmax_pm:.0f} pm"])
plt.gca().set_aspect("equal")
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
plt.title("Plane-subtracted Topography")
flat_plot_path = os.path.join(output_dir, "plane_sub_topography.png")
plt.savefig(flat_plot_path, bbox_inches="tight", dpi=300)
plt.close()

# Plot and save FFT of log-contrast topograph
fft_map = np.fft.fftshift(np.fft.fft2(z_flat))
fft_mag = np.abs(fft_map)

plt.figure()
plt.imshow(np.log(fft_mag + 1e-12), cmap=cmap_bluegrey, origin="lower")
plt.colorbar(label="Log FFT Magnitude")
plt.title("FFT of Log-Contrast Topography")
fft_plot_path = os.path.join(output_dir, "log_contrast_topography_fft.png")
plt.savefig(fft_plot_path, bbox_inches="tight", dpi=300)
plt.close()