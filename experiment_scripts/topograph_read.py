
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# Load file
ds = xr.open_dataset("experimental_data/topograph.sxm", engine="nanonis")

# Output directory for plots
output_dir = r"C:\\Masters\\Code\\QPI_sim\\experiment_output\\read_topography"
os.makedirs(output_dir, exist_ok=True)
print("Data variables:", ds.data_vars)
print(ds["Z"])


z = ds["Z"].sel(dir="forward")   # or .isel(dir=0)
z = z.squeeze()                  # remove any leftover singleton dims

print("Z dims:", z.dims)



plt.figure()
z.plot(cmap="viridis")
plt.gca().set_aspect("equal")
plt.title("Raw Topography")
raw_plot_path = os.path.join(output_dir, "raw_topography.png")
plt.savefig(raw_plot_path, bbox_inches="tight", dpi=300)
plt.close()

x = z.coords["x"].values
y = z.coords["y"].values

X, Y = np.meshgrid(x, y)

Zvals = z.values

# Fit plane ax + by + c
A = np.c_[X.ravel(), Y.ravel(), np.ones(X.size)]
C, _, _, _ = np.linalg.lstsq(A, Zvals.ravel(), rcond=None)

plane = (C[0]*X + C[1]*Y + C[2])

# Subtract plane
z_flat = Zvals - plane



plt.figure()
plt.imshow(
    z_flat,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    cmap="viridis"
)
plt.colorbar(label="Height (m)")
plt.gca().set_aspect("equal")
plt.title("Plane-subtracted Topography")
flat_plot_path = os.path.join(output_dir, "plane_subtracted_topography.png")
plt.savefig(flat_plot_path, bbox_inches="tight", dpi=300)
plt.close()


fft = np.fft.fftshift(np.fft.fft2(z_flat))
fft_mag = np.abs(fft)

plt.imshow(np.log(fft_mag))
plt.savefig(os.path.join(output_dir, "fft_magnitude.png"), bbox_inches="tight", dpi=300)

