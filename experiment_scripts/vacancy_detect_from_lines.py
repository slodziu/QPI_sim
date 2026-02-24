
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Parameters
profile_dir = "experiment_output/line_profiles"
topograph_file = "experimental_data/topograph.sxm"
circle_radius = 5  # pixels

# Load topograph
ds = xr.open_dataset(topograph_file, engine="nanonis")
z = ds["Z"].sel(dir="forward").squeeze()
x = z.coords["x"].values
y = z.coords["y"].values
x_nm = x * 1e9
y_nm = y * 1e9
z_flat = z.values

# Find all line profile files
profile_files = sorted([f for f in os.listdir(profile_dir) if f.endswith(".npy")])

vacancy_pixels = []

for idx, pf in enumerate(profile_files):
	profile = np.load(os.path.join(profile_dir, pf))
	# Find biggest dip (minimum value)
	min_idx = np.argmin(profile)
	# y position in pixels
	y_pix = min_idx
	# x position: corresponds to linecut index
	x_pix = idx
	vacancy_pixels.append((x_pix, y_pix))

# Plot topograph and overlay circles at vacancy positions
plt.figure()
plt.imshow(z_flat, origin="lower", cmap="viridis", extent=[x.min(), x.max(), y.min(), y.max()])

for x_pix, y_pix in vacancy_pixels:
	plt.scatter(x[x_pix], y[y_pix], s=120, facecolors="none", edgecolors="red", linewidths=2)

plt.title("Vacancy Detection from Line Profiles")
plt.colorbar(label="Height (m)")
plt.savefig("experiment_output/vacancy_overlay.png", bbox_inches="tight", dpi=300)
plt.show()
