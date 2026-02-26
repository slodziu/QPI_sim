import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt

# Path to .3ds file
file_path = "experimental_data/spectrum.3ds"

# Output directory for defect analysis
output_dir = "experiment_output/defect_analysis"
os.makedirs(output_dir, exist_ok=True)

# Load .3ds file
ds = xr.open_dataset(file_path, engine="nanonis")

# Find dI/dV channel
di_channel = None
for var in ds.data_vars:
	if "dI/dV" in var or "Input 3" in var:
		di_channel = var
		break
if di_channel is None:
	raise ValueError("dI/dV channel not found.")

# Get spatial axes
x = ds.coords["x"].values
y = ds.coords["y"].values

# Convert spatial axes to pixel indices
def coord_to_pixel(coord, axis):
	return np.abs(axis - coord).argmin()

orig_size = 1024
new_size = 270

# Original vacancy coordinates (from 1024x1024 topograph)
vacancy_coords_orig = [
	(15, 29), (33, 439), (87, 703), (88, 907), (118, 153), (171, 337), (237, 81),
	(341, 680), (371, 84), (408, 472), (426, 780), (461, 850), (510, 834), (573, 12),
	(574, 500), (577, 1002), (660, 477), (663, 949), (729, 870), (627, 620),
	(797, 971), (844, 591), (847, 1000), (877, 560), (958, 239), (993, 147), (641, 138)
]
vacancy_coords_type2_orig = [
	(195, 669), (365, 555), (279, 366), (446, 105), (463, 120), (434, 902), (700, 508)
]

# Rescale coordinates to 270x270 grid
def rescale_coords(coords, orig_size, new_size):
	return [
		(int(round(x * new_size / orig_size)), int(round(y * new_size / orig_size)))
		for (x, y) in coords
	]

vacancy_coords = rescale_coords(vacancy_coords_orig, orig_size, new_size)
vacancy_coords_type2 = rescale_coords(vacancy_coords_type2_orig, orig_size, new_size)

window_size = 120
half = window_size // 2

 # Get bias values
if "bias" in ds.coords:
	bias = ds.coords["bias"].values
elif "Sweep Signal" in ds.coords:
	bias = ds.coords["Sweep Signal"].values
else:
	raise ValueError("Bias/energy coordinate not found.")

print("Bias values:", bias)

# For each bias, extract windows around vacancies
for b_idx, b_val in enumerate(bias):
	# Get dI/dV map at this bias
	didv_map = ds[di_channel].sel(bias=b_val, method="nearest").values
	print(f"Bias index {b_idx}, value {b_val}: dI/dV map shape {didv_map.shape}")

	# Save folder for this bias
	bias_folder = os.path.join(output_dir, f"bias_{b_val:.3f}V")
	os.makedirs(bias_folder, exist_ok=True)


	# Plot and save type 1 vacancies
	for i, (x0, y0) in enumerate(vacancy_coords):
		x_min = x0 - half
		x_max = x0 + half
		y_min = y0 - half
		y_max = y0 + half
		if (
			x_min < 0 or x_max > didv_map.shape[1] or
			y_min < 0 or y_max > didv_map.shape[0]
		):
			print(f"Skipping vacancy {i} at ({x0},{y0}) for bias {b_val:.3f} V: out of bounds")
			continue
		window = didv_map[y_min:y_max, x_min:x_max]
		if window.shape == (window_size, window_size):
			print(f"Processing vacancy {i} at ({x0},{y0}) for bias {b_val:.3f} V")
			np.save(os.path.join(bias_folder, f"vacancy_{i}_type1.npy"), window)
			# Plot
			fig, ax = plt.subplots()
			im = ax.imshow(window, origin="lower", cmap="plasma")
			ax.set_title(f"Type 1 Vacancy (Bias={b_val:.3f} V)")
			fig.colorbar(im, ax=ax, label="dI/dV (V)")
			plt.savefig(os.path.join(bias_folder, f"vacancy_{i}_type1.png"), bbox_inches="tight", dpi=300)
			plt.close(fig)
		else:
			print(f"Skipping vacancy {i} at ({x0},{y0}) for bias {b_val:.3f} V: window shape {window.shape}")

	# Plot and save type 2 vacancies
	for i, (x0, y0) in enumerate(vacancy_coords_type2):
		x_min = x0 - half
		x_max = x0 + half
		y_min = y0 - half
		y_max = y0 + half
		if (
			x_min < 0 or x_max > didv_map.shape[1] or
			y_min < 0 or y_max > didv_map.shape[0]
		):
			print(f"Skipping type2 vacancy {i} at ({x0},{y0}) for bias {b_val:.3f} V: out of bounds")
			continue
		window = didv_map[y_min:y_max, x_min:x_max]
		if window.shape == (window_size, window_size):
			print(f"Processing type2 vacancy {i} at ({x0},{y0}) for bias {b_val:.3f} V")
			np.save(os.path.join(bias_folder, f"vacancy_{i}_type2.npy"), window)
			# Plot
			fig, ax = plt.subplots()
			im = ax.imshow(window, origin="lower", cmap="plasma")
			ax.set_title(f"Type 2 Vacancy (Bias={b_val:.3f} V)")
			fig.colorbar(im, ax=ax, label="dI/dV (V)")
			plt.savefig(os.path.join(bias_folder, f"vacancy_{i}_type2.png"), bbox_inches="tight", dpi=300)
			plt.close(fig)
		else:
			print(f"Skipping type2 vacancy {i} at ({x0},{y0}) for bias {b_val:.3f} V: window shape {window.shape}")

print("Saved vacancy windows for each bias to", output_dir)
