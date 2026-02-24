import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

# =========================
# Parameters
# =========================
c_star = 0.76129292851e-9
angle_deg = 0.39
angle_rad = np.deg2rad(angle_deg)

input_file = "experimental_data/topograph.sxm"
output_dir = "experiment_output/line_profiles"
os.makedirs(output_dir, exist_ok=True)

# =========================
# Load data
# =========================
ds = xr.open_dataset(input_file, engine="nanonis")
z = ds["Z"].sel(dir="forward").squeeze()

x = z.coords["x"].values
y = z.coords["y"].values
x_nm = x * 1e9
y_nm = y * 1e9
z_flat = z.values

dx_nm = np.abs(x_nm[1] - x_nm[0])
dy_nm = np.abs(y_nm[1] - y_nm[0])

# Convert line angle into pixel slope
slope = np.tan(angle_rad) * (dy_nm / dx_nm)

# =========================
# Define linecuts
# =========================
x0_nm_start = 0.225
x0_nm_stop = x_nm[-1]
step_nm = c_star * 1e9 / 2

x0_nm_vals = np.arange(x0_nm_start, x0_nm_stop, step_nm)
x0_indices = [np.argmin(np.abs(x_nm - val)) for val in x0_nm_vals]

num_points = z_flat.shape[0]
line_y_pixels = np.arange(num_points)

# Containers for alternating types
type1_positions = []
type2_positions = []

# =========================
# Process each linecut
# =========================
for idx, x0 in enumerate(x0_indices):

    line_x = x0 + line_y_pixels * slope
    line_y = line_y_pixels

    line_x = np.clip(line_x, 0, z_flat.shape[1] - 1)
    line_y = np.clip(line_y, 0, z_flat.shape[0] - 1)

    profile = map_coordinates(z_flat, [line_y, line_x], order=1)

    # Linear detrend
    p = np.polyfit(line_y, profile, 1)
    baseline = np.polyval(p, line_y)
    detrended = profile - baseline

    noise_level = np.std(detrended)
    expected_spacing_pixels = int((c_star * 1e9) / dy_nm * 0.8)

    minima, _ = find_peaks(
        -detrended,
        prominence=7 * noise_level,
        distance=expected_spacing_pixels,
        width=3
    )

    print(f"Linecut {idx}: found {len(minima)} minima")

    # =========================
    # Save alternating types
    # =========================
    for m in minima:
        x_pix = int(round(line_x[m]))
        y_pix = int(round(line_y[m]))

        if idx % 2 == 0:
            type1_positions.append([x_pix, y_pix])
        else:
            type2_positions.append([x_pix, y_pix])

# Convert to arrays
type1_positions = np.array(type1_positions)
type2_positions = np.array(type2_positions)

# Save to separate files
np.save(os.path.join(output_dir, "type1_minima_pixels.npy"), type1_positions)
np.save(os.path.join(output_dir, "type2_minima_pixels.npy"), type2_positions)

print("Saved:")
print(" - type1_minima_pixels.npy")
print(" - type2_minima_pixels.npy")
print("Done.")