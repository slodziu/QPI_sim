from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import map_coordinates

# Parameters
c_star = 0.78e-9
angle_deg = 0.39  # vertical line
angle_rad = np.deg2rad(angle_deg)
input_file = "experimental_data/topograph.sxm"
output_dir = "experiment_output/line_profiles"
os.makedirs(output_dir, exist_ok=True)

# Load data
ds = xr.open_dataset(input_file, engine="nanonis")
z = ds["Z"].sel(dir="forward").squeeze()
x = z.coords["x"].values
y = z.coords["y"].values
x_nm = x * 1e9
y_nm = y * 1e9
z_flat = z.values

# Pixel size
dx_nm = np.abs(x_nm[1] - x_nm[0])
dy_nm = np.abs(y_nm[1] - y_nm[0])
slope = np.tan(angle_rad) * (dy_nm / dx_nm)

# Linecut x0 values
x0_nm_start = 0.23  # in nm
x0_nm_stop = x_nm[-1]
step_nm = c_star * 1e9 / 2  # c_star/2 in nm
x0_nm_vals = []
cur = x0_nm_start
while cur <= x0_nm_stop:
	x0_nm_vals.append(cur)
	cur += step_nm


from scipy.signal import find_peaks


# Vectorized line profile extraction
num_linecuts = len(x0_nm_vals)
num_points = z_flat.shape[0]
x0_indices = np.array([np.argmin(np.abs(x_nm - x0_nm_val)) for x0_nm_val in x0_nm_vals])

# Build 2D arrays for line_x and line_y
line_y = np.arange(num_points)
line_x_matrix = np.zeros((num_linecuts, num_points))
line_y_matrix = np.zeros((num_linecuts, num_points))
for idx, x0 in enumerate(x0_indices):
	line_x_matrix[idx, :] = x0 + line_y * slope
	line_y_matrix[idx, :] = line_y

# Clip indices to valid range
line_x_matrix = np.clip(line_x_matrix, 0, z_flat.shape[1]-1)
line_y_matrix = np.clip(line_y_matrix, 0, z_flat.shape[0]-1)

# Extract all profiles in batch
profiles = np.empty((num_linecuts, num_points))
for idx in range(num_linecuts):
	profiles[idx] = map_coordinates(z_flat, [line_y_matrix[idx], line_x_matrix[idx]], order=1)

from scipy.signal import find_peaks



# Improved minima detection: always accept the minimum in any region below threshold
all_minima = []
for idx in range(num_linecuts):
	profile = profiles[idx]
	line_y_nm = np.array([y_nm[int(yi)] for yi in line_y])
	# Fit baseline using RANSAC (robust to outliers)
	X = line_y_nm.reshape(-1, 1)
	y_profile = profile
	ransac = make_pipeline(PolynomialFeatures(degree=1), RANSACRegressor())
	ransac.fit(X, y_profile)
	baseline = ransac.predict(X)
	deviation = y_profile - baseline
	std_dev = np.std(deviation)
	threshold = std_dev * 2.7
	below_thresh = deviation < -threshold
	minima = []
	i = 0
	while i < len(below_thresh):
		if below_thresh[i]:
			start = i
			while i < len(below_thresh) and below_thresh[i]:
				i += 1
			region = deviation[start:i]
			if len(region) > 0:
				min_idx = np.argmin(region) + start
				minima.append(min_idx)
		else:
			i += 1
	minima = np.array(minima)
	# Store (x_idx, y_idx, value) for deduplication
	for m in minima:
		all_minima.append((idx, m, profile[m]))

# Deduplicate: keep only the lowest y (deepest) value within a proximity window
proximity_x = 2  # lines
proximity_y = 8  # pixels
all_minima = np.array(all_minima, dtype=[('x', int), ('y', int), ('val', float)])
keep = np.ones(len(all_minima), dtype=bool)
for i in range(len(all_minima)):
	if not keep[i]:
		continue
	xi, yi, vali = all_minima[i]
	for j in range(i+1, len(all_minima)):
		if not keep[j]:
			continue
		xj, yj, valj = all_minima[j]
		if abs(xi - xj) <= proximity_x and abs(yi - yj) <= proximity_y:
			# Keep only the one with the lowest value (deepest dip)
			if vali < valj:
				keep[j] = False
			else:
				keep[i] = False
				break
dedup_minima = all_minima[keep]

# For each linecut, plot and save only deduplicated minima
for idx in range(num_linecuts):
	profile = profiles[idx]
	line_y_nm = np.array([y_nm[int(yi)] for yi in line_y])
	# Fit baseline using RANSAC (robust to outliers)
	X = line_y_nm.reshape(-1, 1)
	y_profile = profile
	ransac = make_pipeline(PolynomialFeatures(degree=1), RANSACRegressor())
	ransac.fit(X, y_profile)
	baseline = ransac.predict(X)
	deviation = y_profile - baseline
	std_dev = np.std(deviation)
	threshold = std_dev * 2.7
	plt.figure()
	plt.plot(line_y_nm, profile, label='Profile')
	plt.plot(line_y_nm, baseline, color='blue', linestyle='--', linewidth=2, label='Baseline')
	plt.fill_between(line_y_nm, baseline - std_dev, baseline + std_dev, color='blue', alpha=0.2, label='±1 std dev')
	# Visualize threshold line
	plt.plot(line_y_nm, baseline - threshold, color='red', linestyle=':', linewidth=2, label='Threshold (2.7 std)')
	# Get deduplicated minima for this linecut
	minima = dedup_minima[dedup_minima['x'] == idx]['y']
	# Merge very close minima (within 3 points)
	minima = np.sort(minima)
	merged_minima = []
	i = 0
	while i < len(minima):
		group = [minima[i]]
		while i + 1 < len(minima) and minima[i+1] - minima[i] <= 3:
			group.append(minima[i+1])
			i += 1
		# Keep the deepest minimum in the group
		if group:
			vals = [profile[m] for m in group]
			merged_minima.append(group[np.argmin(vals)])
		i += 1
	merged_minima = np.array(merged_minima)
	print(f"Linecut {idx}: Deduplicated minima indices: {minima}")
	print(f"Linecut {idx}: Merged minima indices: {merged_minima}")
	print(f"Linecut {idx}: Minima values: {[profile[m] for m in merged_minima]}")
	np.save(os.path.join(output_dir, f"profile_{idx:03d}.npy"), profile)
	if len(merged_minima) <= 3:
		np.save(os.path.join(output_dir, f"minima_{idx:03d}.npy"), merged_minima)
		for m in merged_minima:
			plt.axvline(line_y_nm[m], color='orange', linestyle='--', linewidth=1)
	else:
		print(f"Linecut {idx}: Too many minima ({len(merged_minima)}), skipping vacancy marking.")
	plt.title(f'Line profile at x0={x0_nm_vals[idx]:.2f} nm, angle={angle_deg}°')
	plt.xlabel('y (nm)')
	plt.ylabel('Height (m)')
	plt.legend()
	plt.savefig(os.path.join(output_dir, f'line_profile_{idx:03d}.png'), bbox_inches='tight', dpi=300)
	plt.close()
