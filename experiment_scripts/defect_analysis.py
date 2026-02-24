
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter, label, center_of_mass
# Load file
c_star = 0.78
ds = xr.open_dataset("experimental_data/topograph.sxm", engine="nanonis")

# Output directory for plots
output_dir = "experiment_output/defect_analysis"
os.makedirs(output_dir, exist_ok=True)
z = ds["Z"].sel(dir="forward")   
z = z.squeeze()                  
theta = np.deg2rad(0.01)
x = z.coords["x"].values
y = z.coords["y"].values
x_nm = z.coords["x"].values * 1e9
y_nm = z.coords["y"].values * 1e9
X_nm, Y_nm = np.meshgrid(x_nm, y_nm)


# Smooth and subtract to highlight defects



# --- Defect detection without smoothing ---
z_flat = z.values
vac_mask = z_flat < (np.mean(z_flat) - 3.2* np.std(z_flat))  # threshold for dark defects

# Label connected regions and find their centers
labeled, num = label(vac_mask)
positions = center_of_mass(vac_mask, labeled, range(1, num+1))


# Collect all defect positions (no classification)

# Refine defect positions: for each detected defect, search for the lowest value in a small window
defect_positions = []
window = 3  # search +/-window pixels around detected center
for py, px in positions:

    px_i = int(px)
    py_i = int(py)
    y0 = max(0, py_i - window)
    y1 = min(z_flat.shape[0], py_i + window + 1)
    x0 = max(0, px_i - window)
    x1 = min(z_flat.shape[1], px_i + window + 1)
    local = z_flat[y0:y1, x0:x1]
    min_idx = np.unravel_index(np.argmin(local), local.shape)
    refined_py = y0 + min_idx[0]
    refined_px = x0 + min_idx[1]
    defect_positions.append((refined_px, refined_py))





angle_deg = 0.39
angle_rad = np.deg2rad(angle_deg)
y0 = 0  # start at bottom
num_points = z_flat.shape[0]  # go up full height


# Set x0 in physical units (meters), convert to nearest pixel index
x0_phys = 0.23e-9+50*c_star*1e-9 # 3e-8 meters
y0 = 0  # start at bottom
num_points = z_flat.shape[0]  # go up full height
x0 = np.argmin(np.abs(x_nm - x0_phys))
print(f"x0 index: {x0}, x0 physical: {x[x0]}, requested: {x0_phys}")
y0 = 0  # start at bottom
num_points = z_flat.shape[0]  # go up full height
x0_nm = x0_phys * 1e9  # convert to nm
x0 = np.argmin(np.abs(x_nm - x0_nm))

# Calculate pixel size in nm
dx_nm = np.abs(x_nm[1] - x_nm[0])
dy_nm = np.abs(y_nm[1] - y_nm[0])

# Calculate line coordinates using physical angle
line_x = []
line_y = []
slope = np.tan(angle_rad) * (dy_nm / dx_nm)  # convert physical slope to pixel slope
for i in range(num_points):
    xi = x0 + i * slope
    yi = i
    if 0 <= xi < z_flat.shape[1] and 0 <= yi < z_flat.shape[0]:
        line_x.append(xi)
        line_y.append(yi)

# Interpolate profile values
from scipy.ndimage import map_coordinates
profile = map_coordinates(z_flat, [line_y, line_x], order=1)

# Plot topograph with all defects circled and numbered
plt.figure()
plt.imshow(z_flat, origin="lower", cmap="viridis", extent=[x.min(), x.max(), y.min(), y.max()])

if defect_positions:
    pxs, pys = zip(*defect_positions)
    plt.scatter(x[list(pxs)], y[list(pys)], facecolors="none", edgecolors="red", s=80, label="Defect")
    # Offset for label (in data units, e.g. 2% of x/y range)
    x_offset = 0.02 * (x.max() - x.min())
    y_offset = 0.02 * (y.max() - y.min())
    for idx, (px_i, py_i) in enumerate(defect_positions, 1):
        plt.text(x[px_i] + x_offset, y[py_i] + y_offset, str(idx), color="white", fontsize=8, ha="left", va="bottom", bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"))

# Superpose the extracted line on the topograph
if len(line_x) > 1 and len(line_y) > 1:
    # Clip indices to valid range
    line_x_idx = np.clip(np.round(line_x).astype(int), 0, len(x)-1)
    line_y_idx = np.clip(np.round(line_y).astype(int), 0, len(y)-1)
    line_x_plot = x[line_x_idx]
    line_y_plot = y[line_y_idx]
    plt.plot(line_x_plot, line_y_plot, color='magenta', linewidth=2, label='Profile Line')

plt.gca().set_aspect("equal")
plt.title("Detected Defects (Numbered)")
plt.colorbar(label="Height (m)")
plt.savefig(os.path.join(output_dir, "numbered_defects.png"), bbox_inches="tight", dpi=300)
plt.show()

print('Defect pixels positions:')
print(defect_positions)

num_points = z_flat.shape[0]  # go up full height

# Calculate line coordinates
line_x = []
line_y = []
for i in range(num_points):
    xi = x0 + i * np.tan(angle_rad)
    yi = i
    if 0 <= xi < z_flat.shape[1] and 0 <= yi < z_flat.shape[0]:
        line_x.append(xi)
        line_y.append(yi)

# Interpolate profile values
from scipy.ndimage import map_coordinates
profile = map_coordinates(z_flat, [line_y, line_x], order=1)

# Plot the profile
plt.figure()
# Use y_nm for the y coordinate in physical units
line_y_nm = [y_nm[int(yi)] for yi in line_y]
plt.plot(line_y_nm, profile)
plt.title(f'Line profile at x=3, angle={angle_deg}°')
plt.xlabel('y (nm)')
plt.ylabel('Height (m)')
plt.savefig(os.path.join(output_dir, 'line_profile_x3_angle4.7.png'), bbox_inches='tight', dpi=300)
plt.show()
