import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# =============================
# Load datasets
# =============================

topo_file = "experimental_data/topograph.sxm"
spec_file = "experimental_data/spectrum.3ds"

ds_topo = xr.open_dataset(topo_file, engine="nanonis")
ds_spec = xr.open_dataset(spec_file, engine="nanonis")

# =============================
# Get spatial axes (meters)
# =============================

topo_x = ds_topo.coords["x"].values
topo_y = ds_topo.coords["y"].values

spec_x = ds_spec.coords["x"].values
spec_y = ds_spec.coords["y"].values

# =============================
# Known vacancy in topograph (pixel coords)
# =============================

vac_topo_pix = (729, 870)

# Convert to physical coordinates (meters)
vac_topo_x_m = topo_x[vac_topo_pix[0]]
vac_topo_y_m = topo_y[vac_topo_pix[1]]

print("Topograph vacancy (nm):",
      vac_topo_x_m*1e9,
      vac_topo_y_m*1e9)

# =============================
# Get dI/dV map at chosen bias
# =============================

# Find dI/dV channel
di_channel = None
for var in ds_spec.data_vars:
    if "dI/dV" in var or "Input" in var:
        di_channel = var
        break

if di_channel is None:
    raise ValueError("dI/dV channel not found.")

bias_vals = ds_spec.coords["bias"].values
b_val = bias_vals[len(bias_vals)//2]  # choose middle bias

didv_map = ds_spec[di_channel].sel(bias=b_val).values

# Smooth to avoid noise-based maximum
didv_smooth = gaussian_filter(didv_map, sigma=2)

# =============================
# Restrict search region (x > 10 nm)
# =============================

spec_x_nm = spec_x * 1e9
mask = spec_x_nm > 10

masked_map = didv_smooth.copy()
masked_map[:, ~mask] = -np.inf

brightest_pix = np.unravel_index(np.argmax(masked_map),
                                 masked_map.shape)

brightest_y_pix, brightest_x_pix = brightest_pix

brightest_x_m = spec_x[brightest_x_pix]
brightest_y_m = spec_y[brightest_y_pix]

print("Brightest dI/dV spot (nm):",
      brightest_x_m*1e9,
      brightest_y_m*1e9)

# =============================
# Compute translation (meters)
# =============================

dx_m = brightest_x_m - vac_topo_x_m
dy_m = brightest_y_m - vac_topo_y_m

print("Translation (nm):", dx_m*1e9, dy_m*1e9)

# =============================
# Convert ALL vacancy coordinates
# =============================

vacancy_coords_topo = [
    (729, 870),  # add full list here
]

# Convert to meters
vacancy_coords_m = [
    (topo_x[x], topo_y[y]) for (x, y) in vacancy_coords_topo
]

# Apply translation
aligned_coords_m = [
    (x + dx_m, y + dy_m)
    for (x, y) in vacancy_coords_m
]

# Convert to dI/dV pixel indices
def nearest_pixel(value, axis):
    return np.abs(axis - value).argmin()

aligned_coords_spec = [
    (nearest_pixel(x, spec_x),
     nearest_pixel(y, spec_y))
    for (x, y) in aligned_coords_m
]

print("Aligned dI/dV pixel coords:", aligned_coords_spec)

# =============================
# Plot proper physical overlay
# =============================

topo_img = ds_topo["Z"].sel(dir="forward").values

plt.figure(figsize=(8,8))

# Topograph in physical units
plt.imshow(topo_img,
           cmap="gray",
           alpha=0.5,
           origin="lower",
           extent=[topo_x[0]*1e9, topo_x[-1]*1e9,
                   topo_y[0]*1e9, topo_y[-1]*1e9])

# dI/dV map in physical units
plt.imshow(didv_map,
           cmap="plasma",
           alpha=0.5,
           origin="lower",
           extent=[spec_x[0]*1e9, spec_x[-1]*1e9,
                   spec_y[0]*1e9, spec_y[-1]*1e9])

# Overlay aligned vacancy markers
for (x_pix, y_pix) in aligned_coords_spec:
    x_nm = spec_x[x_pix]*1e9
    y_nm = spec_y[y_pix]*1e9
    plt.plot(x_nm, y_nm, 'ro', markersize=8)

plt.title("Proper Real-Space Alignment")
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
plt.tight_layout()
plt.savefig("experiment_output/aligned_overlay_correct.png", dpi=300)
plt.show()