import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# =============================
# Load datasets
# =============================

topo_file = "experimental_data/topograph.sxm"
spec_file = "experimental_data/spectrum.3ds"

ds_topo = xr.open_dataset(topo_file, engine="nanonis")
ds_spec = xr.open_dataset(spec_file, engine="nanonis")

# =============================
# Extract physical spatial axes (meters)
# =============================

topo_x = ds_topo.coords["x"].values
topo_y = ds_topo.coords["y"].values

spec_x = ds_spec.coords["x"].values
spec_y = ds_spec.coords["y"].values

print("Topograph range (nm):",
      topo_x[0]*1e9, "→", topo_x[-1]*1e9)
print("dI/dV range (nm):",
      spec_x[0]*1e9, "→", spec_x[-1]*1e9)

# =============================
# Your vacancy list in TOPOGRAPH PIXELS
# =============================

vacancy_coords_topo = [
    (729, 870),
    # add full list here
]

# =============================
# Convert vacancy pixels → physical coordinates
# =============================

vacancy_coords_m = [
    (topo_x[x], topo_y[y])
    for (x, y) in vacancy_coords_topo
]

# =============================
# Convert physical → dI/dV pixel indices
# =============================

def nearest_pixel(value, axis):
    return np.abs(axis - value).argmin()

aligned_coords_spec = [
    (
        nearest_pixel(x_m, spec_x),
        nearest_pixel(y_m, spec_y)
    )
    for (x_m, y_m) in vacancy_coords_m
]

print("Mapped vacancy coords in dI/dV pixels:",
      aligned_coords_spec)

# =============================
# Get one dI/dV map for visual check
# =============================

di_channel = None
for var in ds_spec.data_vars:
    if "dI/dV" in var or "Input" in var:
        di_channel = var
        break

if di_channel is None:
    raise ValueError("dI/dV channel not found.")

bias_vals = ds_spec.coords["bias"].values
b_val = bias_vals[len(bias_vals)//2]

didv_map = ds_spec[di_channel].sel(bias=b_val).values

# =============================
# Plot overlay in REAL SPACE
# =============================

topo_img = ds_topo["Z"].sel(dir="forward").values

plt.figure(figsize=(8,8))

# Topograph
plt.imshow(topo_img,
           cmap="gray",
           alpha=0.5,
           origin="lower",
           extent=[topo_x[0]*1e9, topo_x[-1]*1e9,
                   topo_y[0]*1e9, topo_y[-1]*1e9])

# dI/dV
plt.imshow(didv_map,
           cmap="plasma",
           alpha=0.5,
           origin="lower",
           extent=[spec_x[0]*1e9, spec_x[-1]*1e9,
                   spec_y[0]*1e9, spec_y[-1]*1e9])

# Overlay mapped vacancies
for (x_pix, y_pix), (x_m, y_m) in zip(aligned_coords_spec,
                                      vacancy_coords_m):

    plt.plot(x_m*1e9,
             y_m*1e9,
             'ro',
             markersize=8)

plt.title("Header-Based Real Space Alignment")
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
plt.tight_layout()
plt.savefig("experiment_output/aligned_overlay_correct.png", dpi=300)
plt.show()