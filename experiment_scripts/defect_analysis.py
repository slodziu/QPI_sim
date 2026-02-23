
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter, label, center_of_mass
# Load file
c_star = 0.76
ds = xr.open_dataset("experimental_data/topograph.sxm", engine="nanonis")

# Output directory for plots
output_dir = r"experiment_output\\defect_analysis"
os.makedirs(output_dir, exist_ok=True)
z = ds["Z"].sel(dir="forward")   
z = z.squeeze()                  
theta = np.deg2rad(90)
x = z.coords["x"].values
y = z.coords["y"].values
x_nm = z.coords["x"].values * 1e9
y_nm = z.coords["y"].values * 1e9
X_nm, Y_nm = np.meshgrid(x_nm, y_nm)

# Rotate coordinates
Xr = X_nm*np.cos(theta) + Y_nm*np.sin(theta)
Yr = -X_nm*np.sin(theta) + Y_nm*np.cos(theta)
phase = np.mod(Yr, c_star) / c_star

plt.imshow(np.mod(Yr, c_star))
plt.colorbar()


# Smooth and subtract to highlight defects
z_flat = z.values
z_smooth = gaussian_filter(z_flat, sigma=2)
residual = z_flat - z_smooth
vac_mask = residual < -3.7*np.std(residual)  # threshold for dark defects

# Label connected regions and find their centers
labeled, num = label(vac_mask)
positions = center_of_mass(vac_mask, labeled, range(1, num+1))

# Separate into on-chain and between-chain
on_chain = []
between_chain = []

for py, px in positions:
    px_i = int(px)
    py_i = int(py)
    phase_val = phase[py_i, px_i]
    if abs(phase_val - 0.5) < 0.2:
        between_chain.append((px_i, py_i))
    else:
        on_chain.append((px_i, py_i))

# Plot topograph with defects circled
plt.figure()
plt.imshow(z_flat, origin="lower", cmap="viridis", extent=[x.min(), x.max(), y.min(), y.max()])
for px_i, py_i in on_chain:
    plt.scatter(x[px_i], y[py_i], facecolors="none", edgecolors="red", s=80)
for px_i, py_i in between_chain:
    plt.scatter(x[px_i], y[py_i], facecolors="none", edgecolors="cyan", s=80)
plt.gca().set_aspect("equal")
plt.title("Vacancy Classification")
plt.colorbar(label="Height (m)")
plt.savefig(os.path.join(output_dir, "classified_vacancies.png"), bbox_inches="tight", dpi=300)

print('Vacancy pixels positions:')
print(f"On-chain: {on_chain}")
print(f"Between-chain: {between_chain}")
print(f"Total vacancies: {len(on_chain) + len(between_chain)}")
