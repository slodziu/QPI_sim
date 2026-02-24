import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Parameters
profile_dir = "experiment_output/line_profiles"
topograph_file = "experimental_data/topograph.sxm"

# Load topograph
ds = xr.open_dataset(topograph_file, engine="nanonis")
z = ds["Z"].sel(dir="forward").squeeze()
x = z.coords["x"].values
y = z.coords["y"].values
z_flat = z.values

# Load minima arrays
type1_path = os.path.join(profile_dir, "type1_minima_pixels.npy")
type2_path = os.path.join(profile_dir, "type2_minima_pixels.npy")
type1_positions = np.load(type1_path) if os.path.exists(type1_path) else np.empty((0,2), dtype=int)
type2_positions = np.load(type2_path) if os.path.exists(type2_path) else np.empty((0,2), dtype=int)
print(f"Loaded {type1_positions.shape[0]} type 1 minima, {type2_positions.shape[0]} type 2 minima.")

# Plot topograph in pixel coordinates for overlay
fig, ax = plt.subplots()
ax.imshow(z_flat, origin="lower", cmap="viridis")

type1_vacancies = []
type2_vacancies = []

for x_pix, y_pix in type1_positions:
    if 0 <= x_pix < z_flat.shape[1] and 0 <= y_pix < z_flat.shape[0]:
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = x_pix + 5 * np.cos(theta)
        circle_y = y_pix + 5 * np.sin(theta)
        ax.plot(circle_x, circle_y, color="red", linewidth=1, solid_joinstyle='miter', label="Type 1" if len(type1_vacancies) == 0 else None)
        type1_vacancies.append((x_pix, y_pix))
    else:
        print(f"Skipping out-of-bounds Type 1 circle at idx {x_pix}, {y_pix}")

for x_pix, y_pix in type2_positions:
    if 0 <= x_pix < z_flat.shape[1] and 0 <= y_pix < z_flat.shape[0]:
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = x_pix + 5 * np.cos(theta)
        circle_y = y_pix + 5 * np.sin(theta)
        ax.plot(circle_x, circle_y, color="blue", linewidth=1, solid_joinstyle='miter', label="Type 2" if len(type2_vacancies) == 0 else None)
        type2_vacancies.append((x_pix, y_pix))
    else:
        print(f"Skipping out-of-bounds Type 2 circle at idx {x_pix}, {y_pix}")

ax.set_title("Vacancy Detection: Type 1 (red) & Type 2 (blue)")
fig.colorbar(ax.images[0], ax=ax, label="Height (m)")
ax.legend()
plt.savefig("experiment_output/vacancy_overlay.png", bbox_inches="tight", dpi=300)
plt.show()

print("\nType 1 vacancies (x_pix, y_pix):")
for v in type1_vacancies:
    print((int(v[0]), int(v[1])))
print("\nType 2 vacancies (x_pix, y_pix):")
for v in type2_vacancies:
    print((int(v[0]), int(v[1])))

# Show just the topograph in pixel coordinates, no circles
plt.figure()
plt.imshow(z_flat, origin="lower", cmap="viridis")
plt.title("Topograph (pixel coordinates)")
plt.colorbar(label="Height (m)")
plt.show()