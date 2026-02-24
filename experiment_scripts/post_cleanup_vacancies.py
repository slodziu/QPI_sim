import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


topograph_file = "experimental_data/topograph.sxm"
window_size = 100  # pixels (must be even)
half = window_size // 2


ds = xr.open_dataset(topograph_file, engine="nanonis")
z = ds["Z"].sel(dir="forward").squeeze()
z_flat = z.values

vacancy_coords = [
    (15, 29), (33, 439), (87, 703), (88, 907), (118, 153), (171, 337), (237, 81),
    (341, 680), (371, 84), (408, 472), (426, 780), (461, 850), (510, 834), (573, 12),
    (574, 500), (577, 1002), (660, 477), (663, 949), (729, 870), (627, 620),
    (797, 971), (844, 591), (847, 1000), (877, 560), (958, 239), (993, 147), (641, 138)
]


aligned_windows = []

for x0, y0 in vacancy_coords:

    # Define window bounds
    x_min = x0 - half
    x_max = x0 + half
    y_min = y0 - half
    y_max = y0 + half

    # Skip if window would go out of bounds
    if (
        x_min < 0 or x_max > z_flat.shape[1] or
        y_min < 0 or y_max > z_flat.shape[0]
    ):
        print(f"Skipping vacancy at ({x0}, {y0}) - too close to edge")
        continue

    window = z_flat[y_min:y_max, x_min:x_max]

    # Ensure correct shape
    if window.shape == (window_size, window_size):
        aligned_windows.append(window)

aligned_windows = np.array(aligned_windows)

print(f"Number of aligned vacancies used: {len(aligned_windows)}")

# Show topograph with vacancy overlays
fig_topo, ax_topo = plt.subplots()
ax_topo.imshow(z_flat, origin="lower", cmap="viridis")

# Overlay circles at vacancy coordinates
for x_pix, y_pix in vacancy_coords:
    if 0 <= x_pix < z_flat.shape[1] and 0 <= y_pix < z_flat.shape[0]:
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = x_pix + 8 * np.cos(theta)
        circle_y = y_pix + 8 * np.sin(theta)
        ax_topo.plot(circle_x, circle_y, color="red", linewidth=1)
    else:
        print(f"Skipping out-of-bounds circle at ({x_pix}, {y_pix})")

ax_topo.set_title("Topograph with Vacancy Overlays")
fig_topo.colorbar(ax_topo.images[0], ax=ax_topo, label="Height (m)")
plt.savefig("experiment_output/vacancy_overlays.png", bbox_inches="tight", dpi=300)
plt.show()


average_defect = np.mean(aligned_windows, axis=0)


fig, ax = plt.subplots()
im = ax.imshow(average_defect, origin="lower", cmap="viridis")

# Mark common origin (center pixel)
ax.scatter(half, half, color="red", s=60)

ax.set_title("Common-Origin Averaged Vacancy")
fig.colorbar(im, ax=ax, label="Height (m)")
plt.savefig("experiment_output/average_defect.png", bbox_inches="tight", dpi=300)
plt.show()
