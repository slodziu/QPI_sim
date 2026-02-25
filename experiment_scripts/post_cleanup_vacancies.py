import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


topograph_file = "experimental_data/topograph.sxm"
window_size = 120  # pixels (must be even)
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
vacancy_coords_type2 = [(195, 669),
(365, 555),
(279, 366),
(446, 105),
(463, 120),
(434, 902),
(700, 508)]

aligned_windows = []
aligned_windows_type2 = []

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

    # Process type 2 vacancies
    for x0, y0 in vacancy_coords_type2:
        x_min = x0 - half
        x_max = x0 + half
        y_min = y0 - half
        y_max = y0 + half

        if (
            x_min < 0 or x_max > z_flat.shape[1] or
            y_min < 0 or y_max > z_flat.shape[0]
        ):
            print(f"Skipping type 2 vacancy at ({x0}, {y0}) - too close to edge")
            continue

        window = z_flat[y_min:y_max, x_min:x_max]
        if window.shape == (window_size, window_size):
            aligned_windows_type2.append(window)

aligned_windows = np.array(aligned_windows)

print(f"Number of aligned vacancies used: {len(aligned_windows)}")

# Show topograph with vacancy overlays
fig_topo, ax_topo = plt.subplots()
ax_topo.imshow(z_flat, origin="lower", cmap="viridis")

# Overlay circles at vacancy coordinates

# For legend: plot one invisible line for each type
legend_type1, = ax_topo.plot([], [], color="red", linewidth=1, label="Type 1 Vacancy")
legend_type2, = ax_topo.plot([], [], color="cyan", linewidth=1, label="Type 2 Vacancy")

for x_pix, y_pix in vacancy_coords:
    if 0 <= x_pix < z_flat.shape[1] and 0 <= y_pix < z_flat.shape[0]:
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = x_pix + 8 * np.cos(theta)
        circle_y = y_pix + 8 * np.sin(theta)
        ax_topo.plot(circle_x, circle_y, color="red", linewidth=1)
    else:
        print(f"Skipping out-of-bounds circle at ({x_pix}, {y_pix})")

# Overlay circles for type 2 vacancies (different color)
for x_pix, y_pix in vacancy_coords_type2:
    if 0 <= x_pix < z_flat.shape[1] and 0 <= y_pix < z_flat.shape[0]:
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = x_pix + 8 * np.cos(theta)
        circle_y = y_pix + 8 * np.sin(theta)
        ax_topo.plot(circle_x, circle_y, color="cyan", linewidth=1)
    else:
        print(f"Skipping out-of-bounds type 2 circle at ({x_pix}, {y_pix})")

ax_topo.set_title("Topograph with Vacancy Overlays")
fig_topo.colorbar(ax_topo.images[0], ax=ax_topo, label="Height (m)")
# Add legend outside the plot (only one entry per type)
fig_topo.legend([legend_type1, legend_type2], ["Type 1 Vacancy", "Type 2 Vacancy"], loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=2)
plt.savefig("experiment_output/vacancy_overlays.png", bbox_inches="tight", dpi=300)
plt.show()

# Save zoomed-in version
fig_zoom, ax_zoom = plt.subplots()
ax_zoom.imshow(z_flat, origin="lower", cmap="viridis")
# Overlay circles for type 1
for x_pix, y_pix in vacancy_coords:
    if 0 <= x_pix < z_flat.shape[1] and 0 <= y_pix < z_flat.shape[0]:
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = x_pix + 8 * np.cos(theta)
        circle_y = y_pix + 8 * np.sin(theta)
        ax_zoom.plot(circle_x, circle_y, color="red", linewidth=1)
# Overlay circles for type 2
for x_pix, y_pix in vacancy_coords_type2:
    if 0 <= x_pix < z_flat.shape[1] and 0 <= y_pix < z_flat.shape[0]:
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = x_pix + 8 * np.cos(theta)
        circle_y = y_pix + 8 * np.sin(theta)
        ax_zoom.plot(circle_x, circle_y, color="cyan", linewidth=1)
ax_zoom.set_title("Topograph with Vacancy Overlays (Zoomed)")
fig_zoom.colorbar(ax_zoom.images[0], ax=ax_zoom, label="Height (m)")
# Add legend below
legend_type1_zoom, = ax_zoom.plot([], [], color="red", linewidth=1, label="Type 1 Vacancy")
legend_type2_zoom, = ax_zoom.plot([], [], color="cyan", linewidth=1, label="Type 2 Vacancy")
fig_zoom.legend([legend_type1_zoom, legend_type2_zoom], ["Type 1 Vacancy", "Type 2 Vacancy"], loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=2)
ax_zoom.set_xlim(150, 500)
ax_zoom.set_ylim(500, 800)
plt.savefig("experiment_output/vacancy_overlays_zoomed.png", bbox_inches="tight", dpi=300)
plt.show()


average_defect = np.mean(aligned_windows, axis=0)

average_defect_type2 = np.mean(aligned_windows_type2, axis=0) if len(aligned_windows_type2) > 0 else None



# Plot and save averaged type 1 vacancy
fig, ax = plt.subplots()
im = ax.imshow(average_defect, origin="lower", cmap="viridis")
ax.set_title("Common-Origin Averaged Type 1 Vacancy")
fig.colorbar(im, ax=ax, label="Height (m)")
plt.savefig("experiment_output/average_defect_type_1.png", bbox_inches="tight", dpi=300)
plt.show()

# Fourier transform of averaged type 1 vacancy
fft1 = np.fft.fftshift(np.fft.fft2(average_defect))
fft1_mag = np.abs(fft1)
fig_fft1, ax_fft1 = plt.subplots()
ax_fft1.imshow(np.log(fft1_mag + 1e-12), origin="lower", cmap="viridis")
ax_fft1.set_title("FFT Magnitude (Type 1 Vacancy)")
plt.savefig("experiment_output/average_defect_type_1_fft.png", bbox_inches="tight", dpi=300)
plt.close(fig_fft1)

# Plot and save averaged type 2 vacancy
if average_defect_type2 is not None:
    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(average_defect_type2, origin="lower", cmap="viridis")
    ax2.set_title("Common-Origin Averaged Type 2 Vacancy")
    fig2.colorbar(im2, ax=ax2, label="Height (m)")
    plt.savefig("experiment_output/average_defect_type2.png", bbox_inches="tight", dpi=300)
    plt.show()

    # Fourier transform of averaged type 2 vacancy
    fft2 = np.fft.fftshift(np.fft.fft2(average_defect_type2))
    fft2_mag = np.abs(fft2)
    fig_fft2, ax_fft2 = plt.subplots()
    ax_fft2.imshow(np.log(fft2_mag + 1e-12), origin="lower", cmap="viridis")
    ax_fft2.set_title("FFT Magnitude (Type 2 Vacancy)")
    plt.savefig("experiment_output/average_defect_type_2_fft.png", bbox_inches="tight", dpi=300)
    plt.close(fig_fft2)

    # Take log of FFT magnitudes to compress dynamic range
    log_fft1_mag = np.log(fft1_mag + 1e-12)
    log_fft2_mag = np.log(fft2_mag + 1e-12)

    # Normalize log-FFTs using global min and max
    global_min = min(np.min(log_fft1_mag), np.min(log_fft2_mag))
    global_max = max(np.max(log_fft1_mag), np.max(log_fft2_mag))
    log_fft1_mag_norm = (log_fft1_mag - global_min) / (global_max - global_min + 1e-12)
    log_fft2_mag_norm = (log_fft2_mag - global_min) / (global_max - global_min + 1e-12)

    # Take the difference
    fft_diff = log_fft1_mag_norm - log_fft2_mag_norm

    # Plot and save the difference
    fig_diff, ax_diff = plt.subplots()
    im_diff = ax_diff.imshow(fft_diff, origin="lower", cmap="bwr", vmin=-1, vmax=1)
    ax_diff.set_title("Log-Scaled FFT Magnitude Difference (Type 1 - Type 2)")
    fig_diff.colorbar(im_diff, ax=ax_diff, label="Normalized Log Difference")
    plt.savefig("experiment_output/fft_magnitude_difference.png", bbox_inches="tight", dpi=300)
    plt.close(fig_diff)
