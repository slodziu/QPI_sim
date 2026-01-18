import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import gaussian_filter
from UTe2_fixed import H_full   # <-- your Hamiltonian

# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
nk = 151         # number of k-points in each direction
EF = 0.0               # Fermi level
smooth_sigma = 0.0     # Gaussian smoothing (set to 0 for no smoothing)
step_size = 1     # marching cubes step

# Lattice constants (insert correct values)
a = 0.41   
b = 0.61   
c = 1.39   
c_star = 0.76

C0, C1, C2, C3 = 0, 300e-6, 300e-6, 300e-6  # in eV

def B2u_gap(kx, ky, kz):
    # Each term is of shape (N,)
    comp1 = C1 * np.sin(kz * c)
    comp2 = C0 * np.sin(kx * a) * np.sin(ky * b) * np.sin(kz * c)
    comp3 = C3 * np.sin(kx * a)
    # Stack into (3, N)
    return np.vstack([comp1, comp2, comp3])

def B3u_gap(kx, ky, kz):
    comp1 = C0 * np.sin(kx * a) * np.sin(ky * b) * np.sin(kz * c)
    comp2 = C2 * np.sin(kz * c)
    comp3 = C3 * np.sin(ky * b)
    return np.vstack([comp1, comp2, comp3])


# k-grid in crystal units
kx = np.linspace(-np.pi/a, np.pi/a, nk)
ky = np.linspace(-np.pi/b, np.pi/b, nk)
kz = np.linspace(-np.pi/c, np.pi/c, nk)
kz_gap = np.linspace(-np.pi/c, np.pi/c, nk)
kz_fs = np.linspace(-2*np.pi/c, 2*np.pi/c, nk)

# ------------------------------------------------------------
# (1) Compute band energies on a 3D grid
# ------------------------------------------------------------

def compute_band_energies(H_func, kx, ky, kz, nk):
    """Compute band energies on a 3D grid using the provided Hamiltonian function."""
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    H = H_func(KX, KY, KZ)           # shape (nk, nk, nk, 4, 4)
    band_energies = np.linalg.eigvalsh(H)   # shape (nk, nk, nk, 4)
    return band_energies

def extract_fermi_surface(band_energies, EF, kx, ky, kz, smooth_sigma, step_size):
    """Extract Fermi surface vertices and faces for each band crossing EF."""
    all_verts = []
    for band in range(band_energies.shape[-1]):
        B = band_energies[:, :, :, band]
        if not (B.min() <= EF <= B.max()):
            continue
        B_smooth = gaussian_filter(B, sigma=smooth_sigma)
        verts, faces, normals, values = measure.marching_cubes(
            B_smooth,
            level=EF,
            spacing=(kx[1]-kx[0], ky[1]-ky[0], kz[1]-kz[0]),
            step_size=step_size
        )
        verts_kx = verts[:,0] + kx[0]
        verts_ky = verts[:,1] + ky[0]
        verts_kz = verts[:,2] + kz[0]
        all_verts.append((band, verts_kx, verts_ky, verts_kz, faces))
    return all_verts

def plot_fs_projection(
    all_verts, gap_func, gap_label, output_path,
    a, b, c, c_star, theta_deg=24, node_tol=1e-7
):
    """Plot Fermi surface projection and gap nodes for a given gap function."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(10,12))
    ax = fig.add_subplot(111, projection='3d')
    theta = np.deg2rad(theta_deg)
    colours = ["#9635E5", "#FD0000"]
    z_offset = 0.1

    kx_min, kx_max = -np.pi/a, np.pi/a
    kc_min, kc_max = -np.pi/c_star, np.pi/c_star
    pad_x = 0.001 * (kc_max - kc_min)
    pad_y = 0.001 * (kx_max - kx_min)

    # First plot all FS bands
    for idx, (band, verts_kx, verts_ky, verts_kz, faces) in enumerate(all_verts):
        k1 = verts_kx
        k2 = verts_ky*np.sin(theta) + verts_kz*np.cos(theta)
        verts2d = np.column_stack([k2, k1])
        z = np.full(3, idx * z_offset)
        mask_tol = 0.001
        k2_min_mask = kc_min + pad_x - mask_tol * (kc_max - kc_min)
        k2_max_mask = kc_max - pad_x + mask_tol * (kc_max - kc_min)
        k1_min_mask = kx_min + pad_y - mask_tol * (kx_max - kx_min)
        k1_max_mask = kx_max + pad_y + mask_tol * (kx_max - kx_min)

        masked_triangles = []
        for f in faces:
            tri = verts2d[f]
            if np.all((k2_min_mask <= tri[:,0]) & (tri[:,0] <= k2_max_mask) & (k1_min_mask <= tri[:,1]) & (tri[:,1] <= k1_max_mask)):
                masked_triangles.append(np.column_stack([tri, z]))
        if masked_triangles:
            poly = Poly3DCollection(
                masked_triangles,
                facecolor=colours[idx % len(colours)],
                edgecolor='none',
                alpha=0.3
            )
            ax.add_collection3d(poly)

    # Now plot all gap nodes on top
    # Use a z-value above all FS polygons
    node_z = z_offset * (len(all_verts) + 2)
    if gap_label == "B2u":
        N_nodes = 60  # Number of lowest-gap nodes to plot and print (set as needed)
    else:
        N_nodes = 40  # Number of lowest-gap nodes to plot and print (set as needed)
    # Collect all nodes from all bands, filtering by gap kz range
    all_k1 = []
    all_k2 = []
    all_gap_vals = []
    for idx, (band, verts_kx, verts_ky, verts_kz, faces) in enumerate(all_verts):
        k1 = verts_kx
        k2 = verts_ky*np.sin(theta) + verts_kz*np.cos(theta)
        gap_vals = np.linalg.norm(gap_func(verts_kx, verts_ky, verts_kz), axis=0)
        # Filter nodes to gap kz range
        gap_kz_mask = (verts_kz >= -np.pi/c) & (verts_kz <= np.pi/c)
        k1 = k1[gap_kz_mask]
        k2 = k2[gap_kz_mask]
        gap_vals = gap_vals[gap_kz_mask]
        all_k1.append(k1)
        all_k2.append(k2)
        all_gap_vals.append(gap_vals)
    # Concatenate all bands' nodes
    all_k1 = np.concatenate(all_k1)
    all_k2 = np.concatenate(all_k2)
    all_gap_vals = np.concatenate(all_gap_vals)

    # Convert all nodes to reduced units
    all_k1_red = all_k1 / (np.pi/a)
    all_k2_red = all_k2 / (np.pi/c_star)
    # Mask for nodes within plot window
    window_mask = (all_k1_red >= -1) & (all_k1_red <= 1) & (all_k2_red >= -1) & (all_k2_red <= 1)
    k1_win = all_k1[window_mask]
    k2_win = all_k2[window_mask]
    gap_win = all_gap_vals[window_mask]
    k1_win_red = all_k1_red[window_mask]
    k2_win_red = all_k2_red[window_mask]
    N = min(N_nodes, len(gap_win))
    if N == 0:
        print(f"{gap_label}: No nodes within plot window.")
    else:
        lowest_idx = np.argsort(np.abs(gap_win))[:N]
        k1_gap = k1_win[lowest_idx]
        k2_gap = k2_win[lowest_idx]
        gap_vals_lowest = gap_win[lowest_idx]
        k1_gap_red = k1_win_red[lowest_idx]
        k2_gap_red = k2_win_red[lowest_idx]
        print(f"{gap_label}: Plotting {N} lowest-gap nodes within plot window:")
        if gap_label == "B3u":
            # Snap kc* to nearest multiple of 0.5
            k2_gap_red_snapped = np.round(k2_gap_red * 2) / 2
            for i in range(N):
                print(f"{gap_label}: node {i}: kc*={k2_gap_red_snapped[i]:.3f}, kx={k1_gap_red[i]:.3f}, gap={gap_vals_lowest[i]:.2e}")
            # Convert reduced units back to physical units for plotting
            k1_gap_plot = k1_gap_red * (np.pi / a)
            k2_gap_plot = k2_gap_red_snapped * (np.pi / c_star)
        else:
                # For B2u: snap kx to nearest of 0, 1, or -1
                snap_vals = np.array([0, 1, -1])
                k1_gap_red_arr = np.array(k1_gap_red)
                k2_gap_red_arr = np.array(k2_gap_red)
                gap_vals_arr = np.array(gap_vals_lowest)
                # Snap kx to nearest value in snap_vals
                k1_gap_red_snapped = snap_vals[np.argmin(np.abs(k1_gap_red_arr[:, None] - snap_vals), axis=1)]
                tol_kx = 0.02
                mask_kx = np.abs(k1_gap_red_arr - k1_gap_red_snapped) < tol_kx
                k2_gap_red_keep = k2_gap_red_arr[mask_kx]
                k1_gap_red_keep = k1_gap_red_snapped[mask_kx]
                gap_vals_keep = gap_vals_arr[mask_kx]
                print(f"{gap_label}: Keeping only nodes snapped to kx=0, 1, -1 (tol={tol_kx})")
                # For each kx value, remove two lowest gap nodes, then plot next six lowest
                plot_mask = np.zeros_like(k1_gap_red_keep, dtype=bool)
                for kx_target in snap_vals:
                    idx_kx = np.where(k1_gap_red_keep == kx_target)[0]
                    if len(idx_kx) < 8:
                        print(f"{gap_label}: kx={kx_target}: Not enough nodes to remove 2 and plot 6.")
                        continue
                    # Sort by gap magnitude
                    sorted_idx = idx_kx[np.argsort(np.abs(gap_vals_keep[idx_kx]))]
                    # Remove two lowest
                    plot_idx = sorted_idx[2:8]
                    for i, idx in enumerate(plot_idx):
                        print(f"{gap_label}: node {i} (kx={kx_target}): kc*={k2_gap_red_keep[idx]:.3f}, kx={k1_gap_red_keep[idx]:.3f}, gap={gap_vals_keep[idx]:.2e}")
                    # Collect nodes for plotting
                    plot_mask[plot_idx] = True
                k1_gap_plot = k1_gap_red_keep[plot_mask] * (np.pi / a)
                k2_gap_plot = k2_gap_red_keep[plot_mask] * (np.pi / c_star)
        ax.scatter(k2_gap_plot, k1_gap_plot, node_z, c='k', s=60, edgecolors='w', linewidths=1.5, label=f'{gap_label} nodes')

    ax.set_xlim(kc_min + pad_x, kc_max - pad_x)
    ax.set_ylim(kx_min + pad_y, kx_max - pad_y)
    ax.set_zlim(0, z_offset * (len(all_verts)+1))
    ax.view_init(elev=90, azim=-90)
    ax.set_zticks([])
    ax.set_yticks(np.linspace(-np.pi/a, np.pi/a, 5))
    ax.set_xticks(np.linspace(-np.pi/c_star, np.pi/c_star, 5))
    ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(-1, 1, 5)], fontsize=28)
    ax.set_yticklabels([f"{y:.1f}" for y in np.linspace(-1, 1, 5)], fontsize=28)
    ax.set_xlabel(r"$k_c^* \; (\pi/c^*)$", fontsize=32, labelpad=15)
    ax.set_ylabel(r"$k_x \; (\pi/a)$", fontsize=32, labelpad=15)
    ax.set_aspect('equal')
    
    # THE KEY FIX: Manual position adjustment for 3D plots to prevent axis label cutoff
    # Parameters: [left, bottom, width, height] - increased space to prevent cutoff
    ax.set_position([-0.15, -0.2, 1.3, 1.4])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    print(f"Fermi surface projection plot saved to {output_path}.")

    # After plotting nodes, add B2u gap vs kc* plots for kx=0, 1, -1
    if gap_label == "B2u":
        kx_targets = np.array([0.0, 1.0, -1.0])
        tol_kx = 0.001
        for kx_t in kx_targets:
            sel = np.abs(k1_win_red - kx_t) < tol_kx
            if np.any(sel):
                plt.figure(figsize=(8,6))
                plt.plot(k2_win_red[sel], gap_win[sel], 'ko-', label=f'$k_x={kx_t}$')
                plt.xlabel(r"$k_c^*$ ($\pi/c^*$)", fontsize=16)
                plt.ylabel("Gap magnitude", fontsize=16)
                plt.title(f"B2u gap vs $k_c^*$ at $k_x={kx_t}$")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"outputs/FSPROJ/B2u_gap_vs_kcstar_kx_{int(kx_t)}.png", dpi=200)
                plt.close()

# --- Main workflow ---
print("Computing band structure on 3D grid...")
band_energies = compute_band_energies(H_full, kx, ky, kz_fs, nk)
print("Done computing 3D band energies.")

print("Extracting Fermi surface...")
all_verts = extract_fermi_surface(band_energies, EF, kx, ky, kz_fs, smooth_sigma, step_size)
print(f"Total FS bands crossing EF: {len(all_verts)}")

# Plot B2u nodes
plot_fs_projection(
    all_verts, B2u_gap, "B2u", "outputs/FSPROJ/FS_3D_projection_B2u.png",
    a, b, c, c_star, node_tol=1e-16
)

# Plot B3u nodes
plot_fs_projection(
    all_verts, B3u_gap, "B3u", "outputs/FSPROJ/FS_3D_projection_B3u.png",
    a, b, c, c_star, node_tol=1e-16
)
