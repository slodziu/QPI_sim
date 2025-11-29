import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import gaussian_filter
from UTe2_fixed import H_full   # <-- your Hamiltonian

# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
nk = 101          # number of k-points in each direction
EF = 0.0               # Fermi level
smooth_sigma = 0.5     # Gaussian smoothing
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
kz = np.linspace(-2*np.pi/c, 2*np.pi/c, nk)

# ------------------------------------------------------------
# (1) Compute band energies on a 3D grid
# ------------------------------------------------------------
print("Computing band structure on 3D grid...")

band_energies = np.zeros((nk, nk, nk, 4))

# Build full 3D k-grid (shape nk×nk×nk)
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

# Evaluate full Hamiltonian in one vectorised call
H = H_full(KX, KY, KZ)           # shape (nk, nk, nk, 4, 4)
band_energies = np.linalg.eigvalsh(H)   # shape (nk, nk, nk, 4)

print("Done computing 3D band energies.")

# Extract FS
all_verts = []
print("Extracting Fermi surface...")

for band in range(4):

    B = band_energies[:, :, :, band]    # <-- correct now!!

    if not (B.min() <= EF <= B.max()):
        continue

    print(f"  Band {band} crosses EF")

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


    print('Done with band: ', band)


print(f"Total FS vertices: {verts.shape[0]}")

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig = plt.figure(figsize=(10,16))
ax = fig.add_subplot(111, projection='3d')

theta = np.deg2rad(24)
colours = ["#ff66aa", "#6699ff", "#ffaa33", "#55dd88"]

z_offset = 0.1   # tiny separation so polygons do NOT collapse


# === Axes limits: use slightly inside real projection bounds ===
kx_min, kx_max = -np.pi/a, np.pi/a
kc_min, kc_max = -np.pi/c_star, np.pi/c_star
pad_x = 0.001 * (kc_max - kc_min)
pad_y = 0.001 * (kx_max - kx_min)
k2_min, k2_max = kc_min + pad_x, kc_max - pad_x
k1_min, k1_max = kx_min + pad_y, kx_max - pad_y
node_tol = 1e-6
for idx, (band, verts_kx, verts_ky, verts_kz, faces) in enumerate(all_verts):
    # === Projection formula ===
    k1 = verts_kx
    k2 = verts_ky*np.sin(theta) + verts_kz*np.cos(theta)
    verts2d = np.column_stack([k2, k1])   # shape (N,2)

    # Improved triangle masking: allow edge-touching, remove far-out overflow
    z = np.full(3, idx * z_offset)
    mask_tol = 0.001  # 1% tolerance beyond axis bounds
    k2_min_mask = kc_min + pad_x - mask_tol * (kc_max - kc_min)
    k2_max_mask = kc_max - pad_x + mask_tol * (kc_max - kc_min)
    k1_min_mask = kx_min + pad_y - mask_tol * (kx_max - kx_min)
    k1_max_mask = kx_max - pad_y + mask_tol * (kx_max - kx_min)
    B2u_vals = np.linalg.norm(B2u_gap(verts_kx, verts_ky, verts_kz), axis=0)
    B3u_vals = np.linalg.norm(B3u_gap(verts_kx, verts_ky, verts_kz), axis=0)
    # Identify nodes
    B2u_nodes = (np.abs(B2u_vals) < node_tol)
    B3u_nodes = (np.abs(B3u_vals) < node_tol)

    # Filter nodes within mask
    B2u_mask = (
        (k2 >= k2_min_mask) & (k2 <= k2_max_mask) &
        (k1 >= k1_min_mask) & (k1 <= k1_max_mask)
    )
    B3u_mask = B2u_mask  # same mask for both

    B2u_nodes_masked = B2u_nodes & B2u_mask
    B3u_nodes_masked = B3u_nodes & B3u_mask

    # Project nodes to 2D
    k1_B2u = verts_kx[B2u_nodes_masked]
    k2_B2u = verts_ky[B2u_nodes_masked]*np.sin(theta) + verts_kz[B2u_nodes_masked]*np.cos(theta)

    k1_B3u = verts_kx[B3u_nodes_masked]
    k2_B3u = verts_ky[B3u_nodes_masked]*np.sin(theta) + verts_kz[B3u_nodes_masked]*np.cos(theta)

    # Overlay on the 2D plot
    ax.scatter(k2_B2u, k1_B2u, zdir='z', s=20, c='red', marker='o', label='B2u nodes' if band==0 else "")
    ax.scatter(k2_B3u, k1_B3u, zdir='z', s=20, c='green', marker='x', label='B3u nodes' if band==0 else "")
    masked_triangles = []
    for f in faces:
        tri = verts2d[f]
        # Only plot triangles where all vertices are within the mask box
        if np.all((k2_min_mask <= tri[:,0]) & (tri[:,0] <= k2_max_mask) & (k1_min_mask <= tri[:,1]) & (tri[:,1] <= k1_max_mask)):
            masked_triangles.append(np.column_stack([tri, z]))

    if masked_triangles:
        poly = Poly3DCollection(
            masked_triangles,
            facecolor=colours[idx],
            edgecolor='none',
            alpha=0.3
        )
        ax.add_collection3d(poly)


# === Axes limits: use slightly inside real projection bounds ===
kx_min, kx_max = -np.pi/a, np.pi/a
kc_min, kc_max = -np.pi/c_star, np.pi/c_star
pad_x = 0.001 * (kc_max - kc_min)
pad_y = 0.001 * (kx_max - kx_min)
ax.set_xlim(kc_min + pad_x, kc_max - pad_x)
ax.set_ylim(kx_min + pad_y, kx_max - pad_y)
ax.set_zlim(0, z_offset * (len(all_verts)+1))


# === Make it look EXACTLY 2D ===
ax.view_init(elev=90, azim=-90)  # top-down
ax.set_zticks([])
# Use fewer ticks for less cramping
ax.set_yticks(np.linspace(-np.pi/a, np.pi/a, 5))
ax.set_xticks(np.linspace(-np.pi/c_star, np.pi/c_star, 5))
ax.set_xticklabels([f"{x:.2f}" for x in np.linspace(-1, 1, 5)], fontsize=18)
ax.set_yticklabels([f"{y:.2f}" for y in np.linspace(-1, 1, 5)], fontsize=18)
ax.set_xlabel(r"$k_c^* \; (\pi/c^*)$", fontsize=20, labelpad=20)
ax.set_ylabel(r"$k_x \; (\pi/a)$", fontsize=20, labelpad=20)


ax.set_aspect('equal')


plt.savefig("outputs/FSPROJ/FS_3D_projection.png", dpi=300, bbox_inches='tight')
plt.show()

print("Fermi surface projection plot saved.")
