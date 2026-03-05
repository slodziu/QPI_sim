"""
align_maps.py
Align topograph.sxm onto the spectrum.3ds topograph using a single shared
vacancy as the anchor point, then overlay the two images.

Pixel coordinate convention: (col, row_from_bottom) as shown on the axes of
the diagnostic origin="lower" plot.  These are used directly as physical coords:
    x_nm = col * px_size
    y_nm = row_from_bottom * px_size

Anchor vacancies (3 matched pairs, col/row_from_bottom in origin=lower coords):
  spectrum.3ds   → (127,158)  (99,104)  (199,105)
  topograph.sxm  → (510,834)  (365,555)  (877,560)
Translation is computed per-pair and averaged.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

output_dir = "experiment_output/read_topography"
os.makedirs(output_dir, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
def plane_subtract(arr):
    """Fit and subtract a least-squares plane from a 2D array."""
    ny, nx = arr.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    A = np.c_[X.ravel(), Y.ravel(), np.ones(X.size)]
    C, _, _, _ = np.linalg.lstsq(A, arr.ravel(), rcond=None)
    return arr - (C[0]*X + C[1]*Y + C[2])

def log_contrast(arr):
    """Log-scale contrast matching topograph_read.py."""
    flat = plane_subtract(arr)
    return np.log(flat - np.nanmin(flat) + 1e-12)

ds3 = xr.open_dataset("experimental_data/spectrum.3ds", engine="nanonis")
topo3 = ds3.coords["Z"].values                       # (270, 270) metres
x3_nm = ds3.coords["x"].values * 1e9                 # nm, length 270
y3_nm = ds3.coords["y"].values * 1e9
px3 = x3_nm[1] - x3_nm[0]                            # ≈ 0.2454 nm/px
N3 = topo3.shape[0]
topo3_plot = log_contrast(topo3)

ds_sxm = xr.open_dataset("experimental_data/topograph.sxm", engine="nanonis")
topo_sxm = ds_sxm["Z"].values[0]                     # forward scan, (1024,1024)
x_sxm_nm = ds_sxm.coords["x"].values * 1e9           # nm, length 1024
y_sxm_nm = ds_sxm.coords["y"].values * 1e9
px_sxm_x = x_sxm_nm[1] - x_sxm_nm[0]                # ≈ 0.0489 nm/px
px_sxm_y = y_sxm_nm[1] - y_sxm_nm[0]
N_sxm = topo_sxm.shape[0]
topo_sxm_plot = log_contrast(topo_sxm)

print(f"spectrum.3ds  : {topo3.shape}, px={px3:.4f} nm/px, range {x3_nm.max():.1f} nm")
print(f"topograph.sxm : {topo_sxm.shape}, px_x={px_sxm_x:.4f} nm/px, range {x_sxm_nm.max():.1f} nm")

# ── Rotation check ─────────────────────────────────────────────────────────────
# .3ds grid settings field: (center_x, center_y, width, height, angle_deg)
_gs = ds3.attrs.get("Grid settings", None)
angle_3ds = float(_gs[4]) if _gs else float("nan")
angle_sxm = float(ds_sxm.attrs.get("SCAN_ANGLE", float("nan")))
delta_angle = angle_sxm - angle_3ds
print(f"\nScan angles  — .3ds: {angle_3ds}°   .sxm: {angle_sxm}°   Δ={delta_angle:.3f}°")
if abs(delta_angle) < 0.05:
    print("  ✓ Angles match — pure translation alignment is valid, no rotation correction needed.")
else:
    print(f"  ⚠ Angle mismatch of {delta_angle:.2f}° — a rotation must be applied to the .sxm overlay.")
    print(f"    Rotation matrix: [[cos({delta_angle:.2f}°), -sin({delta_angle:.2f}°)], "
          f"[sin({delta_angle:.2f}°), cos({delta_angle:.2f}°)]]")

# ── Anchor positions (3 matched pairs, col/row_from_bottom) ───────────────────
# Each row: (col_3ds, rfb_3ds, col_sxm, rfb_sxm)
anchor_pairs = [
    (127, 158,  510, 834),
    ( 99, 104,  365, 555),
    (199, 105,  877, 560),
]

offsets_x, offsets_y = [], []
print("\nPer-pair translations:")
for i, (c3, r3, cs, rs) in enumerate(anchor_pairs):
    x3  = c3 * px3;       y3  = r3 * px3
    xs  = cs * px_sxm_x;  ys  = rs * px_sxm_y
    dx  = x3 - xs;        dy  = y3 - ys
    offsets_x.append(dx); offsets_y.append(dy)
    print(f"  pair {i+1}: .3ds=({x3:.2f},{y3:.2f}) nm  .sxm=({xs:.2f},{ys:.2f}) nm  "
          f"→ dx={dx:.2f} dy={dy:.2f} nm")

offset_x = float(np.mean(offsets_x))
offset_y = float(np.mean(offsets_y))
print(f"  Mean offset: dx={offset_x:.3f} ± {np.std(offsets_x):.3f}  "
      f"dy={offset_y:.3f} ± {np.std(offsets_y):.3f}  nm")

sxm_left   = offset_x + x_sxm_nm.min()
sxm_right  = offset_x + x_sxm_nm.max()
sxm_bottom = offset_y + y_sxm_nm.min()
sxm_top    = offset_y + y_sxm_nm.max()

print(f"Translation offset : dx={offset_x:.2f} nm, dy={offset_y:.2f} nm")
print(f".sxm extent in .3ds space : x=[{sxm_left:.2f}, {sxm_right:.2f}]  "
      f"y=[{sxm_bottom:.2f}, {sxm_top:.2f}] nm")

# ── Overlay plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))

bg = ax.imshow(
    topo3_plot,
    extent=[x3_nm.min(), x3_nm.max(), y3_nm.min(), y3_nm.max()],
    origin="lower", aspect="equal", cmap="plasma",
    vmin=np.percentile(topo3_plot, 2), vmax=np.percentile(topo3_plot, 98),
    zorder=0,
)
fg = ax.imshow(
    topo_sxm_plot,
    extent=[sxm_left, sxm_right, sxm_bottom, sxm_top],
    origin="lower", aspect="equal", cmap="plasma",
    vmin=np.percentile(topo_sxm_plot, 2), vmax=np.percentile(topo_sxm_plot, 98),
    alpha=0.5, zorder=1,
)

for i, (c3, r3, _, _) in enumerate(anchor_pairs):
    ax.plot(c3 * px3, r3 * px3, 'c+', ms=14, mew=2, zorder=5,
            label="anchor pairs" if i == 0 else "")

rect = mpatches.Rectangle(
    (sxm_left, sxm_bottom), sxm_right - sxm_left, sxm_top - sxm_bottom,
    linewidth=1.5, edgecolor="cyan", facecolor="none", zorder=4,
)
ax.add_patch(rect)

ax.set_xlabel("x (nm)")
ax.set_ylabel("y (nm)")
ax.set_title("Aligned topographs\n(sxm overlay on .3ds background)")
ax.legend(loc="upper right", fontsize=7)

cb_bg = plt.colorbar(bg, ax=ax, fraction=0.03, pad=0.01, location="left")
cb_bg.set_label(".3ds log contrast", fontsize=7)
cb_fg = plt.colorbar(fg, ax=ax, fraction=0.03, pad=0.04)
cb_fg.set_label(".sxm log contrast", fontsize=7)

out_path = os.path.join(output_dir, "aligned_topographs.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print(f"\nSaved overlay to {out_path}")

# ── Diagnostic: each map alone with anchor marked ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(topo3_plot, cmap="plasma", origin="lower",
               vmin=np.percentile(topo3_plot, 2), vmax=np.percentile(topo3_plot, 98))
for c3, r3, _, _ in anchor_pairs:
    axes[0].plot(c3, r3, 'c+', ms=14, mew=2)
axes[0].set_title(".3ds  (origin=lower, anchors marked)")
axes[0].set_xlabel("col (px)"); axes[0].set_ylabel("row from bottom (px)")

axes[1].imshow(topo_sxm_plot, cmap="plasma", origin="lower",
               vmin=np.percentile(topo_sxm_plot, 2), vmax=np.percentile(topo_sxm_plot, 98))
for _, _, cs, rs in anchor_pairs:
    axes[1].plot(cs, rs, 'c+', ms=14, mew=2)
axes[1].set_title(".sxm  (origin=lower, anchors marked)")
axes[1].set_xlabel("col (px)"); axes[1].set_ylabel("row from bottom (px)")

diag_path = os.path.join(output_dir, "anchor_check.png")
plt.savefig(diag_path, dpi=200, bbox_inches="tight")
plt.show()
plt.close()
print(f"Saved diagnostic to {diag_path}")
