#!/usr/bin/env python3
"""
crystal_cleave_figure.py
========================
Publication-style two-panel figure:
  c  –  3-D schematic of a UTe2 crystal block showing the (0̄11) cleavage plane.
         x points UP, y and z lie in the horizontal plane (y into page, z right).
  d  –  Experimental STM topograph loaded from topograph.sxm.

Run from the repo root:
    python crystal_cleave_figure.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import xarray as xr
from numpy.linalg import lstsq

# ── output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs/crystal_cleave_figure"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── UTe2 crystal block dimensions ────────────────────────────────────────────
# Axes:
#   mpl X = paper z = c-axis (horizontal right)
#   mpl Y = paper y = b-axis (depth into page)
#   mpl Z = paper x = a-axis (VERTICAL – tallest visual dimension)
BH = 2.40    # mpl-Z  (a-axis, paper x) ← tallest
BD = 0.85    # mpl-Y  (b-axis, paper y)
BW = 1.30    # mpl-X  (c-axis, paper z)

# ── Cleavage geometry ────────────────────────────────────────────────────────
# The (0-11) cleavage plane cuts into the crystal at 24° from the ab-plane.
# Starting at mpl-X = z_step, the visible face recedes into the crystal at 24°
# angle until the right edge (mpl-X = BW), giving a pentagonal cross-section.
theta_clv = np.radians(24.0)
z_step    = BW * 0.42
dz_clv    = BW - z_step                    # mpl-X span of the cleavage face
y_deep    = dz_clv * np.tan(theta_clv)     # mpl-Y how deep it cuts
y_deep    = min(y_deep, BD * 0.72)         # clamp inside the crystal


# ── Load & process topograph ─────────────────────────────────────────────────
def load_topo(path):
    ds = xr.open_dataset(path, engine="nanonis")
    z  = ds["Z"].sel(dir="forward").squeeze()
    x  = z.coords["x"].values   # metres
    y  = z.coords["y"].values
    Z  = z.values                # shape (Ny, Nx)

    # subtract a plane to flatten background
    X2D, Y2D = np.meshgrid(x, y)
    A = np.c_[X2D.ravel(), Y2D.ravel(), np.ones(X2D.size)]
    C, *_ = lstsq(A, Z.ravel(), rcond=None)
    Z_flat = Z - (C[0] * X2D + C[1] * Y2D + C[2])

    # shift so min = 0, convert to pm
    Z_pm  = (Z_flat - Z_flat.min()) * 1e12
    x_nm  = (x - x.min()) * 1e9
    y_nm  = (y - y.min()) * 1e9
    return Z_pm, x_nm, y_nm


Z_pm, x_nm, y_nm = load_topo("experimental_data/topograph.sxm")

# display range: full scale (98th percentile)
vmax_pm = float(np.percentile(Z_pm, 98))


# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5.8), facecolor="white")

ax3 = fig.add_axes([0.00, 0.02, 0.46, 0.96], projection="3d")
ax2 = fig.add_axes([0.53, 0.10, 0.40, 0.82])


# ═══════════════════════════════════════════════════════════════════════════════
#  PANEL C – 3-D crystal block  (non-cuboid, cleaved shape)
#
#  Cross-section (mpl XY plane) is a PENTAGON, not a rectangle:
#   – from mpl-X=0 to z_step  : front face is at mpl-Y=0 (facing viewer)
#   – from mpl-X=z_step to BW : front face recedes at 24° into the crystal
#  This angled face IS the (0-11) cleavage plane (with atomic dots).
#
#  mpl X = paper z = c-axis (horizontal)   mpl Z = paper x = a-axis (UP)
#  mpl Y = paper y = b-axis (depth)        view: azim=-58, elev=22
# ═══════════════════════════════════════════════════════════════════════════════

# 10 vertices:  bottom pentagon (mpl Z=0)  +  top copy (mpl Z=BH)
#   v0-v4: bottom,  v5-v9: top  (v5=v0+BH, v6=v1+BH, ...)
vc = np.array([
    [0,      0,      0  ],   # 0  front-bottom-left
    [z_step, 0,      0  ],   # 1  cleavage-start bottom
    [BW,     y_deep, 0  ],   # 2  cleavage-end / right-bottom
    [BW,     BD,     0  ],   # 3  back-bottom-right
    [0,      BD,     0  ],   # 4  back-bottom-left
    [0,      0,      BH ],   # 5  front-top-left
    [z_step, 0,      BH ],   # 6  cleavage-start top
    [BW,     y_deep, BH ],   # 7  cleavage-end / right-top
    [BW,     BD,     BH ],   # 8  back-top-right
    [0,      BD,     BH ],   # 9  back-top-left
])

# faces drawn back-to-front (painter's algorithm)
face_specs = [
    ([4, 3, 2, 1, 0],  "#A89060", 0.38),   # bottom pentagon
    ([3, 4, 9, 8],     "#B09570", 0.46),   # back face (mpl Y=BD)
    ([2, 3, 8, 7],     "#C0A870", 0.55),   # right face (mpl X=BW)
    ([0, 4, 9, 5],     "#C8B880", 0.58),   # left face  (mpl X=0)
    ([5, 6, 7, 8, 9],  "#DDD0A0", 0.68),   # top pentagon (lit from above)
    ([0, 1, 6, 5],     "#EEE0B0", 0.88),   # front face, left portion (mpl Y=0)
    ([1, 2, 7, 6],     "#FFF4CC", 1.00),   # CLEAVAGE face (0-11) – freshest/brightest
]

for vi_list, fc, alpha in face_specs:
    poly = Poly3DCollection(
        [vc[vi_list].tolist()],
        facecolor=fc, edgecolor="#8B6914", linewidth=0.8, alpha=alpha,
    )
    ax3.add_collection3d(poly)

# ── atoms on the (0-11) cleavage face  [vc1, vc2, vc7, vc6] ──────────────────
# Parameterise as P(s,t) = vc[1] + s*(vc[2]-vc[1]) + t*(vc[6]-vc[1])
#   s in [0,1] across face width,  t in [0,1] up face height
clv_orig  = vc[1]              # (z_step, 0, 0)
clv_dir_w = vc[2] - vc[1]     # width  direction: (dz_clv, y_deep, 0)
clv_dir_h = vc[6] - vc[1]     # height direction: (0, 0, BH)

n_w, n_h = 5, 10
s_g = np.linspace(0.07, 0.93, n_w)
t_g = np.linspace(0.04, 0.96, n_h)
sG, tG = np.meshgrid(s_g, t_g)
aXa = clv_orig[0] + sG.ravel() * clv_dir_w[0]
aYa = clv_orig[1] + sG.ravel() * clv_dir_w[1]
aZa =               tG.ravel() * clv_dir_h[2]

ds, dt = 0.5 / n_w, 0.5 / n_h
sB = sG.ravel() + ds;  tB = tG.ravel() + dt
mask = (sB < 0.96) & (tB < 0.97)
aXb = clv_orig[0] + sB[mask] * clv_dir_w[0]
aYb = clv_orig[1] + sB[mask] * clv_dir_w[1]
aZb =               tB[mask]  * clv_dir_h[2]

ax3.scatter(aXa, aYa, aZa, c="#DD4444", s=26, depthshade=False, zorder=9, alpha=0.92)
ax3.scatter(aXb, aYb, aZb, c="#4466CC", s=18, depthshade=False, zorder=9, alpha=0.84)

# ── 3D style ──────────────────────────────────────────────────────────────────
ax3.set_axis_off()
ax3.set_box_aspect([BW, BD, BH])   # [mpl-X, mpl-Y, mpl-Z]
ax3.view_init(elev=22, azim=-58)

ax3.set_xlim(-0.12, BW + 0.10)
ax3.set_ylim(-0.06, BD + 0.10)
ax3.set_zlim(-0.10, BH + 0.10)



# ═══════════════════════════════════════════════════════════════════════════════
#  PANEL D – STM topograph
# ═══════════════════════════════════════════════════════════════════════════════

# Custom black → blue-grey colormap
from matplotlib.colors import LinearSegmentedColormap
_cmap_bluegrey = LinearSegmentedColormap.from_list(
    "black_bluegrey",
    [(0.00, "#000000"),
     (0.45, "#2a3a4a"),
     (0.75, "#7090a8"),
     (1.00, "#d8e8f0")],
)

im = ax2.imshow(
    np.clip(Z_pm, 0, vmax_pm),
    extent=[x_nm[0], x_nm[-1], y_nm[0], y_nm[-1]],
    origin="lower",
    cmap=_cmap_bluegrey,
    vmin=0, vmax=vmax_pm,
    aspect="equal",
)

# colorbar (vertical, matching paper style)
cbar = plt.colorbar(im, ax=ax2, pad=0.03, fraction=0.046)
cbar.ax.tick_params(labelsize=8)
cbar.set_label("$T(\\mathbf{r})$", fontsize=10, labelpad=6)
cbar.set_ticks([0, vmax_pm / 2, vmax_pm])
cbar.set_ticklabels(["0 pm", f"{vmax_pm/2:.0f} pm", f"{vmax_pm:.0f} pm"])

# scale bar (2 nm, black, bottom-right)
Lx = x_nm[-1] - x_nm[0]
Ly = y_nm[-1] - y_nm[0]
sb_x0 = x_nm[0] + Lx * 0.72
sb_y0 = y_nm[0] + Ly * 0.04
ax2.plot([sb_x0, sb_x0 + 2.0], [sb_y0, sb_y0],
         color="black", lw=2.5, solid_capstyle="butt")
ax2.text(sb_x0 + 1.0, sb_y0 + Ly * 0.028, "2 nm",
         ha="center", va="bottom", color="black",
         fontsize=9, fontweight="bold")

# axis labels matching paper (c* horizontal, a* vertical)
ax2.set_xlabel("$c^*$", fontsize=11)
ax2.set_ylabel("$a^*$", fontsize=11)
ax2.tick_params(labelsize=8)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f} nm"))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f} nm"))


# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "crystal_cleave_figure.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out_path}")
