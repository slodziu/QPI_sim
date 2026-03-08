"""
bz_gap_map.py
--------------
Side-by-side B2u / B3u gap maps within the first Brillouin zone (±π/a, ±π/b),
with Fermi surface contours overlaid and one unlabelled p6 scattering vector.
Shared diverging colorbar in meV, ticks 0 → ±0.30.

Run:
    python bz_gap_map.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os, sys

sys.path.insert(0, '.')
from UTe2_fixed import H_full, set_parameters, a, b
from phase_character import gap_function

# ── Setup ─────────────────────────────────────────────────────────────────────
set_parameters('odd_parity_paper')

KZ   = 0.0
RES  = 300          # pixels per axis
VMAX = 0.30         # meV colourbar limit

# UTe2 band colours (Fermi surface contours), one per band
FS_COLORS = ['#1E88E5', '#42A5F5', '#9635E5', '#FD0000']

# ── k-space grid strictly inside first BZ ─────────────────────────────────────
kx_vals = np.linspace(-np.pi / a, np.pi / a, RES)
ky_vals = np.linspace(-np.pi / b, np.pi / b, RES)
KX, KY  = np.meshgrid(kx_vals, ky_vals, indexing='ij')  # shape (RES, RES)

# ── Gap function on grid (convert eV → meV) ───────────────────────────────────
gap_b2u = gap_function(KX, KY, KZ, 'B2u') * 1e3   # meV
gap_b3u = gap_function(KX, KY, KZ, 'B3u') * 1e3   # meV

# ── Fermi surface: find band eigenvalue contours at E=0 ───────────────────────
print("Computing Fermi surface eigenvalues …")
eigenvalues = np.zeros((RES, RES, 4))
kx_flat = KX.flatten()
ky_flat = KY.flatten()
H_stack = np.array([H_full(kx, ky, KZ) for kx, ky in zip(kx_flat, ky_flat)])
eigenvalues = np.linalg.eigvalsh(H_stack).reshape(RES, RES, 4)
print("Done.")

# ── p6 vector: hardcoded from phase_resolved_qpi.py known coordinates ─────────
# Origin on Fermi surface: (kx=-0.62*pi/a, ky=0)
# q_p6 = (0.619*2pi/a, 0) -> tip at (kx=+0.62*pi/a, ky=0)
origin_kx = -0.62 * np.pi / a
origin_ky =  0.0
tip_kx    =  0.62 * np.pi / a
tip_ky    =  0.0

# ── Gap along the p6 vector (kx: origin→tip, ky=0 throughout) ────────────────
N_LINE = 500
kx_line = np.linspace(origin_kx, tip_kx, N_LINE)
ky_line = np.zeros(N_LINE)
gap_line_b2u = gap_function(kx_line, ky_line, KZ, 'B2u') * 1e3
gap_line_b3u = gap_function(kx_line, ky_line, KZ, 'B3u') * 1e3
# x-axis: normalised position along vector  0 → 1
t_line = np.linspace(0, 1, N_LINE)
# also keep kx in units of π/a for labelling
kx_norm = kx_line / (np.pi / a)   # range ≈ -0.62 … +0.62

# ── Figure — GridSpec: two maps top, one wide line-cut bottom ─────────────────
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(11, 8.0), dpi=150)
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        height_ratios=[1.0, 0.55],
                        hspace=0.42, wspace=0.28,
                        left=0.07, right=0.87,
                        bottom=0.08, top=0.95)

ax_b2u = fig.add_subplot(gs[0, 0:2])   # top-left map  (spans cols 0-1)
ax_b3u = fig.add_subplot(gs[0, 2])      # top-right map (col 2 — will be resized below)

# remake 1×2 top axes properly
fig.clear()
gs  = gridspec.GridSpec(2, 2, figure=fig,
                        height_ratios=[1.0, 0.32],
                        hspace=0.45, wspace=0.28,
                        left=0.07, right=0.87,
                        bottom=0.08, top=0.95)
ax_b2u  = fig.add_subplot(gs[0, 0])
ax_b3u  = fig.add_subplot(gs[0, 1])
ax_line = fig.add_subplot(gs[1, :])    # spans full width

norm = mcolors.TwoSlopeNorm(vmin=-VMAX, vcenter=0, vmax=VMAX)
cmap = 'RdBu_r'

extent = [ky_vals[0], ky_vals[-1], kx_vals[0], kx_vals[-1]]

for ax, gap_data, title, label in [
        (ax_b2u, gap_b2u, r'$B_{2u}$ gap,  $k_z = 0$', '(a)'),
        (ax_b3u, gap_b3u, r'$B_{3u}$ gap,  $k_z = 0$', '(b)'),
]:
    im = ax.imshow(gap_data.T,
                   origin='lower', extent=extent, aspect='auto',
                   cmap=cmap, norm=norm, interpolation='bilinear')

    for band in range(4):
        try:
            ax.contour(ky_vals, kx_vals, eigenvalues[:, :, band],
                       levels=[0.0], colors=[FS_COLORS[band]],
                       linewidths=2.0, alpha=1.0)
        except Exception:
            pass

    # p6 scattering vector (unlabelled)
    ax.annotate('',
                xy=(tip_ky, tip_kx), xytext=(origin_ky, origin_kx),
                arrowprops=dict(arrowstyle='->', color='black',
                                lw=2.2, mutation_scale=16))

    ax.set_xlabel(r'$k_y$  $(\pi/b)$', fontsize=12)
    ax.set_ylabel(r'$k_x$  $(\pi/a)$', fontsize=12)
    ax.set_xticks(np.linspace(-np.pi / b, np.pi / b, 5))
    ax.set_yticks(np.linspace(-np.pi / a, np.pi / a, 5))
    ax.set_xticklabels(['-1', '-0.5', '0', '0.5', '1'], fontsize=9)
    ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'], fontsize=9)
    ax.set_title(title, fontsize=13, pad=8)
    ax.text(0.04, 0.95, label, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=2))

# ── Shared colorbar ───────────────────────────────────────────────────────────
cax  = fig.add_axes([0.895, 0.505, 0.028, 0.445])   # aligned with top row
cbar = fig.colorbar(im, cax=cax, orientation='vertical', extend='both')
ticks = np.linspace(-VMAX, VMAX, 7)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f'{t:+.2f}' if t != 0 else '0' for t in ticks], fontsize=10)
cbar.set_label(r'$\Delta_\mathbf{k}$ (meV)', fontsize=12, labelpad=8)

# Extend bottom panel to match full width of top row + colorbar
_pos = ax_line.get_position()
_cbar_right = 0.895 + 0.028   # right edge of cax
ax_line.set_position([_pos.x0, _pos.y0, _cbar_right - _pos.x0, _pos.height])

# ── Line-cut panel (c) ────────────────────────────────────────────────────────
ax_line.plot(kx_norm, gap_line_b2u, color='#FD0000', lw=2.2, label=r'$B_{2u}$')
ax_line.plot(kx_norm, gap_line_b3u, color='#1E88E5', lw=2.2, label=r'$B_{3u}$')
ax_line.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.6)
ax_line.axvline(origin_kx / (np.pi / a), color='black', lw=1.0, ls=':', alpha=0.5)
ax_line.axvline(tip_kx   / (np.pi / a), color='black', lw=1.0, ls=':', alpha=0.5)
ax_line.set_xlim(kx_norm[0], kx_norm[-1])
ax_line.set_xlabel(r'$k_x$  $(\pi/a)$', fontsize=12)
ax_line.set_ylabel(r'$\Delta_\mathbf{k}$  (meV)', fontsize=12)
ax_line.set_title(r'(c)  Gap along scattering vector  ($k_y = 0$)', fontsize=13, pad=6)
ax_line.legend(fontsize=11, framealpha=0.9)
ax_line.grid(True, alpha=0.25)
ax_line.set_ylim(-VMAX * 1.12, VMAX * 1.12)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs('outputs/gap_symmetry', exist_ok=True)
out = 'outputs/gap_symmetry/bz_gap_map.png'
fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Saved: {out}")
plt.show()
