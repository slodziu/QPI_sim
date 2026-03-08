"""
gap_symmetry_schematic.py
--------------------------

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import Wedge
import os

# ── UTe2 colour palette ────────────────────────────────────────────────────────
C_RED  = "#9635E5"
C_BLUE = "#FD0000"
C_LBLU = "#0044FF"
C_PURP = '#9635E5'
C_FS   = '#1a1a2e'   # very dark navy for Fermi surface ring
C_BG   = '#F4F4F8'   # panel background

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
})

# ── helpers ───────────────────────────────────────────────────────────────────
def draw_circle(ax, r, **kw):
    ax.add_patch(plt.Circle((0, 0), r, **kw))

def polar_lobe(ax, theta_c, half_width, r_max, color, alpha=0.92, n=300, zorder=2):
    """Fill one d-wave petal shaped by |cos(2θ)|."""
    theta = np.linspace(theta_c - half_width, theta_c + half_width, n)
    r = r_max * np.abs(np.cos(2 * theta))
    xs = np.concatenate([[0], r * np.cos(theta), [0]])
    ys = np.concatenate([[0], r * np.sin(theta), [0]])
    ax.fill(xs, ys, color=color, alpha=alpha, zorder=zorder)

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(5.4, 10.2), dpi=150,
                          subplot_kw=dict(aspect='equal'))
fig.patch.set_facecolor('white')

R_FS = 1.0   # normalised Fermi surface radius

# ══════════════════════════════════════════════════════════════════════════════
# Panel (a) — d_{x²−y²}-wave:   Δ(k) = Δ₀(cos kx − cos ky)  ≈  Δ₀ cos(2φ)
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_facecolor(C_BG)
ax.set_xlim(-1.38, 1.38)
ax.set_ylim(-1.38, 1.38)
ax.axis('off')

# Fermi-sea background disc
draw_circle(ax, R_FS, color='#e8eaf6', zorder=0)

# Four petals: Δ > 0 at 0° / 180° (red), Δ < 0 at 90° / 270° (blue)
for theta_c, color in [(0,           C_RED),
                        (np.pi,       C_RED),
                        (np.pi/2,     C_BLUE),
                        (3*np.pi/2,   C_BLUE)]:
    polar_lobe(ax, theta_c, np.pi/4, R_FS, color)

# Fermi surface ring
draw_circle(ax, R_FS, fill=False, edgecolor=C_FS, linewidth=2.5, zorder=8)

# Nodal lines (white dashed at ±45°)
for ang in [np.pi/4, -np.pi/4]:
    ax.plot([-np.cos(ang), np.cos(ang)],
            [-np.sin(ang),  np.sin(ang)],
            color='white', lw=2.0, ls='--', zorder=9, solid_capstyle='round')

# ± labels inside petals
for x, y, sign in [(0.52, 0, '+'), (-0.52, 0, '+'),
                    (0, 0.52, '−'), (0, -0.52, '−')]:
    ax.text(x, y, sign, ha='center', va='center', fontsize=26,
            color='white', fontweight='bold', zorder=10)

# "Line node" callout
ax.annotate('Line node', xy=(0.62, 0.62), xytext=(0.82, 1.12),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='#444', lw=1.0),
            bbox=dict(facecolor='white', alpha=0.80, edgecolor='none', pad=1.5),
            zorder=11)

# Colour legend
ax.legend(handles=[mpatches.Patch(color=C_RED,  label=r'$\Delta > 0$  (+)'),
                   mpatches.Patch(color=C_BLUE, label=r'$\Delta < 0$  (−)')],
          loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=2,
          fontsize=9, framealpha=0.9)

ax.set_title(r'$d_{x^2-y^2}$-wave: $\Delta_\mathbf{k} \propto \cos k_x - \cos k_y$',
             fontsize=12, pad=14)
ax.text(-1.31, 1.25, '(a)', fontsize=14, fontweight='bold')

# ══════════════════════════════════════════════════════════════════════════════
# Panel (b) — chiral p-wave:   Δ(k) = Δ₀(kx + i ky),   L_z = +1,  nodeless
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]
ax.set_facecolor(C_BG)
ax.set_xlim(-1.38, 1.38)
ax.set_ylim(-1.38, 1.38)
ax.axis('off')

# Phase-cyclic colormap using UTe2 palette
cmap_cyc = mcolors.LinearSegmentedColormap.from_list(
    'ute2_cyclic',
    [C_BLUE, C_PURP, C_RED, '#FF8C00', C_LBLU, C_BLUE], N=512)

r_out = 1.03
r_in  = 0.72

# Background Fermi-sea disc
draw_circle(ax, R_FS, color='#e8eaf6', zorder=0)

# Phase-coloured ring (720 thin Wedge patches for smooth gradient)
N_WED = 720
d_ang = 360.0 / N_WED
for i in range(N_WED):
    ang_deg = i * d_ang
    c = cmap_cyc(i / N_WED)
    ax.add_patch(Wedge((0, 0), r_out, ang_deg, ang_deg + d_ang + 0.25,
                       width=r_out - r_in, color=c, zorder=3))

# Inner nodeless disc (light fill — |Δ| > 0 everywhere)
draw_circle(ax, r_in, color=C_LBLU, alpha=0.22, zorder=2)

# Fermi surface midline
draw_circle(ax, (r_out + r_in) / 2, fill=False,
            edgecolor=C_FS, linewidth=1.5, zorder=7)

# CCW winding arrows → L_z = +1
r_arr, n_arr, dl = (r_out + r_in) / 2, 8, 0.19
for ang in np.linspace(0, 2*np.pi, n_arr, endpoint=False):
    px, py = r_arr * np.cos(ang), r_arr * np.sin(ang)
    tx, ty = -np.sin(ang), np.cos(ang)   # CCW tangent
    ax.annotate('', xy=(px + dl*tx, py + dl*ty),
                xytext=(px - dl*tx, py - dl*ty),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.8),
                zorder=9)



# θ_k arc annotation (illustrates the winding phase = θ_k)
a0 = np.pi / 5
ax.annotate('', xy=(0.48*np.cos(a0), 0.48*np.sin(a0)),
             xytext=(0.14, 0),
             arrowprops=dict(arrowstyle='->', color='#333',
                             connectionstyle='arc3,rad=0.38', lw=1.1),
             zorder=11)
ax.text(0.43, 0.15, r'$\theta_\mathbf{k}$', fontsize=11,
        color='#333', zorder=11)

# Phase gradient bar — vertical, right of panel (b)
import matplotlib.cm as mcm
sm = mcm.ScalarMappable(cmap=cmap_cyc,
                        norm=mcolors.Normalize(vmin=0, vmax=2*np.pi))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                    fraction=0.046, pad=0.04, aspect=18)
cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
cbar.set_ticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'],
                     fontsize=8)
cbar.set_label(r'Phase  arg$(k_x + ik_y)$', fontsize=8, labelpad=4)

ax.set_title(r'Chiral $p$-wave: $\Delta_\mathbf{k} \propto k_x + ik_y$',
             fontsize=12, pad=14)
ax.text(-1.31, 1.25, '(b)', fontsize=14, fontweight='bold')

# ── save ──────────────────────────────────────────────────────────────────────
fig.tight_layout(pad=1.8)
os.makedirs('outputs/gap_symmetry', exist_ok=True)
out = 'outputs/gap_symmetry/gap_symmetry_schematic.png'
fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.15)
print(f"Saved: {out}")
plt.show()
