"""
dIdV_energy_sweep.py
────────────────────
Loop over every bias slice in spectrum.3ds, apply the full cleanup + FFT
pipeline, and save the notched-FFT figure (with scattering-vector arrows and
line-cut panels) to experiment_output/didv_energy_sweep/<bias_mV>/
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.ndimage import uniform_filter1d, gaussian_filter
from scipy.signal import find_peaks
from matplotlib.colors import LogNorm

# ── Parameters ─────────────────────────────────────────────────────────────────
OUTPUT_ROOT   = "experiment_output/didv_energy_sweep"

# Real-space calibration
a      = 0.416   # nm  (lattice constant a)
c_star = 0.8     # nm  (c* periodicity)

# Line-artifact removal
CAP_VALUE      = 0.35
THRESHOLD      = 2   # σ threshold for bad-row detection

# FFT display — derived per-slice from percentiles of the notched magnitude
# matching the hand-tuned zero-energy range
FFT_PLOW  = 30   # percentile → vmin
FFT_PHIGH = 99.5 # percentile → vmax

# Notch filter
NOTCH_HALF  = 2   # half-width in q_{c*} bins
NOTCH_SIGMA = 1   # Gaussian smooth after notch (px); 0 = off

# Peak detection exclusion radius around q=0
PEAK_EXCL = 0.1   # in units of π/c*

# Scattering vectors (in units of 2π/a, 2π/c*): (qx, qy)
SCATTERING_VECTORS = {
    'q2': (0.44,  0.5),
    'q4': (0.0,   1.0),
    'q5': (-0.15, 0.5),
    'q6': (0.59,  0.0),
}
SV_COLORS = {'q2': 'red', 'q4': 'lime', 'q5': 'cyan', 'q6': 'orange'}


# ── Helpers ────────────────────────────────────────────────────────────────────
def detrend_rows(arr):
    """Subtract per-column linear trend (preserves mean)."""
    def _detrend(r):
        t = np.arange(len(r))
        slope = np.polyfit(t, r, 1)[0]
        return r - slope * t
    return np.apply_along_axis(_detrend, axis=0, arr=arr)


def clean_didv(didv):
    """Cap outliers + interpolate dark-stripe row artifacts."""
    didv = np.clip(didv, None, CAP_VALUE)
    row_medians = np.median(didv, axis=1)
    baseline    = uniform_filter1d(row_medians, size=15, mode="mirror")
    deviation   = row_medians - baseline
    sigma       = np.std(deviation)
    bad_rows    = np.where(deviation < -THRESHOLD * sigma)[0]
    bad_rows    = bad_rows[(bad_rows > 0) & (bad_rows < didv.shape[0] - 1)]
    didv_clean  = didv.copy()
    for r in bad_rows:
        didv_clean[r, :] = 0.5 * (didv[r - 1, :] + didv[r + 1, :])
    return didv_clean


def compute_fft(didv_clean, x_nm, y_nm):
    """Detrend → FFT → symmetrize. Returns (fft_map, fft_mag, qx, qy)."""
    arr     = detrend_rows(didv_clean)
    fft_map = np.fft.fftshift(np.fft.fft2(arr))
    fft_mag = np.abs(fft_map)
    fft_mag = 0.25 * (fft_mag + fft_mag[::-1, :] +
                      fft_mag[:, ::-1] + fft_mag[::-1, ::-1])
    cx = fft_mag.shape[0] // 2
    fft_mag[cx - 1, :] = 0.5 * (fft_mag[cx - 2, :] + fft_mag[cx + 1, :])
    fft_mag[cx,     :] = 0.5 * (fft_mag[cx - 2, :] + fft_mag[cx + 1, :])
    Nx, Ny = arr.shape
    qx = np.fft.fftshift(2*np.pi*np.fft.fftfreq(Nx, d=(x_nm[1]-x_nm[0]))) / (np.pi/a)
    qy = np.fft.fftshift(2*np.pi*np.fft.fftfreq(Ny, d=(y_nm[1]-y_nm[0]))) / (np.pi/c_star)
    return fft_map, fft_mag, qx, qy, cx


def notch_filter(fft_map, fft_mag, peaks, cx):
    """Interpolate artifact columns; re-symmetrize; optionally smooth."""
    fft_notched = fft_map.copy()
    for pk in peaks:
        lo    = max(0, pk - NOTCH_HALF)
        hi    = min(fft_notched.shape[1] - 1, pk + NOTCH_HALF)
        left  = fft_notched[:, max(0, lo - 1)]
        right = fft_notched[:, min(fft_notched.shape[1] - 1, hi + 1)]
        for j in range(lo, hi + 1):
            t = (j - lo) / max(hi - lo, 1)
            fft_notched[:, j] = (1 - t) * left + t * right
    mag = np.abs(fft_notched)
    mag = 0.25 * (mag + mag[::-1, :] + mag[:, ::-1] + mag[::-1, ::-1])
    mag[cx - 1, :] = 0.5 * (mag[cx - 2, :] + mag[cx + 1, :])
    mag[cx,     :] = 0.5 * (mag[cx - 2, :] + mag[cx + 1, :])
    if NOTCH_SIGMA > 0:
        mag = gaussian_filter(mag, sigma=NOTCH_SIGMA)
    return mag


def save_notched_figure(fft_mag_notched, qx, qy, bias_mv, out_dir, fname, vmin, vmax):
    """Plot notched FFT + arrows + stacked line-cut panels and save."""
    # Convert scattering vectors: (qx_2pi_a, qy_2pi_cstar) → plot units (π/a, π/c*)
    # plot horizontal = qy (π/c*), plot vertical = qx (π/a)
    sv_plot = {k: (v[1] * 2, v[0] * 2) for k, v in SCATTERING_VECTORS.items()}
    sv_names = list(sv_plot.keys())
    n_sv     = len(sv_names)

    fig = plt.figure(figsize=(14, 6))
    gs  = fig.add_gridspec(n_sv, 2, width_ratios=[3, 1], hspace=0.08, wspace=0.35)
    ax2d = fig.add_subplot(gs[:, 0])

    im = ax2d.imshow(
        fft_mag_notched,
        extent=[qy.min(), qy.max(), qx.min(), qx.max()],
        origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis",
    )
    plt.colorbar(im, ax=ax2d, label="FFT magnitude", fraction=0.046, pad=0.04)
    ax2d.set_xlim(-2.4, 2.4); ax2d.set_ylim(-1.3, 1.3)
    ax2d.set_xlabel(r"$q_{c^*}$  ($\pi/c^*$)")
    ax2d.set_ylabel(r"$q_x$  ($\pi/a$)")
    ax2d.set_title(f"FFT — notch filtered  ({bias_mv:.2f} mV)")

    arrowprops = dict(arrowstyle="-|>", lw=1.5, mutation_scale=12)
    for name, (qy_plot, qx_plot) in sv_plot.items():
        col = SV_COLORS[name]
        ax2d.annotate("", xy=(qy_plot, qx_plot), xytext=(0, 0),
                      arrowprops={**arrowprops, "color": col})
        ax2d.text(qy_plot * 1.05, qx_plot * 1.05 + 0.04, name,
                  color=col, fontsize=8, fontweight="bold",
                  ha="center", va="bottom")

    for i, name in enumerate(sv_names):
        ax_lc = fig.add_subplot(gs[i, 1])
        qy_plot, qx_plot = sv_plot[name]
        col = SV_COLORS[name]
        row_idx = np.argmin(np.abs(qx - qx_plot))
        lc = fft_mag_notched[row_idx, :]
        ax_lc.plot(qy, lc, lw=0.9, color=col)
        ax_lc.axvline(qy_plot, color=col, lw=0.8, ls="--", alpha=0.8)
        ax_lc.set_ylabel("mag.", fontsize=6, labelpad=2)
        ax_lc.set_title(f"{name}  $q_x$={qx_plot:.2f}", fontsize=7, pad=2, color=col)
        ax_lc.tick_params(labelsize=6)
        ax_lc.set_xlim(-2.4, 2.4)
        if i < n_sv - 1:
            ax_lc.set_xticklabels([])
        else:
            ax_lc.set_xlabel(r"$q_{c^*}$  ($\pi/c^*$)", fontsize=7)

    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main loop ──────────────────────────────────────────────────────────────────
ds = xr.open_dataset("experimental_data/spectrum2.3ds", engine="nanonis")
bias = ds.coords["bias"].values

di_channel = next((v for v in ds.data_vars if "dI/dV" in v or "Input" in v), None)
if di_channel is None:
    raise ValueError("dI/dV channel not found.")

x = ds.coords["x"].values
y = ds.coords["y"].values
x_nm = x * 1e9 if x.max() < 1e-5 else x
y_nm = y * 1e9 if y.max() < 1e-5 else y

n_bias = len(bias)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ── Pass 1: compute per-slice notched FFTs and collect colour limits ────────────
print(f"Pass 1/2 — computing FFTs for {n_bias} bias slices …")
slices = []   # list of (bias_mv, fname, fft_mag_notched, qx, qy)
all_vmins, all_vmaxs = [], []

for idx, b in enumerate(bias):
    bias_mv  = b * 1000
    bias_str = f"{bias_mv:+.3f}mV".replace("+", "p").replace("-", "n")
    fname    = f"{idx:03d}_{bias_str}.png"

    didv        = ds[di_channel].isel(bias=idx).values.astype(float)
    didv_clean  = clean_didv(didv)
    fft_map, fft_mag, qx, qy, cx = compute_fft(didv_clean, x_nm, y_nm)

    col_power    = fft_mag.mean(axis=0)
    raw_peaks, _ = find_peaks(col_power, prominence=2 * col_power.std())
    art_qy       = qy[raw_peaks]
    mask         = np.abs(art_qy) > PEAK_EXCL
    peaks        = raw_peaks[mask]

    fft_mag_notched = notch_filter(fft_map, fft_mag, peaks, cx) if len(peaks) > 0 else fft_mag.copy()

    qy_mask = (qy >= -2.4) & (qy <= 2.4)
    qx_mask = (qx >= -1.3) & (qx <= 1.3)
    roi = fft_mag_notched[np.ix_(qx_mask, qy_mask)]
    all_vmins.append(float(np.percentile(roi, FFT_PLOW)))
    all_vmaxs.append(float(np.percentile(roi, FFT_PHIGH)))

    slices.append((bias_mv, fname, fft_mag_notched, qx, qy))
    print(f"  [{idx+1:3d}/{n_bias}]  {bias_mv:+8.3f} mV")

# Global colour limits: widest range across all slices
global_vmin = min(all_vmins)
global_vmax = max(all_vmaxs)
print(f"\nGlobal colour limits: vmin={global_vmin:.3f}  vmax={global_vmax:.3f}")

# ── Pass 2: save figures with consistent colour scale ──────────────────────────
print(f"\nPass 2/2 — saving figures …")
for idx, (bias_mv, fname, fft_mag_notched, qx, qy) in enumerate(slices):
    save_notched_figure(fft_mag_notched, qx, qy, bias_mv, OUTPUT_ROOT, fname,
                        global_vmin, global_vmax)
    print(f"  [{idx+1:3d}/{n_bias}]  {fname}")

print(f"\nDone. All figures saved under {OUTPUT_ROOT}/")
