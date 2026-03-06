import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = "experiment_output/didv_cleanup"
os.makedirs(output_dir, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
ds = xr.open_dataset("experimental_data/spectrum.3ds", engine="nanonis")
bias = ds.coords["bias"].values

di_channel = next((v for v in ds.data_vars if "dI/dV" in v or "Input" in v), None)
if di_channel is None:
    raise ValueError("dI/dV channel not found.")

# ── Select 0 V map ─────────────────────────────────────────────────────────────
idx0 = np.argmin(np.abs(bias))
bias0 = bias[idx0]
print(f"Using bias slice {idx0}: {bias0*1000:.4f} mV  (closest to 0 V)")

didv = ds[di_channel].isel(bias=idx0).values.astype(float)   # (270, 270)
print(f"dI/dV map shape : {didv.shape}")
print(f"Range           : {didv.min():.4e}  –  {didv.max():.4e}")
print(f"Mean / Median   : {didv.mean():.4e}  /  {np.median(didv):.4e}")
print(f"Std             : {didv.std():.4e}")

# ── Raw map ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(didv, origin="lower", cmap="RdBu_r",
               vmin=np.percentile(didv, 2), vmax=np.percentile(didv, 98))
plt.colorbar(im, ax=ax, label="dI/dV (V)")
ax.set_title(f"Raw dI/dV map  ({bias0*1000:.2f} mV)")
ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
plt.savefig(os.path.join(output_dir, "didv_0V_raw.png"), dpi=300, bbox_inches="tight")
plt.close()

# ── Histogram ──────────────────────────────────────────────────────────────────
flat = didv.ravel()
p1, p5, p25, p50, p75, p95, p99 = np.percentile(flat, [1, 5, 25, 50, 75, 95, 99])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Linear histogram
ax = axes[0]
ax.hist(flat, bins=200, color="steelblue", edgecolor="none")
for p, label, col in [(p1,"1%","red"),(p5,"5%","orange"),(p95,"95%","orange"),(p99,"99%","red")]:
    ax.axvline(p, color=col, lw=1.2, ls="--", label=f"{label}: {p:.3e}")
ax.set_xlabel("dI/dV (V)"); ax.set_ylabel("count")
ax.set_title("Intensity histogram (linear)")
ax.legend(fontsize=7)

# Log-y histogram (reveals outlier tails clearly)
ax = axes[1]
counts, edges, _ = ax.hist(flat, bins=200, color="steelblue", edgecolor="none", log=True)
for p, label, col in [(p1,"1%","red"),(p5,"5%","orange"),(p95,"95%","orange"),(p99,"99%","red")]:
    ax.axvline(p, color=col, lw=1.2, ls="--", label=f"{label}: {p:.3e}")
ax.set_xlabel("dI/dV (V)"); ax.set_ylabel("count (log)")
ax.set_title("Intensity histogram (log y)")
ax.legend(fontsize=7)

fig.suptitle(f"dI/dV histogram — {bias0*1000:.2f} mV", fontsize=10)
plt.savefig(os.path.join(output_dir, "didv_0V_histogram.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\nPercentile summary:")
for pct, val in zip([1, 5, 25, 50, 75, 95, 99], [p1,p5,p25,p50,p75,p95,p99]):
    print(f"  {pct:3d}%  {val:.4e}")

# ── Tunable map ────────────────────────────────────────────────────────────────
# Adjust vmin / vmax here to taste (values in same units as dI/dV, i.e. V)
VMIN = 0.2
VMAX = 0.35   # matches CAP_VALUE

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(didv, origin="lower", cmap="plasma", vmin=VMIN, vmax=VMAX)
plt.colorbar(im, ax=ax, label="dI/dV (V)")
ax.set_title(f"dI/dV map  ({bias0*1000:.2f} mV)\nvmin={VMIN:.3e}  vmax={VMAX:.3e}")
ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
plt.savefig(os.path.join(output_dir, "didv_0V_tuned.png"), dpi=300, bbox_inches="tight")
plt.show(); plt.close()
print(f"Tuned map saved  (vmin={VMIN:.3e}, vmax={VMAX:.3e})")

# ── Horizontal line artifact removal ──────────────────────────────────────────
# Strategy (as suggested): for each bad row, replace every pixel with the
# mean of the pixel directly above and directly below.
# Bad rows are auto-detected by comparing each row's median to a smoothed
# baseline of surrounding row medians — rows that deviate MORE THAN THRESHOLD * std
# BELOW the baseline are flagged (dark stripe artifacts only).

THRESHOLD = 1.5  # flag rows deviating more than this many σ from local baseline

# Cap high outlier values before cleanup
CAP_VALUE = 0.35
didv = np.clip(didv, None, CAP_VALUE)
print(f"\nValues capped at {CAP_VALUE}  (max after cap: {didv.max():.4e})")

row_medians = np.median(didv, axis=1)          # one value per row
# smoothed baseline: rolling median over a window of neighbouring rows
from scipy.ndimage import uniform_filter1d
baseline = uniform_filter1d(row_medians, size=15, mode="mirror")
deviation = row_medians - baseline
sigma = np.std(deviation)
bad_rows = np.where(deviation < -THRESHOLD * sigma)[0]

# clamp: never flag the very first or last row (no neighbour on one side)
bad_rows = bad_rows[(bad_rows > 0) & (bad_rows < didv.shape[0] - 1)]
print(f"\nLine-artifact removal  (threshold={THRESHOLD}σ = {THRESHOLD*sigma:.4e})")
print(f"  {len(bad_rows)} bad rows flagged: {bad_rows.tolist()}")

didv_clean = didv.copy()
for r in bad_rows:
    didv_clean[r, :] = 0.5 * (didv[r - 1, :] + didv[r + 1, :])

# Diagnostic: row-median profile with flagged lines marked
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.plot(row_medians, np.arange(len(row_medians)), lw=0.8, color="steelblue", label="row median")
ax.plot(baseline,   np.arange(len(baseline)),    lw=1.2, color="black",     ls="--", label="baseline")
for r in bad_rows:
    ax.axhline(r, color="red", lw=0.8, alpha=0.6)
ax.set_xlabel("row median dI/dV (V)"); ax.set_ylabel("row index")
ax.set_title("Row-median profile\n(red = flagged lines)")
ax.legend(fontsize=7)
ax.invert_yaxis()

ax = axes[1]
ax.plot(deviation, np.arange(len(deviation)), lw=0.8, color="steelblue")
ax.axvline(-THRESHOLD * sigma, color="red", ls="--", lw=1, label=f"−{THRESHOLD}σ (cutoff)")
ax.axvline( THRESHOLD * sigma, color="gray", ls=":", lw=0.8, alpha=0.5, label=f"+{THRESHOLD}σ (ignored)")
ax.scatter(deviation[bad_rows], bad_rows, color="red", s=20, zorder=5)
ax.set_xlabel("deviation from baseline (V)"); ax.set_ylabel("row index")
ax.set_title("Deviation from row-median baseline")
ax.legend(fontsize=7)
ax.invert_yaxis()

plt.savefig(os.path.join(output_dir, "didv_0V_linedetect.png"), dpi=200, bbox_inches="tight")
plt.close()

# Side-by-side before / after (with flagged row markers)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, data, title in zip(axes,
                           [didv,       didv_clean],
                           ["Before cleanup", "After cleanup"]):
    im = ax.imshow(data, origin="lower", cmap="plasma", vmin=VMIN, vmax=VMAX)
    # mark flagged rows
    for r in bad_rows:
        ax.axhline(r, color="cyan", lw=0.6, alpha=0.7)
    plt.colorbar(im, ax=ax, label="dI/dV (V)", fraction=0.046, pad=0.04)
    ax.set_title(title); ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
plt.savefig(os.path.join(output_dir, "didv_0V_cleaned.png"), dpi=300, bbox_inches="tight")
plt.close()

# Clean before / after without row markers
for data, fname, title in [
    (didv,       "didv_0V_before.png", "Before cleanup"),
    (didv_clean, "didv_0V_after.png",  "After cleanup"),
]:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data, origin="lower", cmap="plasma", vmin=VMIN, vmax=VMAX)
    plt.colorbar(im, ax=ax, label="dI/dV (V)", fraction=0.046, pad=0.04)
    ax.set_title(title); ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
    plt.close()

print("Cleaned map saved.")

# ── FFT of cleaned map ─────────────────────────────────────────────────────────
from scipy.ndimage import gaussian_filter
a      = 0.416   # nm
c_star = 0.8   # nm

x = ds.coords["x"].values
y = ds.coords["y"].values
x_nm = x * 1e9 if x.max() < 1e-5 else x
y_nm = y * 1e9 if y.max() < 1e-5 else y

# Plane-detrend line-by-line then apply 2-D Hann window
def detrend_rows(arr):
    def _detrend(r):
        t = np.arange(len(r))
        slope = np.polyfit(t, r, 1)[0]   # fit slope only; keep intercept (mean) intact
        return r - slope * t              # preserves column mean → qx=0 row unaffected
    return np.apply_along_axis(_detrend, axis=0, arr=arr)

arr = detrend_rows(didv_clean)  # subtract linear trend per row and per column

fft_map = np.fft.fftshift(np.fft.fft2(arr))
fft_mag = np.abs(fft_map)

# Symmetrize across both axes: average |F| with reflections in qx and qc*
# enforces 4-fold mirror symmetry, reducing asymmetric noise
fft_mag = 0.25 * (fft_mag + fft_mag[::-1, :] + fft_mag[:, ::-1] + fft_mag[::-1, ::-1])
# Blend qx=0 rows from their immediate neighbors (even N → two centre rows affected)
_cx = fft_mag.shape[0] // 2
fft_mag[_cx - 1, :] = 0.5 * (fft_mag[_cx - 2, :] + fft_mag[_cx + 1, :])
fft_mag[_cx,     :] = 0.5 * (fft_mag[_cx - 2, :] + fft_mag[_cx + 1, :])
Nx, Ny = arr.shape
qx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(Nx, d=(x_nm[1]-x_nm[0]))) / (np.pi / a)
qy = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(Ny, d=(y_nm[1]-y_nm[0]))) / (np.pi / c_star)

# Tunable FFT display range
FFT_VMIN = 6
FFT_VMAX = 90
LOG_SCALE = False   # True = log colormap (much better contrast for faint features)

from matplotlib.colors import LogNorm
_norm = LogNorm(vmin=FFT_VMIN, vmax=FFT_VMAX) if LOG_SCALE else None
_imshow_kw = dict(cmap="viridis", norm=_norm) if LOG_SCALE else dict(cmap="viridis", vmin=FFT_VMIN, vmax=FFT_VMAX)

# ── Detect vertical-line artifacts in FFT (peaks in column-integrated power) ──
from scipy.signal import find_peaks

col_power = fft_mag.mean(axis=0)   # shape (Ny,) — mean |F| for each q_{c*} bin
peaks, _ = find_peaks(col_power, prominence=2 * col_power.std())
artifact_qy = qy[peaks]
    # Exclude peaks within ±0.1 of qy=0
mask = np.abs(artifact_qy) > 0.1
artifact_qy = artifact_qy[mask]
peaks = peaks[mask]
print(f"\nDetected artifact q_{{c*}} positions (\u03c0/c*): {np.round(artifact_qy, 3).tolist()}")
print(f"\nDetected artifact q_{{c*}} positions (π/c*): {np.round(artifact_qy, 3).tolist()}")

# ── Combined 2D FFT + 1D marginal diagnostic ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                         gridspec_kw={"width_ratios": [4, 1]})
ax2d, ax1d = axes

im = ax2d.imshow(
    fft_mag,
    extent=[qy.min(), qy.max(), qx.min(), qx.max()],
    origin="lower", aspect="auto", **_imshow_kw,
)
plt.colorbar(im, ax=ax2d, label="FFT magnitude", fraction=0.046, pad=0.04)
for qy0 in artifact_qy:
    ax2d.axvline(qy0, color="red", lw=0.8, ls="--", alpha=0.7)
ax2d.set_xlim(-2.4, 2.4); ax2d.set_ylim(-1.3, 1.3)
ax2d.set_xlabel(r"$q_{c^*}$  ($\pi/c^*$)")
ax2d.set_ylabel(r"$q_x$  ($\pi/a$)")
ax2d.set_title(f"FFT of cleaned dI/dV  ({bias0*1000:.2f} mV)  — artifacts marked")

ax1d.plot(col_power, qy, lw=0.8, color="steelblue")
for qy0 in artifact_qy:
    ax1d.axhline(qy0, color="red", lw=0.8, ls="--", alpha=0.8,
                 label=f"{qy0:.2f}")
ax1d.scatter(col_power[peaks], qy[peaks], color="red", s=25, zorder=5)
ax1d.set_xlabel(r"$\langle|F|\rangle_{q_x}$", fontsize=8)
ax1d.set_ylim(-1.3, 1.3); ax1d.set_yticklabels([])
ax1d.set_title("col-avg\npower", fontsize=8)
if len(artifact_qy):
    ax1d.legend(fontsize=6, title="q art.", title_fontsize=6)

fft_path = os.path.join(output_dir, "didv_0V_cleaned_fft.png")
plt.savefig(fft_path, dpi=300, bbox_inches="tight")
plt.show(); plt.close()
print(f"FFT saved to {fft_path}")

# ── Notch-filtered FFT (artifact q_{c*} columns interpolated away) ─────────────
# Artifacts are suppressed in FFT space; their locations are still marked.
NOTCH               = True   # set False to skip
NOTCH_HALF          = 2      # half-width in q_{c*} bins for vertical line notch
NOTCH_SIGMA         = 1    # Gaussian smooth sigma (px) applied after notch+symmetrise; 0 = off

if NOTCH and len(peaks) > 0:
    fft_map_notched = fft_map.copy()
    # vertical notches (artifact q_{c*} columns) — done on complex map
    for pk in peaks:
        lo = max(0, pk - NOTCH_HALF)
        hi = min(fft_map_notched.shape[1] - 1, pk + NOTCH_HALF)
        left  = fft_map_notched[:, max(0, lo - 1)]
        right = fft_map_notched[:, min(fft_map_notched.shape[1] - 1, hi + 1)]
        for j in range(lo, hi + 1):
            t = (j - lo) / max(hi - lo, 1)
            fft_map_notched[:, j] = (1 - t) * left + t * right
    fft_mag_notched = np.abs(fft_map_notched)
    fft_mag_notched = 0.25 * (fft_mag_notched + fft_mag_notched[::-1, :] + fft_mag_notched[:, ::-1] + fft_mag_notched[::-1, ::-1])
    fft_mag_notched[_cx - 1, :] = 0.5 * (fft_mag_notched[_cx - 2, :] + fft_mag_notched[_cx + 1, :])
    fft_mag_notched[_cx,     :] = 0.5 * (fft_mag_notched[_cx - 2, :] + fft_mag_notched[_cx + 1, :])
    if NOTCH_SIGMA > 0:
        fft_mag_notched = gaussian_filter(fft_mag_notched, sigma=NOTCH_SIGMA)

    # ── Scattering vectors (given in 2π/a, 2π/c*) — convert to π/a, π/c* by ×2 ──
    # Only q2, q4, q5, q6 are plotted; format: (qx_2pi_a, qy_2pi_cstar)
    scattering_vectors = {
        'q2': (0.44, 0.5),
        'q4': (0.0,  1.0),
        'q5': (-0.15, 0.5),
        'q6': (0.59, 0.0),
    }
    sv_colors = {'q2': 'red', 'q4': 'lime', 'q5': 'cyan', 'q6': 'orange'}
    # Convert to plot units (π/a, π/c*): horizontal=qy, vertical=qx
    sv_plot = {k: (v[1] * 2, v[0] * 2) for k, v in scattering_vectors.items()}
    # sv_plot[k] = (qy_in_pi_cstar, qx_in_pi_a)

    sv_names = list(sv_plot.keys())
    n_sv = len(sv_names)

    fig = plt.figure(figsize=(14, 6))
    # Left: 2D FFT spanning full height; right: 4 stacked line-cut panels
    gs = fig.add_gridspec(n_sv, 2, width_ratios=[3, 1], hspace=0.08, wspace=0.35)
    ax2d = fig.add_subplot(gs[:, 0])

    im = ax2d.imshow(
        fft_mag_notched,
        extent=[qy.min(), qy.max(), qx.min(), qx.max()],
        origin="lower", aspect="auto", vmin=7, vmax=20,
    )
    plt.colorbar(im, ax=ax2d, label="FFT magnitude", fraction=0.046, pad=0.04)
    ax2d.set_xlim(-2.4, 2.4); ax2d.set_ylim(-1.3, 1.3)
    ax2d.set_xlabel(r"$q_{c^*}$  ($\pi/c^*$)")
    ax2d.set_ylabel(r"$q_x$  ($\pi/a$)")
    ax2d.set_title(f"FFT — notch filtered  ({bias0*1000:.2f} mV)")

    # Draw each scattering vector as an arrow from origin
    arrowprops = dict(arrowstyle="-|>", lw=1.5, mutation_scale=12)
    for name, (qy_plot, qx_plot) in sv_plot.items():
        col = sv_colors[name]
        ax2d.annotate("", xy=(qy_plot, qx_plot), xytext=(0, 0),
                      arrowprops={**arrowprops, "color": col})
        # Label near arrowhead
        ax2d.text(qy_plot * 1.05, qx_plot * 1.05 + 0.04, name,
                  color=col, fontsize=8, fontweight="bold",
                  ha="center", va="bottom")

    # ── Line cuts: at each vector's qx row, plot intensity vs qy ────────────
    lc_axes = []
    for i, name in enumerate(sv_names):
        ax_lc = fig.add_subplot(gs[i, 1])
        lc_axes.append(ax_lc)
        qy_plot, qx_plot = sv_plot[name]
        col = sv_colors[name]
        # Find nearest row for this qx value
        row_idx = np.argmin(np.abs(qx - qx_plot))
        lc = fft_mag_notched[row_idx, :]
        ax_lc.plot(qy, lc, lw=0.9, color=col)
        # Mark the vector's qy position
        ax_lc.axvline(qy_plot, color=col, lw=0.8, ls="--", alpha=0.8)
        ax_lc.set_ylabel("mag.", fontsize=6, labelpad=2)
        ax_lc.set_title(f"{name}  $q_x$={qx_plot:.2f}", fontsize=7, pad=2,
                        color=col)
        ax_lc.tick_params(labelsize=6)
        ax_lc.set_xlim(-2.4, 2.4)
        if i < n_sv - 1:
            ax_lc.set_xticklabels([])
        else:
            ax_lc.set_xlabel(r"$q_{c^*}$  ($\pi/c^*$)", fontsize=7)

    notch_path = os.path.join(output_dir, "didv_0V_fft_notched.png")
    plt.savefig(notch_path, dpi=300, bbox_inches="tight")
    plt.show(); plt.close()
    print(f"Notch-filtered FFT saved to {notch_path}")

# ── FFT magnitude histogram ────────────────────────────────────────────────────
log_mag = fft_mag.ravel()
fp1, fp5, fp50, fp95, fp99 = np.percentile(log_mag, [1, 5, 50, 95, 99])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.hist(log_mag[log_mag <= fp99], bins=200, color="steelblue", edgecolor="none")
for p, label, col in [(fp1,"1%","red"),(fp5,"5%","orange"),(fp95,"95%","orange"),(fp99,"99%","red")]:
    ax.axvline(p, color=col, lw=1.2, ls="--", label=f"{label}: {p:.3e}")
ax.set_xlabel("FFT magnitude"); ax.set_ylabel("count")
ax.set_title(f"FFT magnitude histogram (linear, clipped to 99th pct={fp99:.2e})")
ax.legend(fontsize=7)

ax = axes[1]
ax.hist(log_mag, bins=200, color="steelblue", edgecolor="none", log=True)
for p, label, col in [(fp1,"1%","red"),(fp5,"5%","orange"),(fp95,"95%","orange"),(fp99,"99%","red")]:
    ax.axvline(p, color=col, lw=1.2, ls="--", label=f"{label}: {p:.3e}")
ax.set_xlabel("FFT magnitude"); ax.set_ylabel("count (log)")
ax.set_title("FFT magnitude histogram (log y)")
ax.legend(fontsize=7)

fig.suptitle(f"FFT magnitude histogram — cleaned dI/dV  ({bias0*1000:.2f} mV)", fontsize=10)
fft_hist_path = os.path.join(output_dir, "didv_0V_fft_histogram.png")
plt.savefig(fft_hist_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"FFT histogram saved to {fft_hist_path}")
print(f"\nFFT magnitude percentiles:")
for pct, val in zip([1, 5, 50, 95, 99], [fp1, fp5, fp50, fp95, fp99]):
    print(f"  {pct:3d}%  {val:.4e}")
