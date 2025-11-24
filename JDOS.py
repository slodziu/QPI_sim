import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from UTe2_fixed import *
import os
import numpy as np

# =============================================================================
# JDOS Computation following Nature paper methodology
# =============================================================================

# Parameters
nk = 1000
energy_width = 2.0e-3  # 2 meV Gaussian width (wider for smoother features)
thresh = 0.30  # Keep top 70% intensity (less aggressive masking)

# Create k-space grid - USE FIRST BRILLOUIN ZONE ONLY
kx_vals = np.linspace(-np.pi/a, np.pi/a, nk)
ky_vals = np.linspace(-np.pi/b, np.pi/b, nk)  # First BZ, not extended
kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals, indexing='ij')

# Compute band structure on 2D grid at kz=0
print("Computing band structure at kz=0...")
kz = 0.0
num_bands = 4

# Initialize arrays
energies = np.zeros((nk, nk, num_bands))
char5f = np.zeros((nk, nk, num_bands))

# Compute for each k-point
for i in range(nk):
    if i % (nk//10) == 0:
        print(f"  Progress: {100*i/nk:.1f}%")
    for j in range(nk):
        kx = kx_grid[i, j]
        ky = ky_grid[i, j]
        
        # Get band energies and eigenvectors
        H = H_full(kx, ky, kz)
        evals, evecs = np.linalg.eigh(H)
        energies[i, j, :] = evals
        
        # Compute U-5f character for each band
        # Paper says "hybridized U 5f orbital spectral weight" = BOTH U orbitals
        for n in range(num_bands):
            psi = evecs[:, n]
            # Hybridized U-5f character = |c_U1|^2 + |c_U2|^2 (indices 0 and 1)
            char5f[i, j, n] = np.abs(psi[0])**2 + np.abs(psi[1])**2

print("  Progress: 100.0%")

# Apply symmetrization (crucial for clean patterns)
print("\nSymmetrizing data to remove numerical noise...")
for n in range(num_bands):
    E = energies[:, :, n]
    W = char5f[:, :, n]
    # Average (x,y), (-x,y), (x,-y), (-x,-y) for 4-fold symmetry
    E_sym = (E + E[::-1, :] + E[:, ::-1] + E[::-1, ::-1]) / 4.0
    W_sym = (W + W[::-1, :] + W[:, ::-1] + W[::-1, ::-1]) / 4.0
    energies[:, :, n] = E_sym
    char5f[:, :, n] = W_sym

# Step 1: Compute spectral intensity map with Gaussian weighting
# Use ALL bands weighted by proximity to Fermi level
print("\nComputing spectral intensity map...")
print("Using all bands with Gaussian weighting around Fermi level...")
I = np.zeros((nk, nk))
for n in range(num_bands):
    weight = char5f[:, :, n] * np.exp(-(energies[:, :, n] / energy_width)**2)
    I += weight
    if np.max(weight) > 1e-6:  # Only print bands with significant contribution
        print(f"  Band {n}: max weight = {weight.max():.3e}, contribution = {np.sum(weight):.3e}")

# Step 2: Normalize intensity
I /= I.max()

# Step 3: Apply smooth window function instead of hard threshold
# Use a smooth Gaussian-like envelope based on intensity
print(f"Applying smooth spectral window (soft threshold at {thresh:.0%})...")
# Smooth window: gradually suppress low-intensity regions
window = np.tanh(5 * (I / I.max() - thresh))  # Smooth transition
window[window < 0] = 0  # Keep only positive
I_masked = I * window

# Step 4: Compute JDOS via autocorrelation (correct for smooth patterns)
# JDOS(q) = ∫ I(k) I(k+q) dk = Real[IFFT[|FFT[I]|²]]
print("Computing JDOS via autocorrelation (IFFT method)...")
# Get k-space sampling factors
dkx = kx_vals[1] - kx_vals[0]
dky = ky_vals[1] - ky_vals[0]

# Autocorrelation method (gives smooth contours, not spiky FFT artifacts)
F = np.fft.fft2(I_masked)
power_spectrum = np.abs(F)**2
jdos = np.real(np.fft.ifft2(power_spectrum))
jdos = np.fft.fftshift(jdos)
jdos /= jdos.max()

# Apply Gaussian smoothing to JDOS for publication quality
from scipy.ndimage import gaussian_filter
print("Applying Gaussian smoothing for cleaner visualization...")
jdos_smooth = gaussian_filter(jdos, sigma=3.0)  # Stronger smoothing (3 pixels)
jdos = jdos_smooth

# Remove numerical noise floor (values below 1e-10 are numerical artifacts)
noise_floor = 1e-10
jdos[jdos < noise_floor] = noise_floor

# Print intensity statistics to understand the distribution
print(f"\nJDOS intensity statistics:")
print(f"  Max: {jdos.max():.3e}")
print(f"  99th percentile: {np.percentile(jdos, 99):.3e}")
print(f"  95th percentile: {np.percentile(jdos, 95):.3e}")
print(f"  90th percentile: {np.percentile(jdos, 90):.3e}")
print(f"  Median: {np.median(jdos):.3e}")


# Step 5: Compute q-space axes (for autocorrelation, use same grid as k-space)
# For autocorrelation IFFT method, q ranges from -k_max to +k_max
qx = np.fft.fftshift(np.fft.fftfreq(len(kx_vals), dkx))
qy = np.fft.fftshift(np.fft.fftfreq(len(ky_vals), dky))

# Convert q to units of π/a and π/b
qx_units = qx / (np.pi / a)  # qx corresponds to kx in π/a units
qy_units = qy / (np.pi / b)  # qy corresponds to ky in π/b units

# Debug: print actual ranges
print(f"k-space sampling:")
print(f"  kx: [{kx_vals.min():.3f}, {kx_vals.max():.3f}], dk = {dkx:.6f}")
print(f"  ky: [{ky_vals.min():.3f}, {ky_vals.max():.3f}], dk = {dky:.6f}")
print(f"q-space range (physical units):")
print(f"  qx: [{qx.min():.3f}, {qx.max():.3f}]")
print(f"  qy: [{qy.min():.3f}, {qy.max():.3f}]")
print(f"q-space range (normalized units):")
print(f"  qx: [{qx_units.min():.3f}, {qx_units.max():.3f}] (π/a)")
print(f"  qy: [{qy_units.min():.3f}, {qy_units.max():.3f}] (π/b)")

# For imshow: first axis is y (rows), second axis is x (cols)
# jdos shape is (nk_kx, nk_ky) = (qx, qy)
# To have qy on x-axis and qx on y-axis, use jdos directly (no transpose)

# Create figure with better formatting
fig, ax = plt.subplots(figsize=(10, 8))
# x-axis gets qy (π/b), y-axis gets qx (π/a)
# Set vmin/vmax to focus on actual features (clip extreme values)
vmin = np.percentile(jdos[jdos > noise_floor], 10)  # 10th percentile of real data
vmax = np.percentile(jdos, 99.9)  # 99.9th percentile to avoid single hot pixels
print(f"\nColorbar range: [{vmin:.3e}, {vmax:.3e}]")

im = ax.imshow(jdos, extent=[qy_units.min(), qy_units.max(), qx_units.min(), qx_units.max()],
               origin='lower', cmap='magma', norm=LogNorm(vmin=vmin, vmax=vmax), 
               aspect='auto', interpolation='bilinear')  # Smooth interpolation

# Note: axis limits determined by FFT frequency range, not manually set
# Maximum q is limited by Nyquist frequency: q_max ~ ±π/(2*dk)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, label='Normalized JDOS')

# Set labels and title - x-axis is π/b, y-axis is π/a
ax.set_xlabel(r'$q_y$ ($\pi/b$)', fontsize=14)
ax.set_ylabel(r'$q_x$ ($\pi/a$)', fontsize=14)
ax.set_title(r'Joint Density of States (JDOS) - U-5f orbital at $k_z=0$', fontsize=16, pad=20)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Set tick parameters
ax.tick_params(labelsize=12)

plt.tight_layout()

# Save with output directory creation
os.makedirs('outputs/JDOS', exist_ok=True)
filename = f'outputs/JDOS/JDOS.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved {filename}")