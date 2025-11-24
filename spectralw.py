import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from UTe2_fixed import *
import os
import pickle
import hashlib

res = 3000
def calculate_fig1b():
    # Grid Setup (High Res for small eta)
    nk_x = res
    nk_y = res # Higher resolution for extended Y-axis
    
    # Using your requested ranges
    kx_vals = np.linspace(-np.pi/a, np.pi/a, nk_x)
    ky_vals = np.linspace(-3*np.pi/b, 3*np.pi/b, nk_y) # Extended range
    KX, KY = np.meshgrid(kx_vals, ky_vals)
    
    print("Building Hamiltonian...")
    H_grid = H_full(KX, KY, 0.0)
    
    print("Calculating Green's Functions...")
    # Parameters for Spectral Weight
    eta_val = 0.0001  # 1 meV for visibility (0.0001 is physically accurate but very hard to plot)
    target_E = 0.0   # Fermi Level
    
    # Broadcast Identity
    identity = np.eye(4)
    inverse_input = (target_E + 1j * eta_val) * identity - H_grid
    
    # Vectorized Inversion
    G_grid = np.linalg.inv(inverse_input)
    
    # --- PROJECTION ---
    # CRITICAL: Using indices [0, 1] for Uranium based on your H_full structure
    u_indices = [0, 1] 
    
    spectral_weight = np.zeros_like(KX)
    for idx in u_indices:
        spectral_weight += -1.0 / np.pi * np.imag(G_grid[:, :, idx, idx])

    return KX, KY, spectral_weight


def plot_jdos(JDOS_q, kx_vals, ky_vals, with_vectors=False):
    """Plot JDOS in q-space with proper axis scaling"""
    # Calculate q-space grid
    nqx, nqy = JDOS_q.shape[1], JDOS_q.shape[0]
    
    # q-space extends from -2*k_max to 2*k_max
    kx_max = np.max(np.abs(kx_vals))
    ky_max = np.max(np.abs(ky_vals))
    
    qx_vals = np.linspace(-2*kx_max, 2*kx_max, nqx)
    qy_vals = np.linspace(-2*ky_max, 2*ky_max, nqy)
    QX_grid, QY_grid = np.meshgrid(qx_vals, qy_vals)

    # Avoid log(0) issues
    JDOS_plot = np.log10(JDOS_q + 1e-10)
    
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(QY_grid, QX_grid, JDOS_plot, cmap='viridis', shading='auto', 
                   vmax=np.percentile(JDOS_plot, 99))
    plt.colorbar(label='log₁₀(JDOS)', shrink=0.8)

    # Zoom to relevant region
    Q_max_display = 2 * np.pi / a
    plt.xlim(-Q_max_display, Q_max_display)
    plt.ylim(-Q_max_display, Q_max_display)
    
    # Format axes with π/a and π/b units
    qy_ticks = np.linspace(-Q_max_display, Q_max_display, 7)
    qx_ticks = np.linspace(-Q_max_display, Q_max_display, 7)
    qy_labels = [f'{val/(np.pi/b):.1f}' for val in qy_ticks]
    qx_labels = [f'{val/(np.pi/a):.1f}' for val in qx_ticks]
    
    plt.xticks(qy_ticks, qy_labels)
    plt.yticks(qx_ticks, qx_labels)
    plt.xlabel(r'$q_y$ (π/b)', fontsize=12)
    plt.ylabel(r'$q_x$ (π/a)', fontsize=12)
    
    title = 'Joint Density of States (JDOS) at E=0'
    if with_vectors:
        title += ' with Scattering Vectors'
        
        # Define the scattering vectors from the Nature paper
        # Format: (qx in units of 2π/a, qy in units of 2π/b)
        vectors = {
            'p1': (0.29, 0),
            'p2': (0.43, 1),
            'p3': (0.29, 2),
            'p4': (0, 2),
            'p5': (-0.14, 1),
            'p6': (0.57, 0)
        }
        
        # Convert to physical units and plot
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
        
        for i, (label, (qx_frac, qy_frac)) in enumerate(vectors.items()):
            # Convert fractional coordinates to physical units
            qx_phys = qx_frac * 2 * np.pi / a
            qy_phys = qy_frac * 2 * np.pi / b
            
            # Draw arrow from origin to the scattering vector
            plt.annotate('', xy=(qy_phys, qx_phys), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', lw=3, color=colors[i], 
                                      alpha=0.8, shrinkA=0, shrinkB=0),
                        zorder=5)
            
            # Plot point at the end of the arrow
            plt.plot(qy_phys, qx_phys, 'o', color=colors[i], markersize=10, 
                    markeredgecolor='white', markeredgewidth=2, zorder=10)
            
            # Add label with offset
            offset_x = 0.2 * np.pi / b * np.sign(qy_phys) if qy_phys != 0 else 0.2 * np.pi / b
            offset_y = 0.2 * np.pi / a * np.sign(qx_phys) if qx_phys != 0 else 0.2 * np.pi / a
            plt.text(qy_phys + offset_x, qx_phys + offset_y, label, 
                    fontsize=11, fontweight='bold', color=colors[i],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             alpha=0.9, edgecolor=colors[i], linewidth=2),
                    zorder=15)
    
    plt.title(title, fontsize=14)
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    if with_vectors:
        filename = 'outputs/Spectral/JDOS_qspace_with_vectors.png'
    else:
        filename = 'outputs/Spectral/JDOS_qspace.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.show()

# Assuming you have the previous code's setup running:
# 1. Re-run calculate_fig1b() but save the G_grid and k_vals
# 2. Calculate A_Total_k
# 3. Calculate JDOS_q using the convolution method above
# 4. Call plot_jdos(JDOS_q, kx_vals, ky_vals)

# --- 4. Plotting ---
KX, KY, intensity = calculate_fig1b()

plt.figure(figsize=(12, 5)) # Wide aspect ratio for extended ky
plt.pcolormesh(KY, KX, intensity, cmap='inferno', shading='auto', vmin=0.1*np.max(intensity))
plt.colorbar(label='U 5f Spectral Weight', shrink=0.8)

# Formatting axes with π/a and π/b units
plt.title(r'UTe$_2$ Fermi Surface (U 5f weighted) at $k_z=0$', fontsize=14)

# Physical limits
ky_lim_physical = [-3*np.pi/b, 3*np.pi/b]
kx_lim_physical = [-1*np.pi/a, 1*np.pi/a]
plt.xlim(ky_lim_physical)
plt.ylim(kx_lim_physical)

# Custom tick labels in units of π/b and π/a
ky_ticks = np.linspace(ky_lim_physical[0], ky_lim_physical[1], 7)
kx_ticks = np.linspace(kx_lim_physical[0], kx_lim_physical[1], 5)
ky_labels = [f'{val/(np.pi/b):.1f}' for val in ky_ticks]
kx_labels = [f'{val/(np.pi/a):.1f}' for val in kx_ticks]

plt.xticks(ky_ticks, ky_labels)
plt.yticks(kx_ticks, kx_labels)
plt.xlabel(r'$k_y$ (π/b)', fontsize=12)
plt.ylabel(r'$k_x$ (π/a)', fontsize=12)
plt.gca().set_aspect('equal')

# Add BZ boundary lines for reference
plt.axvline(np.pi/b, color='white', linestyle='--', alpha=0.3, linewidth=0.8)
plt.axvline(-np.pi/b, color='white', linestyle='--', alpha=0.3, linewidth=0.8)
plt.axvline(3*np.pi/b, color='white', linestyle='--', alpha=0.3, linewidth=0.8)
plt.axvline(-3*np.pi/b, color='white', linestyle='--', alpha=0.3, linewidth=0.8)
plt.axhline(np.pi/a, color='white', linestyle='--', alpha=0.3, linewidth=0.8)
plt.axhline(-np.pi/a, color='white', linestyle='--', alpha=0.3, linewidth=0.8)

plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save with output directory creation
os.makedirs('outputs/Spectral', exist_ok=True)
filename = f'outputs/Spectral/SpectralNew.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved {filename}")
plt.show()



# Calculate JDOS via proper convolution
# JDOS(q) = ∫ A(k) * A(k+q) dk
# This is the autocorrelation of A(k), which via convolution theorem is:
# Autocorr(A) = IFFT[FFT(A) * conj(FFT(A))] = IFFT[|FFT(A)|²]

print("\nCalculating JDOS via convolution theorem...")

# The spectral weight is A(k) in k-space
# To get JDOS(q), we need the autocorrelation in k-space

# Apply a window to reduce edge artifacts and prevent aliasing
# Use a 2D Hann window to smoothly go to zero at edges
window_x = np.hanning(intensity.shape[0])[:, np.newaxis]
window_y = np.hanning(intensity.shape[1])[np.newaxis, :]
window_2d = window_x * window_y

# Apply window to intensity
intensity_windowed = intensity * window_2d

# Step 1: FFT of the windowed spectral weight
A_fft = fft2(intensity_windowed)

# Step 2: Autocorrelation in Fourier space = |FFT|²
autocorr_fft = A_fft * np.conj(A_fft)

# Step 3: IFFT back to get JDOS(q) - this is the convolution in k-space
JDOS_q_raw = ifft2(autocorr_fft)

# Step 4: Take absolute value (should be real but numerical errors) and shift
JDOS_q = fftshift(np.abs(JDOS_q_raw))

# Normalize to enhance visibility
JDOS_q = JDOS_q / np.max(JDOS_q)

# Get k-space arrays (pass the 1D arrays, not meshgrids)
kx_vals = np.linspace(-np.pi/a, np.pi/a, res)
ky_vals = np.linspace(-3*np.pi/b, 3*np.pi/b, res)

# Plot JDOS without vectors
plot_jdos(JDOS_q, kx_vals, ky_vals, with_vectors=False)

# Plot JDOS with scattering vectors from Nature paper
plot_jdos(JDOS_q, kx_vals, ky_vals, with_vectors=True)