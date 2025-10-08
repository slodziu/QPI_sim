import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter

# Simple model parameters
gridsize = 300
L = 50.0  # System size
V_imp = 1  
eta = 0.5  # Broadening - FIXED physical constant, not grid-dependent!

# Energy sweep parameters for animation
E_min = 3.0   # Minimum energy (increased from 0.2)
E_max = 25.0   # Maximum energy (increased from 2.0)
n_frames = 60  # Number of energy steps

# For parabolic dispersion: E = (ℏk)²/2m, so k_F = sqrt(2mE/ℏ²)
# We'll use simple units where the constant is 1
def energy_to_kF(E):
    """Convert energy to Fermi wavevector for parabolic dispersion"""
    return np.sqrt(E)

# Calculate max k_F for k-space range
k_F_max = energy_to_kF(E_max)
print(f"Energy range: {E_min} to {E_max}")
print(f"k_F range: {energy_to_kF(E_min):.2f} to {k_F_max:.2f}")
print(f"2k_F range: {2*energy_to_kF(E_min):.2f} to {2*k_F_max:.2f}")
# Real space grid - centered at (0,0)
x = np.linspace(-L/2, L/2, gridsize)
y = np.linspace(-L/2, L/2, gridsize)
X, Y = np.meshgrid(x, y)

# Impurity position - at center (0,0)
imp_x, imp_y = 0.0, 0.0

def simple_qpi_pattern(X, Y, imp_x, imp_y, k_F, t):
    """Simple analytical QPI pattern with proper physics"""
    # Distance from impurity
    r = np.sqrt((X - imp_x)**2 + (Y - imp_y)**2)
    
    # Define minimum radius to exclude central region
    r_min = 0.1
    
    # Standard Friedel oscillation formula
    ldos_modulation = V_imp * np.cos(2*k_F*r) / (r**2 + eta**2)
    
    # Exclude central region where r < r_min
    mask = r >= r_min
    ldos_modulation = ldos_modulation * mask
    
    return ldos_modulation

# Setup side-by-side figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

# Real space plot
im1 = ax1.imshow(np.zeros((gridsize, gridsize)), cmap='plasma', origin='lower',
                 extent=[-L/2, L/2, -L/2, L/2], interpolation='bilinear')
plt.colorbar(im1, ax=ax1, label='LDOS Modulation')
ax1.set_xlabel('x (length units)')
ax1.set_ylabel('y (length units)')
ax1.set_title('Real Space: QPI Ripples')
ax1.plot(imp_x, imp_y, 'ko', markersize=3, label='Impurity')

ax1.set_xticks([-25, -12.5, 0, 12.5, 25])
ax1.set_yticks([-25, -12.5, 0, 12.5, 25])
ax1.grid(True, alpha=0.3)
ax1.legend()

# Momentum space plot
# Strategy: Use correct FFT coordinates, but only display a cropped region
dx = L / gridsize  # Real space grid spacing  
k_nyquist = np.pi / dx  # Full FFT k-space range

# Decide on the physical k-range to display (independent of gridsize)
k_display = 2 * k_F_max * 1.5  # Show region around our rings

print(f"FFT Nyquist: ±{k_nyquist:.2f}")
print(f"Display range: ±{k_display:.2f}")
print(f"Max ring at 2k_F = {2*k_F_max:.2f}")

# Check if our display range fits within Nyquist
if k_display > k_nyquist:
    print(f"WARNING: Display range {k_display:.2f} > Nyquist {k_nyquist:.2f}")
    print(f"Increase gridsize to at least {int(np.ceil(L * k_display / np.pi))}")

# For now, use the display range (will crop FFT in animate function)
k_range = k_display

im2 = ax2.imshow(np.zeros((gridsize, gridsize)), cmap='hot', origin='lower',
                 extent=[-k_range, k_range, -k_range, k_range])
plt.colorbar(im2, ax=ax2, label='log|FFT(LDOS)| (enhanced rings)')

ax2.set_xlabel('kx (1/length units)')
ax2.set_ylabel('ky (1/length units)')
ax2.set_title('Momentum Space: QPI Pattern')

# Draw k_F circle with proper k-space coordinates (will be updated in animation)
theta = np.linspace(0, 2*np.pi, 100)
# Initial circle for E_min
k_F_init = energy_to_kF(E_min)
q_kF_init = 2*k_F_init
circle_kF_x = q_kF_init * np.cos(theta)
circle_kF_y = q_kF_init * np.sin(theta)
circle_line, = ax2.plot(circle_kF_x, circle_kF_y, 'r--', linewidth=2, alpha=0.6, label=f'q = 2k_F')

# Set nice tick marks - adjust for larger range
tick_spacing = k_range / 2  # 4 ticks per side
ax2.set_xticks([-k_range, -tick_spacing, 0, tick_spacing, k_range])
ax2.set_yticks([-k_range, -tick_spacing, 0, tick_spacing, k_range])
ax2.grid(True, alpha=0.3)
ax2.legend()

# Add text for energy display (will be updated)
energy_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                       fontsize=14, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def animate(i):
    # Calculate energy for this frame
    E = E_min + (E_max - E_min) * i / (n_frames - 1)
    k_F = energy_to_kF(E)
    
    # No time evolution - static pattern for each energy
    t = 0
    
    # Calculate QPI pattern
    ldos_modulation = simple_qpi_pattern(X, Y, imp_x, imp_y, k_F, t)
    
    # Update real space
    im1.set_data(ldos_modulation)
    vmax = np.percentile(np.abs(ldos_modulation), 95)
    if vmax > 0:
        im1.set_clim(vmin=-vmax, vmax=vmax)
    
    # Update title with energy and k_F
    ax1.set_title(f'Real Space: QPI Ripples (E = {E:.2f}, k_F = {k_F:.2f})')
    energy_text.set_text(f'E = {E:.2f}\n2k_F = {2*k_F:.2f}')
    
    # Update 2k_F circle position
    q_2kF = 2 * k_F
    circle_kF_x_new = q_2kF * np.cos(theta)
    circle_kF_y_new = q_2kF * np.sin(theta)
    circle_line.set_data(circle_kF_x_new, circle_kF_y_new)
    
    # Calculate FFT with proper processing for stable rings
    # Remove DC component properly
    ldos_centered = ldos_modulation - np.mean(ldos_modulation)
    
    # KEY FIX: Remove low-frequency envelope in PHYSICAL units, not pixel units
    # Reduce smoothing for thinner rings
    sigma_physical = L / 200  # Reduced from L/100 for less smoothing
    sigma_pixels = sigma_physical / (L / gridsize)  # Convert to pixel units
    low_freq_envelope = gaussian_filter(ldos_centered, sigma=sigma_pixels)
    
    # Subtract envelope to get pure oscillations
    ldos_oscillations = ldos_centered - low_freq_envelope
    
    # Apply additional windowing to reduce edge effects
    window = np.outer(np.hanning(gridsize), np.hanning(gridsize))
    ldos_windowed = ldos_oscillations * window
    
    fft_data = np.fft.fft2(ldos_windowed)
    fft_shifted = np.fft.fftshift(fft_data)
    
    # CRITICAL: Crop FFT to match display range
    # The FFT spans ±k_nyquist, but we want to display ±k_range
    dx = L / gridsize
    k_nyquist = np.pi / dx
    
    # Calculate how many pixels to crop
    crop_fraction = k_range / k_nyquist  # Fraction of FFT to keep
    pixels_to_keep = int(gridsize * crop_fraction / 2)  # Half-width in pixels
    center = gridsize // 2
    
    # Crop the FFT (extract central region)
    crop_start = center - pixels_to_keep
    crop_end = center + pixels_to_keep
    fft_cropped = fft_shifted[crop_start:crop_end, crop_start:crop_end]
    gridsize_display = fft_cropped.shape[0]  # New size after cropping
    
    # Take magnitude to see the ring structure better
    fft_magnitude = np.abs(fft_cropped)
    
    # Suppress central region in K-SPACE PHYSICAL units, not pixels
    center_display = gridsize_display // 2
    
    # Calculate suppression radius in k-space physical units
    k_suppress = 0.5  # Increased from 0.3 for larger dark center
    k_pixel = k_range / (gridsize_display / 2)  # k per pixel in DISPLAY
    central_radius_pixels = int(k_suppress / k_pixel)  # Convert to pixels
    
    central_radius = max(central_radius_pixels, 5)  # At least 5 pixels
    y_center, x_center = np.ogrid[:gridsize_display, :gridsize_display]
    central_mask = (x_center - center_display)**2 + (y_center - center_display)**2 <= central_radius**2
    
    # Set to even lower percentile to darken center more
    central_value = np.percentile(fft_magnitude, 5)  # Reduced from 20 to 5
    fft_magnitude[central_mask] = central_value
    
    # Apply log scale to enhance ring visibility
    fft_display = -np.log(fft_magnitude + 1e-12)
    
    # Update momentum space
    im2.set_data(fft_display)
    vmax_fft = np.percentile(fft_display, 95)
    vmin_fft = np.percentile(fft_display, 5)
    im2.set_clim(vmin=vmin_fft, vmax=vmax_fft)
    
    return [im1, im2, circle_line, energy_text]

ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=100, blit=True)
ani.save('qpi_energy_sweep.gif', writer='pillow', fps=10)
print(f"Saved animation: {n_frames} frames from E={E_min} to E={E_max}")

plt.show()
