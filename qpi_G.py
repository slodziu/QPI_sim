import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# System parameters
gridsize = 2000       # Number of grid points
L = 50.0            # Physical system size
Nx, Ny = gridsize, gridsize
a = L / gridsize    # lattice spacing (derived from L and gridsize)
t = 0.3             # hopping
mu = 0.0            # chemical potential (Fermi energy)
eta = 0.01          # broadening for Green's function (smaller = sharper rings, more lattice artifacts)

print(f"Grid: {gridsize}x{gridsize} points")
print(f"System size: {L} x {L}")
print(f"Lattice spacing: a = {a:.4f}")

# Energy sweep parameters for animation
E_min = 30.0   # Minimum energy
E_max = 50.0   # Maximum energy
n_frames = 20  # Number of energy steps

# Calculate Fermi wavevector for parabolic dispersion
def energy_to_kF(E):
    """Convert energy to Fermi wavevector for parabolic dispersion E = k²"""
    return np.sqrt(E)  # For E = k², we have k_F = sqrt(E)

k_F_max = energy_to_kF(E_max)
print(f"Energy range: {E_min} to {E_max}")
print(f"k_F range: {energy_to_kF(E_min):.2f} to {k_F_max:.2f}")
print(f"2k_F range: {2*energy_to_kF(E_min):.2f} to {2*k_F_max:.2f}")

# -----------------------------
# 2. k-space setup
# -----------------------------
kx = 2*np.pi*np.fft.fftfreq(Nx, d=a)
ky = 2*np.pi*np.fft.fftfreq(Ny, d=a)
KX, KY = np.meshgrid(kx, ky)

# Use parabolic dispersion for circular Fermi surface
epsilon_k = (KX**2 + KY**2) - mu  # Parabolic dispersion: E = k²
print("Using parabolic dispersion (circular Fermi surface): E = k²")
print(f"k-space grid: kx, ky range from {kx.min():.2f} to {kx.max():.2f}")
print(f"dk (k-space resolution) = {kx[1] - kx[0]:.4f}")
print(f"For reference: 2π/L = {2*np.pi/L:.4f}, 2π/a = {2*np.pi/a:.2f}")

# -----------------------------
# 5. Single impurity
# -----------------------------
V_s = 1  # Impurity strength

# Single impurity at center
imp_positions = [
    (Nx//2, Ny//2),      # Center
]
print(f"Using {len(imp_positions)} impurity")

# -----------------------------
# Setup figure for animation
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

# Real space LDOS plot
im1 = ax1.imshow(np.zeros((Nx, Ny)), origin='lower', cmap='plasma',
                 extent=[0, L, 0, L])
ax1.set_title("LDOS around impurities (parabolic dispersion)")
plt.colorbar(im1, ax=ax1, label='LDOS')
ax1.set_xlabel('x (physical units)')
ax1.set_ylabel('y (physical units)')

# Momentum space plot
# Limit the k-space display to a reasonable range around 2k_F_max
k_display_max = 2 * k_F_max * 1.2  # Show 20% beyond maximum 2k_F
# Calculate cropped size for initial display
dk = kx[1] - kx[0]
n_pixels = int(k_display_max / dk)
n_pixels = min(n_pixels, gridsize//2)
init_size = 2 * n_pixels

im2 = ax2.imshow(np.zeros((init_size, init_size)), origin='lower', cmap='plasma',
                 extent=[-k_display_max, k_display_max, -k_display_max, k_display_max])
plt.colorbar(im2, ax=ax2, label='log|FFT(LDOS)|')
ax2.set_xlabel('kx (1/a)')
ax2.set_ylabel('ky (1/a)')
ax2.set_title('Momentum Space: QPI Pattern')
ax2.grid(True, alpha=0.3)

# Add circle for 2k_F (will be updated)
theta = np.linspace(0, 2*np.pi, 100)
k_F_init = energy_to_kF(E_min)
circle_x_init = 2*k_F_init * np.cos(theta)
circle_y_init = 2*k_F_init * np.sin(theta)
circle_line, = ax2.plot(circle_x_init, circle_y_init, 'r--', linewidth=2, alpha=0.6, label='q = 2k_F')
ax2.legend()

# Energy text display
energy_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                       fontsize=14, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def animate(frame_idx):
    # Calculate energy for this frame
    E = E_min + (E_max - E_min) * frame_idx / (n_frames - 1)
    k_F = energy_to_kF(E)
    
    print(f"Frame {frame_idx+1}/{n_frames}: E={E:.3f}, k_F={k_F:.3f}")
    
    # -----------------------------
    # 3. Calculate Green's function for this energy
    # -----------------------------
    Gk = 1.0 / (E - epsilon_k + 1j*eta)
    G0 = np.fft.fftshift(np.fft.ifft2(Gk))
    
    # Calculate T-matrices for each impurity
    T_matrices = []
    for imp_i, imp_j in imp_positions:
        G0_imp = G0[imp_i, imp_j]
        T = V_s / (1 - V_s*G0_imp)
        T_matrices.append(T)
    
    # Calculate LDOS with multiple impurities
    LDOS = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            G_ii = G0[i,j]
            for idx, (imp_i, imp_j) in enumerate(imp_positions):
                T = T_matrices[idx]
                G_ii += G0[i,imp_j] * T * G0[imp_i,j]
            LDOS[i,j] = -1/np.pi * np.imag(G_ii)
    
    # Update real space plot
    im1.set_data(LDOS)
    vmax = np.percentile(np.abs(LDOS), 99)
    im1.set_clim(vmin=0, vmax=vmax)
    ax1.set_title(f"LDOS (E = {E:.3f}, k_F = {k_F:.2f})")
    energy_text.set_text(f'E = {E:.2f}\n2k_F = {2*k_F:.2f}')
    
    # Calculate FFT with processing to reduce lattice artifacts
    LDOS_centered = LDOS - np.mean(LDOS)
    
    # Apply Gaussian smoothing to reduce high-frequency grid noise
    # This suppresses the sharp lattice structure while preserving smooth QPI oscillations
    from scipy.ndimage import gaussian_filter
    sigma_smooth = 0.05  # Small smoothing in pixel units
    LDOS_smoothed = gaussian_filter(LDOS_centered, sigma=sigma_smooth)
    
    # Apply window to reduce edge effects
    window = np.outer(np.hanning(Nx), np.hanning(Ny))
    LDOS_windowed = LDOS_smoothed * window
    LDOS_fft = np.fft.fftshift(np.fft.fft2(LDOS_windowed))
    
    fft_magnitude = np.abs(LDOS_fft)
    center = gridsize//2
    
    # Suppress central DC component
    fft_magnitude[center-5:center+6, center-5:center+6] = np.percentile(fft_magnitude, 10)
    
    # Apply power law to enhance circular features and suppress sharp peaks
    fft_display = np.log1p(fft_magnitude**0.8)
    
    # Crop the FFT to the display region to avoid showing empty space
    # Calculate how many pixels correspond to k_display_max
    dk = kx[1] - kx[0]  # k-space resolution
    n_pixels = int(k_display_max / dk)
    n_pixels = min(n_pixels, gridsize//2)  # Don't exceed half the lattice size
    
    # Crop around center
    fft_cropped = fft_display[center-n_pixels:center+n_pixels, center-n_pixels:center+n_pixels]
    
    # Update momentum space plot with cropped data
    im2.set_data(fft_cropped)
    im2.set_extent([-k_display_max, k_display_max, -k_display_max, k_display_max])
    vmax_fft = np.percentile(fft_cropped, 95)
    vmin_fft = np.percentile(fft_cropped, 5)
    im2.set_clim(vmin=vmin_fft, vmax=vmax_fft)
    
    # Update 2k_F circle
    circle_x = k_F * np.cos(theta)
    circle_y = k_F * np.sin(theta)
    circle_line.set_data(circle_x, circle_y)
    
    return [im1, im2, circle_line, energy_text]

print("Starting animation...")
ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200, blit=True)
ani.save('qpi_greens_function_sweep.gif', writer='pillow', fps=5)
print(f"Saved animation: {n_frames} frames from E={E_min} to E={E_max}")
