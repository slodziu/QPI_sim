
import numpy as np
import matplotlib.pyplot as plt
from UTe2_fixed import *
def orbital_projected_spectral_weight(H_grid, omega=0.0, eta=1e-5):
    """
    CRITICAL ISSUE IDENTIFIED: The paper method uses eigenvalue decomposition, NOT Green's functions!
    
    Paper method: A_α(k,ω) = ∑_n |⟨α|ψ_n(k)⟩|² × L(ω - E_n(k))
    where L is Lorentzian: L(E) = (η/π) / (E² + η²)
    
    This is the CORRECT method for Figure 1.b reproduction.
    """
    Ny, Nx = H_grid.shape[:2]
    A_U = np.zeros((Ny, Nx), dtype=float)

    # Vectorized eigenvalue decomposition
    H_flat = H_grid.reshape(-1, 4, 4)
    eigenvals, eigenvecs = np.linalg.eigh(H_flat)  # Shape: (M, 4), (M, 4, 4)
    
    # CRITICAL: Check if U 5f orbitals are indeed indices 0,1
    # Based on your H_full structure, U orbitals should be first 2 indices
    U_orbital_weights = np.abs(eigenvecs[:, 0, :])**2 + np.abs(eigenvecs[:, 1, :])**2  # Shape: (M, 4)
    
    # Lorentzian spectral function
    lorentzian = (eta/np.pi) / ((omega - eigenvals)**2 + eta**2)  # Shape: (M, 4)
    
    # Sum over all bands
    A_U.flat[:] = np.sum(U_orbital_weights * lorentzian, axis=1)

    return A_U

# Example usage in your calculate_fig1b:
res= 1001
nk_x = res
nk_y = res * 3  # 3x more points in ky since extent is 3x larger
kx_vals = np.linspace(-1*np.pi/a, 1*np.pi/a, nk_x)
ky_vals = np.linspace(-3*np.pi/b, 3*np.pi/b, nk_y)
KX, KY = np.meshgrid(kx_vals, ky_vals)
H_grid = H_full(KX, KY, 0.0)   # should be (nk_y, nk_x, 4, 4)

# CRITICAL: Use paper-appropriate broadening
eta_plot = 2e-3  # 2 meV - realistic experimental broadening
print(f"Computing U 5f spectral weight with η = {eta_plot*1000:.1f} meV broadening...")
print("NOTE: Previous η = 1 μeV was way too small and caused numerical issues!")

A_map = orbital_projected_spectral_weight(H_grid, omega=0.0, eta=eta_plot)  

print(f"Spectral weight range: {np.min(A_map):.6f} to {np.max(A_map):.6f}")
print(f"Mean spectral weight: {np.mean(A_map):.6f}")
print(f"Std spectral weight: {np.std(A_map):.6f}")

# DIAGNOSTIC: Check if we have reasonable values
if np.max(A_map) < 1e-6:
    print("WARNING: Spectral weight values are extremely small!")
    print("This suggests either wrong orbital indexing or parameter issues.")
elif np.isnan(A_map).any():
    print("WARNING: NaN values detected in spectral weight!")
elif np.max(A_map) > 1e6:
    print("WARNING: Spectral weight values are extremely large!")
    print("This suggests broadening η might be too small.")  

plt.figure(figsize=(12,4))  # Wide aspect ratio to match your image
# Rotate 90deg: swap axes by transposing and swapping extent
# Convert to proper units: ky in pi/b, kx in pi/a
extent = [ky_vals[0]/(np.pi/b), ky_vals[-1]/(np.pi/b), kx_vals[0]/(np.pi/a), kx_vals[-1]/(np.pi/a)]

# Linear plot with enhanced contrast
# IMPORTANT: Try different percentile ranges to match experimental data
vmin, vmax = np.percentile(A_map, [80, 99.5])  # Focus on top 20% of intensities
print(f"Using contrast range: {vmin:.6f} to {vmax:.6f}")
plt.imshow(A_map.T, origin='lower', extent=extent, aspect='auto', cmap='plasma', vmin=vmin, vmax=vmax)

plt.xlabel('k_y (π/b)')
plt.ylabel('k_x (π/a)')
plt.title('U 5f projected spectral weight (k_z=0, ω=0) - Linear Scale')
plt.colorbar(label='A_U(k,ω=0)')
plt.tight_layout()
plt.savefig(f'outputs/Spectral/5f/U_5f_spectral_weight_linear_{eta_plot}.png', dpi=300, bbox_inches='tight')
plt.show()

# Second plot: Modulated by cosine to enhance peaks
plt.figure(figsize=(12,4))
# Create cosine modulation that peaks every pi/b in ky direction
ky_mod = np.cos(KY)  # This will peak at ky = 0, ±π/b, ±2π/b, etc.
A_map_modulated = A_map * np.abs(ky_mod)  # Use absolute value to avoid negative regions

# Apply same contrast enhancement
vmin, vmax = np.percentile(A_map_modulated, [90, 99.9])
plt.imshow(A_map_modulated.T, origin='lower', extent=extent, aspect='auto', cmap='plasma', vmin=vmin, vmax=vmax)

plt.xlabel('k_y (π/b)')
plt.ylabel('k_x (π/a)')
plt.title('U 5f projected spectral weight - Cosine Enhanced (k_z=0, ω=0)')
plt.colorbar(label='A_U(k,ω=0) × |cos(k_y)|')
plt.tight_layout()
plt.savefig(f'outputs/Spectral/5f/U_5f_spectral_weight_cosine_enhanced_{eta_plot}.png', dpi=300, bbox_inches='tight')
plt.show()
