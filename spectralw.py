import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.signal import windows
from UTe2_fixed import *
import os
import pickle
import hashlib
scaleb=1
a = 0.41   
b = scaleb*0.61   
c = 1.39
c_star = scaleb*0.76
theta = np.radians(24)  # Angle for (0-11) projection
def calculate_fig1b(res=1000, eta=0.0001):
    # Grid Setup (High Res for small eta)
    nk_x = res
    nk_y = res # Same resolution for both directions
    scale_val = 1
    # EXTENDED RANGE: Need much larger k-space to capture full JDOS features
    # The paper shows features extending to ±1.5 π/a, so we need k-space that extends at least ±3 π/a
    # to get full autocorrelation range of ±1.5 π/a in q-space
    kx_vals = np.linspace(-scale_val*1.5*np.pi/a, scale_val*1.5*np.pi/a, nk_x)  # Much larger extent
    ky_vals = np.linspace(-scale_val*3*np.pi/b, scale_val*3*np.pi/b, nk_y)  # Much larger extent
    KX, KY = np.meshgrid(kx_vals, ky_vals)
    
    print("Building Hamiltonian...")
    H_grid = H_full(KX, KY, 0.0)
    
    print("Calculating Green's Functions...")
    # Parameters for Spectral Weight
    eta_val = eta  # 1 meV for visibility (0.0001 is physically accurate but very hard to plot)
    target_E = 0.0   # Fermi Level
    
    # Broadcast Identity
    identity = np.eye(4)
    inverse_input = (target_E + 1j * eta_val) * identity - H_grid
    
    # Vectorized Inversion
    G_grid = np.linalg.inv(inverse_input)
    
    # --- PROJECTION ---
    # CRITICAL: Using indices [0, 1] for Uranium based on your H_full structure
    u_indices = [0, 1, 2, 3]  # All orbitals contribute to spectral weight 
    
    spectral_weight = np.zeros_like(KX)
    for idx in u_indices:
        spectral_weight += -1.0 / np.pi * np.imag(G_grid[:, :, idx, idx])
    
    return KX, KY, spectral_weight, kx_vals, ky_vals  # Return the 1D arrays too

def plot_jdos(JDOS_q, kx_vals, ky_vals, with_vectors=False, use_log=True, title_suffix="", project_0m11=False):
    """Plot JDOS in q-space with proper axis scaling
    
    Parameters:
    -----------
    JDOS_q : ndarray
        Joint density of states in q-space
    kx_vals : array
        k_x values (1D array)
    ky_vals : array
        k_y values (1D array)  
    with_vectors : bool
        Whether to overlay scattering vectors
    use_log : bool
        Use log scale (True) or linear scale (False)
    title_suffix : str
        Additional text for plot title
    """
    # Calculate q-space grid - JDOS maps momentum transfer vectors
    nqy, nqx = JDOS_q.shape  # Note: JDOS_q is (ky, kx) from FFT convention
    

    # Use the actual k-space range from the calculation
    kx_max = np.max(np.abs(kx_vals))  
    ky_max = np.max(np.abs(ky_vals))  
    
    # Create q-space arrays with proper range for autocorrelation
    qx_vals = np.linspace(-kx_max, kx_max, nqx)
    qy_vals = np.linspace(-ky_max, ky_max, nqy)
    QX_grid, QY_grid = np.meshgrid(qx_vals, qy_vals)
    # If projection is requested, convert qy to projected axis (0-11 plane)
    if project_0m11:
        # Project qy by sin(theta) only
        QY_grid = QY_grid * np.sin(theta)
    

    # Prepare data for plotting
    if use_log:
        # Avoid log(0) issues - use small positive value
        JDOS_plot = np.log10(np.abs(JDOS_q) + 1e-12)
        colorbar_label = 'log10(JDOS)'
        # Use robust percentile-based scaling for log plots
        vmax = np.max(JDOS_plot)
        vmin = np.min(JDOS_plot)
    else:
        JDOS_plot = np.abs(JDOS_q)
        colorbar_label = 'JDOS (linear)'
        # For linear plots, use max-based scaling
        vmax = np.percentile(JDOS_plot, 99.9)
        vmin = np.percentile(JDOS_plot, 1)
    
    plt.figure(figsize=(10, 8))
    # CORRECT: Nature paper has qy on x-axis, qx on y-axis
    im = plt.pcolormesh(QY_grid, QX_grid, JDOS_plot, cmap='Greys', shading='auto', 
                       vmax=vmax, vmin=vmin)
    plt.colorbar(im, label=colorbar_label, shrink=0.8)

    # Display range and axis labels
    Q_max_display_x = 1.4 * np.pi / a  # qx range: -1.5π/a to +1.5π/a 
    if project_0m11:
        # For projected axis, show in units of pi/c*
        Q_max_display_y = 3.0 * np.pi / c_star
        plt.xlim(-Q_max_display_y, Q_max_display_y)
        plt.ylim(-Q_max_display_x, Q_max_display_x)
        # Axis ticks and labels in pi/c*
        qy_ticks = np.linspace(-Q_max_display_y, Q_max_display_y, 7)
        qx_ticks = np.linspace(-Q_max_display_x, Q_max_display_x, 7)
        qy_labels = [f'{val/(np.pi/c_star):.1f}' for val in qy_ticks]
        qx_labels = [f'{val/(np.pi/a):.1f}' for val in qx_ticks]
        plt.xticks(qy_ticks, qy_labels)
        plt.yticks(qx_ticks, qx_labels)
        plt.xlabel(r'$q_{c^*}$ ($\pi$/c$^*$)', fontsize=12)
        plt.ylabel(r'$q_x$ ($\pi$/a)', fontsize=12)
    else:
        Q_max_display_y = 3.0 * np.pi / b  # qy range: -3π/b to +3π/b
        plt.xlim(-Q_max_display_y, Q_max_display_y)
        plt.ylim(-Q_max_display_x, Q_max_display_x)
        # Axis ticks and labels in pi/b
        qy_ticks = np.linspace(-Q_max_display_y, Q_max_display_y, 7)
        qx_ticks = np.linspace(-Q_max_display_x, Q_max_display_x, 7)
        qy_labels = [f'{val/(np.pi/b):.1f}' for val in qy_ticks]
        qx_labels = [f'{val/(np.pi/a):.1f}' for val in qx_ticks]
        plt.xticks(qy_ticks, qy_labels)
        plt.yticks(qx_ticks, qx_labels)
        plt.xlabel(r'$q_y$ ($\pi$/b)', fontsize=12)
        plt.ylabel(r'$q_x$ ($\pi$/a)', fontsize=12)
    
    # Build title with scale information
    scale_type = 'Log Scale' if use_log else 'Linear Scale'
    title = f'Joint Density of States (JDOS) at E=0 ({scale_type})'
    if with_vectors:
        title += ' with Scattering Vectors'
    if title_suffix:
        title += f' {title_suffix}'
        
        # Define the scattering vectors from the Nature paper - EXACT coordinates from table
        # Coordinates in units of (2π/a, 2π/b) as given in the paper
        vectors = {
            'p1': (0.29, 0),      # From paper table
            'p2': (0.43, 1),      # From paper table  
            'p3': (0.29, 2),      # From paper table
            'p4': (0, 2),         # From paper table
            'p5': (-0.14, 1),     # From paper table
            'p6': (0.57, 0)       # From paper table
        }
        
        # Convert to physical units and plot
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
        
        for i, (label, (qx_frac, qy_frac)) in enumerate(vectors.items()):
            # Convert from (2π/a, 2π/b) units to physical units 
            qx_phys = qx_frac * 2 * np.pi / a  # Factor of 2 because coordinates are in 2π/a units
            qy_phys = qy_frac * 2 * np.pi / b  # Factor of 2 because coordinates are in 2π/b units  
            # If projection, project qy_phys by sin(theta) only
            if project_0m11:
                qy_phys = qy_phys * np.sin(theta)
            # Check if vector endpoint is within display range
            if (abs(qx_phys) <= Q_max_display_x and abs(qy_phys) <= Q_max_display_y):
                # Draw arrow from origin to the scattering vector (qy on x-axis, qx on y-axis)
                plt.annotate('', xy=(qy_phys, qx_phys), xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->', lw=3, color=colors[i], 
                                          alpha=0.8, shrinkA=0, shrinkB=0),
                            zorder=5)
                # Plot point at the end of the arrow
                plt.plot(qy_phys, qx_phys, 'o', color=colors[i], markersize=10, 
                        markeredgecolor='white', markeredgewidth=2, zorder=10)
                # Add label with smart offset to avoid overlaps
                offset_qy = 0.15 * np.pi / b
                if project_0m11:
                    offset_qy = offset_qy * np.sin(theta)
                offset_qx = 0.15 * np.pi / a
                # Adjust offset direction based on vector position
                if qy_phys > 0:
                    offset_qy = abs(offset_qy)
                else:
                    offset_qy = -abs(offset_qy)
                if qx_phys > 0:
                    offset_qx = abs(offset_qx)  
                else:
                    offset_qx = -abs(offset_qx)
                plt.text(qy_phys + offset_qy, qx_phys + offset_qx, label, 
                        fontsize=11, fontweight='bold', color=colors[i],
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 alpha=0.9, edgecolor=colors[i], linewidth=2),
                        zorder=15)
            else:
                print(f"Vector {label} at ({qx_frac:.2f}pi/a, {qy_frac:.2f}pi/b) is outside display range - skipping")
    plt.gca().set_aspect('equal')
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save with descriptive filenames
    scale_suffix = 'log' if use_log else 'linear'
    project_suffix = ' projected' if project_0m11 else scale_suffix
    if with_vectors:
        filename = f'outputs/Spectral/JDOS_qspace_with_vectors_{title_suffix}_{project_suffix}.png'
    else:
        filename = f'outputs/Spectral/JDOS_qspace_{title_suffix}_{project_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.show()

def calculate_jdos_with_windowing(intensity, window_type, alpha=0.25, zero_padding_factor=2):
    """Calculate JDOS with proper windowing to reduce FFT artifacts
    
    Parameters:
    -----------
    intensity : ndarray
        Spectral weight A(k) in k-space
    window_type : str
        Type of window ('tukey', 'hann', 'hamming', 'blackman')
    alpha : float
        Parameter for Tukey window (fraction of cosine taper)
    zero_padding_factor : int
        Factor for zero padding to reduce aliasing
    
    Returns:
    --------
    JDOS_q : ndarray
        Joint density of states in q-space
    """
    print(f"\nCalculating JDOS via convolution theorem with {window_type} window...")
    print(f"Input intensity shape: {intensity.shape}")
    print(f"Input k-space extent: kx=[{np.min(kx_vals):.3f}, {np.max(kx_vals):.3f}] (units: rad/nm)")
    print(f"Input k-space extent: ky=[{np.min(ky_vals):.3f}, {np.max(ky_vals):.3f}] (units: rad/nm)")
    
    # JDOS(q) = integral A(k) * A(k+q) dk
    # This is the autocorrelation of A(k), which via convolution theorem is:
    # Autocorr(A) = IFFT[FFT(A) * conj(FFT(A))] = IFFT[|FFT(A)|^2]
    
    intensity_norm = intensity / np.max(intensity)
    
    # Apply smoothing to reduce high-frequency noise that creates artifacts
    from scipy import ndimage
    intensity_smooth = intensity_norm
    
    # Apply windowing to reduce edge artifacts
    if window_type == 'tukey':
        from scipy.signal import windows
        window_x = windows.tukey(intensity_smooth.shape[0], alpha=alpha)[:, np.newaxis]
        window_y = windows.tukey(intensity_smooth.shape[1], alpha=alpha)[np.newaxis, :]
    elif window_type == 'hann':
        window_x = np.hanning(intensity_smooth.shape[0])[:, np.newaxis]
        window_y = np.hanning(intensity_smooth.shape[1])[np.newaxis, :]
    elif window_type == 'hamming':
        window_x = np.hamming(intensity_smooth.shape[0])[:, np.newaxis]
        window_y = np.hamming(intensity_smooth.shape[1])[np.newaxis, :]
    elif window_type == 'blackman':
        window_x = np.blackman(intensity_smooth.shape[0])[:, np.newaxis]
        window_y = np.blackman(intensity_smooth.shape[1])[np.newaxis, :]
    else:
        # No windowing
        window_x = np.ones((intensity_smooth.shape[0], 1))
        window_y = np.ones((1, intensity_smooth.shape[1]))
    
    window_2d = window_x * window_y
    
    # Apply window and normalize to preserve total intensity
    intensity_windowed = intensity_smooth * window_2d
    

    
    # Zero padding to reduce aliasing artifacts
    if zero_padding_factor > 1:
        pad_x = (intensity_windowed.shape[0] * (zero_padding_factor - 1)) // 2
        pad_y = (intensity_windowed.shape[1] * (zero_padding_factor - 1)) // 2
        intensity_padded = np.pad(intensity_windowed, 
                                 ((pad_x, pad_x), (pad_y, pad_y)), 
                                 mode='constant', constant_values=0)
    else:
        intensity_padded = intensity_windowed
    
    # Step 1: FFT of the windowed and padded spectral weight
    A_fft = fft2(intensity_padded)
    
    # Step 2: Autocorrelation in Fourier space = |FFT|^2
    autocorr_fft = A_fft * np.conj(A_fft)
    
    # Step 3: IFFT back to get JDOS(q) - this is the convolution in k-space
    JDOS_q_raw = ifft2(autocorr_fft)
    
    # Step 4: Take absolute value (should be real but numerical errors) and shift
    JDOS_q = fftshift(np.abs(JDOS_q_raw))
    
    # DON'T crop back! We want the full autocorrelation range to see all JDOS features
    # The extended k-space should give us the full q-space range we need
    print(f"Full JDOS shape (no cropping): {JDOS_q.shape}")
    print(f"K-space was extended to ±{4*np.pi/a:.2f} nm^-1 and ±{4*np.pi/b:.2f} nm^-1")
    print(f"This should show features up to ±{2*np.pi/a:.2f} nm^-1 and ±{2*np.pi/b:.2f} nm^-1 in q-space")
    
    

    
    # Normalize to enhance visibility
    if np.max(JDOS_q) > 0:
        JDOS_q = JDOS_q / np.max(JDOS_q)
    
    print(f"JDOS calculation complete. Range: {np.min(JDOS_q):.6f} to {np.max(JDOS_q):.6f}")
    
    return JDOS_q

def get_spectral(res=1000, eta=0.0001):
    KX, KY, intensity, kx_vals, ky_vals = calculate_fig1b(res, eta)

    plt.figure(figsize=(12, 5)) # Wide aspect ratio for extended ky
    plt.pcolormesh(KY, KX, intensity, cmap='inferno', shading='auto', vmin=0.1*np.max(intensity))
    plt.colorbar(label='U 5f Spectral Weight', shrink=0.8)

    # Formatting axes with pi/a and pi/b units
    plt.title(r'UTe$_2$ Fermi Surface (U 5f weighted) at $k_z=0$', fontsize=14)

    # Physical limits
    ky_lim_physical = [-3*np.pi/b, 3*np.pi/b]
    kx_lim_physical = [-1*np.pi/a, 1*np.pi/a]
    plt.xlim(ky_lim_physical)
    plt.ylim(kx_lim_physical)

    # Custom tick labels in units of pi/b and pi/a
    ky_ticks = np.linspace(ky_lim_physical[0], ky_lim_physical[1], 7)
    kx_ticks = np.linspace(kx_lim_physical[0], kx_lim_physical[1], 5)
    ky_labels = [f'{val/(np.pi/b):.1f}' for val in ky_ticks]
    kx_labels = [f'{val/(np.pi/a):.1f}' for val in kx_ticks]

    plt.xticks(ky_ticks, ky_labels)
    plt.yticks(kx_ticks, kx_labels)
    plt.xlabel(r'$k_y$ ($\pi$/b)', fontsize=12)
    plt.ylabel(r'$k_x$ ($\pi$/a)', fontsize=12)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save with output directory creation
    os.makedirs('outputs/Spectral', exist_ok=True)
    filename = f'outputs/Spectral/SpectralNew_{res}_{eta}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()
    return intensity, kx_vals, ky_vals

import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from scipy.signal import windows
from scipy.interpolate import interp1d

def project_to_0m11(QX, QY, theta):
    """
    Apply the (0-11) projection used in the Nature paper.
    qx stays the same
    qy -> qy * sin(theta)
    """
    QY_proj = QY * np.sin(theta)
    return QX, QY_proj


def compute_JDOS_no_smooth(A):
    # A: 2D spectral weight (kx x ky)
    F = fft2(A)
    J = fftshift(np.abs(ifft2(F * np.conj(F))))
    J = J / np.max(J)
    return J

def measure_fwhm(xvals, profile):
    # profile: 1D array, xvals same units
    # ensure single peak center
    p = profile / np.max(profile)
    half = 0.5
    # interpolate to get accurate crossings
    f = interp1d(xvals, p, kind='cubic', bounds_error=False, fill_value=0.0)
    xs = np.linspace(xvals[0], xvals[-1], 10*len(xvals))
    ys = f(xs)
    inds = np.where(ys >= half)[0]
    if len(inds) == 0:
        return np.nan
    left = xs[inds[0]]
    right = xs[inds[-1]]
    return right - left



res_list = [3001, 4001, 5001]
eta_list = [0.0003, 0.0003,0.0003]
for res, eta in zip(res_list, eta_list):
    intensity, kx_vals, ky_vals = get_spectral(res, eta)
    #JDOS_q = calculate_jdos_with_windowing(intensity,window_type=None, alpha=0.25)

    #print(f"\nGenerating JDOS plots for resolution {res} and eta {eta}...")

    #plot_jdos(JDOS_q, kx_vals, ky_vals, with_vectors=True, use_log=False, 
              #title_suffix=f"(FFT Method, res={res}, eta={eta})", project_0m11=False)






