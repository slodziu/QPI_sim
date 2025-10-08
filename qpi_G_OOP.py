"""
QPI (Quasiparticle Interference) Simulation using Green's Functions

This module simulates QPI patterns around impurities in a 2D tight-binding model
with parabolic dispersion. 

Performance optimizations implemented:
- Vectorized LDOS calculations
- Green's function caching
- Pre-computed windows
- Efficient matrix operations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter, zoom
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Optional: Use numba for JIT compilation if available
try:
    from numba import jit, njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Define dummy decorators if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@dataclass
class SystemParameters:
    """System parameters for the QPI simulation."""
    gridsize: int = 512
    L: float = 50.0  # Physical system size
    t: float = 0.3   # Hopping parameter
    mu: float = 0.0  # Chemical potential
    eta: float = 0.1 # Broadening for Green's function
    V_s: float = 1.0 # Impurity strength
    
    # Energy sweep parameters
    E_min: float = 5.0
    E_max: float = 25.0
    n_frames: int = 20
    
    # Processing parameters
    rotation_angle: float = np.pi/12  # 15 degrees
    disorder_strength: float = 0.05
    physical_broadening_sigma: float = 0.8
    apodization_alpha: float = 0.3
    zoom_factor: float = 1.5
    
    @property
    def a(self) -> float:
        """Lattice spacing."""
        return self.L / self.gridsize
    
    @property
    def k_F_max(self) -> float:
        """Maximum Fermi wavevector."""
        return np.sqrt(self.E_max)


class SignalProcessing:
    """Signal processing utilities for QPI analysis."""
    
    @staticmethod
    def tapered_cosine_window_1d(number_of_points: int, alpha: float = 0.5) -> np.ndarray:
        """
        1D Tapered cosine (Tukey) window with parameter α∈[0,1].
        α=0 -> rectangular; α=1 -> Hann window.
        """
        if alpha <= 0.0:
            return np.ones(number_of_points, dtype=float)
        if alpha >= 1.0:
            n = np.arange(number_of_points, dtype=float)
            return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (number_of_points - 1.0)))
        
        n = np.arange(number_of_points, dtype=float)
        window = np.ones(number_of_points, dtype=float)
        
        left_region = n < alpha * (number_of_points - 1) / 2.0
        right_region = n > (number_of_points - 1) * (1.0 - alpha / 2.0)
        
        window[left_region] = 0.5 * (
            1.0 + np.cos(np.pi * (2.0 * n[left_region] / (alpha * (number_of_points - 1)) - 1.0))
        )
        window[right_region] = 0.5 * (
            1.0 + np.cos(np.pi * (
                2.0 * n[right_region] / (alpha * (number_of_points - 1))
                - 2.0 / alpha + 1.0
            ))
        )
        return window
    
    @staticmethod
    def gaussian_kernel_1d(sigma: float, half_width_sigmas: float = 3.0) -> np.ndarray:
        """Normalized 1D Gaussian kernel for smoothing."""
        if sigma <= 0:
            return np.array([1.0])
        radius = max(1, int(half_width_sigmas * sigma))
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        return kernel / kernel.sum()
    
    @staticmethod
    def gaussian_smooth_1d(array: np.ndarray, sigma: float) -> np.ndarray:
        """Convolve with Gaussian kernel (reflect at boundaries)."""
        kernel = SignalProcessing.gaussian_kernel_1d(sigma)
        pad = (len(kernel) - 1) // 2
        padded = np.pad(array, pad, mode="reflect")
        smoothed = np.convolve(padded, kernel, mode="same")
        return smoothed[pad:-pad] if pad > 0 else smoothed
    
    @staticmethod
    def radial_average(array_2d: np.ndarray, radius_grid: np.ndarray, 
                      bin_edges: np.ndarray) -> np.ndarray:
        """Azimuthal average over annuli."""
        bin_index = np.digitize(radius_grid.ravel(), bin_edges) - 1
        valid = (bin_index >= 0) & (bin_index < len(bin_edges) - 1)
        values = array_2d.ravel()
        
        summed = np.bincount(bin_index[valid], weights=values[valid], 
                           minlength=len(bin_edges) - 1)
        counts = np.bincount(bin_index[valid], minlength=len(bin_edges) - 1)
        
        with np.errstate(invalid="ignore", divide="ignore"):
            averaged = summed / counts
        return averaged


class GreensFunction:
    """Green's function calculations for the tight-binding model."""
    
    def __init__(self, params: SystemParameters):
        self.params = params
        self._setup_k_space()
        # Cache for Green's functions to avoid recalculation
        self._G0_cache = {}
        
    def _setup_k_space(self):
        """Initialize k-space grids and dispersion relation."""
        # Create k-space grids
        kx = 2*np.pi*np.fft.fftfreq(self.params.gridsize, d=self.params.a)
        ky = 2*np.pi*np.fft.fftfreq(self.params.gridsize, d=self.params.a)
        KX, KY = np.meshgrid(kx, ky)
        
        # Rotate k-space grid to break square lattice symmetry
        cos_theta = np.cos(self.params.rotation_angle)
        sin_theta = np.sin(self.params.rotation_angle)
        self.KX_rot = KX * cos_theta - KY * sin_theta
        self.KY_rot = KX * sin_theta + KY * cos_theta
        
        # Add disorder to break perfect lattice coherence
        disorder = self.params.disorder_strength * np.random.normal(
            0, 1, (self.params.gridsize, self.params.gridsize)
        )
        
        # Parabolic dispersion: E = k² + disorder
        self.epsilon_k = (self.KX_rot**2 + self.KY_rot**2) - self.params.mu + disorder
        
        # Store k-space info for later use
        self.kx, self.ky = kx, ky
        self.dk = kx[1] - kx[0]
        
    def calculate_G0(self, energy: float) -> np.ndarray:
        """Calculate bare Green's function in real space with caching."""
        # Check cache first
        energy_key = round(energy, 6)  # Round to avoid floating point precision issues
        if energy_key in self._G0_cache:
            return self._G0_cache[energy_key]
            
        Gk = 1.0 / (energy - self.epsilon_k + 1j*self.params.eta)
        
        # Apply FFT with proper normalization
        # The k-space integral ∫ dk² /(2π)² becomes sum * (dk)² / (2π)²  
        # But FFT already includes 1/N factor, so we need to add back the k-space volume
        dk = self.dk
        normalization = (dk / (2 * np.pi))**2 * self.params.gridsize**2
        
        G0 = normalization * np.fft.fftshift(np.fft.ifft2(Gk))
        
        # Cache result
        self._G0_cache[energy_key] = G0
        
        return G0
    
    def calculate_T_matrix(self, G0: np.ndarray, imp_pos: Tuple[int, int]) -> complex:
        """Calculate T-matrix for a single impurity."""
        imp_i, imp_j = imp_pos
        G0_imp = G0[imp_i, imp_j]
        return self.params.V_s / (1 - self.params.V_s * G0_imp)


class ImpuritySystem:
    """Manages impurities and calculates LDOS modifications."""
    
    def __init__(self, positions: List[Tuple[int, int]]):
        self.positions = positions
        
    def calculate_LDOS(self, G0: np.ndarray, greens_function: 'GreensFunction') -> np.ndarray:
        """Calculate LDOS including all impurity corrections with proper multiple scattering."""
        n_imp = len(self.positions)
        
        if n_imp == 1:
            # Single impurity - use completely non-periodic approach to avoid splitting
            imp_row, imp_col = self.positions[0]
            
            # For single impurity: T = V_s / (1 - V_s * G0(R,R))
            G0_imp = G0[0, 0]  # G0(impurity, impurity) = G0(0,0) due to translational invariance
            T = greens_function.params.V_s / (1 - greens_function.params.V_s * G0_imp)
            
            # Use vectorized calculation with NO PERIODIC BOUNDARIES
            rows, cols = np.mgrid[0:G0.shape[0], 0:G0.shape[1]]
            
            # Direct distances from impurity (absolutely no wrapping)
            delta_rows = np.abs(rows - imp_row)
            delta_cols = np.abs(cols - imp_col)
            
            # Ensure indices are within bounds
            delta_rows = np.minimum(delta_rows, G0.shape[0] - 1)
            delta_cols = np.minimum(delta_cols, G0.shape[1] - 1)
            
            # Get Green's function values using direct indexing
            G0_values = G0[delta_rows, delta_cols]
            
            # Calculate LDOS change: Δρ(r) = -(1/π) Im{G0(r,R_imp) * T * G0(R_imp,r)}
            # For single impurity with symmetric Green's function
            correction = G0_values * T * G0_values
            LDOS_change = -1/np.pi * np.imag(correction)
            
            return LDOS_change
            
        else:
            # Multiple impurities - solve multiple scattering problem
            return self._calculate_multiple_scattering_LDOS(G0, greens_function)
    
    def _calculate_multiple_scattering_LDOS(self, G0: np.ndarray, greens_function: 'GreensFunction') -> np.ndarray:
        """Calculate LDOS with exact multiple scattering T-matrix formalism (optimized)."""
        n_imp = len(self.positions)
        V_s = greens_function.params.V_s
        gridsize = G0.shape[0]
        
        # Start with ZERO - we want only the impurity-induced LDOS change
        # not the total LDOS (bare + change)
        LDOS_change = np.zeros_like(G0, dtype=float)
        
        # Step 1: Build the impurity Green's function matrix G_imp0 (optimized)
        # G_imp0[a,b] = G0(R_a, R_b) - Green's function between impurity sites
        G_imp0 = np.zeros((n_imp, n_imp), dtype=complex)
        
        for a in range(n_imp):
            for b in range(n_imp):
                row_a, col_a = self.positions[a]
                row_b, col_b = self.positions[b]
                # G0[i,j] is Green's function from (0,0) to (i,j) due to translational invariance
                # So G0(R_a, R_b) = G0(R_b - R_a) with periodic boundary conditions
                delta_row = (row_b - row_a) % gridsize
                delta_col = (col_b - col_a) % gridsize
                G_imp0[a, b] = G0[delta_row, delta_col]
        
        # Step 2: Build the impurity potential matrix V
        # V[a,b] = V_a * δ_ab (diagonal matrix for uncoupled impurities)
        V = np.eye(n_imp, dtype=complex) * V_s
        
        # Step 3: Calculate the full T-matrix: T = [I - V * G_imp0]^(-1) * V
        try:
            I = np.eye(n_imp, dtype=complex)
            VG = V @ G_imp0  # Matrix multiplication V * G_imp0
            matrix_to_invert = I - VG
            
            # Check condition number to ensure matrix is well-conditioned
            cond_num = np.linalg.cond(matrix_to_invert)
            
            if cond_num > 1e12:
                raise np.linalg.LinAlgError(f"Matrix is ill-conditioned (cond={cond_num:.2e})")
            
            T_matrix = np.linalg.solve(matrix_to_invert, V)
            
        except np.linalg.LinAlgError as e:
            return self._calculate_single_impurity_sum(G0, greens_function)
        
        # Step 4: OPTIMIZED LDOS calculation using vectorized operations
        # Pre-compute all G0 matrices for each impurity to all grid points
        G0_to_imp = np.zeros((n_imp, gridsize, gridsize), dtype=complex)
        G0_from_imp = np.zeros((n_imp, gridsize, gridsize), dtype=complex)
        
        for a in range(n_imp):
            row_a, col_a = self.positions[a]
            # Create meshgrid for all positions
            rows, cols = np.meshgrid(range(gridsize), range(gridsize), indexing='ij')
            
            # Vectorized calculation of G0(r, R_a) for all r
            delta_rows = (row_a - rows) % gridsize
            delta_cols = (col_a - cols) % gridsize
            G0_to_imp[a] = G0[delta_rows, delta_cols]
            
            # Vectorized calculation of G0(R_a, r) for all r
            delta_rows_from = (rows - row_a) % gridsize
            delta_cols_from = (cols - col_a) % gridsize
            G0_from_imp[a] = G0[delta_rows_from, delta_cols_from]
        
        # Vectorized calculation of LDOS change
        # Δρ(r) = -(1/π) Im{∑_ab G0(r,R_a) * T_ab * G0(R_b,r)}
        delta_rho = np.zeros((gridsize, gridsize), dtype=complex)
        
        for a in range(n_imp):
            for b in range(n_imp):
                # Vectorized: G0(r,R_a) * T_ab * G0(R_b,r) for all r
                delta_rho += G0_to_imp[a] * T_matrix[a, b] * G0_from_imp[b]
        
        LDOS_change = -1/np.pi * np.imag(delta_rho)
        
        return LDOS_change
    
    def _calculate_single_impurity_sum(self, G0: np.ndarray, greens_function: 'GreensFunction') -> np.ndarray:
        """Fallback: sum of independent single-impurity contributions."""
        LDOS = -1/np.pi * np.imag(G0)
        
        for pos in self.positions:
            T = greens_function.calculate_T_matrix(G0, pos)
            imp_i, imp_j = pos
            G_correction = G0[:, imp_j:imp_j+1] * T * G0[imp_i:imp_i+1, :]
            LDOS += -1/np.pi * np.imag(G_correction)
            
        return LDOS


class QPIAnalyzer:
    """Analyzes QPI patterns and extracts dispersion information."""
    
    def __init__(self, params: SystemParameters):
        self.params = params
        self.signal_proc = SignalProcessing()
        # Pre-compute apodization window to avoid recalculation
        self._apod_window = self._create_apodization_window()
        
    def process_LDOS(self, LDOS: np.ndarray) -> np.ndarray:
        """Apply signal processing pipeline to LDOS data."""
        # Center the data
        LDOS_centered = LDOS - np.mean(LDOS)
        
        # Apply gentle edge tapering to reduce boundary artifacts without killing the signal
        # Only apply near the very edges
        edge_width = min(self.params.gridsize // 20, 10)  # Much smaller edge suppression
        if edge_width > 0:
            rows, cols = np.mgrid[0:LDOS.shape[0], 0:LDOS.shape[1]]
            
            # Distance from edges
            dist_from_edges = np.minimum(
                np.minimum(rows, LDOS.shape[0] - 1 - rows),
                np.minimum(cols, LDOS.shape[1] - 1 - cols)
            )
            
            # Very gentle tapering only at the very edges
            edge_taper = np.ones_like(dist_from_edges, dtype=float)
            edge_mask = dist_from_edges < edge_width
            edge_taper[edge_mask] = np.sin(np.pi * dist_from_edges[edge_mask] / (2 * edge_width))**2
            
            LDOS_centered *= edge_taper
        
        # Physical broadening
        LDOS_broadened = gaussian_filter(
            LDOS_centered, sigma=self.params.physical_broadening_sigma
        )
        
        # Apply pre-computed apodization
        LDOS_windowed = LDOS_broadened * self._apod_window
        
        # Sub-pixel interpolation
        LDOS_upsampled = zoom(LDOS_windowed, self.params.zoom_factor, order=3)
        
        return LDOS_upsampled
    
    def _create_apodization_window(self) -> np.ndarray:
        """Create 2D apodization window."""
        window_x = self.signal_proc.tapered_cosine_window_1d(
            self.params.gridsize, self.params.apodization_alpha
        )
        window_y = self.signal_proc.tapered_cosine_window_1d(
            self.params.gridsize, self.params.apodization_alpha
        )
        return np.outer(window_y, window_x)
    
    def calculate_FFT(self, LDOS_processed: np.ndarray) -> np.ndarray:
        """Calculate and process FFT of LDOS (optimized)."""
        # FFT - use numpy's efficient FFT
        LDOS_fft = np.fft.fftshift(np.fft.fft2(LDOS_processed))
        
        # Calculate magnitude in-place to save memory
        np.abs(LDOS_fft, out=LDOS_fft.real)  # Reuse real part for magnitude
        fft_magnitude = LDOS_fft.real
        
        # Crop back to original size
        up_center = LDOS_processed.shape[0] // 2
        crop_size = self.params.gridsize
        start_idx = up_center - crop_size // 2
        end_idx = start_idx + crop_size
        fft_magnitude = fft_magnitude[start_idx:end_idx, start_idx:end_idx].copy()
        
        # Suppress DC component
        center = self.params.gridsize // 2
        dc_suppress_val = np.percentile(fft_magnitude, 10)
        fft_magnitude[center-5:center+6, center-5:center+6] = dc_suppress_val
        
        # Apply power law enhancement in-place
        np.power(fft_magnitude, 0.8, out=fft_magnitude)
        np.log1p(fft_magnitude, out=fft_magnitude)
        
        return fft_magnitude
    
    def extract_peak_q(self, fft_data: np.ndarray, k_display_max: float) -> Optional[float]:
        """Extract QPI peak position from FFT data."""
        center = fft_data.shape[0] // 2
        
        # Create radial distance grid
        y, x = np.ogrid[:fft_data.shape[0], :fft_data.shape[1]]
        r_pixel = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Convert to k-space
        k_pixel = k_display_max / (fft_data.shape[0] / 2)
        r_k = r_pixel * k_pixel
        
        # Create finer radial bins for better resolution
        max_r_k = k_display_max * 0.8
        n_bins = max(int(fft_data.shape[0]/2), 100)  # Ensure sufficient resolution
        r_bins = np.linspace(0, max_r_k, n_bins)
        
        if len(r_bins) < 2:
            return None
        
        # Radial averaging and smoothing
        radial_profile = self.signal_proc.radial_average(fft_data, r_k, r_bins)
        radial_smoothed = self.signal_proc.gaussian_smooth_1d(radial_profile, 0.8)  # Less smoothing
        
        # Find peak (exclude central region)
        exclude_center = max(1, int(len(radial_smoothed) * 0.08))  # Smaller exclusion zone
        if exclude_center < len(radial_smoothed) - 2:
            peak_idx = np.argmax(radial_smoothed[exclude_center:]) + exclude_center
            if peak_idx < len(r_bins) - 1:
                # Interpolate for sub-bin precision
                peak_q = (r_bins[peak_idx] + r_bins[peak_idx+1]) / 2
                
                # Only return if the peak is significant
                peak_height = radial_smoothed[peak_idx]
                background = np.mean(radial_smoothed[exclude_center:exclude_center+5])
                if peak_height > background * 1.2:  # 20% above background
                    return peak_q
        
        return None


class QPISimulation:
    """Main simulation class that orchestrates the QPI calculation."""
    
    def __init__(self, params: SystemParameters, impurity_positions: List[Tuple[int, int]]):
        self.params = params
        self.greens = GreensFunction(params)
        self.impurities = ImpuritySystem(impurity_positions)
        self.analyzer = QPIAnalyzer(params)
        
        # Data storage
        self.extracted_k = []
        self.extracted_E = []
    
    def energy_to_kF(self, E: float) -> float:
        """Convert energy to Fermi wavevector: k_F = √E."""
        return np.sqrt(E)
    
    def run_single_energy(self, energy: float) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        """Run simulation for a single energy value."""
        k_F = self.energy_to_kF(energy)
        
        # Calculate Green's function
        G0 = self.greens.calculate_G0(energy)
        
        # Calculate LDOS with proper multiple scattering
        LDOS = self.impurities.calculate_LDOS(G0, self.greens)
        
        # Process LDOS and calculate FFT
        LDOS_processed = self.analyzer.process_LDOS(LDOS)
        fft_display = self.analyzer.calculate_FFT(LDOS_processed)
        
        # Extract QPI peak from the cropped FFT data (same as what's displayed)
        k_display_max = max(2 * self.params.k_F_max * 1.3, 8.0)
        center = self.params.gridsize // 2
        n_pixels = int(k_display_max / self.greens.dk)
        n_pixels = min(n_pixels, self.params.gridsize//2)
        fft_cropped = fft_display[center-n_pixels:center+n_pixels, center-n_pixels:center+n_pixels]
        
        peak_q = self.analyzer.extract_peak_q(fft_cropped, k_display_max)
        
        return LDOS, fft_display, peak_q
    
    def update_dispersion_data(self, peak_q: Optional[float]):
        """Update accumulated dispersion data."""
        if peak_q is not None and peak_q > 0:
            # Direct correspondence: q = k_F (intra-band scattering)
            k_F_extracted = peak_q
            E_extracted = k_F_extracted**2
            
            # Check if this point is significantly different from existing points
            tolerance = 0.1  # Minimum separation in k-space
            is_new_point = True
            
            if len(self.extracted_k) > 0:
                existing_k = np.array(self.extracted_k)
                existing_positive_k = existing_k[existing_k > 0]  # Only check positive k
                if len(existing_positive_k) > 0:
                    min_distance = np.min(np.abs(existing_positive_k - k_F_extracted))
                    if min_distance < tolerance:
                        is_new_point = False
            
            if is_new_point:
                # Store both positive and negative k
                self.extracted_k.extend([k_F_extracted, -k_F_extracted])
                self.extracted_E.extend([E_extracted, E_extracted])


class QPIVisualizer:
    """Handles visualization and animation of QPI results."""
    
    def __init__(self, simulation: QPISimulation):
        self.sim = simulation
        self.params = simulation.params
        self._setup_figure()
        
    def _setup_figure(self):
        """Initialize the figure and subplots."""
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            1, 3, figsize=(24, 8), dpi=100
        )
        
        # Real space LDOS plot
        self.im1 = self.ax1.imshow(
            np.zeros((self.params.gridsize, self.params.gridsize)), 
            origin='lower', cmap='plasma', extent=[0, self.params.L, 0, self.params.L]
        )
        self.ax1.set_title("LDOS around impurities")
        self.ax1.set_xlabel('x (physical units)')
        self.ax1.set_ylabel('y (physical units)')
        plt.colorbar(self.im1, ax=self.ax1, label='LDOS')
        
        # Momentum space plot
        k_display_max = max(2 * self.params.k_F_max * 1.3, 8.0)
        n_pixels = int(k_display_max / self.sim.greens.dk)
        n_pixels = min(n_pixels, self.params.gridsize//2)
        init_size = 2 * n_pixels
        
        self.im2 = self.ax2.imshow(
            np.zeros((init_size, init_size)), origin='lower', cmap='plasma',
            extent=[-k_display_max, k_display_max, -k_display_max, k_display_max]
        )
        self.ax2.set_title('Momentum Space: QPI Pattern')
        self.ax2.set_xlabel('kx (1/a)')
        self.ax2.set_ylabel('ky (1/a)')
        self.ax2.grid(True, alpha=0.3)
        plt.colorbar(self.im2, ax=self.ax2, label='log|FFT(LDOS)|')
        
        # Add theoretical circle
        theta = np.linspace(0, 2*np.pi, 100)
        k_F_init = self.sim.energy_to_kF(self.params.E_min)
        circle_x = k_F_init * np.cos(theta)
        circle_y = k_F_init * np.sin(theta)
        self.circle_line, = self.ax2.plot(
            circle_x, circle_y, 'r--', linewidth=2, alpha=0.6, label='q = k_F'
        )
        self.ax2.legend()
        
        # Dispersion plot
        self.ax3.set_xlabel('k (1/length units)')
        self.ax3.set_ylabel('Energy E')
        self.ax3.set_title('Dispersion: Theory vs Extracted')
        self.ax3.grid(True, alpha=0.3)
        
        # Theoretical dispersion
        k_theory = np.linspace(-self.params.k_F_max * 1.2, self.params.k_F_max * 1.2, 400)
        E_theory = k_theory**2
        self.ax3.plot(k_theory, E_theory, 'b-', linewidth=2, label='Theory: E = k²')
        
        # Extracted points (will be updated)
        self.extracted_scatter = self.ax3.scatter(
            [], [], c='red', s=50, alpha=0.7, label='Extracted from QPI'
        )
        
        self.ax3.legend()
        self.ax3.set_xlim(-self.params.k_F_max * 1.2, self.params.k_F_max * 1.2)
        self.ax3.set_ylim(0, self.params.E_max * 1.05)
        
        # Energy text
        self.energy_text = self.ax1.text(
            0.02, 0.98, '', transform=self.ax1.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    def animate_frame(self, frame_idx: int):
        """Animate a single frame."""
        # Calculate energy for this frame
        energy = self.params.E_min + (self.params.E_max - self.params.E_min) * frame_idx / (self.params.n_frames - 1)
        k_F = self.sim.energy_to_kF(energy)
        
        # Run simulation
        LDOS, fft_display, peak_q = self.sim.run_single_energy(energy)
        
        # Update plots
        self._update_real_space_plot(LDOS, energy, k_F)
        self._update_momentum_plot(fft_display)
        self._update_circle(k_F)
        self._update_dispersion_plot(peak_q)
        
        return [self.im1, self.im2, self.circle_line, self.energy_text, self.extracted_scatter]
    
    def _update_real_space_plot(self, LDOS: np.ndarray, energy: float, k_F: float):
        """Update the real space LDOS plot."""
        self.im1.set_data(LDOS)
        vmax = np.percentile(np.abs(LDOS), 99)
        self.im1.set_clim(vmin=0, vmax=vmax)
        self.ax1.set_title(f"LDOS (E = {energy:.3f}, k_F = {k_F:.2f})")
        self.energy_text.set_text(f'E = {energy:.2f}\nk_F = {k_F:.2f}')
        
        # Clear previous impurity markers
        for artist in self.ax1.lines:
            artist.remove()
        
        # Note: Impurity markers removed per user request
        # Individual Friedel rings are now clearly visible around each impurity position
        
        # Add legend if multiple impurities
        if len(self.sim.impurities.positions) > 1 and len(self.sim.impurities.positions) <= 5:
            self.ax1.legend(loc='upper right')
    
    def _update_momentum_plot(self, fft_display: np.ndarray):
        """Update the momentum space plot."""
        k_display_max = max(2 * self.params.k_F_max * 1.3, 8.0)
        center = self.params.gridsize // 2
        n_pixels = int(k_display_max / self.sim.greens.dk)
        n_pixels = min(n_pixels, self.params.gridsize//2)
        
        fft_cropped = fft_display[center-n_pixels:center+n_pixels, center-n_pixels:center+n_pixels]
        
        self.im2.set_data(fft_cropped)
        self.im2.set_extent([-k_display_max, k_display_max, -k_display_max, k_display_max])
        vmax_fft = np.percentile(fft_cropped, 95)
        vmin_fft = np.percentile(fft_cropped, 5)
        self.im2.set_clim(vmin=vmin_fft, vmax=vmax_fft)
    
    def _update_circle(self, k_F: float):
        """Update the theoretical circle."""
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = k_F * np.cos(theta)
        circle_y = k_F * np.sin(theta)
        self.circle_line.set_data(circle_x, circle_y)
    
    def _update_dispersion_plot(self, peak_q: Optional[float]):
        """Update the dispersion plot with new extracted points."""
        self.sim.update_dispersion_data(peak_q)
        
        if len(self.sim.extracted_k) > 0:
            self.extracted_scatter.set_offsets(
                np.column_stack([self.sim.extracted_k, self.sim.extracted_E])
            )
    
    def create_animation(self, filename: str = 'qpi_animation.gif'):
        """Create and save the animation."""
        ani = animation.FuncAnimation(
            self.fig, self.animate_frame, frames=self.params.n_frames, 
            interval=200, blit=True
        )
        ani.save(filename, writer='pillow', fps=5)
        return ani


def main():
    """Main function to run the QPI simulation."""
    # System parameters
    params = SystemParameters(
        gridsize=512,
        E_min=5.0,
        E_max=25.0,
        n_frames=20
    )
    
    # Single impurity at center
    impurity_positions = [(params.gridsize//2, params.gridsize//2)]
    
    # Create and run simulation
    simulation = QPISimulation(params, impurity_positions)
    visualizer = QPIVisualizer(simulation)
    
    # Create animation in outputs folder
    import os
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    filename = os.path.join(outputs_dir, 'qpi_greens_function_sweep_clean.gif')
    ani = visualizer.create_animation(filename)
    
    # Show final dispersion data
    if len(simulation.extracted_k) > 0:
        print(f"\nExtracted {len(simulation.extracted_k)} dispersion points")
        print(f"Energy range covered: {min(simulation.extracted_E):.2f} to {max(simulation.extracted_E):.2f}")


if __name__ == "__main__":
    main()