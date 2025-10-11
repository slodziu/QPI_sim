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
    rotation_angle: float = 0.0  # No rotation - perfect square lattice
    disorder_strength: float = 0.0  # No disorder
    zoom_factor: float = 1.0  # No zoom
    
    # FFT artifact suppression parameters
    high_pass_strength: float = 0.0  # No background removal 
    low_freq_suppress_radius: int = 2  # Minimal DC suppression only
    low_freq_transition_radius: int = 3  # Minimal transition zone
    subtract_radial_average: bool = False  # No radial averaging
    
    @property
    def a(self) -> float:
        """Lattice spacing."""
        return self.L / self.gridsize
    
    @property
    def k_F_max(self) -> float:
        """Maximum Fermi wavevector."""
        return np.sqrt(self.E_max)


class SignalProcessing:
    """Signal processing utilities for QPI analysis - NO SMOOTHING VERSION."""
    
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
        
        # No rotation, no disorder - pure parabolic dispersion
        self.KX_rot = KX
        self.KY_rot = KY
        
        # Pure parabolic dispersion: E = k² 
        self.epsilon_k = (KX**2 + KY**2) - self.params.mu
        
        # Store k-space info for later use
        self.kx, self.ky = kx, ky
        self.dk = kx[1] - kx[0]
        
    def calculate_G0(self, energy: float) -> np.ndarray:
        """Calculate bare Green's function in real space with caching."""
        # Check cache first
        energy_key = round(energy, 6)  # Round to avoid floating point precision issues
        if energy_key in self._G0_cache:
            return self._G0_cache[energy_key]
            
        Gk = 1.0 / (energy - self.epsilon_k + 8j*self.params.eta)
        
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
            # Single impurity
            imp_row, imp_col = self.positions[0]
            gridsize = G0.shape[0]
            center = gridsize // 2
            
            # For single impurity: T = V_s / (1 - V_s * G0(R,R))
            # G0(R,R) = G0(0) which is at the center of fftshifted G0
            G0_imp = G0[center, center]
            T = greens_function.params.V_s / (1 - greens_function.params.V_s * G0_imp)
            
            # Vectorized calculation
            rows, cols = np.mgrid[0:gridsize, 0:gridsize]
            
            # Calculate displacements from impurity
            delta_rows = rows - imp_row
            delta_cols = cols - imp_col
            
            # G0 is fftshifted: to get G0(displacement), index with center + displacement
            G0_indices_row = (center + delta_rows) % gridsize
            G0_indices_col = (center + delta_cols) % gridsize
            G0_values = G0[G0_indices_row, G0_indices_col]
            
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
        
        # IMPORTANT: G0 is fftshifted, so we need to account for this
        center = gridsize // 2
        
        for a in range(n_imp):
            for b in range(n_imp):
                row_a, col_a = self.positions[a]
                row_b, col_b = self.positions[b]
                
                # Calculate displacement: R_b - R_a
                delta_row = row_b - row_a
                delta_col = col_b - col_a
                
                # G0 is fftshifted: G0[center, center] = G0(r=0)
                # For displacement (delta_row, delta_col), we need G0[center + delta_row, center + delta_col]
                G0_row = (center + delta_row) % gridsize
                G0_col = (center + delta_col) % gridsize
                    
                G_imp0[a, b] = G0[G0_row, G0_col]
        
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
        
        # IMPORTANT: G0 is fftshifted, so G0[gridsize//2, gridsize//2] corresponds to r=(0,0)
        # We need to undo this shift when indexing
        center = gridsize // 2
        
        for a in range(n_imp):
            row_a, col_a = self.positions[a]
            # Create meshgrid for all positions
            rows, cols = np.meshgrid(range(gridsize), range(gridsize), indexing='ij')
            
            # Calculate relative positions: (row, col) - (row_a, col_a)
            # These are the ACTUAL distances in real space
            delta_rows = rows - row_a
            delta_cols = cols - col_a
            
            # G0[i,j] represents Green's function for displacement (i-center, j-center)
            # So for displacement (delta_row, delta_col), we need G0[delta_row + center, delta_col + center]
            # with periodic wrapping
            G0_indices_row = (delta_rows + center) % gridsize
            G0_indices_col = (delta_cols + center) % gridsize
            
            G0_to_imp[a] = G0[G0_indices_row, G0_indices_col]
            
            # For G0(R_a, r) = G0(r - R_a), use opposite sign
            delta_rows_from = row_a - rows
            delta_cols_from = col_a - cols
            G0_indices_row_from = (delta_rows_from + center) % gridsize
            G0_indices_col_from = (delta_cols_from + center) % gridsize
            G0_from_imp[a] = G0[G0_indices_row_from, G0_indices_col_from]
        
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
        
    def process_LDOS(self, LDOS: np.ndarray) -> np.ndarray:
        """Process LDOS for QPI analysis - completely raw, no smoothing."""
        # For QPI, we want to analyze the LDOS modulations
        # The standard approach is to look at δρ(r) = ρ(r) - ρ₀
        # where ρ₀ is the clean (unperturbed) LDOS
        
        # Since we calculated the LDOS change directly in the impurity system,
        # this is already δρ(r), so return as-is
        return LDOS
    
    def calculate_FFT(self, LDOS_processed: np.ndarray) -> tuple:
        """Calculate FFT for QPI analysis - return both complex and magnitude data."""
        # In QPI analysis, we take FFT of the LDOS modulations δρ(r)
        # Common preprocessing: subtract the spatial average to remove DC component
        LDOS_zero_mean = LDOS_processed - np.mean(LDOS_processed)
        
        # Take 2D FFT and shift zero frequency to center
        LDOS_fft_complex = np.fft.fftshift(np.fft.fft2(LDOS_zero_mean))
        
        # Return both complex FFT and magnitude squared for different uses
        magnitude_squared = np.abs(LDOS_fft_complex)**2
        return LDOS_fft_complex, magnitude_squared
    
    def extract_peak_q(self, fft_data: np.ndarray, k_display_max: float) -> Optional[float]:
        """Extract QPI peak position from FFT data - NO SMOOTHING."""
        center = fft_data.shape[0] // 2
        
        # Create radial distance grid
        y, x = np.ogrid[:fft_data.shape[0], :fft_data.shape[1]]
        r_pixel = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Convert to k-space
        k_pixel = k_display_max / (fft_data.shape[0] / 2)
        r_k = r_pixel * k_pixel
        
        # Create finer radial bins for better resolution
        max_r_k = k_display_max * 0.8
        n_bins = max(int(fft_data.shape[0]/2), 100)
        r_bins = np.linspace(0, max_r_k, n_bins)
        
        if len(r_bins) < 2:
            return None
        
        # Radial averaging - NO SMOOTHING
        radial_profile = self.signal_proc.radial_average(fft_data, r_k, r_bins)
        
        # Find peak (exclude central region) - NO SMOOTHING
        exclude_center = max(1, int(len(radial_profile) * 0.08))
        if exclude_center < len(radial_profile) - 2:
            peak_idx = np.argmax(radial_profile[exclude_center:]) + exclude_center
            if peak_idx < len(r_bins) - 1:
                # Simple peak position
                peak_q = r_bins[peak_idx]
                
                # Only return if the peak is significant
                peak_height = radial_profile[peak_idx]
                background = np.mean(radial_profile[exclude_center:exclude_center+5])
                if peak_height > background * 1.2:
                    return peak_q
        
        return None


class QPISimulation:
    """Main simulation class that orchestrates the QPI calculation."""
    
    def __init__(self, params: SystemParameters, impurity_positions: List[Tuple[int, int]]):
        self.params = params
        self.greens = GreensFunction(params)
        self.impurities = ImpuritySystem(impurity_positions)
        self.analyzer = QPIAnalyzer(params)
        
        # Data storage for 2k_F scattering only (focus on backscattering)
        self.extracted_k = []
        self.extracted_E = []
    
    def energy_to_kF(self, E: float) -> float:
        """Convert energy to Fermi wavevector: k_F = √E."""
        return np.sqrt(E)
    
    def run_single_energy(self, energy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
        """Run simulation for a single energy value."""
        k_F = self.energy_to_kF(energy)
        
        # Calculate Green's function
        G0 = self.greens.calculate_G0(energy)
        
        # Calculate LDOS with proper multiple scattering
        LDOS = self.impurities.calculate_LDOS(G0, self.greens)
        
        # Process LDOS and calculate FFT
        LDOS_processed = self.analyzer.process_LDOS(LDOS)
        fft_complex, fft_display = self.analyzer.calculate_FFT(LDOS_processed)
        
        # Extract QPI peak from the cropped FFT data (same as what's displayed)
        k_display_max = max(2 * self.params.k_F_max * 1.3, 8.0)
        center = self.params.gridsize // 2
        n_pixels = int(k_display_max / self.greens.dk)
        n_pixels = min(n_pixels, self.params.gridsize//2)
        fft_cropped = fft_display[center-n_pixels:center+n_pixels, center-n_pixels:center+n_pixels]
        
        peak_q = self.analyzer.extract_peak_q(fft_cropped, k_display_max)
        
        # Store current energy for dispersion analysis
        self.current_energy = energy
        
        return LDOS, fft_display, fft_complex, peak_q
    
    def update_dispersion_data(self, peak_q: Optional[float]):
        """Update accumulated dispersion data by looking for 2k_F peaks only."""
        if peak_q is not None and peak_q > 0:
            # Use current energy if available, otherwise fall back to E_min
            current_energy = getattr(self, 'current_energy', self.params.E_min)
            
            # Calculate expected k_F for current energy
            k_F_expected = np.sqrt(current_energy)
            
            # Only look for 2k_F peaks (q ≈ 2k_F)
            # Use more generous tolerance and wider range to capture more points
            expected_2kF = 2 * k_F_expected
            tolerance_2kF = 0.6 * expected_2kF  # 60% tolerance for better capture
            
            # Check for 2k_F peak with broader range
            if peak_q >= 0.3 * expected_2kF and peak_q <= 3.0 * expected_2kF:
                # For 2k_F peak, the actual k_F is q/2
                k_F_extracted = peak_q / 2.0
                E_extracted = k_F_extracted**2
                
                # Check if this point is significantly different from existing 2k_F points
                # Use smaller tolerance for uniqueness check to avoid duplicates
                is_new_point = True
                if len(self.extracted_k) > 0:
                    existing_k = np.array(self.extracted_k)
                    existing_positive_k = existing_k[existing_k > 0]
                    if len(existing_positive_k) > 0:
                        min_distance = np.min(np.abs(existing_positive_k - k_F_extracted))
                        if min_distance < 0.03:  # Very small tolerance for uniqueness
                            is_new_point = False
                
                if is_new_point:
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
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(
            1, 4, figsize=(32, 8), dpi=100
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
        
        # Add 2k_F circle only (focus on backscattering)
        theta = np.linspace(0, 2*np.pi, 100)
        k_F_init = self.sim.energy_to_kF(self.params.E_min)
        circle_2kF_x = 2 * k_F_init * np.cos(theta)
        circle_2kF_y = 2 * k_F_init * np.sin(theta)
        self.circle_2kF_line, = self.ax2.plot(
            circle_2kF_x, circle_2kF_y, 'r-', linewidth=3, alpha=0.8, label='q = 2k_F (backscattering)'
        )
        
        self.ax2.legend()
        
        # Inverse FFT plot (new diagnostic window)
        self.im3 = self.ax3.imshow(
            np.zeros((self.params.gridsize, self.params.gridsize)), 
            origin='lower', cmap='RdBu_r', extent=[0, self.params.L, 0, self.params.L]
        )
        self.ax3.set_title('Inverse FFT of Momentum Pattern')
        self.ax3.set_xlabel('x (physical units)')
        self.ax3.set_ylabel('y (physical units)')
        plt.colorbar(self.im3, ax=self.ax3, label='Reconstructed LDOS')
        
        # Dispersion plot (moved to ax4) - focus on 2k_F scattering only
        self.ax4.set_xlabel('k_F (1/length units)')
        self.ax4.set_ylabel('Energy E')
        self.ax4.set_title('Dispersion: Theory vs Extracted (2k_F backscattering only)')
        self.ax4.grid(True, alpha=0.3)
        
        # Theoretical dispersion
        k_theory = np.linspace(-self.params.k_F_max * 1.2, self.params.k_F_max * 1.2, 400)
        E_theory = k_theory**2
        self.ax4.plot(k_theory, E_theory, 'b-', linewidth=2, label='Theory: E = k²')
        
        # Extracted points for 2k_F scattering only
        self.extracted_scatter = self.ax4.scatter(
            [], [], c='red', s=50, alpha=0.7, label='From q=2k_F peaks'
        )
        
        self.ax4.legend()
        self.ax4.set_xlim(-self.params.k_F_max * 1.2, self.params.k_F_max * 1.2)
        self.ax4.set_ylim(0, self.params.E_max * 1.05)
        
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
        LDOS, fft_display, fft_complex, peak_q = self.sim.run_single_energy(energy)
        
        # Update plots
        self._update_real_space_plot(LDOS, energy, k_F)
        self._update_momentum_plot(fft_display)
        self._update_inverse_fft_plot(fft_complex)
        self._update_circle(k_F)
        self._update_dispersion_plot(peak_q)
        
        # Return artists for animation
        artists = [self.im1, self.im2, self.im3, self.circle_2kF_line, self.energy_text, self.extracted_scatter]
        return artists
    
    def _update_real_space_plot(self, LDOS: np.ndarray, energy: float, k_F: float):
        """Update the real space LDOS plot."""
        self.im1.set_data(LDOS)
        # Use symmetric color scale for LDOS
        vmax = np.max(np.abs(LDOS))
        self.im1.set_clim(vmin=-vmax, vmax=vmax)
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
    
    def _update_inverse_fft_plot(self, fft_complex: np.ndarray):
        """Update the inverse FFT plot to show what features contribute to artifacts."""
        # Take the inverse FFT of the complex momentum space pattern
        # This will show us what spatial features the momentum patterns correspond to
        inverse_fft = np.fft.ifft2(np.fft.ifftshift(fft_complex))
        inverse_fft_real = np.real(inverse_fft)
        
        # Update the plot
        self.im3.set_data(inverse_fft_real)
        
        # Use symmetric color scale
        vmax = np.max(np.abs(inverse_fft_real))
        if vmax > 0:
            self.im3.set_clim(vmin=-vmax, vmax=vmax)
        
        # Update title with information
        self.ax3.set_title(f'Inverse FFT of Momentum Pattern\nShows spatial origin of k-space features')
    
    def _update_circle(self, k_F: float):
        """Update the theoretical 2k_F circle."""
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Update 2k_F circle only
        circle_2kF_x = 2 * k_F * np.cos(theta)
        circle_2kF_y = 2 * k_F * np.sin(theta)
        self.circle_2kF_line.set_data(circle_2kF_x, circle_2kF_y)
    
    def _update_dispersion_plot(self, peak_q: Optional[float]):
        """Update the dispersion plot with new extracted 2k_F points."""
        self.sim.update_dispersion_data(peak_q)
        
        # Update 2k_F scattering points only
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
    total_points = len(simulation.extracted_k)
    
    if total_points > 0:
        print(f"\nExtracted {total_points} dispersion points from 2k_F scattering")
        print(f"Energy range: {min(simulation.extracted_E):.2f} to {max(simulation.extracted_E):.2f}")
    else:
        print("\nNo 2k_F scattering points extracted")


if __name__ == "__main__":
    main()