"""
QPI (Quasiparticle Interference) Simulation using Green's Functions

This module simulates QPI patterns around impurities in a 2D system with 
parabolic dispersion using the T-matrix formalism for single and multiple 
impurity scattering.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SystemParameters:
    """
    System parameters for the QPI simulation.
    
    Attributes:
        gridsize: Number of grid points in each spatial dimension
        L: Physical system size in real space
        t: Hopping parameter (unused in parabolic dispersion)
        mu: Chemical potential
        eta: Energy broadening parameter for Green's function
        V_s: Impurity potential strength
        E_min: Minimum energy for sweep
        E_max: Maximum energy for sweep
        n_frames: Number of energy points to simulate
        rotation_angle: Lattice rotation (currently unused)
        disorder_strength: Disorder strength (currently unused)
        zoom_factor: Zoom factor for visualization (currently unused)
        high_pass_strength: High-pass filter strength (currently unused)
        low_freq_suppress_radius: Radius for DC suppression in FFT
        low_freq_transition_radius: Transition width for DC suppression
        subtract_radial_average: Whether to subtract radial average (currently unused)
    """
    gridsize: int = 512
    L: float = 50.0
    t: float = 0.3
    mu: float = 0.0
    eta: float = 0.1
    V_s: float = 1.0
    
    E_min: float = 5.0
    E_max: float = 25.0
    n_frames: int = 20
    
    rotation_angle: float = 0.0
    disorder_strength: float = 0.0
    zoom_factor: float = 1.0
    
    high_pass_strength: float = 0.0
    low_freq_suppress_radius: int = 2
    low_freq_transition_radius: int = 3
    subtract_radial_average: bool = False
    
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
    def radial_average(array_2d: np.ndarray, radius_grid: np.ndarray, 
                      bin_edges: np.ndarray) -> np.ndarray:
        """
        Compute azimuthal average of 2D array over radial bins.
        
        Args:
            array_2d: 2D array to average
            radius_grid: 2D array of radial distances
            bin_edges: 1D array of radial bin edges
            
        Returns:
            1D array of averaged values in each radial bin
        """
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
    """
    Green's function calculations for parabolic dispersion.
    
    Computes the bare Green's function G₀(r,r',E) in real space using FFT,
    with the dispersion relation ε(k) = k² - μ.
    """
    
    def __init__(self, params: SystemParameters):
        """
        Initialize Green's function calculator.
        
        Args:
            params: System parameters including gridsize, broadening, etc.
        """
        self.params = params
        self._setup_k_space()
        self._G0_cache = {}
        
    def _setup_k_space(self):
        """Initialize k-space grids and parabolic dispersion relation."""
        kx = 2*np.pi*np.fft.fftfreq(self.params.gridsize, d=self.params.a)
        ky = 2*np.pi*np.fft.fftfreq(self.params.gridsize, d=self.params.a)
        KX, KY = np.meshgrid(kx, ky)
        
        self.KX_rot = KX
        self.KY_rot = KY
        
        self.epsilon_k = (KX**2 + KY**2) - self.params.mu
        
        self.kx, self.ky = kx, ky
        self.dk = kx[1] - kx[0]
        
    def calculate_G0(self, energy: float) -> np.ndarray:
        """
        Calculate bare Green's function in real space with caching.
        
        G₀(k,E) = 1/(E - ε(k) + iη) is computed in k-space, then transformed
        to real space via inverse FFT with proper normalization.
        
        Args:
            energy: Energy at which to compute Green's function
            
        Returns:
            Complex 2D array of G₀(r,r') in real space
        """
        energy_key = round(energy, 6)
        if energy_key in self._G0_cache:
            return self._G0_cache[energy_key]
            
        Gk = 1.0 / (energy - self.epsilon_k + 1j*self.params.eta)
        
        dk = self.dk
        normalization = (dk / (2 * np.pi))**2 * self.params.gridsize**2
        
        G0 = normalization * np.fft.fftshift(np.fft.ifft2(Gk))
        
        self._G0_cache[energy_key] = G0
        
        return G0
    
    def calculate_T_matrix(self, G0: np.ndarray, imp_pos: Tuple[int, int]) -> complex:
        """
        Calculate single-site T-matrix for an impurity.
        
        T = V_s / (1 - V_s G₀(r_imp, r_imp)) is the exact scattering solution
        including all orders of scattering.
        
        Args:
            G0: Bare Green's function in real space
            imp_pos: Impurity position (row, col) in grid
            
        Returns:
            Complex T-matrix value
        """
        imp_i, imp_j = imp_pos
        G0_imp = G0[imp_i, imp_j]
        return self.params.V_s / (1 - self.params.V_s * G0_imp)


class ImpuritySystem:
    """
    Manages impurities and calculates LDOS modifications via T-matrix formalism.
    
    Handles both single and multiple impurity scattering with exact treatment
    of quantum interference effects.
    """
    
    def __init__(self, positions: List[Tuple[int, int]]):
        """
        Initialize impurity system.
        
        Args:
            positions: List of (row, col) positions for each impurity
        """
        self.positions = positions
        
    def calculate_LDOS(self, G0: np.ndarray, greens_function: 'GreensFunction') -> np.ndarray:
        """
        Calculate LDOS modification due to impurities.
        
        For single impurity: δρ(r) = -(1/π) Im[G₀(r,r_imp) T G₀(r_imp,r)]
        For multiple impurities: Uses full T-matrix with interference effects
        
        Args:
            G0: Bare Green's function in real space
            greens_function: GreensFunction instance for parameters
            
        Returns:
            2D array of LDOS change δρ(r,E)
        """
        n_imp = len(self.positions)
        
        if n_imp == 1:
            imp_row, imp_col = self.positions[0]
            gridsize = G0.shape[0]
            center = gridsize // 2
            
            G0_imp = G0[center, center]
            T = greens_function.params.V_s / (1 - greens_function.params.V_s * G0_imp)
            
            rows, cols = np.mgrid[0:gridsize, 0:gridsize]
            
            delta_rows = rows - imp_row
            delta_cols = cols - imp_col
            
            G0_indices_row = (center + delta_rows) % gridsize
            G0_indices_col = (center + delta_cols) % gridsize
            G0_values = G0[G0_indices_row, G0_indices_col]
            
            correction = G0_values * T * G0_values
            LDOS_change = -1/np.pi * np.imag(correction)
            
            return LDOS_change
            
        else:
            return self._calculate_multiple_scattering_LDOS(G0, greens_function)
    
    def _calculate_multiple_scattering_LDOS(self, G0: np.ndarray, greens_function: 'GreensFunction') -> np.ndarray:
        """
        Calculate LDOS with exact multiple scattering T-matrix formalism.
        
        Solves T = [V⁻¹ - G₀]⁻¹ for the full T-matrix including all interference
        paths between impurities, then computes δρ = -(1/π) Im[Σ_ab G₀_to_imp T_ab G₀_from_imp].
        Uses vectorized einsum for efficiency.
        
        Args:
            G0: Bare Green's function in real space
            greens_function: GreensFunction instance for parameters
            
        Returns:
            2D array of LDOS change δρ(r,E)
        """
        n_imp = len(self.positions)
        V_s = greens_function.params.V_s
        gridsize = G0.shape[0]
        
        LDOS_change = np.zeros_like(G0, dtype=float)
        
        G_imp0 = np.zeros((n_imp, n_imp), dtype=complex)
        
        center = gridsize // 2
        
        for a in range(n_imp):
            for b in range(n_imp):
                row_a, col_a = self.positions[a]
                row_b, col_b = self.positions[b]
                
                delta_row = row_b - row_a
                delta_col = col_b - col_a
                
                G0_row = (center + delta_row) % gridsize
                G0_col = (center + delta_col) % gridsize
                    
                G_imp0[a, b] = G0[G0_row, G0_col]
        
        V = np.eye(n_imp, dtype=complex) * V_s
        
        try:
            I = np.eye(n_imp, dtype=complex)
            VG = V @ G_imp0
            matrix_to_invert = I - VG
            
            cond_num = np.linalg.cond(matrix_to_invert)
            
            max_cond = 1e6
            
            if cond_num > max_cond:
                print(f"Warning: Matrix ill-conditioned (cond={cond_num:.2e}), using independent impurities")
                raise np.linalg.LinAlgError(f"Matrix is ill-conditioned (cond={cond_num:.2e})")
            
            T_matrix = np.linalg.solve(matrix_to_invert, V)
            
        except np.linalg.LinAlgError as e:
            return self._calculate_single_impurity_sum(G0, greens_function)
        
        center = gridsize // 2
        
        rows, cols = np.meshgrid(range(gridsize), range(gridsize), indexing='ij')
        
        imp_rows = np.array([pos[0] for pos in self.positions])
        imp_cols = np.array([pos[1] for pos in self.positions])
        
        delta_rows = rows[np.newaxis, :, :] - imp_rows[:, np.newaxis, np.newaxis]
        delta_cols = cols[np.newaxis, :, :] - imp_cols[:, np.newaxis, np.newaxis]
        
        G0_indices_row = (delta_rows + center) % gridsize
        G0_indices_col = (delta_cols + center) % gridsize
        
        G0_to_imp = G0[G0_indices_row, G0_indices_col]
        
        delta_rows_from = imp_rows[:, np.newaxis, np.newaxis] - rows[np.newaxis, :, :]
        delta_cols_from = imp_cols[:, np.newaxis, np.newaxis] - cols[np.newaxis, :, :]
        
        G0_indices_row_from = (delta_rows_from + center) % gridsize
        G0_indices_col_from = (delta_cols_from + center) % gridsize
        
        G0_from_imp = G0[G0_indices_row_from, G0_indices_col_from]
        
        delta_rho = np.einsum('axy,ab,bxy->xy', G0_to_imp, T_matrix, G0_from_imp, optimize=True)
        
        LDOS_change = -1/np.pi * np.imag(delta_rho)
        
        return LDOS_change
    
    def _calculate_single_impurity_sum(self, G0: np.ndarray, greens_function: 'GreensFunction') -> np.ndarray:
        """
        Fallback method: sum of independent single-impurity contributions.
        
        Used when T-matrix inversion fails due to ill-conditioning.
        
        Args:
            G0: Bare Green's function in real space
            greens_function: GreensFunction instance for parameters
            
        Returns:
            2D array of LDOS change
        """
        LDOS = -1/np.pi * np.imag(G0)
        
        for pos in self.positions:
            T = greens_function.calculate_T_matrix(G0, pos)
            imp_i, imp_j = pos
            G_correction = G0[:, imp_j:imp_j+1] * T * G0[imp_i:imp_i+1, :]
            LDOS += -1/np.pi * np.imag(G_correction)
            
        return LDOS


class QPIAnalyzer:
    """
    Analyzes QPI patterns and extracts dispersion information.
    
    Computes FFT of LDOS, extracts radial profiles, and identifies
    peak positions corresponding to 2k_F scattering vectors.
    """
    
    def __init__(self, params: SystemParameters):
        """
        Initialize QPI analyzer.
        
        Args:
            params: System parameters
        """
        self.params = params
        self.signal_proc = SignalProcessing()
        self.impurity_positions = None  # Will be set by QPISimulation
        
    def process_LDOS(self, LDOS: np.ndarray) -> np.ndarray:
        """
        Process LDOS for QPI analysis.
        
        Args:
            LDOS: Raw LDOS array
            
        Returns:
            Processed LDOS (currently returns unmodified input)
        """
        return LDOS
    
    def calculate_FFT(self, LDOS_processed: np.ndarray) -> tuple:
        """
        Calculate 2D FFT of LDOS for QPI analysis.
        
        Removes DC component and suppresses low frequencies to highlight
        QPI ring structure.
        
        Args:
            LDOS_processed: Processed LDOS array
            
        Returns:
            Tuple of (complex FFT, magnitude squared)
        """
        LDOS_zero_mean = LDOS_processed - np.mean(LDOS_processed)
        
        LDOS_fft_complex = np.fft.fftshift(np.fft.fft2(LDOS_zero_mean))
        center = LDOS_fft_complex.shape[0] // 2
        
        radius = 10
        Y, X = np.ogrid[:LDOS_fft_complex.shape[0], :LDOS_fft_complex.shape[1]]
        mask = (X - center)**2 + (Y - center)**2 < radius**2
        LDOS_fft_complex[mask] = 0
        
        magnitude_squared = np.abs(LDOS_fft_complex)**2
        return LDOS_fft_complex, magnitude_squared
    
    def calculate_multi_atom_FFT(self, LDOS_processed: np.ndarray, impurity_positions: List[Tuple[int, int]]) -> np.ndarray:
        """
        Calculate multi-atom phase-preserving FFT using the shift theorem.
        
        Implements δN_MA(q) = δN(q) * Σ_Ri e^(iq·Ri) as described in Sharma et al.
        This preserves phase information by treating each impurity's contribution
        as if it were centered at the origin.
        
        Args:
            LDOS_processed: Processed LDOS array
            impurity_positions: List of (row, col) impurity positions
            
        Returns:
            Complex 2D array of phase-preserving multi-atom FFT
        """
        # Calculate standard FFT
        LDOS_zero_mean = LDOS_processed - np.mean(LDOS_processed)
        delta_N_q = np.fft.fftshift(np.fft.fft2(LDOS_zero_mean))
        
        # Build k-space grid
        gridsize = LDOS_processed.shape[0]
        dk = 2 * np.pi / self.params.L
        kx = dk * np.fft.fftfreq(gridsize, d=1.0/gridsize)
        ky = dk * np.fft.fftfreq(gridsize, d=1.0/gridsize)
        KX, KY = np.meshgrid(np.fft.fftshift(kx), np.fft.fftshift(ky))
        
        # Calculate phase factor sum: Σ_Ri e^(iq·Ri)
        phase_sum = np.zeros_like(delta_N_q, dtype=complex)
        
        for (row, col) in impurity_positions:
            # Convert grid indices to physical positions
            x_i = col * self.params.a
            y_i = row * self.params.a
            
            # Calculate e^(iq·Ri)
            phase_factor = np.exp(1j * (KX * x_i + KY * y_i))
            phase_sum += phase_factor
        
        # Apply multi-atom formula: δN_MA(q) = δN(q) * Σ e^(iq·Ri)
        delta_N_MA = delta_N_q * phase_sum
        
        return delta_N_MA
    
    def extract_peak_q(self, fft_data: np.ndarray, k_display_max: float) -> Optional[float]:
        """
        Extract QPI peak position from FFT data via radial averaging.
        
        Args:
            fft_data: 2D FFT magnitude data
            k_display_max: Maximum k-value in the data
            
        Returns:
            Peak q-value if found, None otherwise
        """
        center = fft_data.shape[0] // 2
        
        y, x = np.ogrid[:fft_data.shape[0], :fft_data.shape[1]]
        r_pixel = np.sqrt((x - center)**2 + (y - center)**2)
        
        k_pixel = k_display_max / (fft_data.shape[0] / 2)
        r_k = r_pixel * k_pixel
        
        max_r_k = k_display_max * 0.8
        n_bins = max(int(fft_data.shape[0]/2), 100)
        r_bins = np.linspace(0, max_r_k, n_bins)
        
        if len(r_bins) < 2:
            return None
        
        radial_profile = self.signal_proc.radial_average(fft_data, r_k, r_bins)
        
        exclude_center = max(1, int(len(radial_profile) * 0.08))
        if exclude_center < len(radial_profile) - 2:
            peak_idx = np.argmax(radial_profile[exclude_center:]) + exclude_center
            if peak_idx < len(r_bins) - 1:
                peak_q = r_bins[peak_idx]
                
                peak_height = radial_profile[peak_idx]
                background = np.mean(radial_profile[exclude_center:exclude_center+5])
                if peak_height > background * 1.2:
                    return peak_q
        
        return None


class QPISimulation:
    """
    Main simulation class that orchestrates the QPI calculation.
    
    Coordinates Green's function calculation, LDOS computation, FFT analysis,
    and dispersion extraction across multiple energies.
    """
    
    def __init__(self, params: SystemParameters, impurity_positions: List[Tuple[int, int]]):
        """
        Initialize QPI simulation.
        
        Args:
            params: System parameters
            impurity_positions: List of (row, col) positions for impurities
        """
        self.params = params
        self.greens = GreensFunction(params)
        self.impurities = ImpuritySystem(impurity_positions)
        self.analyzer = QPIAnalyzer(params)
        self.analyzer.impurity_positions = impurity_positions  # Pass positions to analyzer
        
        self.extracted_k = []
        self.extracted_E = []
    
    def energy_to_kF(self, E: float) -> float:
        """
        Convert energy to Fermi wavevector for parabolic dispersion.
        
        Args:
            E: Energy value
            
        Returns:
            Fermi wavevector k_F = √E
        """
        return np.sqrt(E)
    
    def run_single_energy(self, energy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
        """
        Run simulation for a single energy value.
        
        Args:
            energy: Energy at which to compute QPI pattern
            
        Returns:
            Tuple of (LDOS, FFT magnitude, FFT complex, peak_q)
        """
        k_F = self.energy_to_kF(energy)
        
        G0 = self.greens.calculate_G0(energy)
        
        LDOS = self.impurities.calculate_LDOS(G0, self.greens)
        
        LDOS_processed = self.analyzer.process_LDOS(LDOS)
        fft_complex, fft_display = self.analyzer.calculate_FFT(LDOS_processed)
        
        k_display_max = max(2 * self.params.k_F_max * 1.3, 8.0)
        center = self.params.gridsize // 2
        n_pixels = int(k_display_max / self.greens.dk)
        n_pixels = min(n_pixels, self.params.gridsize//2)
        fft_cropped = fft_display[center-n_pixels:center+n_pixels, center-n_pixels:center+n_pixels]
        
        peak_q = self.analyzer.extract_peak_q(fft_cropped, k_display_max)
        
        self.current_energy = energy
        
        return LDOS, fft_display, fft_complex, peak_q
    
    def update_dispersion_data(self, peak_q: Optional[float]):
        """
        Update accumulated dispersion data from detected QPI peak.
        
        Extracts k_F from the peak position assuming 2k_F scattering,
        and stores the k_F, E pair for dispersion analysis.
        
        Args:
            peak_q: Detected peak position in momentum space
        """
        if peak_q is not None and peak_q > 0:
            current_energy = getattr(self, 'current_energy', self.params.E_min)
            
            k_F_expected = np.sqrt(current_energy)
            
            expected_2kF = 2 * k_F_expected
            tolerance_2kF = 0.6 * expected_2kF
            
            if peak_q >= 0.3 * expected_2kF and peak_q <= 3.0 * expected_2kF:
                k_F_extracted = peak_q / 2.0
                E_extracted = k_F_extracted**2
                
                is_new_point = True
                if len(self.extracted_k) > 0:
                    existing_k = np.array(self.extracted_k)
                    existing_positive_k = existing_k[existing_k > 0]
                    if len(existing_positive_k) > 0:
                        min_distance = np.min(np.abs(existing_positive_k - k_F_extracted))
                        if min_distance < 0.03:
                            is_new_point = False
                
                if is_new_point:
                    self.extracted_k.extend([k_F_extracted, -k_F_extracted])
                    self.extracted_E.extend([E_extracted, E_extracted])


class QPIvisualiser:
    """
    Handles visualization and animation of QPI results.
    
    Creates multi-panel plots showing LDOS in real space, QPI pattern
    in momentum space, and extracted dispersion relation.
    """
    
    def __init__(self, simulation: QPISimulation):
        """
        Initialize visualiser.
        
        Args:
            simulation: QPISimulation instance to visualize
        """
        self.sim = simulation
        self.params = simulation.params
        self._setup_figure()
        
    def _setup_figure(self):
        """Initialize the figure and subplots with appropriate scaling."""
        self.fig, (self.ax1, self.ax2, self.ax4) = plt.subplots(
            1, 3, figsize=(24, 8), dpi=100
        )
        
        self.im1 = self.ax1.imshow(
            np.zeros((self.params.gridsize, self.params.gridsize)), 
            origin='lower', cmap='seismic', extent=[0, self.params.L, 0, self.params.L]
        )
        self.ax1.set_title("LDOS around impurities")
        self.ax1.set_xlabel('x (physical units)')
        self.ax1.set_ylabel('y (physical units)')
        plt.colorbar(self.im1, ax=self.ax1, label='LDOS')
        
        # Momentum space plot - use actual k-space range based on FFT grid
        # The FFT k-space extent is determined by dk = 2π/L and gridsize
        dk = 2 * np.pi / self.params.L
        k_actual_max = dk * self.params.gridsize / 2
        
        self.im2 = self.ax2.imshow(
            np.zeros((self.params.gridsize, self.params.gridsize)), origin='lower', cmap='plasma',
            extent=[-k_actual_max, k_actual_max, -k_actual_max, k_actual_max]
        )
        self.ax2.set_title('Momentum Space: QPI Pattern')
        self.ax2.set_xlabel('kx (1/a)')
        self.ax2.set_ylabel('ky (1/a)')
        self.ax2.grid(False)
        plt.colorbar(self.im2, ax=self.ax2, label='log|FFT(LDOS)|')
        
        self.ax4.set_xlabel('k_F (1/length units)')
        self.ax4.set_ylabel('Energy E')
        self.ax4.set_title('Dispersion: Theory vs Extracted')
        self.ax4.grid(True, alpha=0.3)
        
        k_theory = np.linspace(-self.params.k_F_max * 1.2, self.params.k_F_max * 1.2, 400)
        E_theory = k_theory**2
        self.ax4.plot(k_theory, E_theory, 'b-', linewidth=2, label='Theory: E = k²')
        
        self.extracted_scatter = self.ax4.scatter(
            [], [], c='red', s=50, alpha=0.7, label='From q=2k_F peaks'
        )
        
        self.ax4.legend()
        self.ax4.set_xlim(-self.params.k_F_max * 1.2, self.params.k_F_max * 1.2)
        self.ax4.set_ylim(0, self.params.E_max * 1.05)
        
        self.energy_text = self.ax1.text(
            0.02, 0.98, '', transform=self.ax1.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    def animate_frame(self, frame_idx: int):
        """
        Animate a single frame.
        
        Args:
            frame_idx: Frame index in the animation sequence
            
        Returns:
            List of artists that were modified
        """
        energy = self.params.E_min + (self.params.E_max - self.params.E_min) * frame_idx / (self.params.n_frames - 1)
        k_F = self.sim.energy_to_kF(energy)
        
        LDOS, fft_display, fft_complex, peak_q = self.sim.run_single_energy(energy)
        
        self._update_real_space_plot(LDOS, energy, k_F)
        self._update_momentum_plot(fft_display)
        self._update_dispersion_plot(peak_q)
        
        artists = [self.im1, self.im2, self.energy_text, self.extracted_scatter]
        return artists
    
    def _update_real_space_plot(self, LDOS: np.ndarray, energy: float, k_F: float):
        """
        Update the real space LDOS plot.
        
        Args:
            LDOS: 2D LDOS array
            energy: Current energy value
            k_F: Fermi wavevector at current energy
        """
        self.im1.set_data(LDOS)
        vmax = np.max(np.abs(LDOS))
        self.im1.set_clim(vmin=-vmax, vmax=vmax)
        self.ax1.set_title(f"LDOS (E = {energy:.3f}, k_F = {k_F:.2f})")
        self.energy_text.set_text(f'E = {energy:.2f}\nk_F = {k_F:.2f}')
        
        for artist in self.ax1.lines:
            artist.remove()
        
        if len(self.sim.impurities.positions) > 1 and len(self.sim.impurities.positions) <= 5:
            self.ax1.legend(loc='upper right')
    
    def _update_momentum_plot(self, fft_display: np.ndarray):
        """
        Update the momentum space plot with correct k-space scaling.
        
        Args:
            fft_display: 2D FFT magnitude data
        """
        fft_log = np.log10(fft_display + 1)
        
        self.im2.set_data(fft_log)
        
        dk = 2 * np.pi / self.params.L
        k_actual_max = dk * self.params.gridsize / 2
        self.im2.set_extent([-k_actual_max, k_actual_max, -k_actual_max, k_actual_max])
        
        vmin_fft = np.min(fft_log)
        vmax_fft = np.max(fft_log)
        self.im2.set_clim(vmin=vmin_fft, vmax=vmax_fft)
    def _update_dispersion_plot(self, peak_q: Optional[float]):
        """
        Update the dispersion plot with new extracted 2k_F points.
        
        Args:
            peak_q: Detected peak position in momentum space
        """
        self.sim.update_dispersion_data(peak_q)
        
        if len(self.sim.extracted_k) > 0:
            self.extracted_scatter.set_offsets(
                np.column_stack([self.sim.extracted_k, self.sim.extracted_E])
            )
    
    def save_fourier_analysis_figure(self, energy: float, frames_dir: str, frame_idx: int):
        """
        Create and save a 6-panel figure showing Fourier transform components, R_MA, and line cuts.
        
        Top row:
          Panel 1: Real part of Fourier transform
          Panel 2: Imaginary part of Fourier transform
          Panel 3: Line cuts of Re[FFT] at θ=0 and θ=π
        
        Bottom row:
          Panel 4: Real part of R_MA (multi-atom phase-preserving)
          Panel 5: Imaginary part of R_MA (multi-atom phase-preserving)
          Panel 6: Line cuts of Re[R_MA] at θ=0 and θ=π
        
        Args:
            energy: Current energy value
            frames_dir: Directory to save the figure
            frame_idx: Frame index for filename
        """
        import os
        
        # Run simulation at this energy
        LDOS, _, _, _ = self.sim.run_single_energy(energy)
        
        # Calculate standard FFT
        LDOS_processed = self.sim.analyzer.process_LDOS(LDOS)
        fft_complex, _ = self.sim.analyzer.calculate_FFT(LDOS_processed)
        
        # Calculate multi-atom phase-preserving FFT
        R_MA = self.sim.analyzer.calculate_multi_atom_FFT(
            LDOS_processed, 
            self.sim.impurities.positions
        )
        
        # Create 2x3 subplot figure
        fig = plt.figure(figsize=(24, 16), dpi=100)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Get k-space extent
        dk = 2 * np.pi / self.params.L
        k_actual_max = dk * self.params.gridsize / 2
        extent = [-k_actual_max, k_actual_max, -k_actual_max, k_actual_max]
        
        # Extract real and imaginary parts
        real_fft = np.real(fft_complex)
        imag_fft = np.imag(fft_complex)
        real_R_MA = np.real(R_MA)
        imag_R_MA = np.imag(R_MA)
        
        # Azimuthal integration as described in Sharma et al.
        # Integrate over angles around θ=0 and θ=π as a function of |q|
        center = self.params.gridsize // 2
        
        # Build q-space polar coordinates
        y_idx, x_idx = np.ogrid[:self.params.gridsize, :self.params.gridsize]
        qx = (x_idx - center) * dk
        qy = (y_idx - center) * dk
        q_mag = np.sqrt(qx**2 + qy**2)
        q_angle = np.arctan2(qy, qx)  # angle from -π to π
        
        # Define angular integration window (in radians)
        # Integrate over ±15 degrees around the target angle
        angular_width = np.pi / 12  # 15 degrees
        
        # Define radial bins for |q|
        n_q_bins = 200
        q_bins = np.linspace(0, k_actual_max * 0.9, n_q_bins)
        
        # Initialize arrays for azimuthally integrated profiles
        cut_right_fft = np.zeros(n_q_bins - 1)
        cut_left_fft = np.zeros(n_q_bins - 1)
        cut_right_R_MA = np.zeros(n_q_bins - 1)
        cut_left_R_MA = np.zeros(n_q_bins - 1)
        
        # For each radial bin, integrate azimuthally around θ=0 and θ=π
        for i in range(n_q_bins - 1):
            q_min, q_max = q_bins[i], q_bins[i+1]
            
            # Mask for this radial shell
            radial_mask = (q_mag >= q_min) & (q_mag < q_max)
            
            # θ=0 (right direction): integrate from -angular_width to +angular_width
            angle_mask_0 = (np.abs(q_angle) <= angular_width)
            mask_0 = radial_mask & angle_mask_0
            
            if np.sum(mask_0) > 0:
                cut_right_fft[i] = np.mean(real_fft[mask_0])
                cut_right_R_MA[i] = np.mean(real_R_MA[mask_0])
            
            # θ=π (left direction): integrate around π
            # Since arctan2 returns [-π, π], angles near π and near -π represent the same direction
            angle_mask_pi = (np.abs(q_angle - np.pi) <= angular_width) | (np.abs(q_angle - (-np.pi)) <= angular_width)
            mask_pi = radial_mask & angle_mask_pi
            
            if np.sum(mask_pi) > 0:
                cut_left_fft[i] = np.mean(real_fft[mask_pi])
                cut_left_R_MA[i] = np.mean(real_R_MA[mask_pi])
        
        # Create q-vector array for plotting (use bin centers)
        q_plot = (q_bins[:-1] + q_bins[1:]) / 2
        
        # Panel 1: Real part of FFT
        ax1 = fig.add_subplot(gs[0, 0])
        # Ensure symmetric color scaling for regular FFT real part
        vmax_real_fft = np.max(np.abs(real_fft))
        im1 = ax1.imshow(real_fft, origin='lower', cmap='RdBu_r', extent=extent,
                        vmin=-3*np.max(cut_left_fft), vmax=3*np.max(cut_right_fft))
        ax1.set_title(f'Re[FFT(δN(r))] at E={energy:.2f}', fontsize=14)
        ax1.set_xlabel('kx (1/a)', fontsize=12)
        ax1.set_ylabel('ky (1/a)', fontsize=12)
        plt.colorbar(im1, ax=ax1, label='Re[FFT]')
        
        # Panel 2: Imaginary part of FFT
        ax2 = fig.add_subplot(gs[0, 1])
        # Ensure symmetric color scaling for regular FFT imaginary part
        vmax_imag_fft = np.max(np.abs(imag_fft))
        im2 = ax2.imshow(imag_fft, origin='lower', cmap='RdBu_r', extent=extent,
                        vmin=-3*np.max(cut_left_fft), vmax=3*np.max(cut_right_fft))
        ax2.set_title(f'Im[FFT(δN(r))] at E={energy:.2f}', fontsize=14)
        ax2.set_xlabel('kx (1/a)', fontsize=12)
        ax2.set_ylabel('ky (1/a)', fontsize=12)
        plt.colorbar(im2, ax=ax2, label='Im[FFT]')
        
        # Panel 3: Azimuthally integrated profiles for Re[FFT]
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(q_plot, cut_right_fft, 'g-', linewidth=2, label='θ=0 (right)', alpha=0.8)
        ax3.plot(q_plot, cut_left_fft, 'r-', linewidth=2, label='θ=π (left)', alpha=0.8)
        ax3.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        # Add expected 2k_F peak position
        k_F = np.sqrt(energy)
        expected_2kF = 2 * k_F
        if expected_2kF <= np.max(q_plot):
            ax3.axvline(x=expected_2kF, color='black', linestyle='--', linewidth=2, alpha=0.8, label=f'2k_F = {expected_2kF:.2f}')
        
        ax3.set_xlabel('|q| (1/a)', fontsize=12)
        ax3.set_ylabel('Re[FFT(δN)] (azimuthally integrated)', fontsize=12)
        ax3.set_title(f'Azimuthal Integration: Re[FFT] at E={energy:.2f}', fontsize=14)
        ax3.legend(loc='best', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Real part of R_MA
        ax4 = fig.add_subplot(gs[1, 0])
        # Ensure symmetric color scaling for multi-atom real part
        vmax_R_MA = np.max(np.abs(real_R_MA))
        im4 = ax4.imshow(real_R_MA, origin='lower', cmap='RdBu_r', extent=extent,
                        vmin=-3*np.max(np.abs(cut_left_R_MA)), vmax=3*np.max(np.abs(cut_right_R_MA)))
        ax4.set_title(f'Re[δN_MA(q)] at E={energy:.2f}', fontsize=14)
        ax4.set_xlabel('kx (1/a)', fontsize=12)
        ax4.set_ylabel('ky (1/a)', fontsize=12)
        plt.colorbar(im4, ax=ax4, label='Re[R_MA]')
        
        # Panel 5: Imaginary part of R_MA
        ax5 = fig.add_subplot(gs[1, 1])
        # Ensure symmetric color scaling for multi-atom imaginary part
        vmax_imag_R_MA = np.max(np.abs(imag_R_MA))
        im5 = ax5.imshow(imag_R_MA, origin='lower', cmap='RdBu_r', extent=extent,
                        vmin=-0.1*vmax_imag_R_MA, vmax=0.1*vmax_imag_R_MA)
        ax5.set_title(f'Im[δN_MA(q)] at E={energy:.2f}', fontsize=14)
        ax5.set_xlabel('kx (1/a)', fontsize=12)
        ax5.set_ylabel('ky (1/a)', fontsize=12)
        plt.colorbar(im5, ax=ax5, label='Im[R_MA]')
        
        # Panel 6: Azimuthally integrated profiles for Re[R_MA]
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(q_plot, cut_right_R_MA, 'g-', linewidth=2, label='θ=0 (right)', alpha=0.8)
        ax6.plot(q_plot, cut_left_R_MA, 'r-', linewidth=2, label='θ=π (left)', alpha=0.8)
        ax6.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        # Add expected 2k_F peak position  
        if expected_2kF <= np.max(q_plot):
            ax6.axvline(x=expected_2kF, color='black', linestyle='--', linewidth=2, alpha=0.8, label=f'2k_F = {expected_2kF:.2f}')
        
        ax6.set_xlabel('|q| (1/a)', fontsize=12)
        ax6.set_ylabel('Re[δN_MA] (azimuthally integrated)', fontsize=12)
        ax6.set_title(f'Azimuthal Integration: Re[δN_MA] at E={energy:.2f}', fontsize=14)
        ax6.legend(loc='best', fontsize=11)
        ax6.grid(True, alpha=0.3)
        
        # Add overall title with number of impurities
        n_imp = len(self.sim.impurities.positions)
        fig.suptitle(f'Fourier Analysis with Azimuthal Integration - {n_imp} Impurities | E={energy:.2f}', 
                     fontsize=18, fontweight='bold', y=0.995)
        
        # Save figure
        fourier_filename = os.path.join(frames_dir, f'fourier_analysis_E{energy:.2f}.png')
        fig.savefig(fourier_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def create_animation(self, filename: str = 'qpi_animation.mp4', frames_dir: str = None):
        """
        Create and save animation with individual frames.
        
        Args:
            filename: Path for output animation file (MP4 or GIF)
            frames_dir: Directory to save individual frames (if None, frames not saved)
            
        Returns:
            Animation object
        """
        import os
        
        if frames_dir is not None:
            os.makedirs(frames_dir, exist_ok=True)
        
        import matplotlib
        writers = matplotlib.animation.writers.list()
        has_ffmpeg = 'ffmpeg' in writers
        
        if has_ffmpeg:
            if filename.endswith('.gif'):
                filename = filename.replace('.gif', '.mp4')
            elif not filename.endswith('.mp4'):
                filename = filename + '.mp4'
            writer = 'ffmpeg'
            writer_args = {'fps': 5, 'bitrate': 1800}
        else:
            if filename.endswith('.mp4'):
                filename = filename.replace('.mp4', '.gif')
            elif not filename.endswith('.gif'):
                filename = filename + '.gif'
            writer = 'pillow'
            writer_args = {'fps': 5}
            print("Note: ffmpeg not available, saving as GIF instead")
        
        if frames_dir is not None:
            print(f"Saving individual frames to {frames_dir}/")
            for frame_idx in range(self.params.n_frames):
                # Calculate energy for this frame
                energy = self.params.E_min + (self.params.E_max - self.params.E_min) * frame_idx / (self.params.n_frames - 1)
                
                # Save standard QPI frame
                self.animate_frame(frame_idx)
                frame_filename = os.path.join(frames_dir, f'qpi_{frame_idx+1}.png')
                self.fig.savefig(frame_filename, dpi=150, bbox_inches='tight')
                
            # Save only one Fourier analysis figure from middle energy
            mid_frame_idx = self.params.n_frames // 2
            mid_energy = self.params.E_min + (self.params.E_max - self.params.E_min) * mid_frame_idx / (self.params.n_frames - 1)
            self.save_fourier_analysis_figure(mid_energy, frames_dir, mid_frame_idx)
                
            print(f"✓ Saved {self.params.n_frames} QPI frames")
            print(f"✓ Saved 1 Fourier analysis frame at mid-energy E={mid_energy:.2f}")
            
            # Reset extracted points after saving frames so animation builds them up fresh
            self.sim.extracted_k = []
            self.sim.extracted_E = []
        
        ani = animation.FuncAnimation(
            self.fig, self.animate_frame, frames=self.params.n_frames, 
            interval=200, blit=True
        )
        
        ani.save(filename, writer=writer, **writer_args)
        
        return ani
    
    def save_mid_energy_snapshot(self, filename: str = 'qpi_snapshot.png'):
        """
        Save a static image at the middle energy of the range.
        
        Args:
            filename: Path for output snapshot file
        """
        mid_energy = (self.params.E_min + self.params.E_max) / 2
        k_F = self.sim.energy_to_kF(mid_energy)
        
        LDOS, fft_display, fft_complex, peak_q = self.sim.run_single_energy(mid_energy)
        
        self._update_real_space_plot(LDOS, mid_energy, k_F)
        self._update_momentum_plot(fft_display)
        self._update_dispersion_plot(peak_q)
        
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved snapshot at E={mid_energy:.2f} to {filename}")


def main():
    """
    Main function to run a simple QPI simulation.
    
    Sets up a single impurity at the center of the grid and runs
    an energy sweep to produce QPI animation.
    """
    params = SystemParameters(
        gridsize=512,
        E_min=5.0,
        E_max=25.0,
        n_frames=20
    )
    
    impurity_positions = [(params.gridsize//2, params.gridsize//2)]
    
    simulation = QPISimulation(params, impurity_positions)
    visualiser = QPIvisualiser(simulation)
    
    import os
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    anim_filename = os.path.join(outputs_dir, 'qpi_greens_function_sweep_clean.mp4')
    ani = visualiser.create_animation(anim_filename)
    
    snapshot_filename = os.path.join(outputs_dir, 'qpi_greens_function_snapshot.png')
    visualiser.save_mid_energy_snapshot(snapshot_filename)
    
    total_points = len(simulation.extracted_k)
    
    if total_points > 0:
        print(f"\nExtracted {total_points} dispersion points from 2k_F scattering")
        print(f"Energy range: {min(simulation.extracted_E):.2f} to {max(simulation.extracted_E):.2f}")
    else:
        print("\nNo 2k_F scattering points extracted")


if __name__ == "__main__":
    main()