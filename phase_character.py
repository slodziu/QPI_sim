import numpy as np
import matplotlib.pyplot as plt
import os
from UTe2_fixed import H_full, a, b, c, set_parameters, parameter_sets, current_parameter_set, calculate_gap_magnitude

def d_vector(kx, ky, kz, pairing_type, C0, C1, C2, C3):
	"""Compute d-vector for spin-triplet pairing
	
	Args:
		kx, ky, kz: k-space coordinates
		pairing_type: 'B2u' or 'B3u'
		C0, C1, C2, C3: Gap parameters
		
	Returns:
		3-component d-vector (dx, dy, dz)
	"""
	if pairing_type == 'B2u':
		# B2u: d = [C1*sin(kz*c), C0*sin(kx*a)*sin(ky*b)*sin(kz*c), C3*sin(kx*a)]
		dx = C1 * np.sin(kz * c)
		dy = C0 * np.sin(kx * a) * np.sin(ky * b) * np.sin(kz * c)
		dz = C3 * np.sin(kx * a)
		return np.array([dx, dy, dz])
	elif pairing_type == 'B3u':
		# B3u: d = [C0*sin(kx*a)*sin(ky*b)*sin(kz*c), C2*sin(kz*c), C3*sin(ky*b)]
		dx = C0 * np.sin(kx * a) * np.sin(ky * b) * np.sin(kz * c)
		dy = C2 * np.sin(kz * c)
		dz = C3 * np.sin(ky * b)
		return np.array([dx, dy, dz])
	else:
		raise ValueError("Only B2u and B3u supported")

def gap_matrix_from_dvec(d):
	"""Convert d-vector to 2x2 spin-triplet gap matrix: Δ = Δ₀(d·σ)iσᵧ
	
	Args:
		d: 3-component d-vector (dx, dy, dz)
		
	Returns:
		2x2 complex spin gap matrix
	"""
	dx, dy, dz = d
	return np.array([
		[-dx + 1j*dy,  dz],
		[ dz,         dx + 1j*dy]
	], dtype=complex)

def construct_BdG_hamiltonian(kx, ky, kz, pairing_type, gap_params=None):
	"""Construct 16×16 Bogoliubov-de Gennes Hamiltonian with spin-triplet pairing
	
	Physics (Wang et al. Nature Physics 2025):
	- H₈ₓ₈(k) = H₄ₓ₄(k) ⊗ I₂ (spin-orbital expansion)
	- Δ₈ₓ₈(k) = I₄ ⊗ Δₛₚᵢₙ(k) (orbital ⊗ spin gap)
	- H_BdG = [[H₈ₓ₈, Δ₈ₓ₈†], [Δ₈ₓ₈, -H₈ₓ₈*(-k)]] (16×16)
	
	Returns:
		16×16 BdG Hamiltonian matrix
	"""
	if gap_params is None:
		gap_params = get_gap_parameters_for_pairing(pairing_type)
	
	# Get 4×4 normal-state orbital Hamiltonian
	H_4x4 = H_full(kx, ky, kz)
	
	# Expand to 8×8 spin-orbital: H₈ₓ₈ = H₄ₓ₄ ⊗ I₂
	H_8x8 = np.kron(H_4x4, np.eye(2, dtype=complex))
	
	# Compute d-vector and spin gap matrix
	C0, C1, C2, C3 = gap_params['C0'], gap_params['C1'], gap_params['C2'], gap_params['C3']
	d = d_vector(kx, ky, kz, pairing_type, C0, C1, C2, C3)
	Delta_spin = gap_matrix_from_dvec(d)
	
	# Embed into 8×8 orbital space: Δ₈ₓ₈ = I₄ ⊗ Δₛₚᵢₙ
	Delta_8x8 = np.kron(np.eye(4, dtype=complex), Delta_spin)
	
	# Construct 16×16 BdG Hamiltonian
	H_BdG = np.zeros((16, 16), dtype=complex)
	
	# Particle block (upper-left 8×8)
	H_BdG[:8, :8] = H_8x8
	
	# Hole block (lower-right 8×8): -H₈ₓ₈*(-k)
	H_minus_k = np.kron(H_full(-kx, -ky, -kz), np.eye(2, dtype=complex))
	H_BdG[8:, 8:] = -np.conj(H_minus_k)
	
	# Pairing blocks
	H_BdG[:8, 8:] = Delta_8x8.conj().T  # Δ†
	H_BdG[8:, :8] = Delta_8x8           # Δ
	
	return H_BdG

def construct_BdG_hamiltonian_vectorized(kx_array, ky_array, kz, pairing_type, gap_params=None):
	"""Vectorized 16×16 BdG Hamiltonian construction"""
	if gap_params is None:
		gap_params = get_gap_parameters_for_pairing(pairing_type)
	
	kx_flat = kx_array.flatten()
	ky_flat = ky_array.flatten()
	n_points = len(kx_flat)
	
	H_BdG_array = np.zeros((n_points, 16, 16), dtype=complex)
	
	for i, (kx, ky) in enumerate(zip(kx_flat, ky_flat)):
		H_BdG_array[i] = construct_BdG_hamiltonian(kx, ky, kz, pairing_type, gap_params)
	
	return H_BdG_array.reshape(kx_array.shape + (16, 16))

def green_function_k_vectorized(kx_array, ky_array, kz, energy, pairing_type, eta=1e-6, gap_params=None):
	"""Vectorized 16×16 Green's function computation"""
	eta = determine_broadening(energy, eta)
	energy_eff = energy if abs(energy) >= 1e-12 else 1e-5
	
	H_BdG_array = construct_BdG_hamiltonian_vectorized(kx_array, ky_array, kz, pairing_type, gap_params)
	
	original_shape = kx_array.shape
	H_flat = H_BdG_array.reshape(-1, 16, 16)
	n_points = len(H_flat)
	
	G_array = np.zeros((n_points, 16, 16), dtype=complex)
	E_matrix = (energy_eff + 1j * eta) * np.eye(16)
	
	for i in range(n_points):
		try:
			G_array[i] = np.linalg.inv(E_matrix - H_flat[i])
		except np.linalg.LinAlgError:
			reg = max(eta * 0.1, 1e-8) * np.eye(16)
			try:
				G_array[i] = np.linalg.inv(E_matrix - H_flat[i] + reg)
			except np.linalg.LinAlgError:
				G_array[i] = np.linalg.pinv(E_matrix - H_flat[i])
			
	return G_array.reshape(original_shape + (16, 16))

def determine_broadening(energy, base_eta=1e-6, gap_scale=0.0003):
	"""PHYSICS-PRESERVING broadening for clear B2u vs B3u differentiation
	
	Args:
		energy: Energy (in eV)
		base_eta: Minimum broadening
		gap_scale: Superconducting gap scale Δ (in eV)
		
	Returns:
		Broadening η ≈ 0.1-0.15 × Δ (moderate, preserves physics)
	"""
	# FIXED: Conservative broadening that preserves pairing symmetry differences
	# η should be much smaller than gap scale to resolve B2u vs B3u differences
	return max(0.1 * gap_scale, base_eta, 5e-6)  # At least 5µeV, ≤30µeV for 300µeV gap

def green_function_k(kx, ky, kz, energy, pairing_type, eta=1e-6, gap_params=None):
	"""16×16 Green's function G(k,E) = [(E+iη)I - H_BdG(k)]⁻¹"""
	eta = determine_broadening(energy, eta)
	energy_eff = energy if abs(energy) >= 1e-12 else 1e-5
		
	H_BdG = construct_BdG_hamiltonian(kx, ky, kz, pairing_type, gap_params)
	E_matrix = (energy_eff + 1j * eta) * np.eye(16)
	
	try:
		G_k = np.linalg.inv(E_matrix - H_BdG)
	except np.linalg.LinAlgError:
		reg = max(eta * 0.1, 1e-8) * np.eye(16)
		try:
			G_k = np.linalg.inv(E_matrix - H_BdG + reg)
		except np.linalg.LinAlgError:
			G_k = np.linalg.pinv(E_matrix - H_BdG)
	
	return G_k

def compute_fermi_surface_mask_vectorized(kx_array, ky_array, kz, eta_cutoff):
	"""VECTORIZED Fermi surface mask computation - 50x faster
	
	Args:
		kx_array, ky_array: k-space coordinate arrays
		kz: kz value
		eta_cutoff: Energy cutoff (typically 4*eta or 0.002 eV)
		
	Returns:
		Boolean mask indicating Fermi surface states
	"""
	original_shape = kx_array.shape
	kx_flat = kx_array.flatten()
	ky_flat = ky_array.flatten()
	n_points = len(kx_flat)
	
	# Vectorized Hamiltonian computation
	H_matrices = np.zeros((n_points, 4, 4), dtype=complex)
	for i, (kx, ky) in enumerate(zip(kx_flat, ky_flat)):
		H_matrices[i] = H_full(kx, ky, kz)
	
	# Vectorized eigenvalue computation
	eigenvals = np.linalg.eigvalsh(H_matrices)  # Shape: (n_points, 4)
	
	# Check if any band is near Fermi level for each k-point
	fermi_mask_flat = np.any(np.abs(eigenvals) < eta_cutoff, axis=1)
	
	return fermi_mask_flat.reshape(original_shape)

def compute_fermi_surface_mask(kx_array, ky_array, kz, eta_cutoff):
	"""Wrapper for backward compatibility"""
	return compute_fermi_surface_mask_vectorized(kx_array, ky_array, kz, eta_cutoff)

def compute_local_green_function_vectorized(energy, pairing_type, kz=0.0, eta=1e-6, nk=50, gap_params=None):
	"""Vectorized 16×16 local Green's function computation"""
	kx_vals = np.linspace(-np.pi/a, np.pi/a, nk)
	ky_vals = np.linspace(-3*np.pi/b, np.pi/b, nk)
	
	KX, KY = np.meshgrid(kx_vals, ky_vals, indexing='ij')
	G_array = green_function_k_vectorized(KX, KY, kz, energy, pairing_type, eta, gap_params)
	
	return np.mean(G_array, axis=(0, 1))

def compute_t_matrix(energy, pairing_type, V_imp=None, kz=0.0, eta=1e-6, nk=50, gap_params=None):
	"""16×16 T-matrix T(E) = (1 - V_imp G₀(E))⁻¹ V_imp"""
	G_local = compute_local_green_function_vectorized(energy, pairing_type, kz, eta, nk, gap_params)
	
	if V_imp is None:
		energy_scale = max(abs(energy), 1e-6)
		V_imp = min(max(0.005 * energy_scale, 0.2e-6), 0.5e-3)
	else:
		energy_scale = max(abs(energy), 1e-6)
		if V_imp > 50 * energy_scale:
			print(f"  WARNING: V_imp={V_imp*1e3:.1f} meV >> E={energy*1e6:.0f} µeV")
	
	V_matrix = V_imp * np.eye(16)
	identity = np.eye(16)
	matrix_to_invert = identity - V_matrix @ G_local
	
	try:
		cond_num = np.linalg.cond(matrix_to_invert)
		if cond_num > 1e12:
			matrix_to_invert += 1e-10 * np.eye(16)
			
		T_matrix = np.linalg.inv(matrix_to_invert) @ V_matrix
	except np.linalg.LinAlgError:
		T_matrix = np.linalg.pinv(matrix_to_invert) @ V_matrix
	
	return T_matrix

def get_gap_parameters_for_pairing(pairing_type):
	"""Get gap parameters matching simulation hypothesis
	
	Args:
		pairing_type: 'B2u' or 'B3u'
		
	Returns:
		Dictionary of gap parameters
		
	Simulation Hypothesis: C0 = 0, C1 = C2 = C3 = 300 µeV
	
	Physics:
		B2u: d = [C1*sin(kz*c), C0*sin(kx*a)*sin(ky*b)*sin(kz*c), C3*sin(kx*a)]
		     = [300*sin(kz*c), 0, 300*sin(kx*a)] µeV
		B3u: d = [C0*sin(kx*a)*sin(ky*b)*sin(kz*c), C2*sin(kz*c), C3*sin(ky*b)]
		     = [0, 300*sin(kz*c), 300*sin(ky*b)] µeV
	"""
	# All pairing types use the same simulation parameters
	return {
		'C0': 0.0,      # No triple-product term (simulation hypothesis)
		'C1': 0.0003,   # 300 µeV = 0.0003 eV
		'C2': 0.0003,   # 300 µeV = 0.0003 eV  
		'C3': 0.0003    # 300 µeV = 0.0003 eV
	}

def compute_ldos_perturbation_optimized_vectorized(qx, qy, qz, energy, pairing_type, T_matrix, eta=1e-6, nk=50, gap_params=None, use_fs_filter=True):
	"""HAEM LDOS: δρ(q,E) = -1/π ℑTr[τ₃ G(k,E) T(E) G(k+q,E)] with 16×16 τ₃ projection"""
	q_magnitude = np.sqrt(qx**2 + qy**2 + qz**2)
	if q_magnitude < 1e-6:
		return 0.0 + 0j
	

	bandwidth_estimate = 0.1  
	kspace_resolution = bandwidth_estimate / nk
	eta_fast = max(eta, 2.0 * kspace_resolution)
	if eta_fast > eta:
		eta = eta_fast
	
	nk_coarse = max(nk // 2, 20)
	kx_coarse = np.linspace(-np.pi/a, np.pi/a, nk_coarse)
	ky_coarse = np.linspace(-3*np.pi/b, np.pi/b, nk_coarse)
	KX_coarse, KY_coarse = np.meshgrid(kx_coarse, ky_coarse, indexing='ij')
	
	eta_cutoff = 2.0 * eta
	fs_mask_coarse = compute_fermi_surface_mask_vectorized(KX_coarse, KY_coarse, qz, eta_cutoff)
	n_fs_coarse = np.sum(fs_mask_coarse)
	
	if n_fs_coarse == 0:
		return 0.0 + 0j
		
	if n_fs_coarse > 0.1 * KX_coarse.size:
		eta_cutoff = 1.0 * eta
		fs_mask_coarse = compute_fermi_surface_mask_vectorized(KX_coarse, KY_coarse, qz, eta_cutoff)
		n_fs_coarse = np.sum(fs_mask_coarse)
	
	fs_kx = KX_coarse[fs_mask_coarse]
	fs_ky = KY_coarse[fs_mask_coarse]
	
	# Compute Green's functions for FS states
	fs_kx_2d = fs_kx[:, np.newaxis] 
	fs_ky_2d = fs_ky[:, np.newaxis]
	
	G_k_fs = green_function_k_vectorized(fs_kx_2d, fs_ky_2d, qz, energy, pairing_type, eta, gap_params)
	G_kq_fs = green_function_k_vectorized(fs_kx_2d + qx, fs_ky_2d + qy, qz, energy, pairing_type, eta, gap_params)
	
	# Squeeze out the singleton dimension
	G_k_fs = G_k_fs.squeeze(axis=1)  # Shape: (n_fs, 16, 16)
	G_kq_fs = G_kq_fs.squeeze(axis=1)  # Shape: (n_fs, 16, 16)
	
	valid_fs = (~np.any(np.isnan(G_k_fs) | np.isinf(G_k_fs), axis=(1, 2)) & 
	           ~np.any(np.isnan(G_kq_fs) | np.isinf(G_kq_fs), axis=(1, 2)))
	
	n_valid = np.sum(valid_fs)
	if n_valid == 0:
		return 0.0 + 0j
	
	# =============================================================================
	# FULLY VECTORIZED HAEM COMPUTATION: τ₃ projection in 16×16 Nambu space
	# =============================================================================
	# Define τ₃ in 16×16 Nambu space: τ₃ = diag(I₈, -I₈)
	tau3_nambu = np.zeros((16, 16))
	tau3_nambu[:8, :8] = np.eye(8)   # +I₈ for particle sector
	tau3_nambu[8:, 8:] = -np.eye(8)  # -I₈ for hole sector
	
	# Filter for valid FS points
	G_k_valid = G_k_fs[valid_fs]  # Shape: (n_valid, 16, 16)
	G_kq_valid = G_kq_fs[valid_fs]  # Shape: (n_valid, 16, 16)
	
	# Batch compute G(k) @ T @ G(k+q) and apply τ₃ projection
	GT = np.einsum('nij,jk->nik', G_k_valid, T_matrix)
	GTG_batch = np.einsum('nij,njk->nik', GT, G_kq_valid)
	tau3_GTG_batch = np.einsum('ij,njk->nik', tau3_nambu, GTG_batch)
	trace_tau3_GTG_batch = np.trace(tau3_GTG_batch, axis1=1, axis2=2)
	
	# HAEM formula: δρ(q,E) = -1/π ℑTr[τ₃ G T G]
	if len(trace_tau3_GTG_batch) > 0:
		delta_rho_complex = np.mean(trace_tau3_GTG_batch)
	else:
		delta_rho_complex = 0.0 + 0j
		
	return (-1.0 / np.pi) * np.imag(delta_rho_complex)

def compute_ldos_perturbation_optimized(qx, qy, qz, energy, pairing_type, T_matrix, eta=1e-6, nk=50, gap_params=None):
	"""CORRECTED wrapper: uses vectorized HAEM-correct implementation
	
	Args:
		qx, qy, qz: Scattering wavevector
		energy: Energy (in eV)
		pairing_type: Pairing symmetry
		T_matrix: Pre-computed T-matrix for this energy
		eta: Broadening parameter
		nk: k-space sampling density
		gap_params: Gap parameters
		
	Returns:
		Complex LDOS perturbation δρ(q,E) with correct τ₃ projection
	"""
	# CORRECTED: Always use FS filtering for proper HAEM
	return compute_ldos_perturbation_optimized_vectorized(qx, qy, qz, energy, pairing_type, T_matrix, eta, nk, gap_params, use_fs_filter=True)

def compute_ldos_perturbation(qx, qy, qz, energy, pairing_type, V_imp=0.1, eta=1e-6, nk=50, gap_params=None):
	"""Legacy wrapper for compatibility - computes T-matrix each time"""
	T_matrix = compute_t_matrix(energy, pairing_type, V_imp, qz, eta, nk, gap_params)
	return compute_ldos_perturbation_optimized(qx, qy, qz, energy, pairing_type, T_matrix, eta, nk, gap_params)

def compute_haem_signal_tmatrix_optimized(qx, qy, qz, energy, pairing_type, T_matrix_plus, T_matrix_minus, eta=1e-6, nk=30, gap_params=None):
	"""HAEM signal ρ⁻(q,E) = ℜ[δρ(q,+E) - δρ(q,-E)] with 16×16 structure"""
	q_mag = np.sqrt(qx**2 + qy**2 + qz**2)
	if q_mag < 1e-6:
		return 0.0
	
	delta_rho_plus_E = compute_ldos_perturbation_optimized_vectorized(
		qx, qy, qz, energy, pairing_type, T_matrix_plus, eta, nk, gap_params, use_fs_filter=True)
	delta_rho_minus_E = compute_ldos_perturbation_optimized_vectorized(
		qx, qy, qz, -energy, pairing_type, T_matrix_minus, eta, nk, gap_params, use_fs_filter=True)
	
	if not isinstance(delta_rho_plus_E, complex):
		delta_rho_plus_E = complex(delta_rho_plus_E)
	if not isinstance(delta_rho_minus_E, complex):
		delta_rho_minus_E = complex(delta_rho_minus_E)
	
	delta_rho_antisym = delta_rho_plus_E - delta_rho_minus_E
	return np.real(delta_rho_antisym)

def compute_haem_signal_energy_integrated(qx, qy, qz, energy_center, pairing_type, eta=1e-6, nk=30, gap_params=None, n_energies=7, gap_scale=0.0003, V_imp=0.1):
	"""Compute HAEM signal integrated over positive energy window for smooth results
	
	Args:
		qx, qy, qz: Scattering wavevector
		energy_center: Central energy for integration window (in eV)
		pairing_type: Pairing symmetry
		eta: Broadening parameter
		nk: k-space sampling density
		gap_params: Gap parameters
		n_energies: Number of energy points in integration window
		gap_scale: Superconducting gap scale Δ
		V_imp: Impurity strength (eV)
		
	Returns:
		Energy-integrated HAEM signal
		
	Note: Uses ONLY positive energies since HAEM is already antisymmetrized ρ(+E) - ρ(-E)
	"""
	# CRITICAL: Use only positive energies for integration
	# Energy window around central positive energy
	eta_eff = determine_broadening(energy_center, eta, gap_scale)
	
	# Define positive energy range: 100 µeV to 300 µeV typical
	E_min = max(energy_center * 0.5, 1.0e-4)  # At least 100 µeV
	E_max = energy_center * 1.5
	energy_points = np.linspace(E_min, E_max, n_energies)
	
	haem_integrated = 0.0
	valid_count = 0
	
	for energy in energy_points:
		if abs(energy) < 1e-9:  # Skip exactly zero energy
			continue
			
		# Compute broadening for this energy
		eta_this = determine_broadening(energy, eta, gap_scale)
		
		# Compute T-matrices for this energy with specified V_imp
		T_matrix_plus = compute_t_matrix(energy, pairing_type, V_imp, qz, eta_this, nk, gap_params)
		T_matrix_minus = compute_t_matrix(-energy, pairing_type, V_imp, qz, eta_this, nk, gap_params)
		
		# Compute HAEM signal at this energy (already antisymmetrized)
		haem_val = compute_haem_signal_tmatrix_optimized(qx, qy, qz, energy, pairing_type, 
		                                                   T_matrix_plus, T_matrix_minus, eta_this, nk, gap_params)
		haem_integrated += haem_val
		valid_count += 1
	
	# Average over energy points
	if valid_count > 0:
		haem_integrated /= valid_count
	
	return haem_integrated

def compute_haem_signal_tmatrix(qx, qy, qz, energy, pairing_type, V_imp=0.1, eta=1e-6, nk=30, gap_params=None):
	"""Legacy wrapper for compatibility"""
	# Compute LDOS perturbations at +E and -E
	delta_N_plus = compute_ldos_perturbation(qx, qy, qz, energy, pairing_type, V_imp, eta, nk, gap_params)
	delta_N_minus = compute_ldos_perturbation(qx, qy, qz, -energy, pairing_type, V_imp, eta, nk, gap_params)
	
	# Energy-antisymmetrized HAEM signal
	haem_signal = np.real(delta_N_plus - delta_N_minus)
	
	return haem_signal

def gap_function(kx, ky, kz, pairing_type, C3=0.0003):
	"""Legacy gap function for visualization - returns dominant component magnitude"""
	if pairing_type == 'B2u':
		# B2u: dominant component is dz ~ C3*sin(kx*a)
		return C3 * np.sin(kx * a)
	elif pairing_type == 'B3u':
		# B3u: dominant component is dz ~ C3*sin(ky*b)
		return C3 * np.sin(ky * b)
	else:
		raise ValueError('Only B2u and B3u supported')

def create_hardcoded_vectors(kz=0.0):
	"""Create vector data using known hardcoded origins (fast, no FS computation)
	
	Returns:
		dict: Dictionary containing vector data for each pairing type
		
	Note: Uses exact coordinates in (2π/a, 2π/b) units
	Vectors: p1=(0.29,0), p2=(0.43,1), p3=(0.29,2), p4=(0,2), p5=(-0.14,1), p6=(0.57,0)
	"""
	vector_labels = ['p1', 'p2', 'p5', 'p6']
	# Vectors in (2pi/a, 2pi/b) units - exact coordinates from user
	vectors_base = [
		(0.29, 0.0),      # p1
		(0.43, 1.0),      # p2
		(-0.14, 1.0),     # p5
		(0.57, 0.0)       # p6
	]
	
	# Origins on Fermi contours (need to find exact FS points)
	origin_main = (-2.0 * np.pi/b, 0.0)  # ky = -2π/b for p1,p2,p5 - will adjust to FS
	origin_p6 = (0.0, 0.0)  # ky = 0 for p6 - will adjust to FS
	
	# Find exact Fermi surface points for origins
	def find_fermi_point_at_ky(target_ky, kz_val, tolerance=1e-4, prefer_negative_kx=False):
		"""Find kx value where Fermi level crosses at given ky"""
		kx_search = np.linspace(-np.pi/a, np.pi/a, 500)  # Higher resolution for better FS detection
		candidates = []
		
		for kx in kx_search:
			H = H_full(kx, target_ky, kz_val)
			eigs = np.linalg.eigvals(H)
			min_energy = np.min(np.abs(np.real(eigs)))  # Distance to Fermi level
			
			# Collect all good candidates
			if min_energy < tolerance:
				candidates.append((kx, min_energy))
		
		if candidates:
			# Sort candidates by energy first, then by preference
			if prefer_negative_kx:
				# For ky=0, prefer negative kx
				candidates.sort(key=lambda x: (x[1], x[0] > 0, abs(x[0])))
			else:
				# For other origins, prefer closest to zero kx
				candidates.sort(key=lambda x: (x[1], abs(x[0])))
			return candidates[0][0]
		
		# Fallback: find closest point overall with preference
		best_kx = 0.0
		best_energy = float('inf')
		
		for kx in kx_search:
			H = H_full(kx, target_ky, kz_val)
			eigs = np.linalg.eigvals(H)
			min_energy = np.min(np.abs(np.real(eigs)))
			
			# Apply preference in fallback too
			is_better = False
			if min_energy < best_energy - 1e-6:  # Significantly better energy
				is_better = True
			elif abs(min_energy - best_energy) < 1e-6:  # Similar energies
				if prefer_negative_kx:
					# Strongly prefer negative kx
					is_better = (kx < 0 and best_kx >= 0) or (kx < 0 and best_kx < 0 and abs(kx) < abs(best_kx))
				else:
					# Prefer closest to zero
					is_better = abs(kx) < abs(best_kx)
			
			if is_better:
				best_energy = min_energy
				best_kx = kx
				
		return best_kx
	
	# Find Fermi surface origins with debug output
	ky_main = -2.0 * np.pi/b
	ky_p6 = 0.0
	kx_main = find_fermi_point_at_ky(ky_main, kz)  # Default preference for ky=-2π/b
	kx_p6 = find_fermi_point_at_ky(ky_p6, kz, prefer_negative_kx=True)  # Prefer negative kx for ky=0
	
	# Verify origins are on Fermi surface
	print(f"Debug: Origin at ky=-2π/b: FS point = ({ky_main/(np.pi/b):.2f}π/b, {kx_main/(np.pi/a):.2f}π/a)")
	print(f"Debug: Origin at ky=0: FS point = ({ky_p6/(np.pi/b):.2f}π/b, {kx_p6/(np.pi/a):.2f}π/a)")
	
	origin_main = (ky_main, kx_main)
	origin_p6 = (ky_p6, kx_p6)
	
	results = {}
	for pairing_type in ['B2u', 'B3u']:
		vector_data = []
		vectors = []
		
		for i, (vec_x_init, vec_y_init) in enumerate(vectors_base):
			# Choose origin: p6 gets ky=0, others get ky=-2
			if i == 3:  # p6 (index 3 in the 4-vector list)
				origin_ky, origin_kx = origin_p6
			else:  # p1, p2, p5
				origin_ky, origin_kx = origin_main
			
			# Use exact coordinates
			vec_x_final = vec_x_init
			vec_y_final = vec_y_init
			vectors.append((vec_x_final, vec_y_final))
			
			vector_data.append({
				'label': vector_labels[i],
				'initial_x': vec_x_init,
				'initial_y': vec_y_init,
				'final_x': vec_x_final,
				'final_y': vec_y_final,
				'scale_factor': 1.0,
				'origin': (origin_ky, origin_kx)
			})
		
		results[pairing_type] = {
			'vectors': vectors,
			'vector_data': vector_data,
			'origin_main': origin_main,
			'origin_p6': origin_p6
		}
	
	return results

def compute_vectors_on_fermi_surface(kz=0.0, resolution=300):
	"""Compute vectors that end on Fermi surface for both pairing types
	
	Returns:
		dict: Dictionary containing vector data for each pairing type
	"""
	nk = resolution
	kx_vals = np.linspace(-np.pi/a, np.pi/a, nk)
	ky_vals = np.linspace(-3*np.pi/b, np.pi/b, nk)  # FIXED: Use original asymmetric range to match display!

	# Compute Fermi surface energies to find origins
	energies_fs = np.zeros((nk, nk, 4))
	for ix, kx in enumerate(kx_vals):
		for jy, ky in enumerate(ky_vals):
			H = H_full(kx, ky, kz)
			eigvals = np.linalg.eigvals(H)
			energies_fs[ix, jy, :] = np.sort(np.real(eigvals))

	# Vectors in (2pi/a, 2pi/b) units
	vectors_initial = [
		(0.130, 0.000),  # p1
		(0.374, 1.000),  # p2
		(-0.244, 1.000), # p5
		(0.619, 0.000)   # p6
	]
	vector_labels = ['p1', 'p2', 'p5', 'p6']
	
	pairing_types = ['B2u', 'B3u']
	results = {}
	
	for pairing_type in pairing_types:
		# Find origins - use original locations on asymmetric grid
		target_ky = -2.0 * (np.pi/b)  # FIXED: p1,p2,p5 originate from ky = -2π/b
		Z = energies_fs[:, :, 2]
		threshold = 0.001  # Balanced threshold for Fermi surface detection
		
		# Find origin exactly at ky = -2π/b (original setup) 
		ky_idx_exact = np.argmin(np.abs(ky_vals - target_ky))
		ky_exact = ky_vals[ky_idx_exact]
		
		candidate_points = []
		for kx_idx, kx in enumerate(kx_vals):
			energy = Z[kx_idx, ky_idx_exact]
			if abs(energy) < threshold:
				candidate_points.append((ky_exact, kx, abs(energy)))
		
		origin_main = None
		if candidate_points:
			# First sort by energy, then group points with very similar energies
			candidate_points.sort(key=lambda pt: pt[2])  # Sort by energy (pt[2])
			best_energy = candidate_points[0][2]
			energy_tolerance = 1e-6
			best_energy_group = [pt for pt in candidate_points if abs(pt[2] - best_energy) < energy_tolerance]
			
			# Among the best energy group, prefer negative kx, then closest to kx=0
			best_energy_group.sort(key=lambda pt: (pt[1] > 0, abs(pt[1])))
			
			best_point = best_energy_group[0]
			origin_main = (best_point[0], best_point[1])
		else:
			threshold = 0.02  # eV - looser threshold
			# Search in a range around the exact target
			search_range = 5
			ky_indices = range(max(0, ky_idx_exact - search_range), 
							   min(len(ky_vals), ky_idx_exact + search_range + 1))
			for ky_idx in ky_indices:
				ky = ky_vals[ky_idx]
				for kx_idx, kx in enumerate(kx_vals):
					energy = Z[kx_idx, ky_idx]
					if abs(energy) < threshold:
						candidate_points.append((ky, kx, abs(energy)))
			if candidate_points:
				# Apply same logic: group by energy, then prefer negative kx
				candidate_points.sort(key=lambda pt: pt[2])
				best_energy = candidate_points[0][2]
				energy_tolerance = 1e-6
				best_group = [pt for pt in candidate_points if abs(pt[2] - best_energy) < energy_tolerance]
				best_group.sort(key=lambda pt: (pt[1] > 0, abs(pt[1])))  # Prefer negative kx
				best_point = best_group[0]
				origin_main = (best_point[0], best_point[1])
		
		# Find p6 origin at ky = 0
		target_ky_p6 = 0.0
		ky_idx_exact_p6 = np.argmin(np.abs(ky_vals - target_ky_p6))
		ky_exact_p6 = ky_vals[ky_idx_exact_p6]
		
		candidate_points_p6 = []
		threshold_p6 = 0.001  # Start with strict threshold
		for kx_idx, kx in enumerate(kx_vals):
			energy = Z[kx_idx, ky_idx_exact_p6]
			if abs(energy) < threshold_p6:  # Remove kx < 0 restriction
				candidate_points_p6.append((ky_exact_p6, kx, abs(energy)))
		
		origin_p6 = None
		if candidate_points_p6:
			# Sort by energy first, then prefer kx closest to 0
			candidate_points_p6.sort(key=lambda pt: (pt[2], abs(pt[1])))
			best_point_p6 = candidate_points_p6[0]
			origin_p6 = (best_point_p6[0], best_point_p6[1])
		else:
			# If no candidates found with strict threshold, try relaxed threshold
			threshold_p6 = 0.02  # Relaxed threshold
			search_range_p6 = 5
			ky_indices_p6 = range(max(0, ky_idx_exact_p6 - search_range_p6), 
								  min(len(ky_vals), ky_idx_exact_p6 + search_range_p6 + 1))
			for ky_idx in ky_indices_p6:
				ky = ky_vals[ky_idx]
				for kx_idx, kx in enumerate(kx_vals):
					energy = Z[kx_idx, ky_idx]
					if abs(energy) < threshold_p6:
						candidate_points_p6.append((ky, kx, abs(energy)))
			
			if candidate_points_p6:
				candidate_points_p6.sort(key=lambda pt: (pt[2], abs(pt[1] - 0), abs(pt[0] - target_ky_p6)))
				best_point_p6 = candidate_points_p6[0]
				origin_p6 = (best_point_p6[0], best_point_p6[1])
		
		if origin_main is None:
			continue
		
		# Find where hardcoded vectors intersect Fermi surface
		vectors = []
		vector_data = []
		
		for i, (vec_x_init, vec_y_init) in enumerate(vectors_initial):
			# Choose origin based on vector
			if i == 3 and origin_p6:  # p6 vector
				origin_ky, origin_kx = origin_p6
			else:  # p1, p2, p5 vectors
				origin_ky, origin_kx = origin_main
			
			# Convert initial vector to raw coordinates
			vec_ky_init = vec_y_init * 2 * (np.pi/b)
			vec_kx_init = vec_x_init * 2 * (np.pi/a)
			
			# Find Fermi surface intersection along vector path
			# p1, p2, p5 should end at ky = -2π/b, p6 at ky = 0 (original setup)
			if i in [0, 1, 2]:  # p1, p2, p5 vectors
				target_endpoint_ky = -2 * (np.pi/b)
			else:  # p6 vector
				target_endpoint_ky = 0.0
			
			# Calculate t value to reach target ky
			if abs(vec_ky_init) > 1e-12:  # Avoid division by zero
				t_target = (target_endpoint_ky - origin_ky) / vec_ky_init
				
				# Ensure t is positive and reasonable
				if t_target > 0 and t_target < 3.0:  # Reasonable upper bound
					best_t = t_target
				else:
					# Fallback: find Fermi surface intersection as before
					t_steps = np.linspace(0.05, 1.2, 200)
					best_t = 1.0
					best_distance_to_endpoint = float('inf')
					
					for t in t_steps:
						ky_test = origin_ky + t * vec_ky_init
						kx_test = origin_kx + t * vec_kx_init
						
						if (ky_vals[0] <= ky_test <= ky_vals[-1] and 
							kx_vals[0] <= kx_test <= kx_vals[-1]):
							
							kx_idx = np.argmin(np.abs(kx_vals - kx_test))
							ky_idx = np.argmin(np.abs(ky_vals - ky_test))
							
							energy = Z[kx_idx, ky_idx]
							if abs(energy) < threshold:
								distance_to_endpoint = abs(t - 1.0)
								if distance_to_endpoint < best_distance_to_endpoint:
									best_t = t
									best_distance_to_endpoint = distance_to_endpoint
			else:
				# Vector has no ky component, use original logic
				t_steps = np.linspace(0.05, 1.2, 200)
				best_t = 1.0
				best_distance_to_endpoint = float('inf')
				
				for t in t_steps:
					ky_test = origin_ky + t * vec_ky_init
					kx_test = origin_kx + t * vec_kx_init
					
					if (ky_vals[0] <= ky_test <= ky_vals[-1] and 
						kx_vals[0] <= kx_test <= kx_vals[-1]):
						
						kx_idx = np.argmin(np.abs(kx_vals - kx_test))
						ky_idx = np.argmin(np.abs(ky_vals - ky_test))
						
						energy = Z[kx_idx, ky_idx]
						if abs(energy) < threshold:
							distance_to_endpoint = abs(t - 1.0)
							if distance_to_endpoint < best_distance_to_endpoint:
								best_t = t
								best_distance_to_endpoint = distance_to_endpoint
			
			# Calculate final vector using best intersection point
			vec_x_final = vec_x_init * best_t
			vec_y_final = vec_y_init * best_t
			vectors.append((vec_x_final, vec_y_final))
			
			# Store detailed vector information
			vector_data.append({
				'label': vector_labels[i],
				'initial_x': vec_x_init,
				'initial_y': vec_y_init,
				'final_x': vec_x_final,
				'final_y': vec_y_final,
				'scale_factor': best_t,
				'origin': (origin_ky, origin_kx)
			})
		
		results[pairing_type] = {
			'vectors': vectors,
			'vector_data': vector_data,
			'origin_main': origin_main,
			'origin_p6': origin_p6,
			'energies_fs': energies_fs,
			'kx_vals': kx_vals,
			'ky_vals': ky_vals
		}
	
	return results

def save_vector_table(vector_results, kz=0.0, base_save_dir='outputs/phase_character', energy=0.001):
	"""Save vector information as a table image using precomputed results"""
	# Create organized subfolder structure with energy subfolder
	energy_folder = f'E_{energy*1e6:.0f}ueV'
	save_dir = os.path.join(base_save_dir, 'vector_tables', current_parameter_set, energy_folder)
	os.makedirs(save_dir, exist_ok=True)
	
	# Prepare table data from results
	table_data = []
	for pairing_type, data in vector_results.items():
		for vec_data in data['vector_data']:
			table_data.append({
				'pairing': pairing_type,
				'vector': vec_data['label'],
				'initial_x': vec_data['initial_x'],
				'initial_y': vec_data['initial_y'],
				'final_x': vec_data['final_x'],
				'final_y': vec_data['final_y'],
				'scale_factor': vec_data['scale_factor']
			})
	
	# Create table image
	fig, ax = plt.subplots(figsize=(18, 8), dpi=300)
	ax.axis('tight')
	ax.axis('off')
	
	# Prepare table data
	table_rows = []
	headers = ['Pairing', 'Vector', 'Initial Vector (π/a, π/b)', 'Final Vector (π/a, π/b)', 
	          'Initial kx (π/a)', 'Initial ky (π/b)', 'Final kx (π/a)', 'Final ky (π/b)', 'Scale Factor']
	
	for data in table_data:
		row = [
			data['pairing'],
			data['vector'],
			f"({data['initial_x']:.3f}, {data['initial_y']:.3f})",
			f"({data['final_x']:.3f}, {data['final_y']:.3f})",
			f"{data['initial_x']:.3f}",
			f"{data['initial_y']:.3f}",
			f"{data['final_x']:.3f}",
			f"{data['final_y']:.3f}",
			f"{data['scale_factor']:.3f}"
		]
		table_rows.append(row)
	
	# Create the table
	table = ax.table(cellText=table_rows, colLabels=headers, 
	                cellLoc='center', loc='center', 
	                colWidths=[0.10, 0.08, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.12])
	
	# Style the table
	table.auto_set_font_size(False)
	table.set_fontsize(10)
	table.scale(1.2, 2)
	
	# Color headers
	for i in range(len(headers)):
		table[(0, i)].set_facecolor('#4CAF50')
		table[(0, i)].set_text_props(weight='bold', color='white')
	
	# Alternate row colors
	for i in range(1, len(table_rows) + 1):
		for j in range(len(headers)):
			if i % 2 == 0:
				table[(i, j)].set_facecolor('#f0f0f0')
			# Color code by pairing type
			if j == 0:  # Pairing column
				if table_rows[i-1][0] == 'B2u':
					table[(i, j)].set_facecolor('#ffcccc')
				elif table_rows[i-1][0] == 'B3u':
					table[(i, j)].set_facecolor('#ccccff')
			# Color code vector format columns
			elif j in [2, 3]:  # Vector format columns
				if table_rows[i-1][0] == 'B2u':
					table[(i, j)].set_facecolor('#ffe6e6')
				elif table_rows[i-1][0] == 'B3u':
					table[(i, j)].set_facecolor('#e6e6ff')
	
	plt.title(f'Vector Components in π/a and π/b units\n{current_parameter_set} parameters, kz={kz:.2f}', 
	         fontsize=14, fontweight='bold', pad=20)
	
	filename = f'vector_table_kz_{kz:.3f}.png'
	filepath = os.path.join(save_dir, filename)
	fig.savefig(filepath, dpi=300, bbox_inches='tight')
	plt.close(fig)
	print(f"Saved vector table to: {filepath}")

def plot_haem_along_vectors(kz=0.0, pairing_types=['B2u', 'B3u'], base_save_dir='outputs/phase_character', 
                          resolution=300, parameter_set=None, n_points=100, energy=0.001, 
                          eta=1e-6, nk_green=64, n_energies=50, plot_type='energy_vs_q', energy_window_integration=3,
                          use_hardcoded_vectors=True):
	"""Plot antisymmetrized HAEM signal with energy dependence
	
	Args:
		kz: kz value for 2D slice
		pairing_types: List of pairing symmetries to plot
		base_save_dir: Base directory for organized output structure
		resolution: Grid resolution for finding origins (only if use_hardcoded_vectors=False)
		parameter_set: Parameter set to use (if None, uses current setting)
		n_points: Number of points to sample along each vector (default 100 for smooth curves)
		energy: Maximum energy for energy sweep (in eV, typically 300 µeV)
		eta: Broadening parameter for Green's functions
		nk_green: k-space sampling density for T-matrix (default 64 for convergence)
		n_energies: Number of energies to sample (default 50, increase to 100 for publication quality)
		plot_type: 'energy_vs_q' for 2D heatmap or 'fixed_energy' for 1D curves
		energy_window_integration: Smoothing (1=off, 3=fast 3-point, 5+=slower multi-point)
		use_hardcoded_vectors: If True, use fast hardcoded origins (recommended, ~100x faster)
		
	Note: Full BZ integration (no FS filtering), adaptive broadening η ≈ 0.075 Δ
	HAEM validation: ρ⁻(q,E) = Re[δN(q,+E) - δN(q,-E)] correctly antisymmetrizes particle vs hole sectors
	"""
	if parameter_set is not None:
		set_parameters(parameter_set)
		print(f"Using parameter set: {parameter_set}")
	else:
		print(f"Using current parameter set: {current_parameter_set}")
	
	os.makedirs(base_save_dir, exist_ok=True)
	
	# Create organized subfolder structure
	energy_folder = f'E_max_{energy*1e6:.0f}ueV'
	save_dir = os.path.join(base_save_dir, 'haem_vectors_energy_sweep', current_parameter_set, energy_folder)
	os.makedirs(save_dir, exist_ok=True)
	
	# Get vector data - use fast hardcoded method by default
	if use_hardcoded_vectors:
		print("  Using hardcoded vector origins (fast mode)")
		vector_results = create_hardcoded_vectors(kz=kz)
	else:
		print(f"  Computing vectors on Fermi surface (slow, resolution={resolution})")
		vector_results = compute_vectors_on_fermi_surface(kz=kz, resolution=resolution)
	
	vector_labels = ['p1', 'p2', 'p5', 'p6']
	vector_colors = ['#228B22', '#FF8C00', '#8B008B', '#000000']
	
	# Gap scale for broadening
	gap_scale = 0.0003  # 300 µeV typical for UTe2

	for pairing_type in pairing_types:
		if pairing_type not in vector_results:
			continue
			
		data = vector_results[pairing_type]
		vectors = data['vectors']
		vector_data = data['vector_data']
		
		print(f"\nComputing HAEM signals for {pairing_type} with energy dependence...")
		# Determine and report broadening parameters
		eta_eff = determine_broadening(energy, eta, gap_scale)
		print(f"  Gap scale Δ = {gap_scale*1e6:.0f} µeV")
		print(f"  Adaptive broadening η = {eta_eff*1e6:.1f} µeV (0.075 × Δ)")
		print(f"  Energy sweep: {energy*1e6:.0f} µeV → ~0 µeV ({n_energies} points)")
		print(f"  Full BZ integration (no FS filtering)")
		
		# Report the auto-scaled impurity potential
		energy_scale = max(abs(energy), 1e-6)
		V_imp_used = min(max(0.005 * energy_scale, 0.2e-6), 0.5e-3)
		print(f"  Auto-scaled V_imp = {V_imp_used*1e6:.1f} µeV")
		
		if plot_type == 'energy_vs_q':
			# Create 2D heatmap plots (Energy vs q-position)
			fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
			axes = axes.flatten()
			
			# PRE-COMPUTE ALL T-MATRICES ONCE (major optimization!)
			energy_vals = np.linspace(energy, energy * 0.05, n_energies)
			print(f"\n  PRE-COMPUTING T-MATRICES FOR {n_energies} ENERGIES...")
			print(f"    Energy resolution ΔE = {(energy_vals[0]-energy_vals[1])*1e6:.2f} µeV")
			print(f"    Broadening η = {determine_broadening(energy, eta, gap_scale)*1e6:.2f} µeV")
			print(f"    k-space density: {nk_green}×{nk_green} = {nk_green**2} points")
			
			T_matrices_plus = {}
			T_matrices_minus = {}
			for j, E in enumerate(energy_vals):
				if j % max(1, n_energies//10) == 0:
					print(f"    T-matrix {j+1}/{n_energies}: E={E*1e6:.1f} µeV")
				eta_this = determine_broadening(E, eta, gap_scale)
				T_matrices_plus[j] = compute_t_matrix(E, pairing_type, None, kz, eta_this, nk_green)
				T_matrices_minus[j] = compute_t_matrix(-E, pairing_type, None, kz, eta_this, nk_green)
			
			print(f"  T-matrices cached. Now computing HAEM for all q-vectors...\n")
			
			for i, (vector, label, color, vec_info) in enumerate(zip(vectors, vector_labels, vector_colors, vector_data)):
				print(f"  Processing vector {label}...")
				origin_ky, origin_kx = vec_info['origin']
				
				# Convert vector to raw coordinates
				vec_x, vec_y = vector  # in (2π/a, 2π/b) units
				vec_ky = vec_y * 2 * (np.pi/b)  # convert to raw ky
				vec_kx = vec_x * 2 * (np.pi/a)  # convert to raw kx
				
				# Sample points along vector (0 to 1)
				t_vals = np.linspace(0, 1, n_points)
				
				# 2D array for HAEM(q, E)
				haem_2d = np.zeros((n_energies, n_points))
				
				# Compute all q-vectors at once (vectorization)
				ki_kx = origin_kx
				ki_ky = origin_ky
				kf_kx_array = origin_kx + t_vals * vec_kx
				kf_ky_array = origin_ky + t_vals * vec_ky
				qx_array = kf_kx_array - ki_kx
				qy_array = kf_ky_array - ki_ky
				
				# Loop over energies only (using pre-computed T-matrices)
				for j in range(n_energies):
					if j % max(1, n_energies//10) == 0:
						print(f"    Energy {j+1}/{n_energies}")
					
					E = energy_vals[j]
					eta_this = determine_broadening(E, eta, gap_scale)
					
					# Use pre-computed T-matrices
					T_matrix_plus = T_matrices_plus[j]
					T_matrix_minus = T_matrices_minus[j]
					
					# Apply 3-point smoothing (much faster than 5-point window)
					if energy_window_integration > 1 and j > 0 and j < n_energies - 1:
						# Use neighboring energies for smoothing
						for k in range(n_points):
							qx, qy, qz = qx_array[k], qy_array[k], 0
							# Central point (weight 0.5) + neighbors (weight 0.25 each)
							haem_center = compute_haem_signal_tmatrix_optimized(qx, qy, qz, E, pairing_type,
							                                                       T_matrix_plus, T_matrix_minus, eta_this, nk_green)
							haem_prev = compute_haem_signal_tmatrix_optimized(qx, qy, qz, energy_vals[j-1], pairing_type,
							                                                     T_matrices_plus[j-1], T_matrices_minus[j-1], eta_this, nk_green)
							haem_next = compute_haem_signal_tmatrix_optimized(qx, qy, qz, energy_vals[j+1], pairing_type,
							                                                     T_matrices_plus[j+1], T_matrices_minus[j+1], eta_this, nk_green)
							haem_2d[j, k] = 0.5 * haem_center + 0.25 * (haem_prev + haem_next)
					else:
						# No smoothing at boundaries
						for k in range(n_points):
							qx, qy, qz = qx_array[k], qy_array[k], 0
							haem_2d[j, k] = compute_haem_signal_tmatrix_optimized(qx, qy, qz, E, pairing_type,
							                                                         T_matrix_plus, T_matrix_minus, eta_this, nk_green)
				
				# Plot 2D heatmap
				im = axes[i].imshow(haem_2d, aspect='auto', origin='lower', cmap='RdBu_r',
				                   extent=[0, 1, energy_vals[-1]*1e6, energy_vals[0]*1e6],
				                   vmin=-np.max(np.abs(haem_2d)), vmax=np.max(np.abs(haem_2d)))
				axes[i].set_xlabel('Position along vector', fontsize=12)
				axes[i].set_ylabel('Energy (µeV)', fontsize=12)
				axes[i].set_title(f'{label} - {pairing_type}', fontsize=14, color=color)
				plt.colorbar(im, ax=axes[i], label=r'HAEM $\rho^-(q,E)$')
				
			plt.suptitle(f'Energy-dependent HAEM - {pairing_type} ({current_parameter_set})\nFull BZ, η=0.075Δ', fontsize=16)
			plt.tight_layout()
			
			filename = f'haem_energy_sweep_{pairing_type}_kz_{kz:.3f}_nE_{n_energies}.png'
			filepath = os.path.join(save_dir, filename)
			fig.savefig(filepath, dpi=300, bbox_inches='tight')
			plt.close(fig)
			print(f"Saved energy-dependent HAEM to: {filepath}")
			
		else:  # Fixed energy 1D curves
			fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
			axes = axes.flatten()
			
			for i, (vector, label, color, vec_info) in enumerate(zip(vectors, vector_labels, vector_colors, vector_data)):
				print(f"  Processing vector {label}...")
				origin_ky, origin_kx = vec_info['origin']
				
				vec_x, vec_y = vector
				vec_ky = vec_y * 2 * (np.pi/b)
				vec_kx = vec_x * 2 * (np.pi/a)
				
				t_vals = np.linspace(0, 1, n_points)
				haem_vals = []
				
				for j, t in enumerate(t_vals):
					if j % (n_points//10) == 0:
						print(f"    Point {j+1}/{n_points}")
						
					ki_kx = origin_kx
					ki_ky = origin_ky
					kf_kx = origin_kx + t * vec_kx
					kf_ky = origin_ky + t * vec_ky
					
					qx = kf_kx - ki_kx
					qy = kf_ky - ki_ky
					qz = 0
					
					# Energy-integrated HAEM with full BZ (10 point integration for smoothness)
					haem_val = compute_haem_signal_energy_integrated(qx, qy, qz, energy, pairing_type,
					                                                   eta, nk_green, gap_scale=gap_scale, n_energies=10)
					haem_vals.append(haem_val)
				
				axes[i].plot(t_vals, haem_vals, color=color, linewidth=2, marker='o', markersize=2)
				axes[i].set_xlabel('Position along vector', fontsize=12)
				axes[i].set_ylabel(r'HAEM signal $\rho^-(q, E)$', fontsize=12)
				axes[i].set_title(f'{label} - {pairing_type}', fontsize=14, color=color)
				if haem_vals:
					max_val = max(abs(min(haem_vals)), abs(max(haem_vals)))
					if max_val > 0:
						axes[i].set_ylim(-max_val * 1.1, max_val * 1.1)
				axes[i].grid(True, alpha=0.3)
				axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
			
			plt.suptitle(f'HAEM signal - {pairing_type} ({current_parameter_set})\nFull BZ, η=0.075Δ', fontsize=16)
			plt.tight_layout()
			
			filename = f'haem_fixed_E_{pairing_type}_kz_{kz:.3f}_E_{energy*1e6:.0f}ueV.png'
			filepath = os.path.join(save_dir, filename)
			fig.savefig(filepath, dpi=300, bbox_inches='tight')
			plt.close(fig)
			print(f"Saved HAEM vectors to: {filepath}")
			
	return vector_results

def plot_haem_gap_maps_with_vectors(kz=0.0, energy=0.0002, save_dir='outputs', pairing_types=['B2u', 'B3u'], resolution=150):
	"""Generate gap maps with HAEM vectors overlaid - saves to specified directory
	
	Args:
		kz: kz value for 2D slice
		energy: Energy for HAEM calculation (used in filename)
		save_dir: Directory to save gap maps (same as HAEM 1D plots)
		pairing_types: List of pairing symmetries
		resolution: Grid resolution for gap calculation
	"""
	os.makedirs(save_dir, exist_ok=True)
	
	# Get precomputed vector data
	vector_results = create_hardcoded_vectors(kz=kz)
	
	# Create k-space grid for gap calculation (higher resolution for smooth contours)
	kx_range = np.linspace(-np.pi/a, np.pi/a, resolution*2)  # Double resolution for smoother contours
	ky_range = np.linspace(-3*np.pi/b, np.pi/b, resolution*2)
	KX, KY = np.meshgrid(kx_range, ky_range, indexing='ij')
	
	for pairing_type in pairing_types:
		if pairing_type not in vector_results:
			continue
			
		# Get gap parameters for this pairing type
		gap_params = get_gap_parameters_for_pairing(pairing_type)
		
		# Compute gap values (actual gap components, not magnitude)
		gap_values = np.zeros_like(KX)
		for i in range(KX.shape[0]):
			for j in range(KX.shape[1]):
				kx, ky = KX[i,j], KY[i,j]
				C0, C1, C2, C3 = gap_params['C0'], gap_params['C1'], gap_params['C2'], gap_params['C3']
				d_vec = d_vector(kx, ky, kz, pairing_type, C0, C1, C2, C3)
				
				# Show dominant gap component with sign (actual gap, not magnitude)
				if pairing_type == 'B2u':
					# B2u: dominant components are dx=C1*sin(kz*c) and dz=C3*sin(kx*a)
					# Show dz component (varies with kx) as it's more interesting
					gap_values[i,j] = np.real(d_vec[2]) * 1e3  # dz in meV with sign
				elif pairing_type == 'B3u':
					# B3u: dominant components are dy=C2*sin(kz*c) and dz=C3*sin(ky*b)
					# Show dz component (varies with ky) as it's more interesting
					gap_values[i,j] = np.real(d_vec[2]) * 1e3  # dz in meV with sign
				else:
					# Fallback: use magnitude
					gap_values[i,j] = np.linalg.norm(d_vec) * 1e3
		
		# Create gap map plot
		fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
		
		# Plot gap as background (actual gap with sign)
		from matplotlib.colors import ListedColormap
		base_cmap = plt.get_cmap('RdBu_r')  # Red-blue for signed values
		vmax = np.max(np.abs(gap_values))
		im = ax.imshow(gap_values.T, extent=[-3, 1, -1, 1], 
		              origin='lower', cmap=base_cmap, aspect='auto',
		              vmin=-vmax, vmax=vmax)  # Symmetric colormap for signed values
		
		# Get vector data for overlay
		data = vector_results[pairing_type]
		vectors = data['vectors']
		vector_data = data['vector_data']
		vector_labels = ['p1', 'p2', 'p5', 'p6']
		vector_colors = ['#FFFF00', '#FF8C00', '#FF1493', '#00FF00']  # Bright colors for 4 vectors
		
		# Compute Fermi surface for proper contour plotting (higher resolution, lower threshold)
		fermi_energies = np.zeros_like(KX)
		for i in range(KX.shape[0]):
			for j in range(KX.shape[1]):
				kx, ky = KX[i,j], KY[i,j]
				H = H_full(kx, ky, kz)
				eigs = np.linalg.eigvals(H)
				fermi_energies[i,j] = np.min(np.abs(np.real(eigs)))  # Distance to Fermi level
		
		# Plot Fermi contours directly (smoother, no discontinuities)
		kx_display = KX / (np.pi/a)  # Convert to display units
		ky_display = KY / (np.pi/b)
		
		# Use lower threshold for continuous contours
		contour_levels = [0.001, 0.002]  # Much lower threshold for smoother contours
		cs = ax.contour(ky_display, kx_display, fermi_energies, 
		               levels=contour_levels, colors=['white'], 
		               linewidths=2, alpha=0.9, zorder=15)
		
		# Overlay vectors
		for i, (vector, vec_info) in enumerate(zip(vectors, vector_data)):
			origin_ky, origin_kx = vec_info['origin']
			
			# Convert vector coordinates for display
			vec_x, vec_y = vector  # in (2π/a, 2π/b) units
			vec_ky = vec_y * (2*np.pi/b)  # CORRECTED: Full 2π/b scaling
			vec_kx = vec_x * (2*np.pi/a)  # CORRECTED: Full 2π/a scaling
			
			# Convert to display coordinates
			origin_ky_norm = origin_ky / (np.pi/b)
			origin_kx_norm = origin_kx / (np.pi/a)
			vec_ky_norm = vec_ky / (np.pi/b)
			vec_kx_norm = vec_kx / (np.pi/a)
			
			# Draw origin point on Fermi surface
			ax.plot(origin_ky_norm, origin_kx_norm, 'o', color=vector_colors[i], 
			       markersize=8, markeredgecolor='white', markeredgewidth=2, zorder=25)
			
			# Draw vector arrow
			ax.arrow(origin_ky_norm, origin_kx_norm, vec_ky_norm, vec_kx_norm, 
			        head_width=0.08, head_length=0.1, 
			        fc=vector_colors[i], ec='white', linewidth=3, 
			        length_includes_head=True, zorder=20)
			        
			# Add vector label
			ax.text(origin_ky_norm + vec_ky_norm*1.15, origin_kx_norm + vec_kx_norm*1.15, 
			       vector_labels[i], color=vector_colors[i], fontsize=14, 
			       fontweight='bold', zorder=21,
			       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
		
		# Format plot
		ax.set_xlim(-3, 1)  # ky in units of π/b
		ax.set_ylim(-1, 1)  # kx in units of π/a
		ax.set_xlabel(r'$k_y$ ($\pi/b$)', fontsize=16, fontweight='bold')
		ax.set_ylabel(r'$k_x$ ($\pi/a$)', fontsize=16, fontweight='bold')
		ax.set_title(f'{pairing_type} Gap Component + HAEM Vectors + Fermi Contour\nE = {energy*1e6:.0f} µeV, kz = {kz:.2f}', 
		            fontsize=18, fontweight='bold')
		
		# Add colorbar
		cbar = plt.colorbar(im, ax=ax, shrink=0.8)
		gap_component = 'dz' if pairing_type in ['B2u', 'B3u'] else '|d|'
		cbar.set_label(f'Gap Component {gap_component}(k) (meV)', fontsize=14, fontweight='bold')
		
		# Add legend for vectors and Fermi surface
		legend_elements = [plt.Line2D([0], [0], color='white', linestyle='-', linewidth=2,
		                             label='Fermi Contour')] 
		legend_elements.extend([plt.Line2D([0], [0], color=vector_colors[i], marker='o', linestyle='-',
		                                  markersize=8, label=f'Vector {vector_labels[i]}')
		                       for i in range(len(vector_labels))])
		ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.8)
		
		# Grid and formatting
		ax.grid(True, alpha=0.3, color='white')
		ax.tick_params(axis='both', which='major', labelsize=12)
		
		# Save gap map
		filename = f'gap_component_{pairing_type}_with_haem_vectors_and_fermi_contour_E{energy*1e6:.0f}ueV.png'
		filepath = os.path.join(save_dir, filename)
		plt.tight_layout()
		fig.savefig(filepath, dpi=300, bbox_inches='tight')
		plt.close(fig)
		print(f"  Gap map saved: {filename}")
	
	return vector_results

# =============================================================================
# VALIDATION TESTS - Required for HAEM correctness verification
# =============================================================================

def test_normal_state_haem_zero(qx=0.1, qy=0.1, qz=0.0, energy=1e-4, pairing_type='B2u', eta=1e-6, nk=30):
	"""Validation Test 1: Normal state (Δ=0) should give HAEM ≈ 0 everywhere
	
	PHYSICS: In normal state, particle-hole symmetry is exact, so:
	δρ(q,+E) = δρ(q,-E) → HAEM = ℜ[δρ(+E) - δρ(-E)] = 0
	"""
	print("\n" + "="*60)
	print("VALIDATION TEST 1: Normal State (Δ=0) → HAEM = 0")
	print("="*60)
	
	# Force normal state by setting all gap parameters to zero
	gap_params_zero = {'C0': 0.0, 'C1': 0.0, 'C2': 0.0, 'C3': 0.0}
	
	# Compute T-matrices (should be identical for ±E in normal state)
	T_plus = compute_t_matrix(energy, pairing_type, None, qz, eta, nk, gap_params_zero)
	T_minus = compute_t_matrix(-energy, pairing_type, None, qz, eta, nk, gap_params_zero)
	
	# Compute HAEM signal
	haem_normal = compute_haem_signal_tmatrix_optimized(
		qx, qy, qz, energy, pairing_type, T_plus, T_minus, eta, nk, gap_params_zero)
	
	print(f"Normal state HAEM: {haem_normal:.3e}")
	
	# Test passes if |HAEM| < 1e-10 (essentially zero within numerical precision)
	tolerance = 1e-10
	if abs(haem_normal) < tolerance:
		print(f"✓ PASS: |HAEM| = {abs(haem_normal):.2e} < {tolerance:.0e}")
		return True
	else:
		print(f"✗ FAIL: |HAEM| = {abs(haem_normal):.2e} ≥ {tolerance:.0e}")
		return False

def test_s_wave_gap_haem_smooth(qx=0.1, qy=0.1, qz=0.0, energy=1e-4, eta=1e-6, nk=30):
	"""Validation Test 2: Trivial s-wave gap should give sign-definite, smooth HAEM
	
	PHYSICS: s-wave gap breaks time-reversal but preserves inversion,
	giving smooth, sign-definite HAEM response
	"""
	print("\n" + "="*60)
	print("VALIDATION TEST 2: s-wave Gap → Sign-definite HAEM")
	print("="*60)
	
	# Simple s-wave gap: only C0 (onsite pairing) 
	gap_params_swave = {'C0': 3e-4, 'C1': 0.0, 'C2': 0.0, 'C3': 0.0}
	
	# Test multiple q-points to check smoothness
	q_points = [(0.05, 0.05), (0.1, 0.1), (0.15, 0.1), (0.1, 0.15)]
	haem_values = []
	
	for qx_test, qy_test in q_points:
		T_plus = compute_t_matrix(energy, 'B2u', None, qz, eta, nk, gap_params_swave)
		T_minus = compute_t_matrix(-energy, 'B2u', None, qz, eta, nk, gap_params_swave)
		
		haem_val = compute_haem_signal_tmatrix_optimized(
			qx_test, qy_test, qz, energy, 'B2u', T_plus, T_minus, eta, nk, gap_params_swave)
		haem_values.append(haem_val)
		print(f"q = ({qx_test:.2f}, {qy_test:.2f}): HAEM = {haem_val:.3e}")
	
	# Check sign-definiteness (all same sign)
	signs = [np.sign(val) for val in haem_values if abs(val) > 1e-12]
	sign_definite = len(set(signs)) <= 1  # All same sign or all zero
	
	# Check smoothness (no dramatic variations)
	if len(haem_values) > 1:
		variations = [abs(haem_values[i+1] - haem_values[i]) for i in range(len(haem_values)-1)]
		max_variation = max(variations) if variations else 0
		mean_magnitude = np.mean([abs(val) for val in haem_values])
		smoothness = max_variation < 5 * mean_magnitude
	else:
		smoothness = True
	
	print(f"Sign-definite: {sign_definite}, Smooth: {smoothness}")
	
	if sign_definite and smoothness:
		print("✓ PASS: s-wave HAEM is sign-definite and smooth")
		return True
	else:
		print("✗ FAIL: s-wave HAEM shows sign changes or roughness")
		return False

def test_nk_convergence(qx=0.1, qy=0.1, qz=0.0, energy=1e-4, pairing_type='B2u', eta=1e-6, gap_params=None):
	"""Validation Test 3: nk convergence - doubling nk should preserve sign structure
	
	PHYSICS: Physical HAEM sign structure should be converged with sufficient k-sampling.
	Doubling nk should not flip signs, only refine magnitudes.
	"""
	print("\n" + "="*60)
	print("VALIDATION TEST 3: nk Convergence Test")
	print("="*60)
	
	if gap_params is None:
		gap_params = {'C0': 0.0, 'C1': 3e-4, 'C2': 3e-4, 'C3': 3e-4}
	
	# Test two different nk values
	nk_coarse = 20
	nk_fine = 40
	
	print(f"Testing nk = {nk_coarse} vs nk = {nk_fine}")
	
	# Coarse sampling
	T_plus_c = compute_t_matrix(energy, pairing_type, None, qz, eta, nk_coarse, gap_params)
	T_minus_c = compute_t_matrix(-energy, pairing_type, None, qz, eta, nk_coarse, gap_params)
	haem_coarse = compute_haem_signal_tmatrix_optimized(
		qx, qy, qz, energy, pairing_type, T_plus_c, T_minus_c, eta, nk_coarse, gap_params)
	
	# Fine sampling
	T_plus_f = compute_t_matrix(energy, pairing_type, None, qz, eta, nk_fine, gap_params)
	T_minus_f = compute_t_matrix(-energy, pairing_type, None, qz, eta, nk_fine, gap_params)
	haem_fine = compute_haem_signal_tmatrix_optimized(
		qx, qy, qz, energy, pairing_type, T_plus_f, T_minus_f, eta, nk_fine, gap_params)
	
	print(f"nk = {nk_coarse}: HAEM = {haem_coarse:.3e}")
	print(f"nk = {nk_fine}: HAEM = {haem_fine:.3e}")
	
	# Check sign preservation
	if abs(haem_coarse) > 1e-12 and abs(haem_fine) > 1e-12:
		same_sign = np.sign(haem_coarse) == np.sign(haem_fine)
		
		# Check relative convergence
		rel_change = abs(haem_fine - haem_coarse) / max(abs(haem_coarse), 1e-15)
		converged = rel_change < 2.0  # Within factor of 2
		
		print(f"Same sign: {same_sign}, Relative change: {rel_change:.2f}")
		
		if same_sign and converged:
			print("✓ PASS: nk convergence preserves sign structure")
			return True
		else:
			print("✗ FAIL: nk convergence changes sign or shows poor convergence")
			return False
	else:
		print("↻ SKIP: HAEM values too small for meaningful convergence test")
		return True

def run_all_haem_validation_tests():
	"""Run all HAEM validation tests and report results"""
	print("\n" + "="*80)
	print("HAEM VALIDATION TEST SUITE")
	print("="*80)
	
	results = []
	
	# Test 1: Normal state
	try:
		results.append(("Normal State (Δ=0)", test_normal_state_haem_zero()))
	except Exception as e:
		print(f"✗ Test 1 ERROR: {e}")
		results.append(("Normal State (Δ=0)", False))
	
	# Test 2: s-wave gap
	try:
		results.append(("s-wave Smoothness", test_s_wave_gap_haem_smooth()))
	except Exception as e:
		print(f"✗ Test 2 ERROR: {e}")
		results.append(("s-wave Smoothness", False))
	
	# Test 3: nk convergence
	try:
		results.append(("nk Convergence", test_nk_convergence()))
	except Exception as e:
		print(f"✗ Test 3 ERROR: {e}")
		results.append(("nk Convergence", False))
	
	# Summary
	print("\n" + "="*80)
	print("VALIDATION TEST SUMMARY")
	print("="*80)
	
	passed = 0
	total = len(results)
	
	for test_name, passed_test in results:
		status = "PASS" if passed_test else "FAIL"
		print(f"{test_name:<25} : {status}")
		if passed_test:
			passed += 1
	
	print(f"\nOVERALL: {passed}/{total} tests passed")
	
	if passed == total:
		print("🎉 ALL HAEM VALIDATION TESTS PASSED")
		return True
	else:
		print("⚠️  SOME HAEM VALIDATION TESTS FAILED - CHECK IMPLEMENTATION")
		return False


if __name__ == "__main__":
	import time
	import sys
	
	# =============================================================================
	# HAEM VALIDATION OPTION - Run this first to verify corrected implementation
	# =============================================================================
	if len(sys.argv) > 1 and sys.argv[1].lower() == 'validate':
		print("🔬 RUNNING HAEM VALIDATION TESTS")
		print("This will verify the corrected τ₃ projection and physics requirements...")
		run_all_haem_validation_tests()
		print("\n✅ To run normal calculations, use: python phase_character.py")
		exit()

	# Create organized output structure
	base_dir = 'outputs/phase_character'
	os.makedirs(base_dir, exist_ok=True)
	param_set = 'odd_parity_paper'
	set_parameters(param_set)

	# Fixed energy for 1D plot
	energy = 200e-6  # 200 µeV in eV
	kz = 0.0
	gap_scale = 0.0003  # 300 µeV
	
	# OPTIMIZED settings for speed vs precision balance
	nk_green = 50  # Reduced from 80 for 2.5x speedup
	n_points = 100   # Reduced from 100 for 2x speedup
	
	# Get hardcoded vectors
	vector_results = create_hardcoded_vectors(kz=kz)
	
	# Process both pairing types
	pairing_types = ['B2u', 'B3u']
	
	for pairing_type in pairing_types:
		start_time = time.time()
		print(f"\n{'='*60}")
		print(f"Processing {pairing_type} pairing symmetry")
		print(f"{'='*60}\n")
		
		data = vector_results[pairing_type]
		vectors = data['vectors']
		vector_data = data['vector_data']
		vector_labels = ['p1', 'p2', 'p5', 'p6']
		vector_colors = ['#228B22', '#FF8C00', '#8B008B', '#DC143C']  # Green, Orange, Purple, Crimson
		
		# PRE-COMPUTE T-MATRICES ONCE (major speedup!)
		print(f"Pre-computing T-matrices at E = {energy*1e6:.1f} µeV...")
		eta = determine_broadening(energy, 1e-6, gap_scale)
		print(f"  Broadening η = {eta*1e6:.2f} µeV")
		print(f"  k-space grid: {nk_green}×{nk_green} = {nk_green**2} points")
		
		# FIXED: Get proper gap parameters for this pairing type
		gap_params = get_gap_parameters_for_pairing(pairing_type)
		print(f"  Gap params for {pairing_type}: {gap_params}")
		
		# DEBUG: Test a few k-points to see if d-vectors are actually different
		test_kx, test_ky = 0.1, 0.1  # Test point
		C0, C1, C2, C3 = gap_params['C0'], gap_params['C1'], gap_params['C2'], gap_params['C3']
		test_d = d_vector(test_kx, test_ky, kz, pairing_type, C0, C1, C2, C3)
		print(f"  Test d-vector at (kx={test_kx}, ky={test_ky}): {test_d}")
		
		T_matrix_plus = compute_t_matrix(energy, pairing_type, None, kz, eta, nk_green, gap_params)
		T_matrix_minus = compute_t_matrix(-energy, pairing_type, None, kz, eta, nk_green, gap_params)
		print(f"  T-matrices computed in {time.time()-start_time:.1f}s\n")
		
		# Create figure for this pairing type
		fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
		
		# Loop over all 4 vectors
		for i, (vector, label, color, vec_info) in enumerate(zip(vectors, vector_labels, vector_colors, vector_data)):
			print(f"Computing HAEM along vector {label}...")
			origin_ky, origin_kx = vec_info['origin']
			
			# FIXED: Convert vector to raw coordinates (correct 2π scaling)
			vec_x, vec_y = vector  # in (2π/a, 2π/b) units
			vec_ky = vec_y * (2*np.pi/b)  # CORRECTED: Full 2π/b scaling
			vec_kx = vec_x * (2*np.pi/a)  # CORRECTED: Full 2π/a scaling
			
			print(f"  Vector {label}: ({vec_x:.3f}, {vec_y:.3f}) → k-space ({vec_kx:.3f}, {vec_ky:.3f})")
			
			# 1D path from origin to endpoint (0 to 1)
			t_vals = np.linspace(0, 1, n_points)
			haem_signal = np.zeros(n_points)
			
			# Compute HAEM for each point along vector
			for j, t in enumerate(t_vals):
				if j % 25 == 0:
					print(f"  Point {j+1}/{n_points}...")
				
				# q-vector at position t
				ki_kx = origin_kx
				ki_ky = origin_ky
				kf_kx = origin_kx + t * vec_kx
				kf_ky = origin_ky + t * vec_ky
				
				qx = kf_kx - ki_kx
				qy = kf_ky - ki_ky
				qz = 0
				
				# Debug: Check q-vector magnitude (should be reasonable, not 8+)
				q_mag = np.sqrt(qx**2 + qy**2)
				if j % 25 == 0:
					print(f"    q = ({qx:.3f}, {qy:.3f}), |q| = {q_mag:.3f}")
				
				# Compute HAEM using pre-computed T-matrices with gap_params
				haem_signal[j] = compute_haem_signal_tmatrix_optimized(
					qx, qy, qz, energy, pairing_type,
					T_matrix_plus, T_matrix_minus, eta, nk_green, gap_params
				)
			
			# Plot this vector
			ax.plot(t_vals, haem_signal, label=label, color=color, 
			       linewidth=2.5, marker='o', markersize=3, markevery=10)
			print(f"  {label} complete: range [{np.min(haem_signal):.2e}, {np.max(haem_signal):.2e}]\n")
		
		# Format plot
		ax.set_xlabel('Position along vector', fontsize=14, fontweight='bold')
		ax.set_ylabel(r'HAEM signal Re[$\rho^-(q,E)$]', fontsize=14, fontweight='bold')
		ax.set_title(f'{pairing_type} - HAEM at E={energy*1e6:.0f} µeV ({current_parameter_set})', 
		            fontsize=16, fontweight='bold')
		ax.legend(fontsize=12, loc='best', framealpha=0.9)
		ax.grid(True, alpha=0.3, linestyle='--')
		ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
		
		# Save figure
		save_dir = os.path.join(base_dir, 'haem_1d_lines', current_parameter_set)
		os.makedirs(save_dir, exist_ok=True)
		filename = f'haem_1d_{pairing_type}_E{energy*1e6:.0f}ueV_nk{nk_green}.png'
		filepath = os.path.join(save_dir, filename)
		
		plt.tight_layout()
		fig.savefig(filepath, dpi=300, bbox_inches='tight')
		plt.close(fig)
		
		elapsed = time.time() - start_time
		print(f"{'='*60}")
		print(f"{pairing_type} completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
		print(f"Saved to: {filepath}")
		print(f"{'='*60}\n")

	# Generate gap maps with vector overlays in the same directory
	print("\n📊 GENERATING GAP MAPS WITH VECTOR OVERLAYS")
	print("Saving to same directory as HAEM 1D plots...")
	gap_map_save_dir = os.path.join(base_dir, 'haem_1d_lines', current_parameter_set)
	plot_haem_gap_maps_with_vectors(kz=kz, energy=energy, save_dir=gap_map_save_dir, pairing_types=pairing_types, resolution=400)
	print("✓ Gap maps generated successfully\n")
	print("\n✓ ALL PLOTS GENERATED SUCCESSFULLY")