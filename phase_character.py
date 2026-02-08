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
		return np.array([
			C1 * np.sin(kz * c),
			C0 * np.sin(kx * a) * np.sin(ky * b) * np.sin(kz * c),
			C3 * np.sin(kx * a)
		])
	elif pairing_type == 'B3u':
		return np.array([
			C0 * np.sin(kx * a) * np.sin(ky * b) * np.sin(kz * c),
			C2 * np.sin(kz * c),
			C3 * np.sin(ky * b)
		])
	else:
		raise ValueError("Only B2u and B3u supported")

def gap_matrix_from_dvec(d):
	"""Convert d-vector to 2x2 gap matrix in spin space
	
	Args:
		d: 3-component d-vector (dx, dy, dz)
		
	Returns:
		2x2 complex gap matrix
	"""
	dx, dy, dz = d
	return np.array([
		[-dx + 1j*dy,  dz],
		[ dz,         dx + 1j*dy]
	], dtype=complex)

def construct_BdG_hamiltonian(kx, ky, kz, pairing_type, gap_params=None):
	"""Construct Bogoliubov-de Gennes Hamiltonian with superconducting pairing
	
	Args:
		kx, ky, kz: k-space coordinates
		pairing_type: Pairing symmetry ('B2u' or 'B3u')
		gap_params: Gap parameters (C1, C2, C3)
		
	Returns:
		8x8 BdG Hamiltonian matrix
	"""
	if gap_params is None:
		gap_params = {'C1': 0.0003, 'C2': 0.0003, 'C3': 0.0003}
	
	# Get normal-state 4x4 Hamiltonian
	H_normal = H_full(kx, ky, kz)
	
	# Calculate d-vector for spin-triplet pairing
	C0 = gap_params.get('C0', 0.0)
	C1 = gap_params.get('C1', 0.0003)
	C2 = gap_params.get('C2', 0.0003)
	C3 = gap_params['C3']
	
	d = d_vector(kx, ky, kz, pairing_type, C0, C1, C2, C3)
	
	# Convert d-vector to 2x2 gap matrix in spin space
	gap_2x2 = gap_matrix_from_dvec(d)
	
	# Embed 2x2 spin-space gap matrix into 4x4 orbital-spin structure
	# Assuming bands 0,1 and 2,3 are two orbital pairs with spin structure
	Delta_matrix = np.zeros((4, 4), dtype=complex)
	
	# Apply spin-triplet pairing to first orbital pair (bands 0,1)
	Delta_matrix[0:2, 0:2] = gap_2x2
	
	# Apply same pairing to second orbital pair (bands 2,3)
	Delta_matrix[2:4, 2:4] = gap_2x2
	
	# Construct 8x8 BdG Hamiltonian
	H_BdG = np.zeros((8, 8), dtype=complex)
	
	# Particle block (upper-left)
	H_BdG[:4, :4] = H_normal
	
	# Hole block (lower-right) 
	H_BdG[4:, 4:] = -np.conj(H_normal)
	
	# Pairing blocks
	H_BdG[:4, 4:] = Delta_matrix
	H_BdG[4:, :4] = np.conj(Delta_matrix.T)
	
	return H_BdG

def construct_BdG_hamiltonian_vectorized(kx_array, ky_array, kz, pairing_type, gap_params=None):
	"""Vectorized construction of BdG Hamiltonians for multiple k-points
	
	Args:
		kx_array, ky_array: Arrays of k-space coordinates
		kz: kz value 
		pairing_type: Pairing symmetry ('B2u' or 'B3u')
		gap_params: Gap parameters
		
	Returns:
		Array of 8x8 BdG Hamiltonian matrices
	"""
	if gap_params is None:
		gap_params = {'C1': 0.0003, 'C2': 0.0003, 'C3': 0.0003}
	
	# Flatten for vectorized operations
	kx_flat = kx_array.flatten()
	ky_flat = ky_array.flatten()
	n_points = len(kx_flat)
	
	# Pre-allocate result array
	H_BdG_array = np.zeros((n_points, 8, 8), dtype=complex)
	
	for i, (kx, ky) in enumerate(zip(kx_flat, ky_flat)):
		H_BdG_array[i] = construct_BdG_hamiltonian(kx, ky, kz, pairing_type, gap_params)
	
	return H_BdG_array.reshape(kx_array.shape + (8, 8))

def green_function_k_vectorized(kx_array, ky_array, kz, energy, pairing_type, eta=1e-6, gap_params=None):
	"""Vectorized computation of Green's functions for multiple k-points - OPTIMIZED
	
	Args:
		kx_array, ky_array: Arrays of k-space coordinates
		kz: kz value
		energy: Energy (in eV)
		pairing_type: Pairing symmetry
		eta: Broadening parameter
		gap_params: Gap parameters
		
	Returns:
		Array of 8x8 Green's function matrices
	"""
	# Use adaptive broadening
	eta = determine_broadening(energy, eta)
	
	# Handle E=0 case
	energy_eff = energy if abs(energy) >= 1e-12 else 1e-5
	
	# Get vectorized Hamiltonians
	H_BdG_array = construct_BdG_hamiltonian_vectorized(kx_array, ky_array, kz, pairing_type, gap_params)
	
	# Flatten arrays for batch processing
	original_shape = kx_array.shape
	H_flat = H_BdG_array.reshape(-1, 8, 8)
	n_points = len(H_flat)
	
	# Pre-allocate result array
	G_array = np.zeros((n_points, 8, 8), dtype=complex)
	
	# Vectorized computation using broadcasting
	E_matrix = (energy_eff + 1j * eta) * np.eye(8)
	
	for i in range(n_points):
		try:
			G_array[i] = np.linalg.inv(E_matrix - H_flat[i])
		except np.linalg.LinAlgError:
			# Handle singular matrices
			regularization = max(eta * 0.1, 1e-8) * np.eye(8)
			try:
				G_array[i] = np.linalg.inv(E_matrix - H_flat[i] + regularization)
			except np.linalg.LinAlgError:
				G_array[i] = np.linalg.pinv(E_matrix - H_flat[i])
	
	return G_array.reshape(original_shape + (8, 8))
	"""Vectorized computation of Green's functions for multiple k-points - OPTIMIZED
	
	Args:
		kx_array, ky_array: Arrays of k-space coordinates
		kz: kz value
		energy: Energy (in eV)
		pairing_type: Pairing symmetry
		eta: Broadening parameter
		gap_params: Gap parameters
		
	Returns:
		Array of 8x8 Green's function matrices
	"""
	# Use adaptive broadening
	eta = determine_broadening(energy, eta)
	
	# Handle E=0 case
	energy_eff = energy if abs(energy) >= 1e-12 else 1e-5
	
	# Get vectorized Hamiltonians
	H_BdG_array = construct_BdG_hamiltonian_vectorized(kx_array, ky_array, kz, pairing_type, gap_params)
	
	# Flatten arrays for batch processing
	original_shape = kx_array.shape
	H_flat = H_BdG_array.reshape(-1, 8, 8)
	n_points = len(H_flat)
	
	# Pre-allocate result array
	G_array = np.zeros((n_points, 8, 8), dtype=complex)
	
	# Vectorized computation using broadcasting
	E_matrix = (energy_eff + 1j * eta) * np.eye(8)
	
	for i in range(n_points):
		try:
			G_array[i] = np.linalg.inv(E_matrix - H_flat[i])
		except np.linalg.LinAlgError:
			# Handle singular matrices
			regularization = max(eta * 0.1, 1e-8) * np.eye(8)
			try:
				G_array[i] = np.linalg.inv(E_matrix - H_flat[i] + regularization)
			except np.linalg.LinAlgError:
				G_array[i] = np.linalg.pinv(E_matrix - H_flat[i])
	
	return G_array.reshape(original_shape + (8, 8))

def determine_broadening(energy, base_eta=1e-6, gap_scale=0.0003):
	"""Determine appropriate broadening parameter based on gap scale
	
	Args:
		energy: Energy (in eV)
		base_eta: Minimum broadening
		gap_scale: Superconducting gap scale Δ (in eV)
		
	Returns:
		Broadening η ≈ 0.05-0.1 × Δ
	"""
	# Use gap scale for broadening: η ≈ 0.075 × Δ (midpoint of 0.05-0.1 range)
	return max(0.075 * gap_scale, base_eta)

def green_function_k(kx, ky, kz, energy, pairing_type, eta=1e-6, gap_params=None):
	"""Compute momentum-resolved Green's function G^0_k(E)
	
	Args:
		kx, ky, kz: k-space coordinates
		energy: Energy (in eV)
		pairing_type: Pairing symmetry
		eta: Broadening parameter
		gap_params: Gap parameters
		
	Returns:
		8x8 Green's function matrix
	"""
	# Use adaptive broadening
	eta = determine_broadening(energy, eta)
		
	# Handle E=0 case with better numerical stability  
	energy_eff = energy
	if abs(energy) < 1e-12:
		energy_eff = 1e-5  # Use 10 µeV instead of tiny value
		
	H_BdG = construct_BdG_hamiltonian(kx, ky, kz, pairing_type, gap_params)
	E_matrix = (energy_eff + 1j * eta) * np.eye(8)
	
	try:
		G_k = np.linalg.inv(E_matrix - H_BdG)
	except np.linalg.LinAlgError:
		# Handle singular matrices with adaptive regularization
		regularization = max(eta * 0.1, 1e-8) * np.eye(8)
		try:
			G_k = np.linalg.inv(E_matrix - H_BdG + regularization)
		except np.linalg.LinAlgError:
			# Last resort: use pseudo-inverse
			G_k = np.linalg.pinv(E_matrix - H_BdG)
	
	return G_k

def compute_fermi_surface_mask(kx_array, ky_array, kz, eta_cutoff):
	"""Identify k-points on Fermi surface for focused summation
	
	Args:
		kx_array, ky_array: k-space coordinate arrays
		kz: kz value
		eta_cutoff: Energy cutoff (typically 4*eta or 0.002 eV)
		
	Returns:
		Boolean mask indicating Fermi surface states
	"""
	# Flatten arrays
	kx_flat = kx_array.flatten()
	ky_flat = ky_array.flatten()
	n_points = len(kx_flat)
	
	# Compute normal-state energies for all k-points
	fermi_mask_flat = np.zeros(n_points, dtype=bool)
	
	for i, (kx, ky) in enumerate(zip(kx_flat, ky_flat)):
		H_normal = H_full(kx, ky, kz)
		eigenvalues = np.linalg.eigvalsh(H_normal)
		# Check if any band is near Fermi level
		if np.any(np.abs(eigenvalues) < eta_cutoff):
			fermi_mask_flat[i] = True
	
	return fermi_mask_flat.reshape(kx_array.shape)

def compute_local_green_function_vectorized(energy, pairing_type, kz=0.0, eta=1e-6, nk=50, gap_params=None):
	"""Vectorized computation of local Green's function - MUCH FASTER
	
	Args:
		energy: Energy (in eV)
		pairing_type: Pairing symmetry
		kz: kz value for 2D slice
		eta: Broadening parameter
		nk: k-space sampling density
		gap_params: Gap parameters
		
	Returns:
		8x8 local Green's function matrix
	"""
	# Use original k-space grid (asymmetric in ky as user intended)
	kx_vals = np.linspace(-np.pi/a, np.pi/a, nk)
	ky_vals = np.linspace(-3*np.pi/b, np.pi/b, nk)
	
	# Create meshgrid for vectorized computation
	KX, KY = np.meshgrid(kx_vals, ky_vals, indexing='ij')
	
	# Vectorized Green's function computation
	G_array = green_function_k_vectorized(KX, KY, kz, energy, pairing_type, eta, gap_params)
	
	# Sum over all k-points and normalize
	G_local = np.mean(G_array, axis=(0, 1))  # More numerically stable than sum + divide
	
	return G_local

def compute_t_matrix(energy, pairing_type, V_imp=None, kz=0.0, eta=1e-6, nk=50, gap_params=None):
	"""Compute T-matrix T(E) = (1 - V_imp G_0(E))^-1 V_imp
	
	Args:
		energy: Energy (in eV)
		pairing_type: Pairing symmetry
		V_imp: Impurity potential strength (in eV) - if None, auto-scale to energy
		kz: kz value for 2D slice
		eta: Broadening parameter
		nk: k-space sampling density
		gap_params: Gap parameters
		
	Returns:
		8x8 T-matrix
	"""
	G_local = compute_local_green_function_vectorized(energy, pairing_type, kz, eta, nk, gap_params)
	
	# Auto-scale impurity potential to energy scale for stable weak scattering 
	if V_imp is None:
		# Use very small fraction of energy scale for stable numerics
		energy_scale = max(abs(energy), 1e-6)  # At least 1 µeV
		# Much weaker: 0.5% instead of 2% to avoid numerical issues
		V_imp = min(max(0.005 * energy_scale, 0.2e-6), 0.5e-3)  # 0.2-500 µeV range
	else:
		# Check if provided V_imp is reasonable
		energy_scale = max(abs(energy), 1e-6)
		if V_imp > 50 * energy_scale:
			print(f"  WARNING: V_imp={V_imp*1e3:.1f} meV >> E={energy*1e6:.0f} µeV (near unitary limit)")
	
	# Assume scalar impurity potential (non-magnetic)
	V_matrix = V_imp * np.eye(8)
	
	# Compute T-matrix: T = (1 - V G_0)^-1 V with regularization
	identity = np.eye(8)
	matrix_to_invert = identity - V_matrix @ G_local
	
	try:
		# Add small regularization for numerical stability
		cond_num = np.linalg.cond(matrix_to_invert)
		if cond_num > 1e12:  # Ill-conditioned
			regularization = 1e-10 * np.eye(8)
			matrix_to_invert += regularization
			
		T_matrix = np.linalg.inv(matrix_to_invert) @ V_matrix
	except np.linalg.LinAlgError:
		# Handle singular matrices with pseudoinverse
		T_matrix = np.linalg.pinv(matrix_to_invert) @ V_matrix
	
	return T_matrix

def compute_ldos_perturbation_optimized_vectorized(qx, qy, qz, energy, pairing_type, T_matrix, eta=1e-6, nk=50, gap_params=None, use_fs_filter=False):
	"""FULLY VECTORIZED LDOS perturbation calculation (full BZ by default)
	
	Args:
		qx, qy, qz: Scattering wavevector
		energy: Energy (in eV)
		pairing_type: Pairing symmetry
		T_matrix: Pre-computed T-matrix for this energy
		eta: Broadening parameter
		nk: k-space sampling density
		gap_params: Gap parameters
		use_fs_filter: If True, restrict to Fermi surface states (disabled by default)
		
	Returns:
		Complex LDOS perturbation δN(q,E)
	"""
	# Use original k-space grid consistent with user's setup
	kx_vals = np.linspace(-np.pi/a, np.pi/a, nk)
	ky_vals = np.linspace(-3*np.pi/b, np.pi/b, nk)
	
	# For small q-vectors, use more stable computation
	q_magnitude = np.sqrt(qx**2 + qy**2)
	if q_magnitude < 1e-8:
		return 1e-10 + 0j
	
	# Create meshgrids for vectorized computation
	KX, KY = np.meshgrid(kx_vals, ky_vals, indexing='ij')
	
	# Compute Fermi surface mask for focused summation
	if use_fs_filter:
		eta_cutoff = 4.0 * eta  # Looser cutoff: |ε_k| < 4η for enough states
		fs_mask = compute_fermi_surface_mask(KX, KY, qz, eta_cutoff)
		if not np.any(fs_mask):
			return 0.0 + 0j
	else:
		fs_mask = np.ones(KX.shape, dtype=bool)
	
	# Vectorized Green's function computations (only for FS states)
	G_k_array = green_function_k_vectorized(KX, KY, qz, energy, pairing_type, eta, gap_params)
	G_kq_array = green_function_k_vectorized(KX + qx, KY + qy, qz, energy, pairing_type, eta, gap_params)
	
	# Check for valid values
	valid_mask = (fs_mask & 
	             ~np.any(np.isnan(G_k_array) | np.isinf(G_k_array), axis=(2, 3)) & 
	             ~np.any(np.isnan(G_kq_array) | np.isinf(G_kq_array), axis=(2, 3)))
	
	if not np.any(valid_mask):
		return 0.0 + 0j
	
	# Vectorized matrix multiplication: G_k @ T_matrix @ G_kq
	# Use Einstein summation for efficient batched matrix multiplication
	contributions = np.zeros(KX.shape, dtype=complex)
	
	for i in range(KX.shape[0]):
		for j in range(KX.shape[1]):
			if valid_mask[i, j]:
				# G_k T G_{k+q} contribution
				temp = G_k_array[i, j] @ T_matrix
				contribution = temp @ G_kq_array[i, j]
				# Take trace of particle sector (upper-left 4x4 block)
				particle_block = contribution[:4, :4]
				contributions[i, j] = np.trace(particle_block)
	
	# Sum valid contributions and normalize
	valid_contributions = contributions[valid_mask]
	if len(valid_contributions) > 0:
		delta_N = np.mean(valid_contributions)
	else:
		delta_N = 0.0 + 0j
	
	delta_N = (1.0 / np.pi) * np.imag(delta_N)
	
	return delta_N

def compute_ldos_perturbation_optimized(qx, qy, qz, energy, pairing_type, T_matrix, eta=1e-6, nk=50, gap_params=None):
	"""Optimized LDOS perturbation calculation using pre-computed T-matrix
	
	Args:
		qx, qy, qz: Scattering wavevector
		energy: Energy (in eV)
		pairing_type: Pairing symmetry
		T_matrix: Pre-computed T-matrix for this energy
		eta: Broadening parameter
		nk: k-space sampling density
		gap_params: Gap parameters
		
	Returns:
		Complex LDOS perturbation δN(q,E)
	"""
	# Use vectorized version for much better performance
	return compute_ldos_perturbation_optimized_vectorized(qx, qy, qz, energy, pairing_type, T_matrix, eta, nk, gap_params)

def compute_ldos_perturbation(qx, qy, qz, energy, pairing_type, V_imp=0.1, eta=1e-6, nk=50, gap_params=None):
	"""Legacy wrapper for compatibility - computes T-matrix each time"""
	T_matrix = compute_t_matrix(energy, pairing_type, V_imp, qz, eta, nk, gap_params)
	return compute_ldos_perturbation_optimized(qx, qy, qz, energy, pairing_type, T_matrix, eta, nk, gap_params)

def compute_haem_signal_tmatrix_optimized(qx, qy, qz, energy, pairing_type, T_matrix_plus, T_matrix_minus, eta=1e-6, nk=30, gap_params=None):
	"""Optimized HAEM signal using pre-computed T-matrices
	
	Args:
		qx, qy, qz: Scattering wavevector  
		energy: Energy (in eV)
		pairing_type: Pairing symmetry
		T_matrix_plus: Pre-computed T-matrix for +E
		T_matrix_minus: Pre-computed T-matrix for -E
		eta: Broadening parameter
		nk: k-space sampling density
		gap_params: Gap parameters
		
	Returns:
		Real HAEM signal ρ⁻(q,E) = Re[δN(q,+E) - δN(q,-E)]
		
	Validation: The antisymmetrization ρ(+E) - ρ(-E) correctly accesses:
		- δN(q,+E): Particle sector (upper-left 4×4 block) for positive energies
		- δN(q,-E): Hole sector (lower-right 4×4 block mapped to negative E) 
		The 8×8 BdG structure ensures proper particle-hole antisymmetry
	"""
	# Compute LDOS perturbations at +E and -E using pre-computed T-matrices
	delta_N_plus = compute_ldos_perturbation_optimized(qx, qy, qz, energy, pairing_type, T_matrix_plus, eta, nk, gap_params)
	delta_N_minus = compute_ldos_perturbation_optimized(qx, qy, qz, -energy, pairing_type, T_matrix_minus, eta, nk, gap_params)
	
	# Energy-antisymmetrized HAEM signal
	haem_signal = np.real(delta_N_plus - delta_N_minus)
	
	return haem_signal

def compute_haem_signal_energy_integrated(qx, qy, qz, energy_center, pairing_type, eta=1e-6, nk=30, gap_params=None, n_energies=7, gap_scale=0.0003):
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
		
		# Compute T-matrices for this energy
		T_matrix_plus = compute_t_matrix(energy, pairing_type, None, qz, eta_this, nk, gap_params)
		T_matrix_minus = compute_t_matrix(-energy, pairing_type, None, qz, eta_this, nk, gap_params)
		
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
		
	Note: Uses literature coordinates in (2π/a, 2π/b) units
	Vectors: p1=(0.29,0), p2=(0.43,1), p5=(-0.14,1), p6=(0.57,0)
	"""
	vector_labels = ['p1', 'p2', 'p5', 'p6']
	# Vectors in (2pi/a, 2pi/b) units from literature
	vectors_base = [
		(0.29, 0.0),      # p1
		(0.43, 1.0),      # p2
		(-0.14, 1.0),     # p5
		(0.57, 0.0)       # p6
	]
	
	# Known Fermi surface origins (empirically determined)
	origin_main = (-2.0 * np.pi/b, 0.0)  # ky = -2π/b, kx ≈ 0 for p1,p2,p5
	origin_p6 = (0.0, 0.0)  # ky = 0, kx ≈ 0 for p6
	
	results = {}
	for pairing_type in ['B2u', 'B3u']:
		vector_data = []
		vectors = []
		
		for i, (vec_x_init, vec_y_init) in enumerate(vectors_base):
			# Choose origin
			if i == 3:  # p6
				origin_ky, origin_kx = origin_p6
			else:  # p1, p2, p5
				origin_ky, origin_kx = origin_main
			
			# Use scale factor of 1.0 (vectors already correct)
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

	# Vectors in (2pi/a, 2pi/b) units - these will be adjusted to end on Fermi surface
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

def plot_haem_sign_2d(kz=0.0, pairing_types=['B2u', 'B3u'], base_save_dir='outputs/phase_character', resolution=300, 
                     parameter_set=None, energy=0.001, eta=1e-6):
	"""Plot gap function as background with Fermi surface overlay and vector origins
	
	Args:
		kz: kz value for 2D slice
		pairing_types: List of pairing symmetries to plot
		base_save_dir: Base directory for organized output structure
		resolution: Grid resolution
		parameter_set: Parameter set to use (if None, uses current setting)
		energy: Energy for HAEM calculation (in eV) - used for filename
		eta: Regularization parameter - used for filename
	"""
	if parameter_set is not None:
		set_parameters(parameter_set)
		print(f"Using parameter set: {parameter_set}")
	else:
		print(f"Using current parameter set: {current_parameter_set}")
	
	os.makedirs(base_save_dir, exist_ok=True)
	
	# Create organized subfolder structure with energy subfolder
	energy_folder = f'E_{energy*1e6:.0f}ueV'
	save_dir = os.path.join(base_save_dir, 'gap_maps', current_parameter_set, energy_folder)
	os.makedirs(save_dir, exist_ok=True)
	
	# Get precomputed vector data
	vector_results = compute_vectors_on_fermi_surface(kz=kz, resolution=resolution)
	
	# Use first result's grid data since they're all the same
	first_result = list(vector_results.values())[0]
	kx_vals = first_result['kx_vals']
	ky_vals = first_result['ky_vals'] 
	energies_fs = first_result['energies_fs']
	KX, KY = np.meshgrid(kx_vals, ky_vals, indexing='ij')

	for pairing_type in pairing_types:
		if pairing_type not in vector_results:
			continue
			
		# Compute gap function for background (original approach)
		gap = gap_function(KX, KY, kz, pairing_type)
		gap_meV = gap * 1e3  # Convert eV to meV
		
		from matplotlib.colors import ListedColormap
		base_cmap = plt.get_cmap('RdBu_r')
		colors = base_cmap(np.linspace(0, 1, 256))
		colors[128] = [1, 1, 1, 1]
		custom_cmap = ListedColormap(colors)

		fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
		# Use gap function for background intensity with original coordinate system
		im = ax.imshow(gap_meV, extent=[-3, 1, -1, 1],  # Original ky:-3π/b to +π/b, kx:-π/a to +π/a
		              origin='lower', cmap=custom_cmap, vmin=-np.max(np.abs(gap_meV)), vmax=np.max(np.abs(gap_meV)))
		# Overlay Fermi surface contours for bands 2 and 3 with normalized coordinates
		fs_colors = ['#9635E5', '#FD0000']
		for band, color in zip([2, 3], fs_colors):
			Z = energies_fs[:, :, band]
			if Z.min() <= 0 <= Z.max():
				# Convert coordinates to original units (ky: -3 to 1, kx: -1 to 1)
				KY_norm = KY / (np.pi/b)
				KX_norm = KX / (np.pi/a)
				ax.contour(KY_norm, KX_norm, Z, levels=[0], colors=[color], linewidths=2, alpha=1.0, zorder=8)
		# Get vector data for this pairing type
		data = vector_results[pairing_type]
		vectors = data['vectors']
		vector_data = data['vector_data']
		
		# Overlay vectors with proper coordinate conversion
		# vectors contain final scaled vectors in (2π/a, 2π/b) units
		vector_labels = ['p1', 'p2', 'p5', 'p6']
		vector_colors = ['#228B22', '#FF8C00', '#8B008B', '#000000']  # Forest Green, Dark Orange, Dark Magenta, Black
		
		for i, (vector, vec_info) in enumerate(zip(vectors, vector_data)):
			origin_ky, origin_kx = vec_info['origin']
			
			# Convert vector from (2π/a, 2π/b) units to raw momentum coordinates
			vec_x, vec_y = vector  # in (2π/a, 2π/b) units
			vec_ky_raw = vec_y * 2 * (np.pi/b)  # convert to raw ky
			vec_kx_raw = vec_x * 2 * (np.pi/a)  # convert to raw kx
			
			# Convert everything to normalized display coordinates
			origin_ky_norm = origin_ky / (np.pi/b)
			origin_kx_norm = origin_kx / (np.pi/a)
			vec_ky_norm = vec_ky_raw / (np.pi/b)
			vec_kx_norm = vec_kx_raw / (np.pi/a)
			
			ax.arrow(origin_ky_norm, origin_kx_norm, vec_ky_norm, vec_kx_norm, 
					head_width=0.08, head_length=0.12, 
					fc=vector_colors[i], ec=vector_colors[i], linewidth=2.5, 
					length_includes_head=True, zorder=20)
			ax.text(origin_ky_norm + vec_ky_norm*1.15, origin_kx_norm + vec_kx_norm*1.15, 
			       vector_labels[i], color=vector_colors[i], fontsize=12, fontweight='bold', zorder=21)

		# Axis formatting - restore original coordinate system
		ax.set_xlim(-3, 1)  # ky in units of π/b (original asymmetric range)
		ax.set_ylim(-1, 1)  # kx in units of π/a
		ky_ticks = np.linspace(-3, 1, 5)
		kx_ticks = np.linspace(-1, 1, 5)
		ax.set_xticks(ky_ticks)
		ax.set_yticks(kx_ticks)  
		ax.set_xticklabels([f'{val:.1f}' for val in ky_ticks])
		ax.set_yticklabels([f'{val:.1f}' for val in kx_ticks])
		ax.set_xlabel(r'$k_y$ ($\pi/b$)', fontsize=14)
		ax.set_ylabel(r'$k_x$ ($\pi/a$)', fontsize=14)
		ax.set_title(f'{pairing_type} gap with vectors at $k_z$={kz:.2f} ({current_parameter_set})\nHAEM energy: ±{energy*1e6:.0f} µeV', fontsize=16)
		ax.grid(False)
		cbar = plt.colorbar(im, ax=ax, shrink=0.8, format='%.1e')
		cbar.set_label(r'$\Delta_k$ (meV)', fontsize=12)
		
		plt.tight_layout()
		filename = f'gap_with_haem_vectors_{pairing_type}_kz_{kz:.3f}_res_{resolution}.png'
		filepath = os.path.join(save_dir, filename)
		fig.savefig(filepath, dpi=300, bbox_inches='tight')
		plt.close(fig)
		print(f"Saved gap map to: {filepath}")
	
	return vector_results



if __name__ == "__main__":
	import time
	
	# Create organized output structure
	base_dir = 'outputs/phase_character'
	os.makedirs(base_dir, exist_ok=True)
	param_set = 'odd_parity_paper'
	set_parameters(param_set)
	
	print("=== 1D HAEM Line Plot (Publication Quality) ===")
	print("CONFIGURATION:")
	print("  • Fixed energy: E = 200 µeV")
	print("  • Vectors: p1, p2, p5, p6 (literature coordinates)")
	print("  • Path sampling: 100 points per vector")
	print("  • k-space density: nk = 80 (high precision)")
	print("  • T-matrices pre-computed once per pairing type")
	print("  • Output: Single 2D line plot with all vectors\n")
	
	# Fixed energy for 1D plot
	energy = 200e-6  # 200 µeV in eV
	kz = 0.0
	gap_scale = 0.0003  # 300 µeV
	
	# High precision settings
	nk_green = 80  # High k-space density for smooth curves
	n_points = 100  # Points along each vector
	
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
		
		T_matrix_plus = compute_t_matrix(energy, pairing_type, None, kz, eta, nk_green)
		T_matrix_minus = compute_t_matrix(-energy, pairing_type, None, kz, eta, nk_green)
		print(f"  T-matrices computed in {time.time()-start_time:.1f}s\n")
		
		# Create figure for this pairing type
		fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
		
		# Loop over all 4 vectors
		for i, (vector, label, color, vec_info) in enumerate(zip(vectors, vector_labels, vector_colors, vector_data)):
			print(f"Computing HAEM along vector {label}...")
			origin_ky, origin_kx = vec_info['origin']
			
			# Convert vector to raw coordinates
			vec_x, vec_y = vector  # in (2π/a, 2π/b) units
			vec_ky = vec_y * 2 * (np.pi/b)
			vec_kx = vec_x * 2 * (np.pi/a)
			
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
				
				# Compute HAEM using pre-computed T-matrices
				haem_signal[j] = compute_haem_signal_tmatrix_optimized(
					qx, qy, qz, energy, pairing_type,
					T_matrix_plus, T_matrix_minus, eta, nk_green
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
	
	print("\n✓ ALL PLOTS GENERATED SUCCESSFULLY")