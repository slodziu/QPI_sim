import numpy as np
import matplotlib.pyplot as plt
import os
from UTe2_fixed import H_full, a, b, set_parameters, parameter_sets, current_parameter_set

def gap_function(kx, ky, kz, pairing_type, C1=0.0003, C2=0.0003, C3=0.0003):
	if pairing_type == 'B2u':
		# d ∝ (sin(kz·c), 0, sin(kx·a)), at kz=0: dz = C3*sin(kx*a)
		return C3 * np.sin(kx * a)
	elif pairing_type == 'B3u':
		# d ∝ (0, sin(kz·c), sin(ky·b)), at kz=0: dz = C3*sin(ky*b)
		return C3 * np.sin(ky * b)
	else:
		raise ValueError('Only B2u and B3u supported')

def plot_gap_along_vectors(kz=0.0, pairing_types=['B2u', 'B3u'], save_dir='outputs/phase_character', 
                          resolution=300, parameter_set=None, n_points=100):
	"""Plot gap values along each vector
	
	Args:
		kz: kz value for 2D slice
		pairing_types: List of pairing symmetries to plot
		save_dir: Directory to save plots
		resolution: Grid resolution for finding origins
		parameter_set: Parameter set to use (if None, uses current setting)
		n_points: Number of points to sample along each vector
	"""
	if parameter_set is not None:
		set_parameters(parameter_set)
		print(f"Using parameter set: {parameter_set}")
	else:
		print(f"Using current parameter set: {current_parameter_set}")
	
	os.makedirs(save_dir, exist_ok=True)
	nk = resolution
	kx_vals = np.linspace(-np.pi/a, np.pi/a, nk)
	ky_vals = np.linspace(-3*np.pi/b, np.pi/b, nk)

	# Compute Fermi surface energies to find origins
	energies_fs = np.zeros((nk, nk, 4))
	for ix, kx in enumerate(kx_vals):
		for jy, ky in enumerate(ky_vals):
			H = H_full(kx, ky, kz)
			eigvals = np.linalg.eigvals(H)
			energies_fs[ix, jy, :] = np.sort(np.real(eigvals))

	# Vectors in (2pi/a, 2pi/b) units
	vectors = [
		(0.130, 0.003),  # p1
		(0.374, 1.000),  # p2
		(-0.244, 1.000), # p5
		(0.619, 0.000)   # p6
	]
	vector_labels = ['p1', 'p2', 'p5', 'p6']
	vector_colors = ['#228B22', '#FF8C00', '#8B008B', '#000000']

	for pairing_type in pairing_types:
		# Find origins (same logic as main plotting function)
		target_ky = -2 * (np.pi/b)
		Z = energies_fs[:, :, 2]
		threshold = 0.005
		
		# Find origin at ky = -2π/b
		ky_mask = np.abs(ky_vals - target_ky) < 0.1 * (np.pi/b)
		ky_indices = np.where(ky_mask)[0]
		
		candidate_points = []
		for ky_idx in ky_indices:
			ky = ky_vals[ky_idx]
			for kx_idx, kx in enumerate(kx_vals):
				energy = Z[kx_idx, ky_idx]
				if abs(energy) < threshold:
					candidate_points.append((ky, kx, abs(energy)))
		
		origin_main = None
		if candidate_points:
			candidate_points.sort(key=lambda pt: (pt[2], abs(pt[1])))
			best_point = candidate_points[0]
			origin_main = (best_point[0], best_point[1])
		
		# Find p6 origin at ky = 0
		target_ky_p6 = 0.0
		ky_idx_exact_p6 = np.argmin(np.abs(ky_vals - target_ky_p6))
		ky_exact_p6 = ky_vals[ky_idx_exact_p6]
		
		candidate_points_p6 = []
		for kx_idx, kx in enumerate(kx_vals):
			energy = Z[kx_idx, ky_idx_exact_p6]
			if abs(energy) < threshold and kx < 0:
				candidate_points_p6.append((ky_exact_p6, kx, abs(energy)))
		
		origin_p6 = None
		if candidate_points_p6:
			candidate_points_p6.sort(key=lambda pt: (pt[2], abs(pt[1])))
			best_point_p6 = candidate_points_p6[0]
			origin_p6 = (best_point_p6[0], best_point_p6[1])
		
		if origin_main is None:
			print(f"[{pairing_type}] Could not find main origin, skipping vector plots")
			continue
			
		# Create 4-panel plot
		fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
		axes = axes.flatten()
		
		for i, (vector, label, color) in enumerate(zip(vectors, vector_labels, vector_colors)):
			# Choose origin
			if i == 3 and origin_p6:  # p6 vector
				origin_ky, origin_kx = origin_p6
			else:  # p1, p2, p5 vectors
				origin_ky, origin_kx = origin_main
			
			# Convert vector to raw coordinates
			vec_x, vec_y = vector  # in (2π/a, 2π/b) units
			vec_ky = vec_y * 2 * (np.pi/b)  # convert to raw ky
			vec_kx = vec_x * 2 * (np.pi/a)  # convert to raw kx
			
			# Sample points along vector (0 to 1)
			t_vals = np.linspace(0, 1, n_points)
			gap_vals = []
			
			for t in t_vals:
				ky_point = origin_ky + t * vec_ky
				kx_point = origin_kx + t * vec_kx
				gap_val = gap_function(kx_point, ky_point, kz, pairing_type)
				gap_vals.append(gap_val * 1000)  # Convert to meV
			
			# Plot
			axes[i].plot(t_vals, gap_vals, color=color, linewidth=2, marker='o', markersize=2)
			axes[i].set_xlabel('Position along vector', fontsize=12)
			axes[i].set_ylabel('Gap (meV)', fontsize=12)
			axes[i].set_title(f'{label} - {pairing_type}', fontsize=14, color=color)
			axes[i].grid(True, alpha=0.3)
			axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
		
		plt.suptitle(f'Gap along vectors - {pairing_type} ({current_parameter_set})', fontsize=16)
		plt.tight_layout()
		
		filename = f'gap_along_vectors_{pairing_type}_kz_{kz:.3f}_{current_parameter_set}.png'
		filepath = os.path.join(save_dir, filename)
		fig.savefig(filepath, dpi=300, bbox_inches='tight')
		plt.close(fig)
		print(f"Saved vector gap plot: {filepath}")

def plot_gap_sign_2d(kz=0.0, pairing_types=['B2u', 'B3u'], save_dir='outputs/phase_character', resolution=300, parameter_set=None):
	"""Plot gap sign with Fermi surface overlay and vector origins
	
	Args:
		kz: kz value for 2D slice
		pairing_types: List of pairing symmetries to plot
		save_dir: Directory to save plots
		resolution: Grid resolution
		parameter_set: Parameter set to use (if None, uses current setting)
	"""
	if parameter_set is not None:
		set_parameters(parameter_set)
		print(f"Using parameter set: {parameter_set}")
	else:
		print(f"Using current parameter set: {current_parameter_set}")
	
	os.makedirs(save_dir, exist_ok=True)
	nk = resolution
	kx_vals = np.linspace(-np.pi/a, np.pi/a, nk)
	ky_vals = np.linspace(-3*np.pi/b, np.pi/b, nk)
	KX, KY = np.meshgrid(kx_vals, ky_vals, indexing='ij')

	# Compute Fermi surface energies
	energies_fs = np.zeros((nk, nk, 4))
	for ix, kx in enumerate(kx_vals):
		for jy, ky in enumerate(ky_vals):
			H = H_full(kx, ky, kz)
			eigvals = np.linalg.eigvals(H)
			energies_fs[ix, jy, :] = np.sort(np.real(eigvals))


	for pairing_type in pairing_types:
		gap = gap_function(KX, KY, kz, pairing_type)
		gap_meV = gap * 1e3  # Convert eV to meV
		from matplotlib.colors import ListedColormap
		base_cmap = plt.get_cmap('RdBu_r')
		colors = base_cmap(np.linspace(0, 1, 256))
		colors[128] = [1, 1, 1, 1]
		custom_cmap = ListedColormap(colors)

		fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
		im = ax.imshow(gap_meV, extent=[ky_vals.min(), ky_vals.max(), kx_vals.min(), kx_vals.max()],
			      origin='lower', cmap=custom_cmap, vmin=-np.max(np.abs(gap_meV)), vmax=np.max(np.abs(gap_meV)))

		# Overlay Fermi surface contours for bands 2 and 3
		fs_colors = ['#9635E5', '#FD0000']
		contour_data = {}
		for band, color in zip([2, 3], fs_colors):
			Z = energies_fs[:, :, band]
			if Z.min() <= 0 <= Z.max():
				cs = ax.contour(KY, KX, Z, levels=[0], colors=[color], linewidths=2, alpha=1.0, zorder=8)
				# Store contour data for band 2 (electron-like, red contour)
				if band == 2:
					contour_data[band] = cs

		# Overlay vectors for both B2u and B3u
		# Vectors in (2pi/a, 2pi/b) units - updated coordinates
		vectors = [
			(0.130, 0.003),  # p1
			(0.374, 1.000),  # p2
			(-0.244, 1.000), # p5
			(0.619, 0.000)   # p6
		]
		vectors_plot = [(y*2, x*2) for (x, y) in vectors]
		
		# Find a point on the Fermi contour at ky = -2π/b
		target_ky = -2 * (np.pi/b)
		origin_point = None
		
		# Direct approach: find actual Fermi surface points near target_ky
		Z = energies_fs[:, :, 2]  # band 2 (electron-like, red contour)
		threshold = 0.005  # eV - very strict threshold to be on Fermi surface
		
		# Find ky indices near target
		ky_mask = np.abs(ky_vals - target_ky) < 0.1 * (np.pi/b)
		ky_indices = np.where(ky_mask)[0]
		
		candidate_points = []
		for ky_idx in ky_indices:
			ky = ky_vals[ky_idx]
			for kx_idx, kx in enumerate(kx_vals):
				energy = Z[kx_idx, ky_idx]
				if abs(energy) < threshold:  # On Fermi surface
					candidate_points.append((ky, kx, abs(energy)))
		
		print(f"[{pairing_type}] Found {len(candidate_points)} Fermi surface points near ky/(pi/b) = -2")
		
		if candidate_points:
			# Sort by energy (closest to zero first), then by distance from kx=0
			candidate_points.sort(key=lambda pt: (pt[2], abs(pt[1])))
			best_point = candidate_points[0]
			origin_point = (best_point[0], best_point[1])
			print(f"[{pairing_type}] Best point: energy = {best_point[2]:.6f} eV")
		else:
			print(f"[{pairing_type}] No Fermi surface points found, trying looser threshold")
			threshold = 0.02  # eV - looser threshold
			for ky_idx in ky_indices:
				ky = ky_vals[ky_idx]
				for kx_idx, kx in enumerate(kx_vals):
					energy = Z[kx_idx, ky_idx]
					if abs(energy) < threshold:
						candidate_points.append((ky, kx, abs(energy)))
			if candidate_points:
				candidate_points.sort(key=lambda pt: (pt[2], abs(pt[1])))
				best_point = candidate_points[0]
				origin_point = (best_point[0], best_point[1])
				print(f"[{pairing_type}] Looser search: found point with energy = {best_point[2]:.6f} eV")
		
		if origin_point:
			origin_plot_ky, origin_plot_kx = origin_point
			print(f"[{pairing_type}] Vector origin (ky, kx) = ({origin_plot_ky:.4f}, {origin_plot_kx:.4f}) [raw], (ky/(pi/b), kx/(pi/a)) = ({origin_plot_ky/(np.pi/b):.4f}, {origin_plot_kx/(np.pi/a):.4f})")
			
			# Find second origin for p4 vector at ky = 0 exactly, below kx axis
			target_ky_p4 = 0.0
			# Find the exact ky index closest to target
			ky_idx_exact_p4 = np.argmin(np.abs(ky_vals - target_ky_p4))
			ky_exact_p4 = ky_vals[ky_idx_exact_p4]
			
			candidate_points_p4 = []
			# Look only at the exact ky slice
			for kx_idx, kx in enumerate(kx_vals):
				energy = Z[kx_idx, ky_idx_exact_p4]
				if abs(energy) < threshold and kx < 0:  # On Fermi surface and below kx axis
					candidate_points_p4.append((ky_exact_p4, kx, abs(energy)))
			
			origin_p4 = None
			if candidate_points_p4:
				# Sort by energy first, then by distance from kx=0 (but staying negative)
				candidate_points_p4.sort(key=lambda pt: (pt[2], abs(pt[1])))
				best_point_p4 = candidate_points_p4[0]
				origin_p4 = (best_point_p4[0], best_point_p4[1])
				print(f"[{pairing_type}] p6 origin (ky, kx) = ({origin_p4[0]:.4f}, {origin_p4[1]:.4f}) [raw], (ky/(pi/b), kx/(pi/a)) = ({origin_p4[0]/(np.pi/b):.4f}, {origin_p4[1]/(np.pi/a):.4f})")
			
			# Vector labels and colors (p1, p2, p5, p6)
			vector_labels = ['p1', 'p2', 'p5', 'p6']
			vector_colors = ['#228B22', '#FF8C00', '#8B008B', '#000000']  # Forest Green, Dark Orange, Dark Magenta, Black
			
			for i, (dky, dkx) in enumerate(vectors_plot):
				# Use different origin for p6 (index 3, which is p6)
				if i == 3 and origin_p4:  # p6 vector
					origin_ky, origin_kx = origin_p4
				else:  # p1, p2, p5 vectors
					origin_ky, origin_kx = origin_plot_ky, origin_plot_kx
				
				# Use raw coordinates for arrow plotting with smaller heads and different colors
				ax.arrow(origin_ky, origin_kx, dky*(np.pi/b), dkx*(np.pi/a), 
						head_width=0.04*(np.pi/b), head_length=0.06*(np.pi/a), 
						fc=vector_colors[i], ec=vector_colors[i], linewidth=2, 
						length_includes_head=True, zorder=20)
				ax.text(origin_ky + dky*1.05*(np.pi/b), origin_kx + dkx*1.05*(np.pi/a), 
					   vector_labels[i], color=vector_colors[i], fontsize=12, fontweight='bold', zorder=21)
		else:
			print(f"[{pairing_type}] No Fermi contour point found near ky/(pi/b) = -2.")

		# Axis formatting
		ax.set_xlim(ky_vals.min(), ky_vals.max())
		ax.set_ylim(kx_vals.min(), kx_vals.max())
		ky_ticks = np.linspace(ky_vals.min(), ky_vals.max(), 5)
		kx_ticks = np.linspace(kx_vals.min(), kx_vals.max(), 5)
		ky_labels = [f'{val/(np.pi/b):.1f}' for val in ky_ticks]
		kx_labels = [f'{val/(np.pi/a):.1f}' for val in kx_ticks]
		ax.set_xticks(ky_ticks)
		ax.set_yticks(kx_ticks)
		ax.set_xticklabels(ky_labels)
		ax.set_yticklabels(kx_labels)
		ax.set_xlabel(r'$k_y$ ($\pi/b$)', fontsize=14)
		ax.set_ylabel(r'$k_x$ ($\pi/a$)', fontsize=14)
		ax.set_title(f'{pairing_type} gap sign at $k_z$={kz:.2f} ({current_parameter_set})', fontsize=16)
		ax.grid(False)
		cbar = plt.colorbar(im, ax=ax, shrink=0.8, format='%.1e')
		cbar.set_label(r'$\Delta_k$ (meV)', fontsize=12)

		plt.tight_layout()
		filename = f'gap_sign_{pairing_type}_kz_{kz:.3f}_{current_parameter_set}.png'
		filepath = os.path.join(save_dir, filename)
		fig.savefig(filepath, dpi=300, bbox_inches='tight')
		plt.close(fig)
		print(f"Saved: {filepath}")

if __name__ == "__main__":
	# You can specify a parameter set here, or leave as None to use current default
	# Available: 'DFT', 'QuantumOscillation', 'QPIFS', 'odd_parity_paper'
	plot_gap_sign_2d(kz=0.0, parameter_set='odd_parity_paper')
	
	# Plot gap values along vectors
	plot_gap_along_vectors(kz=0.0, parameter_set='odd_parity_paper')
	
	# Example: plot for multiple parameter sets
	# for param_set in ['DFT', 'QuantumOscillation', 'QPIFS', 'odd_parity_paper']:
	#     plot_gap_sign_2d(kz=0.0, parameter_set=param_set)
	#     plot_gap_along_vectors(kz=0.0, parameter_set=param_set)
