#!/usr/bin/env python3
"""
UTe2 5f Orbital Character at Fermi Surface - Correct Implementation
Following Nature paper method: Calculate W_5f(k) = |⟨5f|ψ_n(k)⟩|² for Fermi surface states
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from UTe2_fixed import *
import os
import pickle
import hashlib

def compute_5f_character_on_fermi_contours(kz=0.0, resolution=512):
    """
    Compute 5f orbital character along exact Fermi surface contours.
    Upgraded with Cubic Spline Interpolation and 4-fold Symmetrization.
    """
    print(f"Computing 5f character on exact Fermi surface contours with {resolution}x{resolution} resolution...")
    
    # 1. Create momentum grid
    # Note: Using endpoint=True is safer for exact index mapping
    kx_vals = np.linspace(-1*np.pi/a, 1*np.pi/a, resolution)
    ky_vals = np.linspace(-3*np.pi/b, 3*np.pi/b, resolution)
    
    # Pre-calculate grid bounds for index mapping later
    kx_min, kx_range = kx_vals[0], kx_vals[-1] - kx_vals[0]
    ky_min, ky_range = ky_vals[0], ky_vals[-1] - ky_vals[0]
    
    # 2. Calculate band energies and 5f character on full grid
    print("Computing band energies and 5f character across k-space...")
    energies = np.zeros((len(kx_vals), len(ky_vals), 4))
    character_5f = np.zeros((len(kx_vals), len(ky_vals), 4))
    
    # (Optimized loop: You could vectorize this, but keeping your structure for safety)
    for i, kx in enumerate(kx_vals):
        if i % 50 == 0: print(f"  Progress: {i}/{len(kx_vals)} ({100*i/len(kx_vals):.1f}%)")     
        for j, ky in enumerate(ky_vals):
            H = H_full(kx, ky, kz)
            eigenvals, eigenvecs = np.linalg.eigh(H)
            energies[i, j, :] = eigenvals
            # Calculate 5f character (assuming index 0 is the relevant f-orbital)
            for n in range(4):
                character_5f[i, j, n] = abs(eigenvecs[0, n])**2 

    # 3. SYMMETRIZATION STEP (Crucial for "Publication Quality")
    # UTe2 (Immm) has mirror symmetries. We enforce them to remove numerical noise.
    print("Symmetrizing data to ensure perfect orbital shapes...")
    for n in range(4):
        E = energies[:,:,n]
        W = character_5f[:,:,n]
        # Average (x,y), (-x,y), (x,-y), (-x,-y)
        E_sym = (E + E[::-1,:] + E[:,::-1] + E[::-1,::-1]) / 4.0
        W_sym = (W + W[::-1,:] + W[:,::-1] + W[::-1,::-1]) / 4.0
        energies[:,:,n] = E_sym
        character_5f[:,:,n] = W_sym

    print("Finding exact Fermi surface contours with Cubic Interpolation...")
    
    contour_data = []
    
    for band in range(4):
        E_band = energies[:, :, band]
        char_band = character_5f[:, :, band]
        
        if np.min(E_band) <= 0 <= np.max(E_band):
            
            # Find contours (suppress plot creation)
            fig_temp, ax_temp = plt.subplots()
            # Note: We pass (ky, kx) to contour so output is (x=ky, y=kx)
            CS = ax_temp.contour(ky_vals, kx_vals, E_band, levels=[0.0])
            plt.close(fig_temp)
            
            for contours in CS.allsegs:
                for path in contours:
                    if len(path) > 10:
                        # path is (N, 2) array: [ky_coord, kx_coord]
                        ky_path = path[:, 0]
                        kx_path = path[:, 1]
                        
                        # --- THE FIX: COORDINATE MAPPING ---
                        
                        # Convert Physical Coords -> Fractional Grid Indices
                        # Formula: index = (value - min) / range * (N-1)
                        y_indices = (kx_path - kx_min) / kx_range * (len(kx_vals) - 1) # Row indices (kx)
                        x_indices = (ky_path - ky_min) / ky_range * (len(ky_vals) - 1) # Col indices (ky)
                        
                        # --- THE FIX: CUBIC SPLINE INTERPOLATION ---
                        
                        # map_coordinates expects [row_indices, col_indices]
                        # order=3 is Cubic Spline (smooth), order=1 is Linear
                        char_contour = map_coordinates(
                            char_band, 
                            [y_indices, x_indices], 
                            order=3, 
                            mode='nearest'
                        )
                        
                        contour_data.append({
                            'band': band,
                            'kx': kx_path,
                            'ky': ky_path,
                            'character_5f': char_contour,
                            'length': len(kx_path)
                        })
                        
    print(f"Found {len(contour_data)} Fermi surface contours")
    return contour_data, kx_vals, ky_vals, energies, character_5f


def plot_colored_fermi_contours(contour_data, kx_vals, ky_vals, title="5f Character on Fermi Surface Contours"):
    """
    Plot Fermi surface contours colored by 5f orbital character.
    
    This creates the sharp, varying lines seen in Nature papers by plotting
    the exact geometric contours as colored line segments.
    """
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors
    
    print(f"Plotting colored Fermi surface contours...")
    
    if not contour_data:
        print("No contour data available!")
        return None
    
    # Collect all 5f character values to set colormap range
    all_char_values = []
    for contour in contour_data:
        all_char_values.extend(contour['character_5f'])
    
    if len(all_char_values) == 0:
        print("No valid character values found!")
        return None
    
    char_min, char_max = np.min(all_char_values), np.max(all_char_values)
    print(f"5f character range on Fermi contours: {char_min:.6f} to {char_max:.6f}")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Colored contour lines with scattering vectors
    ax1.set_title(f'{title}\n(with Scattering Vectors)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Create colormap
    cmap = plt.cm.Greys  # Good for scientific visualization
    norm = mcolors.Normalize(vmin=char_min, vmax=char_max)
    
    total_segments = 0
    
    # Plot each contour as colored line segments with thick lines
    # Anti-overlap strategy: round caps/joins + high alpha for smooth blending
    for contour in contour_data:
        kx_contour = contour['kx']
        ky_contour = contour['ky']
        char_contour = contour['character_5f']
        band = contour['band']
        
        if len(kx_contour) < 2:
            continue
            
        # Create line segments
        points = np.array([ky_contour, kx_contour]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Average character values for each segment
        char_segments = (char_contour[:-1] + char_contour[1:]) / 2
        
        # Create line collection with thicker lines and better blending
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=4.0, alpha=0.9, 
                           capstyle='round', joinstyle='round')
        lc.set_array(char_segments)
        ax1.add_collection(lc)
        
        total_segments += len(segments)
        
        print(f"  Band {band}: {len(segments)} segments, "
              f"char range: {np.min(char_contour):.4f} - {np.max(char_contour):.4f}")
    
    print(f"Total segments plotted: {total_segments}")
    
    # Find and plot scattering vectors on main plot
    scattering_vectors, hot_spots = find_scattering_vectors(contour_data)
    
    # Mark hot spots on main plot
    if len(hot_spots) > 0:
        ax1.scatter(hot_spots[:, 1], hot_spots[:, 0], c='red', s=100, 
                   alpha=1.0, edgecolors='white', linewidth=2, zorder=10,
                   label=f'Hot spots ({len(hot_spots)})')
        
        # Draw specific scattering vectors as connecting lines with different colors and labels
        if len(scattering_vectors) > 0:
            # Define colors for each vector
            vector_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
            vector_labels = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']
            
            # Draw vectors from hot spot 5 to all others (first 5 vectors)
            for i, vector in enumerate(scattering_vectors[:5]):
                if np.linalg.norm(vector) > 0.1:  # Skip near-zero vectors
                    start_point = hot_spots[4]  # Hot spot 5 (index 4)
                    end_point = hot_spots[4] + vector
                    color = vector_colors[i]
                    label = vector_labels[i]
                    
                    # Draw vector as arrow
                    ax1.annotate('', xy=[end_point[1], end_point[0]], 
                               xytext=[start_point[1], start_point[0]],
                               arrowprops=dict(arrowstyle='->', lw=3, 
                                             color=color, alpha=0.9),
                               zorder=5)
                    
                    # Position label near arrow end, away from hot spot
                    # Use 80% along vector from start + small perpendicular offset
                    vector_dir = np.array([end_point[1] - start_point[1], end_point[0] - start_point[0]])
                    vector_length = np.linalg.norm(vector_dir)
                    if vector_length > 0:
                        vector_unit = vector_dir / vector_length
                        perp_unit = np.array([-vector_unit[1], vector_unit[0]])  # Perpendicular
                        
                        label_pos = np.array([start_point[1], start_point[0]]) + 0.8 * vector_dir + 0.3 * perp_unit
                        
                        ax1.text(label_pos[0], label_pos[1], label, fontsize=11, fontweight='bold',
                                color=color, ha='center', va='center', 
                                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.95, edgecolor=color, linewidth=1),
                                zorder=15)
            
            # Draw vector from hot spot 2 to hot spot 1 (6th vector)
            if len(scattering_vectors) >= 6:
                vector = scattering_vectors[5]  # 6th vector (2→1)
                if np.linalg.norm(vector) > 0.1:
                    start_point = hot_spots[1]  # Hot spot 2 (index 1)
                    end_point = hot_spots[1] + vector
                    color = vector_colors[5]
                    label = vector_labels[5]
                    
                    # Draw vector as arrow
                    ax1.annotate('', xy=[end_point[1], end_point[0]], 
                               xytext=[start_point[1], start_point[0]],
                               arrowprops=dict(arrowstyle='->', lw=3, 
                                             color=color, alpha=0.9),
                               zorder=5)
                    
                    # Position label near arrow end, away from hot spot
                    vector_dir = np.array([end_point[1] - start_point[1], end_point[0] - start_point[0]])
                    vector_length = np.linalg.norm(vector_dir)
                    if vector_length > 0:
                        vector_unit = vector_dir / vector_length
                        perp_unit = np.array([-vector_unit[1], vector_unit[0]])  # Perpendicular
                        
                        label_pos = np.array([start_point[1], start_point[0]]) + 0.8 * vector_dir - 0.4 * perp_unit
                        
                        ax1.text(label_pos[0], label_pos[1], label, fontsize=11, fontweight='bold',
                                color=color, ha='center', va='center',
                                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.95, edgecolor=color, linewidth=1),
                                zorder=15)
    
    # Add legend
    ax1.legend(loc='upper right')
    
    # Add colorbar for contour plot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar1 = plt.colorbar(sm, ax=ax1, shrink=0.8)
    cbar1.set_label('5f Character W₅f(k)', fontsize=12)
    
    # Plot 2: Traditional contour comparison
    ax2.set_title('Traditional Fermi Surface\\n(Reference)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Plot traditional contours for comparison
    colors = ['#1E88E5', '#42A5F5', "#9635E5", "#FD0000"]
    bands_plotted = set()
    
    for contour in contour_data:
        band = contour['band']
        if band not in bands_plotted:
            ax2.plot(contour['ky'], contour['kx'], 
                    color=colors[band % len(colors)], linewidth=4.5, 
                    alpha=0.9, label=f'Band {band+1}', solid_capstyle='round', solid_joinstyle='round')
            bands_plotted.add(band)
        else:
            ax2.plot(contour['ky'], contour['kx'], 
                    color=colors[band % len(colors)], linewidth=4.5, alpha=0.9,
                    solid_capstyle='round', solid_joinstyle='round')
    
    ax2.legend()
    
    # Format both plots
    for ax in [ax1, ax2]:
        # Set physical axis limits
        ky_lim_physical = [-3*np.pi/b, 3*np.pi/b]
        kx_lim_physical = [-1*np.pi/a, 1*np.pi/a]
        ax.set_xlim(ky_lim_physical)
        ax.set_ylim(kx_lim_physical)
        
        # Custom tick formatting
        ky_ticks = np.linspace(ky_lim_physical[0], ky_lim_physical[1], 7)
        kx_ticks = np.linspace(kx_lim_physical[0], kx_lim_physical[1], 5)
        ky_labels = [f'{val/(np.pi/b):.1f}' for val in ky_ticks]
        kx_labels = [f'{val/(np.pi/a):.1f}' for val in kx_ticks]
        ax.set_xticks(ky_ticks)
        ax.set_yticks(kx_ticks)
        ax.set_xticklabels(ky_labels)
        ax.set_yticklabels(kx_labels)
        ax.set_xlabel('ky (π/b)', fontsize=12)
        ax.set_ylabel('kx (π/a)', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
    
    return fig


def find_scattering_vectors(contour_data, max_vectors=6):
    """
    Find scattering vectors between high-intensity 5f character points on Fermi surface.
    Similar to QPI analysis but for orbital character hot spots.
    """
    print(f"\nAnalyzing scattering vectors between high 5f character regions...")
    
    # Collect all high-intensity points
    all_points = []
    all_chars = []
    
    for contour in contour_data:
        kx_contour = contour['kx']
        ky_contour = contour['ky'] 
        char_contour = contour['character_5f']
        
        for i, char in enumerate(char_contour):
            all_points.append([kx_contour[i], ky_contour[i]])
            all_chars.append(char)
    
    all_points = np.array(all_points)
    all_chars = np.array(all_chars)
    
    # Start with broader threshold (like the 24 points case)
    char_threshold = np.percentile(all_chars, 99.5)  # Back to the 24-point threshold
    high_intensity_mask = all_chars >= char_threshold
    candidate_points = all_points[high_intensity_mask]
    candidate_chars = all_chars[high_intensity_mask]
    
    print(f"Found {len(candidate_points)} candidate high-intensity points (≥{char_threshold:.4f})")
    
    # Apply spatial clustering to remove nearby points
    separation_threshold = 0.5  # Minimum separation in kx and ky (increased)
    clustered_points = []
    clustered_chars = []
    used_indices = set()
    
    # Sort by character value (highest first)
    sorted_indices = np.argsort(candidate_chars)[::-1]
    
    for idx in sorted_indices:
        if idx in used_indices:
            continue
            
        point = candidate_points[idx]
        char = candidate_chars[idx]
        
        # Find all nearby points within separation threshold
        cluster_indices = []
        cluster_points = []
        cluster_chars = []
        
        for j, other_point in enumerate(candidate_points):
            if j in used_indices:
                continue
                
            # Check Manhattan distance (separate kx and ky thresholds)
            kx_diff = abs(point[0] - other_point[0])
            ky_diff = abs(point[1] - other_point[1])
            
            if kx_diff <= separation_threshold and ky_diff <= separation_threshold:
                cluster_indices.append(j)
                cluster_points.append(other_point)
                cluster_chars.append(candidate_chars[j])
        
        if len(cluster_points) > 0:
            # Find the 'middle' point of the cluster (centroid)
            cluster_points = np.array(cluster_points)
            cluster_chars = np.array(cluster_chars)
            
            # Weight by character value to find center of mass
            weights = cluster_chars / np.sum(cluster_chars)
            centroid = np.average(cluster_points, axis=0, weights=weights)
            avg_char = np.average(cluster_chars, weights=weights)
            
            clustered_points.append(centroid)
            clustered_chars.append(avg_char)
            
            # Mark all points in this cluster as used
            for ci in cluster_indices:
                used_indices.add(ci)
    
    # Take top 6 clusters
    if len(clustered_points) > 6:
        clustered_points = np.array(clustered_points)
        clustered_chars = np.array(clustered_chars)
        
        # Sort by character and take top 6
        sort_indices = np.argsort(clustered_chars)[::-1]
        hot_spots = clustered_points[sort_indices[:6]]
        hot_chars = clustered_chars[sort_indices[:6]]
    else:
        hot_spots = np.array(clustered_points)
        hot_chars = np.array(clustered_chars)
    
    final_threshold = np.min(hot_chars) if len(hot_chars) > 0 else char_threshold
    print(f"After spatial clustering: {len(hot_spots)} well-separated hot spots (≥{final_threshold:.4f})")
    
    # Output hot spot coordinates
    if len(hot_spots) > 0:
        print(f"\\nHot spot coordinates:")
        for i, (point, char) in enumerate(zip(hot_spots, hot_chars)):
            kx, ky = point
            print(f"  Hot spot {i+1}: kx={kx:.4f}, ky={ky:.4f}, W5f={char:.6f}")
    
    # Calculate specific vectors: 5 from hot spot 5 to all others, plus 1 from hot spot 2 to 1
    specific_vectors = []
    if len(hot_spots) >= 6:
        print(f"\\nSpecific scattering vectors:")
        
        # 5 vectors from hot spot 5 (index 4) to all other hot spots
        vector_count = 1
        for i in range(6):
            if i != 4:  # Skip self
                vector = hot_spots[i] - hot_spots[4]  # From hot spot 5 to hot spot i+1
                specific_vectors.append(vector)
                kx, ky = vector
                magnitude = np.linalg.norm(vector)
                print(f"p{vector_count} (hot spot 5 → {i+1}): Δk=({kx:.4f}, {ky:.4f}) |Δk|={magnitude:.4f}")
                vector_count += 1
        
        # 1 vector from hot spot 2 to hot spot 1
        vector_2_to_1 = hot_spots[0] - hot_spots[1]  # From hot spot 2 to hot spot 1
        specific_vectors.append(vector_2_to_1)
        kx, ky = vector_2_to_1
        magnitude = np.linalg.norm(vector_2_to_1)
        print(f"p6 (hot spot 2 → 1): Δk=({kx:.4f}, {ky:.4f}) |Δk|={magnitude:.4f}")
    
    return specific_vectors, hot_spots
    
    # Calculate all pairwise scattering vectors
    scattering_vectors = []
    vector_weights = []
    
    for i in range(len(hot_spots)):
        for j in range(i+1, len(hot_spots)):
            # Vector from point i to point j
            vector = hot_spots[j] - hot_spots[i]
            vector_magnitude = np.linalg.norm(vector)
            
            # Skip very small vectors (same point clustering)
            if vector_magnitude < 0.1:  # Minimum meaningful vector length
                continue
                
            # Weight by product of intensities
            weight = hot_chars[i] * hot_chars[j]
            
            scattering_vectors.append(vector)
            vector_weights.append(weight)
    
    scattering_vectors = np.array(scattering_vectors)
    vector_weights = np.array(vector_weights)
    
    # Cluster similar vectors (within 10% of Brillouin zone)
    clustered_vectors = []
    clustered_weights = []
    clustering_tolerance = 0.1 * np.pi / min(a, b)  # 10% of smallest BZ dimension
    
    used_indices = set()
    
    for i, vec in enumerate(scattering_vectors):
        if i in used_indices:
            continue
            
        # Find all vectors similar to this one
        cluster_vectors = [vec]
        cluster_weights = [vector_weights[i]]
        used_indices.add(i)
        
        for j, other_vec in enumerate(scattering_vectors):
            if j in used_indices:
                continue
                
            # Check if vectors are similar (accounting for symmetry)
            distance = min(np.linalg.norm(vec - other_vec),
                          np.linalg.norm(vec + other_vec))  # Also check reversed direction
            
            if distance < clustering_tolerance:
                cluster_vectors.append(other_vec)
                cluster_weights.append(vector_weights[j])
                used_indices.add(j)
        
        # Calculate cluster average
        avg_vector = np.average(cluster_vectors, axis=0, weights=cluster_weights)
        total_weight = np.sum(cluster_weights)
        
        clustered_vectors.append(avg_vector)
        clustered_weights.append(total_weight)
    
    # Sort by total weight and take top vectors
    if len(clustered_vectors) > 0:
        clustered_vectors = np.array(clustered_vectors)
        clustered_weights = np.array(clustered_weights)
        
        # Sort by weight (most important first)
        sort_indices = np.argsort(clustered_weights)[::-1]
        top_vectors = clustered_vectors[sort_indices[:max_vectors]]
        top_weights = clustered_weights[sort_indices[:max_vectors]]
        
        print(f"\nTop {len(top_vectors)} scattering vectors:")
        for i, (vec, weight) in enumerate(zip(top_vectors, top_weights)):
            kx, ky = vec
            magnitude = np.linalg.norm(vec)
            print(f"  {i+1}: Δk=({kx:.4f}, {ky:.4f}) |Δk|={magnitude:.4f} weight={weight:.6f}")
        
        return top_vectors, hot_spots
    else:
        print("No clustered vectors found")
        return [], hot_spots


def plot_scattering_analysis(contour_data, kx_vals, ky_vals):
    """
    Create comprehensive scattering vector analysis plot.
    """
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors
    
    # Find scattering vectors
    scattering_vectors, hot_spots = find_scattering_vectors(contour_data)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Fermi surface with hot spots marked
    ax1.set_title('5f Character Hot Spots\n(High-Intensity Regions)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot contours
    all_char_values = []
    for contour in contour_data:
        all_char_values.extend(contour['character_5f'])
    
    char_min, char_max = np.min(all_char_values), np.max(all_char_values)
    cmap = plt.cm.plasma
    norm = mcolors.Normalize(vmin=char_min, vmax=char_max)
    
    for contour in contour_data:
        kx_contour = contour['kx']
        ky_contour = contour['ky'] 
        char_contour = contour['character_5f']
        
        if len(kx_contour) >= 2:
            points = np.array([ky_contour, kx_contour]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            char_segments = (char_contour[:-1] + char_contour[1:]) / 2
            
            lc = LineCollection(segments, cmap=cmap, norm=norm, 
                              linewidths=3.0, alpha=0.7)
            lc.set_array(char_segments)
            ax1.add_collection(lc)
    
    # Mark hot spots
    if len(hot_spots) > 0:
        ax1.scatter(hot_spots[:, 1], hot_spots[:, 0], c='red', s=50, 
                   alpha=0.8, edgecolors='white', linewidth=1,
                   label=f'High 5f character ({len(hot_spots)} points)')
    
    # Right plot: Scattering vectors
    ax2.set_title('Scattering Vectors\n(Hot Spot Connections)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Plot lighter contours as reference
    for contour in contour_data:
        kx_contour = contour['kx']
        ky_contour = contour['ky'] 
        ax2.plot(ky_contour, kx_contour, 'gray', alpha=0.3, linewidth=1)
    
    # Plot scattering vectors
    if len(scattering_vectors) > 0:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i, vector in enumerate(scattering_vectors):
            kx_vec, ky_vec = vector
            
            # Plot vector from origin
            ax2.arrow(0, 0, ky_vec, kx_vec, 
                     head_width=0.1, head_length=0.1, 
                     fc=colors[i % len(colors)], ec=colors[i % len(colors)],
                     alpha=0.8, linewidth=2,
                     label=f'q{i+1}: ({kx_vec:.3f}, {ky_vec:.3f})')
        
        ax2.legend(loc='upper right', fontsize=10)
    
    # Format both plots
    for ax in [ax1, ax2]:
        ky_lim_physical = [-3*np.pi/b, 3*np.pi/b]
        kx_lim_physical = [-1*np.pi/a, 1*np.pi/a]
        ax.set_xlim(ky_lim_physical)
        ax.set_ylim(kx_lim_physical)
        
        ky_ticks = np.linspace(ky_lim_physical[0], ky_lim_physical[1], 7)
        kx_ticks = np.linspace(kx_lim_physical[0], kx_lim_physical[1], 5)
        ky_labels = [f'{val/(np.pi/b):.1f}' for val in ky_ticks]
        kx_labels = [f'{val/(np.pi/a):.1f}' for val in kx_ticks]
        ax.set_xticks(ky_ticks)
        ax.set_yticks(kx_ticks)
        ax.set_xticklabels(ky_labels)
        ax.set_yticklabels(kx_labels)
        ax.set_xlabel('ky (π/b)', fontsize=12)
        ax.set_ylabel('kx (π/a)', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
    
    # Add colorbar to left plot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, shrink=0.8)
    cbar.set_label('5f Character W₅f(k)', fontsize=10)
    
    if len(hot_spots) > 0:
        ax1.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def test_5f_fermi_surface_character():
    """Test the correct 5f orbital character calculation using geometric contours."""
    
    print("Testing 5f orbital character on Fermi surface contours - Correct geometric method...")
    
    # Configuration
    kz = 0.0
    resolution = 512  # Optimized resolution for thick contours
    
    print(f"Using geometric contour method with {resolution}x{resolution} resolution")
    
    # Create cache directory
    cache_dir = "outputs/ute2_fixed/cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key
    cache_params = {
        'kz': kz,
        'resolution': resolution,
        'method': '5f_geometric_contours'
    }
    cache_key = hashlib.md5(str(sorted(cache_params.items())).encode()).hexdigest()[:8]
    cache_file = os.path.join(cache_dir, f"5f_contours_{cache_key}.pkl")
    
    # Check cache
    if os.path.exists(cache_file):
        print("\\nLoading 5f contour data from cache...")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            contour_data = cache_data['contour_data']
            kx_vals = cache_data['kx_vals']
            ky_vals = cache_data['ky_vals']
            energies = cache_data['energies']
            character_5f = cache_data['character_5f']
    else:
        print("\\nComputing 5f character on Fermi surface contours...")
        contour_data, kx_vals, ky_vals, energies, character_5f = compute_5f_character_on_fermi_contours(
            kz=kz, resolution=resolution)
        
        # Save to cache
        print("Saving to cache...")
        cache_data = {
            'contour_data': contour_data,
            'kx_vals': kx_vals,
            'ky_vals': ky_vals,
            'energies': energies,
            'character_5f': character_5f,
            'cache_params': cache_params
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Cached 5f contour data: {cache_file}")
    
    # Check Fermi crossing bands
    print(f"\\nAt kz = {kz}:")
    fermi_crossing_bands = []
    for i in range(4):
        E_min = np.min(energies[:,:,i])
        E_max = np.max(energies[:,:,i])
        crosses_fermi = E_min <= 0 <= E_max
        print(f"Band {i}: {E_min:.3f} to {E_max:.3f} eV - {'CROSSES FERMI' if crosses_fermi else 'no crossing'}")
        if crosses_fermi:
            fermi_crossing_bands.append(i)
    
    # Plot the colored contours
    if contour_data:
        print(f"\\nPlotting {len(contour_data)} Fermi surface contours...")
        
        # Create main plot
        fig = plot_colored_fermi_contours(contour_data, kx_vals, ky_vals)
        
        if fig is not None:
            plt.tight_layout()
            plt.savefig('outputs/ute2_fixed/5f_geometric_contours.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Create standalone high-quality plot
            fig_standalone = plt.figure(figsize=(10, 8))
            ax = fig_standalone.add_subplot(111)
            
            from matplotlib.collections import LineCollection
            import matplotlib.colors as mcolors
            
            # Collect all character values for colormap
            all_char_values = []
            for contour in contour_data:
                all_char_values.extend(contour['character_5f'])
            
            if len(all_char_values) > 0:
                char_min, char_max = np.min(all_char_values), np.max(all_char_values)
                
                # Create colormap
                cmap = plt.cm.Greys
                norm = mcolors.Normalize(vmin=char_min, vmax=char_max)
                
                # Plot colored contours
                for contour in contour_data:
                    kx_contour = contour['kx']
                    ky_contour = contour['ky'] 
                    char_contour = contour['character_5f']
                    
                    if len(kx_contour) >= 2:
                        # Create line segments
                        points = np.array([ky_contour, kx_contour]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        char_segments = (char_contour[:-1] + char_contour[1:]) / 2
                        
                        # Create and add line collection with thick lines and anti-overlap
                        lc = LineCollection(segments, cmap=cmap, norm=norm, 
                                          linewidths=4.0, alpha=0.9,
                                          capstyle='round', joinstyle='round')
                        lc.set_array(char_segments)
                        ax.add_collection(lc)
                
                ax.set_title(f'UTe₂ 5f Character on Fermi Surface\nW₅f(k) Range: {char_min:.4f} - {char_max:.4f}', 
                           fontsize=16)
                ax.grid(True, alpha=0.3)
                
                # Format axes
                ky_lim_physical = [-3*np.pi/b, 3*np.pi/b]
                kx_lim_physical = [-1*np.pi/a, 1*np.pi/a] 
                ax.set_xlim(ky_lim_physical)
                ax.set_ylim(kx_lim_physical)
                
                ky_ticks = np.linspace(ky_lim_physical[0], ky_lim_physical[1], 7)
                kx_ticks = np.linspace(kx_lim_physical[0], kx_lim_physical[1], 5)
                ky_labels = [f'{val/(np.pi/b):.1f}' for val in ky_ticks]
                kx_labels = [f'{val/(np.pi/a):.1f}' for val in kx_ticks]
                ax.set_xticks(ky_ticks)
                ax.set_yticks(kx_ticks)
                ax.set_xticklabels(ky_labels)
                ax.set_yticklabels(kx_labels)
                ax.set_xlabel('ky (π/b)', fontsize=14)
                ax.set_ylabel('kx (π/a)', fontsize=14)
                ax.set_aspect('equal', adjustable='box')
                
                # Colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
                cbar.set_label('5f Orbital Character W₅f(k)', fontsize=14)
                
                plt.tight_layout()
                plt.savefig('outputs/ute2_fixed/5f_geometric_contours_standalone.png', 
                          dpi=300, bbox_inches='tight')
                plt.close(fig_standalone)
                
                # Create version WITHOUT hot spots
                fig_clean = plt.figure(figsize=(10, 8))
                ax_clean = fig_clean.add_subplot(111)
                
                # Plot colored contours only
                for contour in contour_data:
                    kx_contour = contour['kx']
                    ky_contour = contour['ky'] 
                    char_contour = contour['character_5f']
                    
                    if len(kx_contour) >= 2:
                        # Create line segments
                        points = np.array([ky_contour, kx_contour]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        char_segments = (char_contour[:-1] + char_contour[1:]) / 2
                        
                        # Create and add line collection
                        lc = LineCollection(segments, cmap=cmap, norm=norm, 
                                          linewidths=4.0, alpha=0.9,
                                          capstyle='round', joinstyle='round')
                        lc.set_array(char_segments)
                        ax_clean.add_collection(lc)
                
                # Add vectors and labels (without hot spots)
                scattering_vectors, _ = find_scattering_vectors(contour_data)
                if len(scattering_vectors) > 0:
                    vector_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
                    vector_labels = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']
                    
                    # Get hot spots for vector calculation (but don't plot them)
                    _, temp_hot_spots = find_scattering_vectors(contour_data)
                    
                    # Draw vectors from hot spot 5 to all others
                    for i, vector in enumerate(scattering_vectors[:5]):
                        if np.linalg.norm(vector) > 0.1:
                            start_point = temp_hot_spots[4]
                            end_point = temp_hot_spots[4] + vector
                            color = vector_colors[i]
                            label = vector_labels[i]
                            
                            # Draw arrow
                            ax_clean.annotate('', xy=[end_point[1], end_point[0]], 
                                           xytext=[start_point[1], start_point[0]],
                                           arrowprops=dict(arrowstyle='->', lw=3, 
                                                         color=color, alpha=0.9), zorder=5)
                            
                            # Position label
                            vector_dir = np.array([end_point[1] - start_point[1], end_point[0] - start_point[0]])
                            vector_length = np.linalg.norm(vector_dir)
                            if vector_length > 0:
                                vector_unit = vector_dir / vector_length
                                perp_unit = np.array([-vector_unit[1], vector_unit[0]])
                                label_pos = np.array([start_point[1], start_point[0]]) + 0.8 * vector_dir + 0.3 * perp_unit
                                
                                ax_clean.text(label_pos[0], label_pos[1], label, fontsize=11, fontweight='bold',
                                            color=color, ha='center', va='center', 
                                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.95, edgecolor=color, linewidth=1),
                                            zorder=15)
                    
                    # Draw vector from hot spot 2 to hot spot 1
                    if len(scattering_vectors) >= 6:
                        vector = scattering_vectors[5]
                        if np.linalg.norm(vector) > 0.1:
                            start_point = temp_hot_spots[1]
                            end_point = temp_hot_spots[1] + vector
                            color = vector_colors[5]
                            label = vector_labels[5]
                            
                            ax_clean.annotate('', xy=[end_point[1], end_point[0]], 
                                           xytext=[start_point[1], start_point[0]],
                                           arrowprops=dict(arrowstyle='->', lw=3, 
                                                         color=color, alpha=0.9), zorder=5)
                            
                            vector_dir = np.array([end_point[1] - start_point[1], end_point[0] - start_point[0]])
                            vector_length = np.linalg.norm(vector_dir)
                            if vector_length > 0:
                                vector_unit = vector_dir / vector_length
                                perp_unit = np.array([-vector_unit[1], vector_unit[0]])
                                label_pos = np.array([start_point[1], start_point[0]]) + 0.8 * vector_dir - 0.4 * perp_unit
                                
                                ax_clean.text(label_pos[0], label_pos[1], label, fontsize=11, fontweight='bold',
                                            color=color, ha='center', va='center',
                                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.95, edgecolor=color, linewidth=1),
                                            zorder=15)
                
                ax_clean.set_title(f'UTe₂ 5f Character with Scattering Vectors\nW₅f(k) Range: {char_min:.4f} - {char_max:.4f}', 
                           fontsize=16)
                ax_clean.grid(True, alpha=0.3)
                
                # Format axes (same as main plot)
                ky_lim_physical = [-3*np.pi/b, 3*np.pi/b]
                kx_lim_physical = [-1*np.pi/a, 1*np.pi/a] 
                ax_clean.set_xlim(ky_lim_physical)
                ax_clean.set_ylim(kx_lim_physical)
                
                ky_ticks = np.linspace(ky_lim_physical[0], ky_lim_physical[1], 7)
                kx_ticks = np.linspace(kx_lim_physical[0], kx_lim_physical[1], 5)
                ky_labels = [f'{val/(np.pi/b):.1f}' for val in ky_ticks]
                kx_labels = [f'{val/(np.pi/a):.1f}' for val in kx_ticks]
                ax_clean.set_xticks(ky_ticks)
                ax_clean.set_yticks(kx_ticks)
                ax_clean.set_xticklabels(ky_labels)
                ax_clean.set_yticklabels(kx_labels)
                ax_clean.set_xlabel('ky (π/b)', fontsize=14)
                ax_clean.set_ylabel('kx (π/a)', fontsize=14)
                ax_clean.set_aspect('equal', adjustable='box')
                
                # Colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax_clean, shrink=0.8)
                cbar.set_label('5f Orbital Character W₅f(k)', fontsize=14)
                
                plt.tight_layout()
                plt.savefig('outputs/ute2_fixed/5f_geometric_contours_clean.png', 
                          dpi=300, bbox_inches='tight')
                plt.close(fig_clean)
                
                # Create scattering vector analysis
                print("\nPerforming scattering vector analysis...")
                fig_scattering = plot_scattering_analysis(contour_data, kx_vals, ky_vals)
                plt.savefig('outputs/ute2_fixed/5f_scattering_vectors.png', 
                          dpi=300, bbox_inches='tight')
                plt.show()
                plt.close(fig_scattering)
                
                print(f"\nGeometric contour analysis completed!")
                print("Main result: outputs/ute2_fixed/5f_geometric_contours_standalone.png")
                print("Clean version (no hot spots): outputs/ute2_fixed/5f_geometric_contours_clean.png")
                print("Comparison: outputs/ute2_fixed/5f_geometric_contours.png")
                print("Scattering analysis: outputs/ute2_fixed/5f_scattering_vectors.png")
                
                print(f"\\nStatistics:")
                print(f"5f character range: {char_min:.6f} to {char_max:.6f}")
                print(f"Dynamic range: {char_max/char_min:.1f}×")
                print(f"Total contour segments: {sum(len(c['kx']) for c in contour_data)}")
            
        else:
            print("Error: Could not create contour plot")
    else:
        print("Error: No contour data found!")


if __name__ == "__main__":
    test_5f_fermi_surface_character()