#!/usr/bin/env python3
"""
Create side-by-side comparison of B2u and B3u Fermi surface projections
with specific font size requirements and figure width constraints.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

def create_fsproj_comparison():
    """Create side-by-side comparison of B2u and B3u plots."""
    
    # Check if the plots exist
    b2u_path = "outputs/FSPROJ/FS_3D_projection_B2u.png"
    b3u_path = "outputs/FSPROJ/FS_3D_projection_B3u.png"
    
    if not os.path.exists(b2u_path):
        print(f"Error: {b2u_path} not found. Please run FSProj.py first.")
        return
    
    if not os.path.exists(b3u_path):
        print(f"Error: {b3u_path} not found. Please run FSProj.py first.")
        return
    
    # Load the images
    b2u_img = mpimg.imread(b2u_path)
    b3u_img = mpimg.imread(b3u_path)
    
    # Create the comparison figure with specific width constraint
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))  # Back to reasonable height
    
    # Display B2u plot
    ax1.imshow(b2u_img)
    ax1.set_title("B2u", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Display B3u plot  
    ax2.imshow(b3u_img)
    ax2.set_title("B3u", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Add subplot labels with medium font size
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=11, 
             fontweight='bold', verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=11, 
             fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    
    # Save the comparison plot
    output_path = "outputs/FSPROJ/B2u_B3u_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Side-by-side comparison saved to: {output_path}")
    
    # Also save as PDF for better quality
    pdf_path = "outputs/FSPROJ/B2u_B3u_comparison.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"High-quality PDF saved to: {pdf_path}")
    
    plt.show()

def create_enhanced_fsproj_comparison():
    """Create enhanced side-by-side comparison with additional details."""
    
    # Check if the plots exist
    b2u_path = "outputs/FSPROJ/FS_3D_projection_B2u.png"
    b3u_path = "outputs/FSPROJ/FS_3D_projection_B3u.png"
    
    if not os.path.exists(b2u_path) or not os.path.exists(b3u_path):
        print("Error: One or both projection plots not found. Please run FSProj.py first.")
        return
    
    # Load the images
    b2u_img = mpimg.imread(b2u_path)
    b3u_img = mpimg.imread(b3u_path)
    
    # Create the comparison figure with specific constraints
    fig = plt.figure(figsize=(8, 5))  # Back to reasonable height
    
    # Create custom subplot layout for better control
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.08)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Display images
    ax1.imshow(b2u_img)
    ax1.set_title("B2u", fontsize=12, fontweight='bold', pad=10)
    ax1.axis('off')
    
    ax2.imshow(b3u_img)
    ax2.set_title("B3u", fontsize=12, fontweight='bold', pad=10)
    ax2.axis('off')
    
    # Add panel labels with appropriate font size
    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=11, 
             fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='black'))
    
    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=11, 
             fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='black'))
    

    

    # Add common legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#9635E5', alpha=0.3, label='Hole-like band'),
        Patch(facecolor='#FD0000', alpha=0.3, label='Electron-like band'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, 
               markeredgecolor='w', markeredgewidth=1.5, label='Gap nodes')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
               ncol=3, fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Save both PNG and PDF versions
    png_output = "outputs/FSPROJ/B2u_B3u_sidebyside_comparison.png"
    pdf_output = "outputs/FSPROJ/B2u_B3u_sidebyside_comparison.pdf"
    
    plt.savefig(png_output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_output, bbox_inches='tight', facecolor='white')
    
    print(f"Enhanced side-by-side comparison saved to:")
    print(f"  PNG: {png_output}")
    print(f"  PDF: {pdf_output}")
    
    plt.show()

if __name__ == "__main__":
    print("Creating FSPROJ B2u vs B3u comparison plots...")
    
    # Check if FSPROJ directory exists
    os.makedirs("outputs/FSPROJ", exist_ok=True)
    
    # Create both versions of the comparison
    create_fsproj_comparison()
    create_enhanced_fsproj_comparison()
    
    print("\nComparison plots created successfully!")