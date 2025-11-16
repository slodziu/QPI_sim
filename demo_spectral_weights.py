#!/usr/bin/env python3
"""
Demo script for orbital-projected spectral weight analysis in UTe2.

This demonstrates the proper physics-based calculation of spectral weights:
A(k,ω) = Σ_n |⟨orbital|ψ_n(k)⟩|² δ(ω - E_n(k))

For Fermi surface analysis, we evaluate at ω = E_F with a small energy window.
"""

from UTe2_fixed import plot_2d_with_spectral_weight, verify_model_parameters
import matplotlib.pyplot as plt

def demo_spectral_weights():
    """Demonstrate different types of spectral weight analysis."""
    
    print("UTe2 Orbital-Projected Spectral Weight Analysis")
    print("=" * 50)
    
    # Verify the model is working correctly
    print("\n1. Verifying UTe2 model parameters...")
    verify_model_parameters()
    
    print("\n2. Computing spectral weights with different orbital projections...")
    
    # Energy windows to test
    energy_windows = [0.02, 0.05, 0.1]  # eV
    
    # Test different energy windows
    for energy_window in energy_windows:
        print(f"\n--- Energy window: ±{energy_window:.3f} eV around E_F ---")
        
        # Te orbital contributions (relevant for superconductivity in UTe2)
        print("Computing Te orbital spectral weight...")
        plot_2d_with_spectral_weight(
            kz=0.0, 
            weight_type='Te_only',
            energy_window=energy_window,
            colormap='plasma',
            save_dir=f'outputs/spectral_demo_Ewin_{energy_window:.3f}eV'
        )
        
        # Orbital character ratio (Te vs U)
        print("Computing Te/U orbital ratio...")
        plot_2d_with_spectral_weight(
            kz=0.0,
            weight_type='orbital_ratio', 
            energy_window=energy_window,
            colormap='RdBu_r',
            save_dir=f'outputs/spectral_demo_Ewin_{energy_window:.3f}eV'
        )
    
    print("\n3. Analysis complete!")
    print("\nKey physics insights:")
    print("• Spectral weight A(k,ω) shows orbital character of electronic states")
    print("• Te orbitals are more relevant for Fermi surface properties in UTe2")
    print("• Energy window around E_F determines which states contribute")
    print("• Orbital ratio reveals which parts of FS are more Te-like vs U-like")
    print("\nPlots saved to outputs/spectral_demo_Ewin_*/ directories")

if __name__ == "__main__":
    demo_spectral_weights()