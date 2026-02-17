#!/usr/bin/env python3
"""
Load and analyze previously saved HAEM results

Usage:
    python analyze_haem_results.py
"""

import numpy as np
import matplotlib.pyplot as plt
from HAEMUTe2 import load_haem_results, plot_haem_simple_overlay

def main():
    # Load saved results
    print("Loading saved HAEM results...")
    results, vectors_dict = load_haem_results('outputs/haem_ute2/haem_results.npz')
    
    # Print summary
    print("\n" + "="*60)
    print("HAEM Results Summary")
    print("="*60)
    
    print("\nVectors analyzed:")
    for label, (qx, qy) in vectors_dict.items():
        print(f"  {label}: ({qx:+.3f}, {qy:+.3f}) × 2π/(a,b)")
    
    print("\nPairing symmetries:")
    for pairing_type in results.keys():
        print(f"  {pairing_type}")
        energies = results[pairing_type]['energies']
        print(f"    Energy range: {energies[0]:.1f} - {energies[-1]:.1f} µeV")
        print(f"    Number of points: {len(energies)}")
    
    print("\nSignal statistics:")
    for pairing_type in results.keys():
        print(f"\n{pairing_type}:")
        for vector_label in vectors_dict.keys():
            data = results[pairing_type]['vectors'][vector_label]
            print(f"  {vector_label}:")
            print(f"    max|ρ⁻| = {np.max(np.abs(data)):.3e}")
            print(f"    mean|ρ⁻| = {np.mean(np.abs(data)):.3e}")
            if np.max(np.abs(data)) > 0:
                # Find energy of maximum signal
                max_idx = np.argmax(np.abs(data))
                energies = results[pairing_type]['energies']
                print(f"    max at E = {energies[max_idx]:.1f} µeV")
    
    # Create plots
    print("\n" + "="*60)
    print("Creating plots...")
    plot_haem_simple_overlay(results, vectors_dict)
    
    print("\n✅ Analysis completed!")
    
    # Example: Access specific data
    print("\n" + "="*60)
    print("Example: Accessing specific data")
    print("="*60)
    print("\n# Get B2u energies and p2 vector signal:")
    print("energies = results['B2u']['energies']")
    print("p2_signal = results['B2u']['vectors']['p2']")
    
    energies = results['B2u']['energies']
    p2_signal = results['B2u']['vectors']['p2']
    print(f"\nShape: energies={energies.shape}, signal={p2_signal.shape}")
    print(f"First 5 energy points (µeV): {energies[:5]}")
    print(f"First 5 signal values: {p2_signal[:5]}")


if __name__ == "__main__":
    main()
