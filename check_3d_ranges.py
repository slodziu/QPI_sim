#!/usr/bin/env python3
"""
Diagnostic script to check energy ranges of our 3D models.
"""
import numpy as np
from model_factory import create_model

def check_3d_model_ranges():
    """Check the actual energy ranges of our 3D models."""
    
    models_to_test = [
        ('true_3d_cubic', {'t': 1.0, 't_prime': 0.2, 'a': 1.0}),
        ('true_3d_aniso', {'tx': 1.0, 'ty': 0.6, 'tz': 0.4, 'a': 1.0}),
        ('true_3d_complex', {'t1': 1.0, 't2': 0.3, 't3': 0.1, 'a': 1.0}),
    ]
    
    # Create small 3D k-space grid - SAME AS VISUALIZATION
    resolution = 40  # Same as visualization
    k_range = np.pi  # Same as visualization
    kx = np.linspace(-k_range, k_range, resolution)
    ky = np.linspace(-k_range, k_range, resolution) 
    kz = np.linspace(-k_range, k_range, resolution)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')  # Same indexing
    
    print("3D Model Energy Range Analysis")
    print("="*50)
    
    for model_name, params in models_to_test:
        try:
            model = create_model(model_name, **params)
            print(f"\n{model_name.upper()}:")
            print(f"Parameters: {params}")
            
            if hasattr(model, 'get_3d_band_structure'):
                eigenvalues, _ = model.get_3d_band_structure(KX, KY, KZ)
                n_bands = eigenvalues.shape[0]
                
                for band_idx in range(n_bands):
                    energy_3d = eigenvalues[band_idx]
                    print(f"  Band {band_idx+1}: [{energy_3d.min():.3f}, {energy_3d.max():.3f}]")
                    
                    # Suggest good μ values
                    e_min, e_max = energy_3d.min(), energy_3d.max()
                    suggested_mu = [
                        e_min + 0.1 * (e_max - e_min),
                        e_min + 0.3 * (e_max - e_min), 
                        e_min + 0.5 * (e_max - e_min),
                        e_min + 0.7 * (e_max - e_min),
                        e_min + 0.9 * (e_max - e_min)
                    ]
                    print(f"  Suggested μ values: {[f'{mu:.3f}' for mu in suggested_mu]}")
            else:
                print("  No 3D band structure method available")
                
        except Exception as e:
            print(f"  Error creating {model_name}: {e}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    check_3d_model_ranges()