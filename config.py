"""
Configuration file for QPI simulations.

This file contains parameter settings for different simulation scenarios.
Simply modify the parameters here to run different types of simulations.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class SimulationConfig:
    """Configuration for a specific QPI simulation scenario."""
    name: str
    description: str
    
    # Grid parameters
    gridsize: int = 512
    L: float = 50.0
    
    # Physical parameters
    t: float = 0.3
    mu: float = 0.0
    eta: float = 0.7  # Much smaller broadening for sharp QPI features
    V_s: float = 1.0
    
    # Energy sweep
    E_min: float = 3.0
    E_max: float = 5.0
    n_frames: int = 20
    
    # Processing parameters
    rotation_angle: float = 0.0  # No rotation
    disorder_strength: float = 0.0  # No disorder
    zoom_factor: float = 1.0  # No zoom
    
    # Tight-binding specific parameters
    model_type: str = "parabolic"  # Default to parabolic for backward compatibility
    t_prime: float = 0.0  # Next-nearest neighbor hopping
    use_all_bands: bool = False  # Whether to sum over all bands
    band_index: int = 0  # Which band to use if not using all bands
    
    # Advanced tight-binding parameters
    ty: float = 0.6  # Hopping in y-direction (for anisotropic models)
    t_z: float = 0.5  # Out-of-plane hopping (for 3D models)  
    kz_slice: float = 0.0  # kz value for 2D slice visualization
    
    # Impurity configuration
    impurity_positions: List[Tuple[int, int]] = None
    
    def get_impurity_positions(self) -> List[Tuple[int, int]]:
        """Get impurity positions, calculating if needed."""
        if self.impurity_positions is None:
            # Default: single impurity at CENTER
            center = self.gridsize // 2
            return [(center, center)]
        return self.impurity_positions


# ============================================================================
# PREDEFINED SIMULATION CONFIGURATIONS
# ============================================================================

# High-quality single impurity simulation
HIGH_QUALITY_SINGLE = SimulationConfig(
    name="high_quality_single",
    description="High-resolution single impurity simulation",
    gridsize=1024,
    n_frames=30,
    E_min=3.0,
    E_max=30.0,
    V_s=1.0,  # Moderate strength for clean scattering
)

# Fast preview simulation
FAST_PREVIEW = SimulationConfig(
    name="fast_preview",
    description="Quick preview with reduced quality",
    gridsize=512,  # Much smaller for fast computation (16x faster than 512!)
    n_frames=10,   # Fewer frames for quick preview
    E_min=3.0,     # Higher energy for better defined Fermi surface
    E_max=20.0,    # Narrower range for more focused analysis
    V_s=1.0,       # Slightly weaker to avoid over-scattering
    zoom_factor=1.0  # No zoom
)

# N-impurity random configurations
RANDOM_5_IMPURITIES = SimulationConfig(
    name="random_5_impurities",
    description="Five randomly placed impurities",
    gridsize=512,
    n_frames=20,
    E_min=5.0,
    E_max=50.0,
    V_s=1.5,  # Moderately attractive
)

RANDOM_10_IMPURITIES = SimulationConfig(
    name="random_10_impurities",
    description="Ten randomly placed impurities",
    gridsize=512,
    n_frames=10,
    E_min=5.0,
    E_max=25.0,
    V_s=1.5,  
)

RANDOM_30_IMPURITIES = SimulationConfig(
    name="random_30_impurities",
    description="Thirty randomly placed impurities",
    gridsize=1024,  # Larger grid for better spacing
    n_frames=20,    # Reduced frames for faster computation
    E_min=5.0,
    E_max=25.0,
    V_s=1.5,       # Much weaker impurities for numerical stability
)

# ============================================================================
# TIGHT-BINDING MODEL CONFIGURATIONS
# ============================================================================

# Square lattice preview
SQUARE_LATTICE_PREVIEW = SimulationConfig(
    name="square_lattice_preview", 
    description="Square lattice tight-binding model preview",
    gridsize=512,
    n_frames=15,
    E_min=-3.0,  # Adjusted for tight-binding energies
    E_max=3.0,
    t=1.0,       # Hopping parameter
    mu=0.0,      # Half-filling
    eta=0.1,     # Sharper features
    V_s=0.5,     # Weaker impurity for tight-binding
    model_type="square_lattice",
    t_prime=0.0, # No next-nearest neighbor hopping
    use_all_bands=False,
    band_index=0
)

# Graphene preview
GRAPHENE_PREVIEW = SimulationConfig(
    name="graphene_preview",
    description="Graphene tight-binding model preview (single band)",
    gridsize=512,
    n_frames=15,
    E_min=-2.0,  # Around Dirac point
    E_max=2.0,
    t=2.7,       # Graphene hopping
    mu=0.1,      # Slightly doped
    eta=0.05,    # Very sharp for Dirac physics
    V_s=0.3,     # Weak impurity
    model_type="graphene",
    use_all_bands=False,
    band_index=0  # Conduction band
)

# High-quality square lattice
SQUARE_LATTICE_HIGH_QUALITY = SimulationConfig(
    name="square_lattice_high_quality",
    description="High-quality square lattice with next-nearest neighbors",
    gridsize=1024,
    n_frames=25,
    E_min=-2.0,
    E_max=2.0,
    t=1.0,
    t_prime=0.2,  # Next-nearest neighbor hopping
    mu=0.5,       # Quarter-filled
    eta=0.05,
    V_s=0.4,
    model_type="square_lattice",
    use_all_bands=False,
    band_index=0
)

# Graphene both bands
GRAPHENE_BOTH_BANDS = SimulationConfig(
    name="graphene_both_bands",
    description="Graphene with both conduction and valence bands",
    gridsize=512,
    n_frames=20,
    E_min=-1.0,
    E_max=1.0,
    t=2.7,
    mu=0.0,      # Undoped
    eta=0.03,    # Very sharp
    V_s=0.2,     # Very weak impurity
    model_type="graphene",
    use_all_bands=True  # Sum over both bands
)

# Multi-impurity tight-binding configurations
SQUARE_LATTICE_5_IMPURITIES = SimulationConfig(
    name="square_lattice_5_impurities",
    description="Square lattice with 5 random impurities",
    gridsize=512,
    n_frames=15,
    E_min=-5.0,
    E_max=5.0,
    t=1.0,
    mu=0.0,
    eta=0.1,
    V_s=0.4,     # Moderate strength
    model_type="square_lattice",
    t_prime=0.1,
    use_all_bands=False,
    band_index=0
)

GRAPHENE_10_IMPURITIES = SimulationConfig(
    name="graphene_10_impurities", 
    description="Graphene with 10 random impurities",
    gridsize=512,
    n_frames=15,
    E_min=-1.5,
    E_max=1.5,
    t=2.7,
    mu=0.2,      # Slightly doped
    eta=0.05,
    V_s=0.3,     # Weak impurities
    model_type="graphene",
    use_all_bands=False,
    band_index=0  # Conduction band only
)

SQUARE_LATTICE_DISORDER = SimulationConfig(
    name="square_lattice_disorder",
    description="Square lattice with many weak impurities (disorder)",
    gridsize=1024,  # Larger for better statistics
    n_frames=12,
    E_min=-1.5,
    E_max=1.5,
    t=1.0,
    mu=0.5,      # Quarter filled
    eta=0.08,
    V_s=0.2,     # Very weak for many impurities
    model_type="square_lattice",
    t_prime=0.0,
    use_all_bands=False,
    band_index=0
)

# Dynamic N-impurity configuration template
RANDOM_N_IMPURITIES_TEMPLATE = SimulationConfig(
    name="random_N_impurities",
    description="N randomly placed impurities (dynamic)",
    gridsize=512,
    n_frames=20,
    E_min=5.0,
    E_max=25.0,
    V_s=1.5,  # Default moderately attractive
)


def setup_n_random_positions(config: SimulationConfig, n_impurities: int, seed: int = 42, 
                             margin: int = None, distributed: bool = True):
    """
    Setup n randomly placed impurity positions.
    
    Args:
        config: Configuration object to modify
        n_impurities: Number of impurities to place
        seed: Random seed for reproducibility
        margin: Margin from edges in pixels (if None, use gridsize//6 for better edge avoidance)
        distributed: If True, distribute across grid; if False, cluster in center
    """
    np.random.seed(seed)
    config.impurity_positions = []
    
    # Use larger default margin to avoid periodic boundary artifacts
    if margin is None:
        margin = config.gridsize // 32  
    # Calculate appropriate minimum distance based on density
    available_area = (config.gridsize - 2*margin)**2
    if not distributed:
        # For clustered placement, use smaller area but still reasonable
        region_size = min(config.gridsize // 3, 150)  # Larger clustering region
        available_area = region_size**2
    
    # Calculate density and adjust min_distance accordingly
    density = n_impurities / available_area
    # Use sqrt of inverse density as rough min_distance, with reasonable bounds
    # Much larger minimum distances for numerical stability without smoothing
    min_distance = max(25, min(60, int(np.sqrt(1.0 / density) * 1.2)))
    
    max_attempts = 1000
    
    if distributed:
        # Distribute across the entire grid with proper margins
        x_min = margin
        x_max = config.gridsize - margin
        y_min = margin  
        y_max = config.gridsize - margin
        print(f"Distributing {n_impurities} impurities across grid ({y_min}-{y_max}, {x_min}-{x_max}), min_distance={min_distance}")
    else:
        # Place impurities in a central region
        center = config.gridsize // 2
        # Use margin-based region size for consistency
        region_radius = (config.gridsize - 2*margin) // 3  # 1/3 of available space
        x_min = center - region_radius
        x_max = center + region_radius
        y_min = center - region_radius
        y_max = center + region_radius
        print(f"Clustering {n_impurities} impurities in central region ({y_min}-{y_max}, {x_min}-{x_max}), min_distance={min_distance}")
    
    for i in range(n_impurities):
        attempts = 0
        placed = False
        
        while attempts < max_attempts and not placed:
            # Generate random position in the appropriate region
            # Note: Using (row, col) convention to match array indexing
            row = np.random.randint(y_min, y_max)
            col = np.random.randint(x_min, x_max)
            
            # Check distance to existing impurities
            valid_position = True
            for existing_row, existing_col in config.impurity_positions:
                distance = np.sqrt((row - existing_row)**2 + (col - existing_col)**2)
                if distance < min_distance:
                    valid_position = False
                    break
            
            if valid_position:
                config.impurity_positions.append((row, col))
                print(f"  Impurity {i+1} placed at (row={row}, col={col})")
                placed = True
            
            attempts += 1
        
        # If we couldn't place this impurity, try with relaxed distance constraint
        if not placed and len(config.impurity_positions) > 0:
            relaxed_min_distance = min_distance * 0.7  # Relax by 30%
            print(f"  Relaxing distance constraint to {relaxed_min_distance:.1f} for impurity {i+1}")
            
            for attempt in range(max_attempts):
                row = np.random.randint(y_min, y_max)
                col = np.random.randint(x_min, x_max)
                
                valid_position = True
                for existing_row, existing_col in config.impurity_positions:
                    distance = np.sqrt((row - existing_row)**2 + (col - existing_col)**2)
                    if distance < relaxed_min_distance:
                        valid_position = False
                        break
                
                if valid_position:
                    config.impurity_positions.append((row, col))
                    print(f"  Impurity {i+1} placed at (row={row}, col={col}) with relaxed constraint")
                    placed = True
                    break
        
        if not placed:
            print(f"Warning: Could not place impurity {i+1} after {max_attempts} attempts")
            break
    
    print(f"Successfully placed {len(config.impurity_positions)} impurities")
    return config.impurity_positions


def create_random_n_config(n_impurities: int) -> SimulationConfig:
    """
    Create a dynamic configuration for N randomly placed impurities.
    
    Args:
        n_impurities: Number of impurities to place
    
    Returns:
        SimulationConfig: Configured simulation with N impurities
    """
    # Create a copy of the template
    import copy
    config = copy.deepcopy(RANDOM_N_IMPURITIES_TEMPLATE)
    
    # Update name and description
    config.name = f"random_{n_impurities}_impurities"
    config.description = f"{n_impurities} randomly placed impurities"
    
    # Adjust parameters based on number of impurities
    if n_impurities == 1:
        config.V_s = -1.0  # Moderate strength for sharp scattering  
        config.n_frames = 20
        config.gridsize = max(512, config.gridsize)  # Ensure good resolution
    elif n_impurities <= 5:
        config.V_s = -1.0  # Few impurities - moderate strength for sharpness
        config.n_frames = 20
    elif n_impurities <= 15:
        config.V_s = -1.0  # Many impurities - reduce strength
        config.n_frames = 15
    elif n_impurities <= 25:
        config.V_s = -0.7  # More impurities - weaker for stability
        config.n_frames = 12
        config.gridsize = 1024 if config.gridsize < 1024 else config.gridsize
    else:
        config.V_s = -0.3  # Many impurities - very weak to avoid numerical issues
        config.n_frames = 10  # Faster for large numbers
        config.gridsize = 1024 if config.gridsize < 1024 else config.gridsize
    
    # Setup random positions
    setup_n_random_positions(config, n_impurities, distributed=True)
    
    return config


# ============================================================================
# TRUE 3D FERMI SURFACE SHOWCASE CONFIGURATIONS  
# ============================================================================

# True 3D cubic lattice
TRUE_3D_CUBIC_SHOWCASE = SimulationConfig(
    name="true_3d_cubic_showcase",
    description="True 3D cubic lattice with full kx-ky-kz Fermi surface",
    gridsize=512,
    n_frames=15,
    E_min=-6.0,
    E_max=6.0,
    t=1.0,       # Nearest neighbor
    t_prime=0.2, # Next-nearest neighbor
    mu=-2.3,     # Middle of energy range [-8.253, 3.600]
    eta=0.05,
    V_s=0.3,
    model_type="true_3d_cubic",
    use_all_bands=False,
    band_index=0
)

# True 3D anisotropic showcase  
TRUE_3D_ANISO_SHOWCASE = SimulationConfig(
    name="true_3d_aniso_showcase",
    description="True 3D anisotropic lattice - ellipsoidal Fermi surface",
    gridsize=512,
    n_frames=15,
    E_min=-5.0,
    E_max=5.0,
    t=1.0,       # tx hopping
    ty=0.6,      # ty hopping (weaker)
    t_z=0.4,     # tz hopping (weakest)
    mu=-3.9,     # Set near minimum for visible Fermi surface [-4.0, -3.8]
    eta=0.05,
    V_s=0.3,
    model_type="true_3d_aniso",
    use_all_bands=False,
    band_index=0
)

# True 3D complex showcase
TRUE_3D_COMPLEX_SHOWCASE = SimulationConfig(
    name="true_3d_complex_showcase", 
    description="True 3D complex lattice - exotic Fermi surface topology",
    gridsize=512,
    n_frames=15,
    E_min=-8.0,
    E_max=4.0,
    t=1.0,       # t1 (nearest neighbor)
    t_z=0.3,     # t2 (next-nearest) 
    t_prime=0.1, # t3 (third-nearest)
    mu=-3.5,     # Within range [-10.188, 3.200]
    eta=0.04,
    V_s=0.25,
    model_type="true_3d_complex",
    use_all_bands=False,
    band_index=0
)


# True 3D anisotropic showcase with multiple Fermi levels
TRUE_3D_MULTI_FERMI = SimulationConfig(
    name="true_3d_multi_fermi",
    description="True 3D anisotropic - multiple Fermi surface shapes", 
    gridsize=512,
    n_frames=15,
    E_min=-5.0,
    E_max=5.0,
    t=1.0,       # tx hopping
    ty=0.6,      # ty hopping (weaker)
    t_z=0.4,     # tz hopping (weakest) 
    mu=-3.95,    # Very close to minimum for small closed surface
    eta=0.05,
    V_s=0.3,
    model_type="true_3d_aniso",
    use_all_bands=False,
    band_index=0
)

# Multi-band showcase: 2-band system with electron/hole pockets
MULTIBAND_2BAND_SHOWCASE = SimulationConfig(
    name="multiband_2band_showcase",
    description="Two-band 3D model - electron and hole pockets",
    gridsize=512,
    n_frames=15,
    E_min=-8.0,
    E_max=8.0,
    t=1.0,        # t1 (electron band)
    t_z=0.8,      # t2 (hole band) 
    t_prime=0.3,  # hybridization
    mu=-5.9,      # In band 1 range to show electron pocket
    eta=0.05,
    V_s=0.3,
    model_type="multiband_2band",
    use_all_bands=True,  # Show both bands
    band_index=0
)

# Multi-band showcase: 2-band system with smaller gap for both surfaces
MULTIBAND_2BAND_BOTH = SimulationConfig(
    name="multiband_2band_both",
    description="Two-band 3D model - both electron and hole surfaces",
    gridsize=512,
    n_frames=15,
    E_min=-8.0,
    E_max=8.0,
    t=1.0,        # t1 (electron band)
    t_z=0.8,      # t2 (hole band) 
    t_prime=0.3,  # hybridization
    mu=5.7,       # In band 2 range to show hole pocket
    eta=0.05,
    V_s=0.3,
    model_type="multiband_2band",
    use_all_bands=True,  # Show both bands
    band_index=0
)

# Multi-band showcase: 3-band Kagome-like system
MULTIBAND_3BAND_SHOWCASE = SimulationConfig(
    name="multiband_3band_showcase", 
    description="Three-band 3D Kagome - flat band and dispersive bands",
    gridsize=512,
    n_frames=15,
    E_min=-6.0,
    E_max=4.0,
    t=1.0,        # in-plane hopping
    t_z=0.3,      # inter-layer hopping
    t_prime=0.2,  # inter-orbital coupling
    mu=1.0,       # In band 3 range for flat-band-like topology
    eta=0.04,
    V_s=0.25,
    model_type="multiband_3band",
    use_all_bands=True,  # Show all three bands
    band_index=0
)

# Multi-band showcase: 4-band d-p model
MULTIBAND_4BAND_SHOWCASE = SimulationConfig(
    name="multiband_4band_showcase",
    description="Four-band 3D d-p model - transition metal compound",
    gridsize=512,
    n_frames=15,
    E_min=-8.0,
    E_max=6.0,
    t=1.0,        # t_d (d-orbital hopping)
    t_z=0.6,      # t_p (p-orbital hopping)
    t_prime=0.8,  # t_pd (p-d hybridization)
    mu=-0.35,     # In p-band range to show hybridized surfaces
    eta=0.05,
    V_s=0.3,
    model_type="multiband_4band",
    use_all_bands=True,  # Show all four bands
    band_index=0
)

# UTe₂ showcase: Full 5-orbital model
UTE2_FULL_SHOWCASE = SimulationConfig(
    name="ute2_full_showcase",
    description="UTe₂ full model - U 5f + Te 5p multi-orbital physics",
    gridsize=512,
    n_frames=15,
    E_min=-6.0,
    E_max=8.0,
    t=1.0,        # t_ff (U 5f-5f hopping)
    t_z=0.6,      # t_fp (U 5f - Te 5p hybridization) 
    t_prime=0.4,  # t_pp (Te 5p-5p hopping)
    mu=2.2,       # Near Fermi level for UTe₂
    eta=0.04,
    V_s=0.25,
    model_type="ute2_full",
    use_all_bands=True,  # Show all five orbitals
    band_index=0
)

# UTe₂ showcase: Simplified 3-band model  
UTE2_SIMPLIFIED_SHOWCASE = SimulationConfig(
    name="ute2_simplified_showcase",
    description="UTe₂ simplified - U-dominant 3-band physics",
    gridsize=512, 
    n_frames=15,
    E_min=-5.0,
    E_max=5.0,
    t=1.0,        # t_U (U-U hopping)
    t_z=0.4,      # t_UTe (U-Te hybridization)
    t_prime=0.7,  # anisotropy ratio
    mu=-0.5,      # In band 2 range for U-dominated surface
    eta=0.05,
    V_s=0.3,
    model_type="ute2_simplified",
    use_all_bands=True,  # Show all three bands
    band_index=0
)

# ============================================================================
# 3D LATTICE SHOWCASE CONFIGURATIONS  
# ============================================================================

# 3D Cubic lattice slice
CUBIC_3D_SHOWCASE = SimulationConfig(
    name="cubic_3d_showcase",
    description="3D cubic lattice with anisotropic hopping (2D slice)",
    gridsize=512,
    n_frames=15,
    E_min=-5.0,
    E_max=3.0,
    t=1.0,       # In-plane hopping
    mu=0.8,      # Interesting Fermi surface shape
    eta=0.05,
    V_s=0.3,
    model_type="cubic_3d",
    t_z=0.5,     # Out-of-plane hopping (weaker)
    kz_slice=0.0, # 2D slice at kz=0
    use_all_bands=False,
    band_index=0
)

# Anisotropic lattice showcase
ANISOTROPIC_SHOWCASE = SimulationConfig(
    name="anisotropic_showcase", 
    description="Anisotropic lattice with elliptical Fermi surfaces",
    gridsize=512,
    n_frames=15,
    E_min=-4.0,
    E_max=2.0,
    t=1.0,       # Hopping in x
    ty=0.6,      # Weaker hopping in y (anisotropy)
    t_prime=0.3, # Next-nearest neighbor coupling
    mu=1.2,      # Creates interesting FS topology
    eta=0.05,
    V_s=0.3,
    model_type="anisotropic",
    use_all_bands=False,
    band_index=0
)

# Honeycomb lattice showcase
HONEYCOMB_SHOWCASE = SimulationConfig(
    name="honeycomb_showcase",
    description="Honeycomb lattice with Dirac-like dispersion (two bands)",
    gridsize=512,
    n_frames=20,
    E_min=-3.0,
    E_max=3.0,
    t=2.8,       # Hopping strength
    mu=0.5,      # Slightly above Dirac point
    eta=0.03,    # Very sharp for Dirac physics
    V_s=0.2,
    model_type="honeycomb",
    use_all_bands=True  # Show both bands
)

# Complex 3D showcase (cubic with strong anisotropy)
COMPLEX_3D_SHOWCASE = SimulationConfig(
    name="complex_3d_showcase",
    description="Complex 3D cubic lattice with strong anisotropy",
    gridsize=512,
    n_frames=15,
    E_min=-6.0,
    E_max=2.0,
    t=1.2,       # Strong in-plane
    mu=-1.5,     # Creates multiple Fermi surface sheets
    eta=0.04,
    V_s=0.25,
    model_type="cubic_3d",
    t_z=0.2,     # Very weak out-of-plane (quasi-2D)
    kz_slice=3.14159/4,  # Interesting slice angle (π/4)
    use_all_bands=False,
    band_index=0
)

# Multi-band anisotropic showcase
MULTI_ANISO_SHOWCASE = SimulationConfig(
    name="multi_aniso_showcase",
    description="Extreme anisotropy creating exotic Fermi surface shapes",
    gridsize=512,
    n_frames=15,
    E_min=-5.0,
    E_max=1.0,
    t=1.5,       # Strong x-hopping
    ty=0.3,      # Very weak y-hopping  
    t_prime=0.5, # Strong diagonal coupling
    mu=-0.8,     # Multiple disconnected FS pieces
    eta=0.05,
    V_s=0.3,
    model_type="anisotropic",
    use_all_bands=False,
    band_index=0
)


# ============================================================================
# CONFIGURATION SELECTION FUNCTION
# ============================================================================

def get_config(config_name: str) -> SimulationConfig:
    """
    Get a predefined configuration by name or create a dynamic N-impurity configuration.
    
    Available configurations:
    - 'high_quality_single': High-resolution single impurity
    - 'fast_preview': Quick preview simulation
    - 'random_5_impurities': Five randomly placed impurities
    - 'random_10_impurities': Ten randomly placed impurities
    - 'random_30_impurities': Thirty randomly placed impurities
    - 'random_N_impurities' where N is any integer: Dynamic N impurities
    
    Examples:
    - get_config('random_7_impurities') -> 7 randomly placed impurities
    - get_config('random_25_impurities') -> 25 randomly placed impurities
    - get_config('random_5_impurities') -> Uses preset configuration
    """
    
    # Predefined configurations
    preset_configs = {
        'high_quality_single': HIGH_QUALITY_SINGLE,
        'fast_preview': FAST_PREVIEW,
        'random_5_impurities': RANDOM_5_IMPURITIES,
        'random_10_impurities': RANDOM_10_IMPURITIES,
        'random_30_impurities': RANDOM_30_IMPURITIES,
        'square_lattice_preview': SQUARE_LATTICE_PREVIEW,
        'graphene_preview': GRAPHENE_PREVIEW,
        'square_lattice_high_quality': SQUARE_LATTICE_HIGH_QUALITY,
        'graphene_both_bands': GRAPHENE_BOTH_BANDS,
        'square_lattice_5_impurities': SQUARE_LATTICE_5_IMPURITIES,
        'graphene_10_impurities': GRAPHENE_10_IMPURITIES,
        'square_lattice_disorder': SQUARE_LATTICE_DISORDER,
        # True 3D Showcase Configurations
        'true_3d_cubic_showcase': TRUE_3D_CUBIC_SHOWCASE,
        'true_3d_aniso_showcase': TRUE_3D_ANISO_SHOWCASE,
        'true_3d_multi_fermi': TRUE_3D_MULTI_FERMI,
        'true_3d_complex_showcase': TRUE_3D_COMPLEX_SHOWCASE,
        # Multi-band Showcase Configurations
        'multiband_2band_showcase': MULTIBAND_2BAND_SHOWCASE,
        'multiband_2band_both': MULTIBAND_2BAND_BOTH,
        'multiband_3band_showcase': MULTIBAND_3BAND_SHOWCASE,
        'multiband_4band_showcase': MULTIBAND_4BAND_SHOWCASE,
        # UTe₂ Showcase Configurations
        'ute2_full_showcase': UTE2_FULL_SHOWCASE,
        'ute2_simplified_showcase': UTE2_SIMPLIFIED_SHOWCASE,
        # 3D Lattice Showcase Configurations
        'cubic_3d_showcase': CUBIC_3D_SHOWCASE,
        'anisotropic_showcase': ANISOTROPIC_SHOWCASE,
        'honeycomb_showcase': HONEYCOMB_SHOWCASE,
        'complex_3d_showcase': COMPLEX_3D_SHOWCASE,
        'multi_aniso_showcase': MULTI_ANISO_SHOWCASE,
    }
    
    # Check if it's a preset configuration
    if config_name in preset_configs:
        config = preset_configs[config_name]
        
        # Setup special position configurations for presets
        if config_name == 'random_5_impurities':
            setup_n_random_positions(config, 5, distributed=True)
        elif config_name == 'random_10_impurities':
            setup_n_random_positions(config, 10, distributed=True)
        elif config_name == 'random_30_impurities':
            setup_n_random_positions(config, 30, distributed=True)
        elif config_name == 'square_lattice_5_impurities':
            setup_n_random_positions(config, 5, distributed=True)
        elif config_name == 'graphene_10_impurities':
            setup_n_random_positions(config, 10, distributed=True)
        elif config_name == 'square_lattice_disorder':
            setup_n_random_positions(config, 25, distributed=True)  # Many weak impurities
        
        return config
    
    # Check if it's a dynamic random_N_impurities configuration
    if config_name.startswith('random_') and config_name.endswith('_impurities'):
        try:
            # Extract N from 'random_N_impurities'
            n_str = config_name[7:-11]  # Remove 'random_' and '_impurities'
            n_impurities = int(n_str)
            
            if n_impurities <= 0:
                raise ValueError(f"Number of impurities must be positive, got {n_impurities}")
            
            # Check if this should fall back to a preset
            fallback_name = f"random_{n_impurities}_impurities"
            if fallback_name in preset_configs:
                print(f"Using preset configuration for {fallback_name}")
                return get_config(fallback_name)
            
            # Create dynamic configuration
            print(f"Creating dynamic configuration for {n_impurities} impurities")
            return create_random_n_config(n_impurities)
            
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError(f"Invalid format '{config_name}'. Use 'random_N_impurities' where N is an integer.")
            else:
                raise e
    
    # If we get here, it's an unknown configuration
    available_presets = ', '.join(preset_configs.keys())
    raise ValueError(f"Unknown config '{config_name}'. Available presets: {available_presets}. "
                    f"Or use 'random_N_impurities' format where N is any positive integer.")


def list_available_configs():
    """Print all available configuration options."""
    configs = [
        ('high_quality_single', HIGH_QUALITY_SINGLE),
        ('fast_preview', FAST_PREVIEW),
        ('random_5_impurities', RANDOM_5_IMPURITIES),
        ('random_10_impurities', RANDOM_10_IMPURITIES),
        ('random_30_impurities', RANDOM_30_IMPURITIES),
        ('square_lattice_preview', SQUARE_LATTICE_PREVIEW),
        ('graphene_preview', GRAPHENE_PREVIEW),
        ('square_lattice_high_quality', SQUARE_LATTICE_HIGH_QUALITY),
        ('graphene_both_bands', GRAPHENE_BOTH_BANDS),
        ('square_lattice_5_impurities', SQUARE_LATTICE_5_IMPURITIES),
        ('graphene_10_impurities', GRAPHENE_10_IMPURITIES),
        ('square_lattice_disorder', SQUARE_LATTICE_DISORDER),
        # True 3D Showcase Configurations  
        ('true_3d_cubic_showcase', TRUE_3D_CUBIC_SHOWCASE),
        ('true_3d_aniso_showcase', TRUE_3D_ANISO_SHOWCASE),
        ('true_3d_multi_fermi', TRUE_3D_MULTI_FERMI),
        ('true_3d_complex_showcase', TRUE_3D_COMPLEX_SHOWCASE),
        # Multi-band Showcase Configurations
        ('multiband_2band_showcase', MULTIBAND_2BAND_SHOWCASE),
        ('multiband_2band_both', MULTIBAND_2BAND_BOTH),
        ('multiband_3band_showcase', MULTIBAND_3BAND_SHOWCASE),
        ('multiband_4band_showcase', MULTIBAND_4BAND_SHOWCASE),
        # UTe₂ Showcase Configurations
        ('ute2_full_showcase', UTE2_FULL_SHOWCASE),
        ('ute2_simplified_showcase', UTE2_SIMPLIFIED_SHOWCASE),
        # 3D Lattice Showcase Configurations
        ('cubic_3d_showcase', CUBIC_3D_SHOWCASE),
        ('anisotropic_showcase', ANISOTROPIC_SHOWCASE),
        ('honeycomb_showcase', HONEYCOMB_SHOWCASE),
        ('complex_3d_showcase', COMPLEX_3D_SHOWCASE),
        ('multi_aniso_showcase', MULTI_ANISO_SHOWCASE),
    ]
    
    print("Available QPI Simulation Configurations:")
    print("=" * 50)
    for name, config in configs:
        print(f"{name:25}: {config.description}")
        if hasattr(config, 'model_type') and config.model_type != "parabolic":
            print(f"{' ' * 25}  Model: {config.model_type}")
    
    print("\nDynamic Configurations:")
    print("-" * 25)
    print("random_N_impurities      : N randomly placed impurities (any positive integer)")
    print("                          Examples: random_1_impurities, random_7_impurities, random_50_impurities")
    print("                          Note: Falls back to preset if N=5,10,30")


if __name__ == "__main__":
    list_available_configs()