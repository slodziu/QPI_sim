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
    eta: float = 0.1
    V_s: float = 1.0
    
    # Energy sweep
    E_min: float = 3.0
    E_max: float = 5.0
    n_frames: int = 20
    
    # Processing parameters
    rotation_angle: float = np.pi/12
    disorder_strength: float = 0.05
    physical_broadening_sigma: float = 0.8
    apodization_alpha: float = 0.3
    zoom_factor: float = 1.5
    
    # Impurity configuration
    impurity_positions: List[Tuple[int, int]] = None
    
    def get_impurity_positions(self) -> List[Tuple[int, int]]:
        """Get impurity positions, calculating if needed."""
        if self.impurity_positions is None:
            # Default: single impurity OFF-CENTER to avoid periodic boundary splitting
            center = self.gridsize // 2
            # Place at 3/8 of the way across to avoid center and edge issues
            offset = self.gridsize // 8
            return [(center - offset, center - offset)]
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
    V_s=-2.0,  # Attractive impurity
    physical_broadening_sigma=0.5,
    apodization_alpha=0.25
)

# Fast preview simulation
FAST_PREVIEW = SimulationConfig(
    name="fast_preview",
    description="Quick preview with reduced quality",
    gridsize=256,
    n_frames=10,
    E_min=5.0,
    E_max=15.0,
    zoom_factor=1.2,
    apodization_alpha=0.5  # More aggressive windowing to suppress boundary artifacts
)

# N-impurity random configurations
RANDOM_2_IMPURITIES = SimulationConfig(
    name="random_2_impurities",
    description="Two randomly placed impurities",
    gridsize=512,
    n_frames=20,
    E_min=5.0,
    E_max=25.0,
    V_s=-2.0,  # Attractive impurities for LDOS enhancement
    disorder_strength=0.02
)

RANDOM_3_IMPURITIES = SimulationConfig(
    name="random_3_impurities", 
    description="Three randomly placed impurities",
    gridsize=512,
    n_frames=20,
    E_min=5.0,
    E_max=25.0,
    V_s=-2.0,  # Attractive impurities
    disorder_strength=0.02
)

RANDOM_5_IMPURITIES = SimulationConfig(
    name="random_5_impurities",
    description="Five randomly placed impurities",
    gridsize=512,
    n_frames=20,
    E_min=5.0,
    E_max=25.0,
    V_s=-1.5,  # Moderately attractive
    disorder_strength=0.02
)

RANDOM_10_IMPURITIES = SimulationConfig(
    name="random_10_impurities",
    description="Ten randomly placed impurities",
    gridsize=512,
    n_frames=10,
    E_min=5.0,
    E_max=25.0,
    V_s=-1.2,  # Mildly attractive
    disorder_strength=0.01
)

RANDOM_30_IMPURITIES = SimulationConfig(
    name="random_30_impurities",
    description="Thirty randomly placed impurities",
    gridsize=512,
    n_frames=30,
    E_min=5.0,
    E_max=25.0,
    V_s=-1.0,  # Weakly attractive
    disorder_strength=0.01
)
# Distributed impurity configurations
DISTRIBUTED_5_IMPURITIES = SimulationConfig(
    name="distributed_5_impurities",
    description="Five impurities distributed across the grid",
    gridsize=512,
    n_frames=20,
    E_min=5.0,
    E_max=25.0,
    V_s=-3.0,  # Strong attractive for clear visibility
    disorder_strength=0.01
)

DISTRIBUTED_10_IMPURITIES = SimulationConfig(
    name="distributed_10_impurities", 
    description="Ten impurities distributed across the grid",
    gridsize=512,
    n_frames=10,  # Reduced from 20 for faster execution
    E_min=5.0,
    E_max=25.0,
    V_s=-2.5,  # Strong attractive for visibility
    disorder_strength=0.01
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
    min_distance = max(8, min(25, int(np.sqrt(1.0 / density) * 0.5)))
    
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

# Strong scattering regime
STRONG_SCATTERING = SimulationConfig(
    name="strong_scattering",
    description="Strong impurity scattering (unitarity limit)",
    gridsize=512,
    n_frames=20,
    V_s=10.0,  # Very strong impurity
    eta=0.05,  # Sharper features
    E_min=5.0,
    E_max=25.0
)

# Weak scattering regime
WEAK_SCATTERING = SimulationConfig(
    name="weak_scattering",
    description="Weak impurity scattering (Born approximation)",
    gridsize=512,
    n_frames=20,
    V_s=0.1,   # Weak impurity
    eta=0.15,  # Broader features
    E_min=5.0,
    E_max=25.0
)

# High energy resolution
HIGH_ENERGY_RESOLUTION = SimulationConfig(
    name="high_energy_resolution",
    description="Fine energy steps for detailed dispersion",
    gridsize=512,
    n_frames=50,  # Many energy steps
    E_min=5.0,
    E_max=20.0,
    eta=0.05  # Sharp energy resolution
)

# Wide energy range
WIDE_ENERGY_RANGE = SimulationConfig(
    name="wide_energy_range",
    description="Exploration of wide energy range",
    gridsize=512,
    n_frames=25,
    E_min=1.0,
    E_max=50.0
)


# ============================================================================
# CONFIGURATION SELECTION FUNCTION
# ============================================================================

def get_config(config_name: str) -> SimulationConfig:
    """
    Get a predefined configuration by name.
    
    Available configurations:
    - 'high_quality_single': High-resolution single impurity
    - 'fast_preview': Quick preview simulation
    - 'random_2_impurities': Two randomly placed impurities (clustered)
    - 'random_3_impurities': Three randomly placed impurities (clustered)
    - 'random_5_impurities': Five randomly placed impurities (clustered)
    - 'random_10_impurities': Ten randomly placed impurities (clustered)
    - 'distributed_5_impurities': Five impurities distributed across grid
    - 'distributed_10_impurities': Ten impurities distributed across grid
    - 'strong_scattering': Strong impurity strength
    - 'weak_scattering': Weak impurity strength
    - 'high_energy_resolution': Many energy steps
    - 'wide_energy_range': Large energy range
    """
    
    configs = {
        'high_quality_single': HIGH_QUALITY_SINGLE,
        'fast_preview': FAST_PREVIEW,
        'random_2_impurities': RANDOM_2_IMPURITIES,
        'random_3_impurities': RANDOM_3_IMPURITIES,
        'random_5_impurities': RANDOM_5_IMPURITIES,
        'random_10_impurities': RANDOM_10_IMPURITIES,
        'random_30_impurities': RANDOM_30_IMPURITIES,
        'distributed_5_impurities': DISTRIBUTED_5_IMPURITIES,
        'distributed_10_impurities': DISTRIBUTED_10_IMPURITIES,
        'strong_scattering': STRONG_SCATTERING,
        'weak_scattering': WEAK_SCATTERING,
        'high_energy_resolution': HIGH_ENERGY_RESOLUTION,
        'wide_energy_range': WIDE_ENERGY_RANGE,
    }
    
    if config_name not in configs:
        available = ', '.join(configs.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    config = configs[config_name]
    
    # Setup special position configurations
    # All "random" configs now use distributed=True for truly random placement
    if config_name == 'random_2_impurities':
        setup_n_random_positions(config, 2, distributed=True)
    elif config_name == 'random_3_impurities':
        setup_n_random_positions(config, 3, distributed=True)
    elif config_name == 'random_5_impurities':
        setup_n_random_positions(config, 5, distributed=True)
    elif config_name == 'random_10_impurities':
        setup_n_random_positions(config, 10, distributed=True)
    elif config_name == 'random_30_impurities':
        setup_n_random_positions(config, 30, distributed=True)
    elif config_name == 'distributed_5_impurities':
        setup_n_random_positions(config, 5, distributed=True)
    elif config_name == 'distributed_10_impurities':
        setup_n_random_positions(config, 10, distributed=True)
    
    return config


def list_available_configs():
    """Print all available configuration options."""
    configs = [
        ('high_quality_single', HIGH_QUALITY_SINGLE),
        ('fast_preview', FAST_PREVIEW),
        ('random_2_impurities', RANDOM_2_IMPURITIES),
        ('random_3_impurities', RANDOM_3_IMPURITIES),
        ('random_5_impurities', RANDOM_5_IMPURITIES),
        ('random_10_impurities', RANDOM_10_IMPURITIES),
        ('random_30_impurities', RANDOM_30_IMPURITIES),
        ('distributed_5_impurities', DISTRIBUTED_5_IMPURITIES),
        ('distributed_10_impurities', DISTRIBUTED_10_IMPURITIES),
        ('strong_scattering', STRONG_SCATTERING),
        ('weak_scattering', WEAK_SCATTERING),
        ('high_energy_resolution', HIGH_ENERGY_RESOLUTION),
        ('wide_energy_range', WIDE_ENERGY_RANGE),
    ]
    
    print("Available QPI Simulation Configurations:")
    print("=" * 50)
    for name, config in configs:
        print(f"{name:25}: {config.description}")


if __name__ == "__main__":
    list_available_configs()