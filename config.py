"""
QPI Simulation Configuration System

This module defines simulation parameters and configurations for QPI (Quasiparticle Interference) 
analysis using parabolic dispersion and Green's function methods.

Key configurations:
- HIGH_QUALITY_SINGLE: Single impurity high-resolution simulation
- RANDOM_30_IMPURITIES: Multiple impurity simulation for statistical QPI patterns
- FAST_PREVIEW: Quick preview with reduced resolution
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import copy


@dataclass
class SimulationConfig:
    """Configuration parameters for QPI simulation"""
    # Basic simulation parameters
    name: str = "default"
    description: str = "Default QPI simulation"
    
    # Grid and energy parameters  
    L: float = 100.0  # System size (physical units)
    gridsize: int = 1024  # Grid resolution
    n_frames: int = 20  # Number of energy frames for animations
    E_min: float = 10.0  # Minimum energy
    E_max: float = 20.0  # Maximum energy
    
    # Physical parameters (parabolic dispersion)
    t: float = 1.0  # Hopping parameter (sets energy scale)
    mu: float = 15.0  # Chemical potential
    eta: float = 0.05  # Broadening parameter
    V_s: float = 1.0  # Impurity scattering strength
    
    # Visualization parameters
    rotation_angle: float = 0.0  # No rotation
    disorder_strength: float = 0.0  # No disorder
    zoom_factor: float = 1.0  # No zoom
    
    # Impurity configuration
    impurity_positions: List[Tuple[int, int]] = None
    
    def get_impurity_positions(self) -> List[Tuple[int, int]]:
        """Get impurity positions, calculating if needed."""
        if self.impurity_positions is None:
            # Default: single impurity at center
            center = self.gridsize // 2
            self.impurity_positions = [(center, center)]
        return self.impurity_positions


# ============================================================================
# ESSENTIAL CONFIGURATION PRESETS
# ============================================================================

# High-quality single impurity simulation
HIGH_QUALITY_SINGLE = SimulationConfig(
    name="high_quality_single",
    description="High-resolution single impurity simulation",
    gridsize=1024,
    n_frames=10,
    E_min=1.0,
    E_max=5.0,
    V_s=0.7,  # Moderate strength for clean scattering
)

# Fast preview simulation
FAST_PREVIEW = SimulationConfig(
    name="fast_preview",
    description="Quick preview with reduced quality",
    gridsize=512,
    n_frames=5,
    E_min=10.0,
    E_max=20.0,
    V_s=1.0,  # Standard strength
)

# Random impurity configurations
RANDOM_5_IMPURITIES = SimulationConfig(
    name="random_5_impurities",
    description="Five randomly placed impurities",
    gridsize=512,
    n_frames=10,
    E_min=10.0,
    E_max=20.0,
    V_s=1.0,  # Standard strength for multiple impurities
)

RANDOM_10_IMPURITIES = SimulationConfig(
    name="random_10_impurities", 
    description="Ten randomly placed impurities",
    gridsize=1024,
    n_frames=10,
    E_min=10.0,
    E_max=20.0,
    V_s=1.0,
)

RANDOM_30_IMPURITIES = SimulationConfig(
    name="random_30_impurities",
    description="Thirty randomly placed impurities",
    gridsize=1024,  # Larger grid for better spacing
    n_frames=10,    # Reduced frames for faster computation
    E_min=10.0,
    E_max=20.0,
    V_s=1.5,       # Much weaker impurities for numerical stability
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


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_n_random_positions(config: SimulationConfig, n_impurities: int, seed: int = 42, 
                             margin: int = None, distributed: bool = True):
    """
    Setup n randomly placed impurity positions.
    
    Args:
        config: Configuration object to modify
        n_impurities: Number of impurities to place
        seed: Random seed for reproducibility
        margin: Margin from edges in pixels (if None, use gridsize//32)
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
        # For clustered placement, use smaller area
        region_size = min(config.gridsize // 3, 150)
        available_area = region_size**2
    
    # Calculate density and adjust min_distance accordingly
    density = n_impurities / available_area
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
        region_radius = (config.gridsize - 2*margin) // 3
        x_min = center - region_radius
        x_max = center + region_radius
        y_min = center - region_radius
        y_max = center + region_radius
        print(f"Clustering {n_impurities} impurities in central region ({y_min}-{y_max}, {x_min}-{x_max}), min_distance={min_distance}")
    
    for i in range(n_impurities):
        attempts = 0
        placed = False
        
        while attempts < max_attempts and not placed:
            # Generate random position (row, col convention)
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
        
        # If placement fails, try with relaxed constraint
        if not placed and len(config.impurity_positions) > 0:
            relaxed_min_distance = min_distance * 0.7
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
    config = copy.deepcopy(RANDOM_N_IMPURITIES_TEMPLATE)
    
    # Update name and description
    config.name = f"random_{n_impurities}_impurities"
    config.description = f"{n_impurities} randomly placed impurities"
    
    # Adjust parameters based on number of impurities
    if n_impurities == 1:
        config.V_s = -1.0
        config.n_frames = 20
        config.gridsize = max(512, config.gridsize)
    elif n_impurities <= 5:
        config.V_s = -1.0
        config.n_frames = 20
    elif n_impurities <= 15:
        config.V_s = -1.0
        config.n_frames = 15
    elif n_impurities <= 25:
        config.V_s = -0.7
        config.n_frames = 12
        config.gridsize = 1024 if config.gridsize < 1024 else config.gridsize
    else:
        config.V_s = -0.3
        config.n_frames = 10
        config.gridsize = 1024 if config.gridsize < 1024 else config.gridsize
    
    # Setup random positions
    setup_n_random_positions(config, n_impurities, distributed=True)
    
    return config


# ============================================================================
# CONFIGURATION ACCESS FUNCTIONS
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
        
        return config
    
    # Handle dynamic N-impurity configurations
    if config_name.startswith('random_') and config_name.endswith('_impurities'):
        try:
            n_str = config_name.replace('random_', '').replace('_impurities', '')
            n_impurities = int(n_str)
            return create_random_n_config(n_impurities)
        except ValueError:
            pass
    
    # If not found, raise error
    available = list(preset_configs.keys()) + ['random_N_impurities (where N is any integer)']
    raise ValueError(f"Unknown configuration '{config_name}'. Available: {available}")


def list_configurations():
    """List all available configurations."""
    print("Available QPI configurations:")
    print("- high_quality_single: High-resolution single impurity")
    print("- fast_preview: Quick preview with reduced resolution") 
    print("- random_5_impurities: Five randomly placed impurities")
    print("- random_10_impurities: Ten randomly placed impurities")
    print("- random_30_impurities: Thirty randomly placed impurities")
    print("- random_N_impurities: Dynamic N impurities (e.g., random_7_impurities)")