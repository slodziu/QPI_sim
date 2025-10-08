# QPI Simulation Suite

A comprehensive, modular suite for simulating Quasiparticle Interference (QPI) patterns in 2D materials using Green's function methods.

## Overview

This codebase simulates QPI around impurities in a 2D tight-binding model with parabolic dispersion. The code is designed for easy extension to multiple impurities, different scattering mechanisms, and various analysis methods.

## Key Features

- **Modular Architecture**: Clean separation of concerns with dedicated classes for different aspects
- **Multiple Impurity Support**: Easy configuration of single or multiple impurity systems
- **Advanced Signal Processing**: Sophisticated FFT processing with apodization and sub-pixel interpolation
- **Real-time Analysis**: Automatic dispersion extraction from QPI patterns
- **Flexible Configuration**: Predefined simulation scenarios with easy parameter modification
- **Professional Visualization**: High-quality animated outputs suitable for presentations

## File Structure

```
qpi_G_OOP.py           # Main simulation classes and core functionality
config.py              # Configuration system with predefined scenarios
run_qpi.py             # Simple command-line runner
outputs/               # Generated animations and analysis results
README.md              # This file
```

## Quick Start

### Basic Usage

```bash
# Run a fast preview simulation
python run_qpi.py fast_preview

# Run a high-quality single impurity simulation
python run_qpi.py high_quality_single

# See all available configurations
python run_qpi.py list
```

**Note**: All output animations and analysis results are saved to the `outputs/` folder.

### Available Configurations

- `fast_preview`: Quick simulation with reduced quality (256×256 grid, 10 frames)
- `high_quality_single`: High-resolution single impurity (1024×1024 grid, 30 frames)
- `double_impurity`: Two impurities showing interference patterns
- `triangular_cluster`: Three impurities in triangular arrangement
- `random_impurities`: Multiple randomly placed impurities
- `strong_scattering`: Strong impurity strength (unitarity limit)
- `weak_scattering`: Weak impurity strength (Born approximation)
- `high_energy_resolution`: Fine energy steps for detailed dispersion
- `wide_energy_range`: Exploration of wide energy range

## Code Architecture

### Core Classes

#### `SystemParameters`
Centralized container for all simulation parameters including:
- Grid size and physical dimensions
- Material parameters (hopping, chemical potential, broadening)
- Energy sweep configuration
- Signal processing parameters

#### `GreensFunction`
Handles all Green's function calculations:
- k-space grid setup with rotation for symmetry breaking
- Disorder generation for realistic lattice effects
- Bare Green's function calculation
- T-matrix calculation for individual impurities

#### `ImpuritySystem`
Manages impurity configurations:
- Flexible positioning system
- LDOS calculation with multiple impurities
- Easy extension for variable impurity strengths

#### `QPIAnalyzer`
Advanced signal processing for QPI analysis:
- Sophisticated apodization using tapered cosine windows
- Sub-pixel interpolation for smoother results
- Radial averaging for robust peak extraction
- Automatic dispersion extraction

#### `QPISimulation`
Main orchestration class:
- Coordinates all components
- Manages energy sweeps
- Accumulates dispersion data

#### `QPIVisualizer`
Professional visualization system:
- Three-panel animated layout
- Real-time parameter display
- Dispersion curve building
- High-quality output generation

### Signal Processing Pipeline

1. **LDOS Calculation**: Full Green's function method with T-matrix approach
2. **Physical Broadening**: Gaussian filtering to simulate experimental conditions
3. **Apodization**: Tapered cosine windowing to suppress edge artifacts
4. **Sub-pixel Interpolation**: Zoom interpolation for smoother FFT results
5. **FFT Processing**: Optimized 2D Fourier transform with DC suppression
6. **Radial Analysis**: Azimuthal averaging for robust peak detection

## Extending the Code

### Adding New Impurity Configurations

```python
from qpi_G_refactored import SystemParameters, QPISimulation, QPIVisualizer

# Define your configuration
params = SystemParameters(gridsize=512, n_frames=20)

# Define impurity positions (grid coordinates)
positions = [
    (256, 256),  # Center
    (200, 256),  # Left
    (312, 256),  # Right
]

# Run simulation
simulation = QPISimulation(params, positions)
visualizer = QPIVisualizer(simulation)
ani = visualizer.create_animation('my_config.gif')
```

### Adding New Parameters

Extend `SystemParameters` dataclass:

```python
@dataclass
class ExtendedParameters(SystemParameters):
    new_parameter: float = 1.0
    another_param: int = 5
```

### Custom Analysis

Extend `QPIAnalyzer` for new analysis methods:

```python
class CustomAnalyzer(QPIAnalyzer):
    def custom_analysis_method(self, data):
        # Your custom analysis here
        return results
```

### Variable Impurity Strengths

Extend `ImpuritySystem` for individual impurity strengths:

```python
class VariableStrengthSystem(ImpuritySystem):
    def __init__(self, positions, strengths):
        super().__init__(positions)
        self.strengths = strengths
    
    def calculate_T_matrices(self, G0, greens_function):
        # Custom T-matrix calculation with individual strengths
        pass
```

## Physics Background

### Model
- **Lattice**: 2D square lattice with nearest-neighbor hopping
- **Dispersion**: Parabolic ε(k) = k² for circular Fermi surface
- **Impurities**: Point scatterers with strength V_s
- **Method**: Full Green's function calculation using T-matrix approach

### QPI Physics
- **Mechanism**: Elastic scattering between k and k' states on Fermi surface
- **Pattern**: Interference between scattered and unscattered electrons
- **Ring Structure**: QPI rings appear at q-vectors connecting Fermi surface points
- **Energy Dependence**: Ring radius scales with √E for parabolic dispersion

### Observed Features
- **Intra-band Scattering**: Rings at q ≈ k_F (observed in this simulation)
- **Inter-band Scattering**: Would show rings at q ≈ 2k_F (in multi-band systems)
- **Dispersion Extraction**: Automatic fitting of E vs k relationship

## Performance Notes

- **Grid Size**: 512×512 provides good quality/speed balance
- **Energy Resolution**: 20 frames typically sufficient for dispersion extraction
- **Processing**: Signal processing pipeline adds ~2x computational overhead but dramatically improves quality
- **Memory**: Peak usage ~200MB for 512×512 grid
- **Speed**: ~1-2 seconds per energy frame on modern hardware

## Output Files

Each simulation generates:
- **Animation GIF**: Three-panel animation showing evolution
- **Console Output**: Real-time progress and parameter information
- **Dispersion Data**: Accumulated in simulation object for further analysis

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure numpy, scipy, matplotlib are installed
2. **Memory Issues**: Reduce gridsize for large simulations
3. **Slow Performance**: Use fast_preview config for testing
4. **Poor Quality**: Increase gridsize and/or zoom_factor

### Parameter Tuning

- **eta**: Controls energy broadening (0.05-0.2 typical)
- **gridsize**: Balance between quality and speed (256-1024)
- **apodization_alpha**: Controls edge suppression (0.2-0.4)
- **physical_broadening_sigma**: Simulates experimental resolution (0.5-1.0)

## Contributing

To add new features:
1. Identify the appropriate class to extend
2. Add new parameters to `SystemParameters` if needed
3. Add configuration to `config.py` for easy access
4. Test with existing configurations to ensure compatibility

## References

- Green's function methods in condensed matter physics
- T-matrix approach for impurity scattering
- QPI theory and experimental techniques
- Advanced signal processing for spectroscopic data