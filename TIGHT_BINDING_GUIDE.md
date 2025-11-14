# Tight-Binding QPI Simulation Guide

This guide shows how to run QPI simulations with tight-binding models and multiple impurities.

## Quick Start

### 1. List Available Configurations
```bash
python run_qpi.py list
```

### 2. Single Impurity Tight-Binding Models
```bash
# Square lattice with single impurity
python run_qpi.py square_lattice_preview

# Graphene with single impurity
python run_qpi.py graphene_preview

# High-quality square lattice with next-nearest neighbor hopping
python run_qpi.py square_lattice_high_quality

# Graphene with both valence and conduction bands
python run_qpi.py graphene_both_bands
```

### 3. Multi-Impurity Tight-Binding Models
```bash
# Square lattice with 5 random impurities
python run_qpi.py square_lattice_5_impurities

# Graphene with 10 random impurities  
python run_qpi.py graphene_10_impurities

# Square lattice with many weak impurities (disorder study)
python run_qpi.py square_lattice_disorder
```

### 4. Dynamic Multi-Impurity Configurations
You can create any number of random impurities with any tight-binding model:

```bash
# 7 random impurities (uses default parabolic model)
python run_qpi.py random_7_impurities

# 15 random impurities
python run_qpi.py random_15_impurities

# 50 random impurities (weak disorder)
python run_qpi.py random_50_impurities
```

## Available Tight-Binding Models

### 1. Square Lattice (`square_lattice`)
- **Dispersion**: ε(k) = -2t(cos(kx) + cos(ky)) - 4t'cos(kx)cos(ky)  
- **Parameters**:
  - `t`: Nearest-neighbor hopping (default: 1.0)
  - `t_prime`: Next-nearest-neighbor hopping (default: 0.0)
- **Energy Range**: Typically -4t to 4t (without t')

### 2. Graphene (`graphene`)
- **Dispersion**: Two bands with Dirac cones at K points
- **Parameters**:
  - `t`: Hopping parameter (default: 2.7 eV)
- **Features**:
  - Two sublattices (A and B)
  - Linear dispersion near Fermi level
  - Can analyze single band or both bands

### 3. Parabolic (`parabolic`)
- **Dispersion**: ε(k) = k² (for backward compatibility)
- Default model when no `model_type` is specified

## Configuration Parameters

### Key Parameters for Tight-Binding Models:
- `model_type`: "square_lattice", "graphene", or "parabolic" 
- `t`: Hopping parameter
- `t_prime`: Next-nearest-neighbor hopping (square lattice only)
- `mu`: Chemical potential (Fermi level)
- `use_all_bands`: Sum over all bands (True) or use single band (False)
- `band_index`: Which band to analyze if use_all_bands=False

### Multi-Impurity Parameters:
- `V_s`: Impurity potential strength
- `eta`: Energy broadening (smaller = sharper features)
- `E_min`, `E_max`: Energy range for sweep
- `n_frames`: Number of energy points

## Creating Custom Multi-Impurity Configurations

### Method 1: Modify config.py
Add a new configuration in `config.py`:

```python
MY_CUSTOM_CONFIG = SimulationConfig(
    name="my_custom_config",
    description="Custom tight-binding configuration", 
    gridsize=512,
    n_frames=20,
    E_min=-2.0,
    E_max=2.0,
    t=1.0,
    mu=0.5,
    eta=0.05,
    V_s=0.3,
    model_type="square_lattice",  # or "graphene" 
    t_prime=0.1,
    use_all_bands=False,
    band_index=0
)
```

Then add it to the `preset_configs` dictionary and handle impurity placement.

### Method 2: Use Dynamic Configuration
For N random impurities, just run:
```bash
python run_qpi.py random_N_impurities
```
This automatically creates N randomly placed impurities with reasonable parameters.

## Understanding the Output

### Files Generated:
- `qpi_[config]_[type].mp4`: Main animation
- `qpi_[config]_fourier.mp4`: Fourier space analysis
- `qpi_[config]_only.mp4`: QPI patterns only
- `qpi_[config]_snapshot.png`: Single energy snapshot
- `frames/qpi/`: Individual QPI frames
- `frames/fourier/`: Individual Fourier analysis frames

### Interpreting Results:
1. **Real Space LDOS**: Shows interference patterns around impurities
2. **Momentum Space FFT**: Shows scattering vectors (2k_F peaks)
3. **Dispersion Extraction**: Analyzes energy dependence of QPI patterns

## Tips for Multi-Impurity Simulations

### Impurity Strength Guidelines:
- **Single impurity**: V_s ~ 0.5-1.0 (moderate to strong)
- **Few impurities (2-5)**: V_s ~ 0.3-0.5 (moderate) 
- **Many impurities (10+)**: V_s ~ 0.1-0.3 (weak to avoid numerical issues)
- **Disorder studies (20+)**: V_s ~ 0.05-0.2 (very weak)

### Performance Considerations:
- Larger grids (1024x1024) for many impurities
- Fewer frames for quick preview
- Higher broadening (eta) for many impurities

### Physical Considerations:
- Multiple impurities create complex interference patterns
- Weak disorder preserves band structure features
- Strong impurities can create localized states
- Multiple scattering between impurities is included exactly

## Examples

### Simple Square Lattice Study:
```bash
# Single impurity
python run_qpi.py square_lattice_preview

# Multiple impurities  
python run_qpi.py square_lattice_5_impurities

# Disorder
python run_qpi.py square_lattice_disorder
```

### Graphene Comparison:
```bash
# Single band
python run_qpi.py graphene_preview

# Both bands
python run_qpi.py graphene_both_bands

# Multiple impurities
python run_qpi.py graphene_10_impurities
```

### Custom Disorder Study:
```bash
# Different impurity densities
python run_qpi.py random_5_impurities
python run_qpi.py random_15_impurities  
python run_qpi.py random_30_impurities
```

The simulations will automatically handle the tight-binding band structure, multi-impurity scattering, and generate comprehensive visualizations showing how the QPI patterns evolve with energy.