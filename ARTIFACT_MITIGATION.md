# QPI Artifact Mitigation Strategies

This document explains the circular artifact features you see within the 2k_F ring and how to mitigate them.

## What Causes Circular Artifacts?

1. **Periodic Boundary Effects**: FFT assumes periodic boundaries, creating interference patterns
2. **DC Component**: Strong zero-frequency component creates radial streaks
3. **Low-Frequency Leakage**: Large-scale LDOS variations create low-q artifacts
4. **Impurity Clustering**: Multiple nearby impurities create sub-structures
5. **Window Function Artifacts**: Apodization can cause ringing

## Mitigation Strategies Implemented

### 1. Real-Space High-Pass Filtering
**Parameter**: `high_pass_strength` (default: 0.3)

Removes large-scale background trends before FFT.
- 0.0 = no filtering (shows all artifacts)
- 0.2-0.4 = mild filtering (recommended)
- 0.5+ = strong filtering (may remove real features)

```python
params = SystemParameters(
    high_pass_strength=0.3  # Adjust between 0.0 and 0.5
)
```

### 2. Enhanced Low-Frequency Suppression
**Parameters**: 
- `low_freq_suppress_radius` (default: 10 pixels)
- `low_freq_transition_radius` (default: 20 pixels)

Creates a gradual high-pass filter in momentum space.
- Heavily suppresses DC component (r < suppress_radius)
- Gradual transition (suppress_radius < r < transition_radius)
- Full signal (r > transition_radius)

```python
params = SystemParameters(
    low_freq_suppress_radius=10,    # Increase for more aggressive suppression
    low_freq_transition_radius=20   # Larger = smoother transition
)
```

### 3. Improved Edge Tapering
Smoother cosine taper at real-space edges reduces FFT artifacts.

### 4. Radial Average Subtraction (Optional)
**Parameters**:
- `subtract_radial_average` (default: False)
- `radial_average_strength` (default: 0.5)

Subtracts azimuthally-averaged background to remove radially symmetric artifacts.
⚠️ WARNING: This can also remove real radially-symmetric QPI features!

```python
params = SystemParameters(
    subtract_radial_average=True,   # Enable radial subtraction
    radial_average_strength=0.5     # 0.0 = none, 1.0 = full subtraction
)
```

## Recommended Settings

### For Clean QPI Patterns (Recommended)
```python
params = SystemParameters(
    gridsize=512,
    high_pass_strength=0.3,
    low_freq_suppress_radius=10,
    low_freq_transition_radius=20,
    subtract_radial_average=False,  # Keep False to preserve QPI features
    physical_broadening_sigma=0.8,
    apodization_alpha=0.3
)
```

### For Very Noisy Data
```python
params = SystemParameters(
    gridsize=512,
    high_pass_strength=0.4,         # Stronger filtering
    low_freq_suppress_radius=15,    # More aggressive
    low_freq_transition_radius=30,
    subtract_radial_average=True,   # Try enabling
    radial_average_strength=0.3,    # Partial subtraction
    physical_broadening_sigma=1.2,  # More smoothing
    apodization_alpha=0.4
)
```

### For Minimal Processing (See All Features + Artifacts)
```python
params = SystemParameters(
    gridsize=512,
    high_pass_strength=0.0,
    low_freq_suppress_radius=5,
    low_freq_transition_radius=10,
    subtract_radial_average=False,
    physical_broadening_sigma=0.5,
    apodization_alpha=0.2
)
```

## Testing Different Settings

Use the test script to compare different settings:

```bash
python test_artifact_suppression.py
```

This creates a comparison figure showing:
- Top row: Real-space LDOS
- Bottom row: Momentum-space FFT with 2k_F circles

Look for:
- ✓ Cleaner region inside 2k_F circle
- ✓ Preserved ring at 2k_F radius  
- ✓ Less radial streaking from DC
- ✗ Don't over-suppress real QPI features!

## Additional Physical Solutions

Beyond signal processing, you can reduce artifacts at the source:

1. **Use Larger Grids**: `gridsize=1024` or `2048`
   - More k-space resolution
   - Edge effects are relatively smaller

2. **Increase Edge Margins**: Keep impurities far from edges
   ```python
   setup_n_random_positions(config, n_impurities, margin=gridsize//4)
   ```

3. **Optimize Impurity Positions**: 
   - Avoid clustering (use `distributed=True`)
   - Maintain minimum separation

4. **Tune Physical Parameters**:
   - Smaller `eta` (broadening) = sharper features
   - Larger `V_s` (impurity strength) = stronger signal

## Theory: What You Should See

For a 2D system with parabolic dispersion (E = k²):

- **Single impurity**: Ring at q = 2k_F (backscattering)
- **Multiple impurities**: 
  - Ring at q = 2k_F (intra-band backscattering)
  - Weaker features at q = k_F (forward scattering between impurities)
  - Random interference patterns from multiple scattering

The 2k_F ring should be the dominant feature. Circular patterns inside this ring are typically artifacts, not physics.
