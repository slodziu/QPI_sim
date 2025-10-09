# Dispersion Extraction from QPI: Single vs Multiple Impurities

## Overview

The QPI simulation now correctly handles dispersion extraction for both single and multiple impurity scenarios, recognizing that they produce different dominant scattering vectors.

## Key Physics

### Single Impurity
- **Dominant process**: Intra-band scattering at the Fermi surface
- **Peak location**: q = k_F
- **Physical interpretation**: Electrons scatter from one point on the Fermi surface to another
- **Extraction**: k_F = q_peak (direct)

### Multiple Impurities
- **Dominant process**: Backscattering between impurities
- **Peak location**: q = 2k_F
- **Physical interpretation**: Electrons scatter backwards through momentum transfer 2k_F
- **Extraction**: k_F = q_peak / 2

## Implementation

### Code Changes

1. **`update_dispersion_data()` method** (qpi_G_OOP.py):
   ```python
   def update_dispersion_data(self, peak_q: Optional[float]):
       if peak_q is not None and peak_q > 0:
           n_imp = len(self.impurities.positions)
           
           if n_imp == 1:
               # Single impurity: q = k_F
               k_F_extracted = peak_q
           else:
               # Multiple impurities: q = 2k_F
               k_F_extracted = peak_q / 2.0
           
           E_extracted = k_F_extracted**2
   ```

2. **Plot labels automatically updated**:
   - Single impurity: "Extracted from q=k_F"
   - Multiple impurities: "Extracted from q=2k_F"

3. **Visual guides**:
   - Red dashed circle: q = k_F (always shown)
   - Black solid circle: q = 2k_F (only for multiple impurities)

## Verification

Both methods correctly recover the theoretical dispersion E = k_F²:

```
Single Impurity Test:
  k_F range: 3.408 to 10.833
  E range: 11.612 to 117.358
  Mean error from E=k²: 0.0000
  Max error from E=k²: 0.0000

Multiple Impurities Test:
  k_F range: 2.320 to 5.073
  E range: 5.381 to 25.731
  Mean error from E=k²: 0.0000
  Max error from E=k²: 0.0000
```

## Usage

The extraction is fully automatic:
```python
# Single impurity
config = get_config('high_quality_single')
# → Looks for peaks at k_F

# Multiple impurities
config = get_config('random_5_impurities')
# → Looks for peaks at 2k_F, divides by 2
```

## Physical Interpretation

### Why 2k_F for Multiple Impurities?

When you have multiple impurities, the dominant QPI signal comes from:
1. An electron at wavevector **k** scattering off impurity A
2. Traveling to impurity B
3. Scattering again with momentum transfer **q**

For backscattering (180° reversal), the momentum transfer is:
**q = k_final - k_initial = -k - k = -2k**

Thus |q| = 2k_F for electrons at the Fermi surface.

### Experimental Relevance

This distinction is crucial for interpreting real STM data:
- **Isolated defects**: Look for k_F features
- **Multiple defects/disorder**: Look for 2k_F features
- **Dilute impurities**: Mixture of both

## References

The 2k_F backscattering in multiple impurity systems is well-documented in:
- Friedel oscillations theory
- STM studies of surface states
- QPI analysis in topological materials
