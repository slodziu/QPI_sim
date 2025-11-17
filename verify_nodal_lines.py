#!/usr/bin/env python3
"""
Verify that gap magnitudes are zero along the theoretically predicted nodal lines.

Nodal lines (where gap = 0):
- B1u: kx = 0, ±π/a  AND  ky = 0, ±π/b  (4 lines parallel to kz axis)
- B2u: kx = 0, ±π/a  AND  kz = 0, ±π/c  (4 lines parallel to ky axis)  
- B3u: ky = 0, ±π/b  AND  kz = 0, ±π/c  (4 lines parallel to kx axis)
"""

import numpy as np
from UTe2_fixed import calculate_gap_magnitude

# UTe2 lattice parameters (Angstroms)
a = 4.07
b = 5.83
c = 13.78

def test_nodal_line(pairing_type, fixed_coords, varied_coord_name, description):
    """Test if gap is zero along a nodal line."""
    print(f"\n{pairing_type} - {description}:")
    print("-" * 60)
    
    # Create test points along the line
    if varied_coord_name == 'kx':
        kx_vals = np.linspace(-np.pi/a, np.pi/a, 20)
        ky_vals = np.full_like(kx_vals, fixed_coords['ky'])
        kz_vals = np.full_like(kx_vals, fixed_coords['kz'])
    elif varied_coord_name == 'ky':
        ky_vals = np.linspace(-np.pi/b, np.pi/b, 20)
        kx_vals = np.full_like(ky_vals, fixed_coords['kx'])
        kz_vals = np.full_like(ky_vals, fixed_coords['kz'])
    elif varied_coord_name == 'kz':
        kz_vals = np.linspace(-np.pi/c, np.pi/c, 20)
        kx_vals = np.full_like(kz_vals, fixed_coords['kx'])
        ky_vals = np.full_like(kz_vals, fixed_coords['ky'])
    
    # Calculate gap along the line
    gaps = []
    for kx, ky, kz in zip(kx_vals, ky_vals, kz_vals):
        gap = calculate_gap_magnitude(
            np.array([[kx]]), np.array([[ky]]), kz, pairing_type=pairing_type
        )[0, 0]
        gaps.append(gap)
    
    gaps = np.array(gaps)
    max_gap = np.max(gaps)
    mean_gap = np.mean(gaps)
    
    # Check if line is nodal (gap should be ~0 everywhere)
    is_nodal = max_gap < 1e-10
    
    if is_nodal:
        print(f"  ✓ NODAL LINE confirmed: max gap = {max_gap:.2e} eV")
    else:
        print(f"  ✗ NOT a nodal line: max gap = {max_gap:.6f} eV, mean = {mean_gap:.6f} eV")
    
    return is_nodal

print("=" * 60)
print("NODAL LINE VERIFICATION TEST")
print("=" * 60)
print("\nTesting if gap magnitude is zero along predicted nodal lines...")

# B1u: d ∝ (sin ky·b, sin kx·a, 0)
# Nodal lines: kx=0, ky=0 (and at BZ boundaries kx=±π/a, ky=±π/b)
print("\n" + "="*60)
print("B1u: d ∝ (sin ky·b, sin kx·a, 0)")
print("Expected nodal lines parallel to kz axis at:")
print("  kx=0, ky=0  and  kx=±π/a, ky=±π/b")
print("="*60)

b1u_tests = [
    ({'kx': 0, 'ky': 0}, 'kz', "Line at kx=0, ky=0"),
    ({'kx': np.pi/a, 'ky': 0}, 'kz', "Line at kx=π/a, ky=0"),
    ({'kx': 0, 'ky': np.pi/b}, 'kz', "Line at kx=0, ky=π/b"),
    ({'kx': np.pi/a, 'ky': np.pi/b}, 'kz', "Line at kx=π/a, ky=π/b"),
]

b1u_results = [test_nodal_line("B1u", *test) for test in b1u_tests]

# B2u: d ∝ (sin kz·c, 0, sin kx·a)
# Nodal lines: kx=0, kz=0 (and at BZ boundaries)
print("\n" + "="*60)
print("B2u: d ∝ (sin kz·c, 0, sin kx·a)")
print("Expected nodal lines parallel to ky axis at:")
print("  kx=0, kz=0  and  kx=±π/a, kz=±π/c")
print("="*60)

b2u_tests = [
    ({'kx': 0, 'kz': 0}, 'ky', "Line at kx=0, kz=0"),
    ({'kx': np.pi/a, 'kz': 0}, 'ky', "Line at kx=π/a, kz=0"),
    ({'kx': 0, 'kz': np.pi/c}, 'ky', "Line at kx=0, kz=π/c"),
    ({'kx': np.pi/a, 'kz': np.pi/c}, 'ky', "Line at kx=π/a, kz=π/c"),
]

b2u_results = [test_nodal_line("B2u", *test) for test in b2u_tests]

# B3u: d ∝ (0, sin kz·c, sin ky·b)
# Nodal lines: ky=0, kz=0 (and at BZ boundaries)
print("\n" + "="*60)
print("B3u: d ∝ (0, sin kz·c, sin ky·b)")
print("Expected nodal lines parallel to kx axis at:")
print("  ky=0, kz=0  and  ky=±π/b, kz=±π/c")
print("="*60)

b3u_tests = [
    ({'ky': 0, 'kz': 0}, 'kx', "Line at ky=0, kz=0"),
    ({'ky': np.pi/b, 'kz': 0}, 'kx', "Line at ky=π/b, kz=0"),
    ({'ky': 0, 'kz': np.pi/c}, 'kx', "Line at ky=0, kz=π/c"),
    ({'ky': np.pi/b, 'kz': np.pi/c}, 'kx', "Line at ky=π/b, kz=π/c"),
]

b3u_results = [test_nodal_line("B3u", *test) for test in b3u_tests]

# Summary
print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"B1u: {sum(b1u_results)}/{len(b1u_results)} nodal lines confirmed")
print(f"B2u: {sum(b2u_results)}/{len(b2u_results)} nodal lines confirmed")
print(f"B3u: {sum(b3u_results)}/{len(b3u_results)} nodal lines confirmed")

if all(b1u_results + b2u_results + b3u_results):
    print("\n✓✓✓ All nodal lines are correctly positioned! ✓✓✓")
    print("\nNow the 3D plots should show gap nodes where these lines")
    print("intersect the Fermi surface.")
else:
    print("\n✗✗✗ Some nodal lines are missing or incorrect! ✗✗✗")
