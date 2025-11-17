#!/usr/bin/env python3
"""
Verify that gap nodes appear at the theoretically predicted locations.

From the paper:
- B1u: d ∝ (sin ky·b, sin kx·a, 0) → zeros at ky = 0, ±π/b, kx = 0, ±π/a
- B2u: d ∝ (sin kz·c, 0, sin kx·a) → zeros at kz = 0, ±π/c, kx = 0, ±π/a
- B3u: d ∝ (0, sin kz·c, sin ky·b) → zeros at kz = 0, ±π/c, ky = 0, ±π/b
"""

import numpy as np
from UTe2_fixed import calculate_gap_magnitude

# UTe2 lattice parameters (Angstroms)
a = 4.07
b = 5.83
c = 13.78

def test_gap_nodes(pairing_type, test_points, description):
    """Test if gap magnitude is zero at specified k-points."""
    print(f"\n{pairing_type} Gap Node Test:")
    print(f"Expected zeros at: {description}")
    print("-" * 60)
    
    all_zero = True
    for kx, ky, kz, label in test_points:
        gap = calculate_gap_magnitude(
            np.array([[kx]]), 
            np.array([[ky]]), 
            kz, 
            pairing_type=pairing_type
        )[0, 0]
        
        is_zero = gap < 1e-10
        status = "✓ ZERO" if is_zero else f"✗ NOT ZERO (|Δ|={gap:.6f})"
        print(f"  {label:30s}: {status}")
        
        if not is_zero:
            all_zero = False
    
    if all_zero:
        print(f"\n✓ All {pairing_type} nodes are correctly at zero!")
    else:
        print(f"\n✗ Some {pairing_type} nodes are NOT at zero - gap function may be incorrect!")
    
    return all_zero

# Test B1u: d ∝ (sin ky·b, sin kx·a, 0)
# Nodes at intersections where BOTH sin(ky·b)=0 AND sin(kx·a)=0
b1u_points = [
    (0, 0, 0, "kx=0 ∩ ky=0, kz=0"),
    (0, 0, np.pi/c, "kx=0 ∩ ky=0, kz=π/c"),
    (np.pi/a, 0, 0, "kx=π/a ∩ ky=0, kz=0"),
    (0, np.pi/b, 0, "kx=0 ∩ ky=π/b, kz=0"),
    (-np.pi/a, 0, 0, "kx=-π/a ∩ ky=0, kz=0"),
    (0, -np.pi/b, 0, "kx=0 ∩ ky=-π/b, kz=0"),
    (np.pi/a, np.pi/b, 0, "kx=π/a ∩ ky=π/b, kz=0"),
    (-np.pi/a, -np.pi/b, 0, "kx=-π/a ∩ ky=-π/b, kz=0"),
]

# Test B2u: d ∝ (sin kz·c, 0, sin kx·a)
# Nodes at intersections where BOTH sin(kz·c)=0 AND sin(kx·a)=0
b2u_points = [
    (0, 0, 0, "kx=0 ∩ kz=0, any ky"),
    (0, np.pi/b, 0, "kx=0 ∩ kz=0, ky=π/b"),
    (np.pi/a, 0, 0, "kx=π/a ∩ kz=0, ky=0"),
    (0, 0, np.pi/c, "kx=0 ∩ kz=π/c, ky=0"),
    (-np.pi/a, 0, 0, "kx=-π/a ∩ kz=0, ky=0"),
    (0, 0, -np.pi/c, "kx=0 ∩ kz=-π/c, ky=0"),
    (np.pi/a, 0, np.pi/c, "kx=π/a ∩ kz=π/c, ky=0"),
    (-np.pi/a, 0, -np.pi/c, "kx=-π/a ∩ kz=-π/c, ky=0"),
]

# Test B3u: d ∝ (0, sin kz·c, sin ky·b)
# Nodes at intersections where BOTH sin(kz·c)=0 AND sin(ky·b)=0
b3u_points = [
    (0, 0, 0, "ky=0 ∩ kz=0, any kx"),
    (np.pi/a, 0, 0, "ky=0 ∩ kz=0, kx=π/a"),
    (0, np.pi/b, 0, "ky=π/b ∩ kz=0, kx=0"),
    (0, 0, np.pi/c, "ky=0 ∩ kz=π/c, kx=0"),
    (0, -np.pi/b, 0, "ky=-π/b ∩ kz=0, kx=0"),
    (0, 0, -np.pi/c, "ky=0 ∩ kz=-π/c, kx=0"),
    (0, np.pi/b, np.pi/c, "ky=π/b ∩ kz=π/c, kx=0"),
    (0, -np.pi/b, -np.pi/c, "ky=-π/b ∩ kz=-π/c, kx=0"),
]

print("=" * 60)
print("GAP NODE VERIFICATION TEST")
print("=" * 60)

b1u_ok = test_gap_nodes("B1u", b1u_points, "(ky = 0, ±π/b) ∩ (kx = 0, ±π/a)")
b2u_ok = test_gap_nodes("B2u", b2u_points, "(kz = 0, ±π/c) ∩ (kx = 0, ±π/a)")
b3u_ok = test_gap_nodes("B3u", b3u_points, "(kz = 0, ±π/c) ∩ (ky = 0, ±π/b)")

print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)
print(f"B1u: {'✓ PASS' if b1u_ok else '✗ FAIL'}")
print(f"B2u: {'✓ PASS' if b2u_ok else '✗ FAIL'}")
print(f"B3u: {'✓ PASS' if b3u_ok else '✗ FAIL'}")

if b1u_ok and b2u_ok and b3u_ok:
    print("\n✓✓✓ All gap functions match theoretical predictions! ✓✓✓")
else:
    print("\n✗✗✗ Some gap functions need correction! ✗✗✗")
