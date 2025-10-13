"""
Quick profiling to see where time is spent.
"""
import time
import numpy as np
from qpi_G_OOP import SystemParameters, GreensFunction, ImpuritySystem, QPIAnalyzer
from config import FAST_PREVIEW

config = FAST_PREVIEW
params = SystemParameters(
    gridsize=config.gridsize,
    L=config.L,
    t=config.t,
    mu=config.mu,
    eta=config.eta,
    V_s=config.V_s,
    E_min=config.E_min,
    E_max=config.E_max,
    n_frames=config.n_frames
)

print(f"Config: gridsize={params.gridsize}, n_frames={params.n_frames}, eta={params.eta}")

# Single impurity at center
impurity_positions = [(params.gridsize//2, params.gridsize//2)]

# Test single frame
energy = 10.0

t0 = time.time()
greens = GreensFunction(params)
t1 = time.time()
print(f"GreensFunction init: {t1-t0:.3f}s")

t0 = time.time()
G0 = greens.calculate_G0(energy)
t1 = time.time()
print(f"Calculate G0: {t1-t0:.3f}s")

t0 = time.time()
impurities = ImpuritySystem(impurity_positions)
LDOS = impurities.calculate_LDOS(G0, greens)
t1 = time.time()
print(f"Calculate LDOS: {t1-t0:.3f}s")

t0 = time.time()
analyzer = QPIAnalyzer(params)
LDOS_processed = analyzer.process_LDOS(LDOS)
t1 = time.time()
print(f"Process LDOS: {t1-t0:.3f}s")

t0 = time.time()
fft_complex, fft_display = analyzer.calculate_FFT(LDOS_processed)
t1 = time.time()
print(f"Calculate FFT: {t1-t0:.3f}s")

print(f"\nTotal for single frame: ~{t1-time.time()+5*(t1-t0):.3f}s")
print(f"Estimated for {params.n_frames} frames: ~{params.n_frames * 0.5:.1f}s (just computation)")
print(f"\nNote: Animation rendering/saving adds significant overhead!")
print(f"      High DPI (current: 500) makes rendering VERY slow")
