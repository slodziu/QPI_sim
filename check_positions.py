"""
Quick script to visualize where impurities are actually placed.
"""
import numpy as np
import matplotlib.pyplot as plt
from config import get_config

# Get configuration
config = get_config('random_10_impurities')
positions = config.get_impurity_positions()

print(f"Grid size: {config.gridsize}")
print(f"Number of impurities: {len(positions)}")
print(f"Positions: {positions}")

# Create a simple visualization
grid = np.zeros((config.gridsize, config.gridsize))

# Mark impurity positions
for row, col in positions:
    # Mark a 5x5 region around each impurity
    r_min = max(0, row - 2)
    r_max = min(config.gridsize, row + 3)
    c_min = max(0, col - 2)
    c_max = min(config.gridsize, col + 3)
    grid[r_min:r_max, c_min:c_max] = 1.0

# Plot
plt.figure(figsize=(10, 10))
plt.imshow(grid, origin='lower', cmap='Reds', interpolation='nearest')
plt.colorbar(label='Impurity presence')
plt.title(f'Impurity Positions (N={len(positions)})')
plt.xlabel('Column (x)')
plt.ylabel('Row (y)')

# Mark center
center = config.gridsize // 2
plt.axhline(y=center, color='blue', linestyle='--', alpha=0.3, label='Center')
plt.axvline(x=center, color='blue', linestyle='--', alpha=0.3)

# Add position labels
for i, (row, col) in enumerate(positions):
    plt.text(col, row, str(i+1), color='white', ha='center', va='center', 
             fontsize=8, fontweight='bold')

plt.legend()
plt.tight_layout()
plt.savefig('outputs/impurity_positions.png', dpi=150)
print("\nSaved visualization to outputs/impurity_positions.png")
plt.close()

# Also print distance from center for each impurity
center = config.gridsize // 2
print(f"\nDistances from center ({center}, {center}):")
for i, (row, col) in enumerate(positions):
    dist = np.sqrt((row - center)**2 + (col - center)**2)
    print(f"  Impurity {i+1}: ({row}, {col}) - distance = {dist:.1f}")
