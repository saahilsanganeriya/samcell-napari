"""
Sample data for SAMCell napari plugin
"""
import numpy as np
from pathlib import Path
import os

def sample_2d():
    """Generate a sample 2D image"""
    # Create a simple simulated cell image
    # In a real plugin, you'd typically ship sample images
    
    # Create a blank image
    image = np.zeros((512, 512), dtype=np.uint8)
    
    # Add some simulated cells (circles)
    rng = np.random.RandomState(42)  # for reproducibility
    
    # Add ~50 cells with random positions and sizes
    for _ in range(50):
        x = rng.randint(50, 462)
        y = rng.randint(50, 462)
        radius = rng.randint(15, 30)
        intensity = rng.randint(100, 200)
        
        # Draw cell
        for i in range(512):
            for j in range(512):
                dist = np.sqrt((i - y) ** 2 + (j - x) ** 2)
                if dist < radius:
                    # Gradient intensity from center to edge
                    cell_intensity = intensity * (1 - dist / radius)
                    image[i, j] = max(image[i, j], cell_intensity)
    
    # Add noise
    noise = rng.normal(0, 10, size=image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # Add background gradient
    bg = np.zeros_like(image, dtype=float)
    for i in range(512):
        for j in range(512):
            bg[i, j] = (i + j) / 10
    
    # Combine
    image = np.clip(image + bg, 0, 255).astype(np.uint8)
    
    # Create a scale bar
    return [(image, {'name': 'Sample Cells 2D'})] 