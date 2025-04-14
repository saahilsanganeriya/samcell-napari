"""
SAMCell standalone application entry point
"""
import sys
import os
from pathlib import Path

# Add the package directory to the Python path
package_dir = Path(__file__).parent
sys.path.insert(0, str(package_dir.parent))

import napari
from ._dock_widget import samcell_widget

def main():
    """Launch napari with the SAMCell plugin"""
    # Create viewer
    viewer = napari.Viewer()
    
    # Add the SAMCell widget
    viewer.window.add_dock_widget(samcell_widget(viewer))
    
    # Show the viewer
    viewer.show()
    
    # Start the event loop
    napari.run()

if __name__ == "__main__":
    main() 