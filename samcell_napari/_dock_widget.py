"""
SAMCell napari plugin dock widget
"""
import os
import time
import sys
from pathlib import Path
import importlib
import numpy as np
import logging
import traceback
import threading

# Check for the required dependencies
required_packages = {
    'napari': '0.4.14',
    'transformers': '4.26.0',
    'torch': '1.9.0',
    'timm': '0.6.0',
    'scikit-image': '0.19.0',
    'opencv-python': '4.5.0'
}

missing_packages = []
for package, min_version in required_packages.items():
    try:
        importlib.import_module(package)
    except ImportError:
        missing_packages.append(f"{package}>={min_version}")

if missing_packages:
    package_list = ', '.join(missing_packages)
    install_command = ' '.join(missing_packages)
    error_message = f"Missing required dependencies: {package_list}. Please install using: pip install {install_command}"
    print(error_message, file=sys.stderr)
    logging.error(error_message)

try:
    from napari.layers import Image, Labels
    from napari_plugin_engine import napari_hook_implementation
    from napari.utils.notifications import show_info, show_warning, show_error
    from napari import Viewer
    from magicgui import magic_factory
    import napari.utils.notifications as notifications

    # Import threading utilities
    from napari.utils.events import Event
    from napari.qt.threading import thread_worker
    from qtpy.QtWidgets import QProgressBar, QVBoxLayout, QWidget, QLabel
    from magicgui.widgets import Container
except ImportError as e:
    error_message = f"Error importing napari components: {e}. Please ensure napari>=0.4.14 is installed."
    print(error_message, file=sys.stderr)
    logging.error(error_message)

# Try importing our own modules with graceful error handling
try:
    from .model import SAMCellModel
    from .pipeline import SAMCellPipeline
except ImportError as e:
    error_message = f"Error importing SAMCell modules: {e}. This could be due to missing dependencies."
    print(error_message, file=sys.stderr)
    logging.error(error_message)

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Global model and pipeline instances
model = None
pipeline = None

# Progress tracking widget
class ProgressWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.label = QLabel("Segmentation Progress")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.progress_bar)
        
        # Hide initially
        self.hide()
        
    def set_progress(self, value):
        self.progress_bar.setValue(int(value))
        if value > 0 and not self.isVisible():
            self.show()
        if value >= 100:
            # Auto-hide after completion
            self.hide()
            
    def reset(self):
        self.progress_bar.setValue(0)

# Store progress widget instance
progress_widget = None

# Worker function for asynchronous processing
@thread_worker
def run_segmentation(image, model_path, threshold_max, threshold_fill, crop_size, output_distance):
    global model, pipeline
    
    try:
        logger.info("Starting SAMCell segmentation asynchronously")
        
        # Image validation
        if image is None:
            raise ValueError("No image provided")
            
        # Make a copy to avoid modifying original
        image = image.copy()
        
        # Check if image is 2D
        if image.ndim > 2:
            # Handle RGB or multi-channel images
            if image.ndim == 3 and image.shape[2] in [3, 4]:  # RGB/RGBA
                # Use a safer method to convert to grayscale
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = np.mean(image[:, :, :3], axis=2).astype(np.uint8)
                logger.info(f"Converted RGB image to grayscale, shape: {image.shape}")
            else:
                raise ValueError("Only 2D images are supported. For multi-channel images, please select a single channel.")
                
        # Make sure image is the right type
        if image.dtype != np.uint8:
            logger.info(f"Converting image from {image.dtype} to uint8")
            if np.max(image) > 1.0 and np.max(image) <= 255:
                image = image.astype(np.uint8)
            else:
                # Normalize to 0-255 range
                image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
        
        # Check model path
        if not model_path or not os.path.exists(model_path):
            raise ValueError("Please select a valid model file (.pt, .bin, or .safetensors).")
            
        # Check file extension
        file_ext = Path(model_path).suffix.lower()
        if file_ext not in ['.pt', '.bin', '.safetensors']:
            show_warning(f"Unusual file extension: {file_ext}. Supported formats are .pt, .bin, and .safetensors. Attempting to load anyway.")
            logger.warning(f"Unusual file extension: {file_ext}. Supported formats are .pt, .bin, and .safetensors.")
            
        # Initialize model if not already done
        if model is None or not model.is_initialized():
            # Progress update - model loading
            yield {"progress": 5, "status": "Loading model..."}
            
            # Clean up existing model if any
            if model is not None:
                try:
                    model.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up model: {str(e)}")
            
            # Load model
            model_instance = SAMCellModel()
            try:
                logger.info("Loading model...")
                success = model_instance.load_model(model_path)
                logger.info(f"Model loading {'succeeded' if success else 'failed'}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                logger.error(traceback.format_exc())
                raise ValueError(f"Error loading model: {str(e)}")
                
            if success:
                model = model_instance
                yield {"progress": 10, "status": "Model loaded successfully!"}
            else:
                raise ValueError("Failed to load model. Please check the model path.")
                
            # Initialize pipeline
            try:
                logger.info("Initializing pipeline...")
                pipeline = SAMCellPipeline(model, crop_size=crop_size)
                logger.info("Pipeline initialized")
                yield {"progress": 15, "status": "Pipeline initialized"}
            except Exception as e:
                logger.error(f"Error initializing pipeline: {str(e)}")
                logger.error(traceback.format_exc())
                raise ValueError(f"Error initializing pipeline: {str(e)}")
        else:
            yield {"progress": 15, "status": "Using existing model"}
        
        # Update pipeline settings
        pipeline.crop_size = crop_size
        pipeline.cells_max_threshold = threshold_max
        pipeline.cell_fill_threshold = threshold_fill
        logger.info(f"Running with thresholds: max={threshold_max}, fill={threshold_fill}, crop_size={crop_size}")
        
        # Register progress callback with a shared data object to track progress
        progress_data = {"percent": 15, "status": "Starting segmentation..."}
        
        def progress_callback(current, total, step_info):
            # Extract percentage info from step_info if present
            current_percent = 15  # Default starting point
            if "(" in step_info and ")" in step_info:
                # Try to extract percentage from string like "Processing... (75%)"
                try:
                    percent_str = step_info.split("(")[1].split(")")[0].replace("%", "")
                    current_percent = float(percent_str)
                    logger.info(f"Extracted percentage: {current_percent}% from status: {step_info}")
                except:
                    # If extraction fails, use the calculation
                    current_percent = 15 + (current / total * 80) if total > 0 else 15
                    logger.info(f"Calculated percentage: {current_percent}% from current/total: {current}/{total}")
            else:
                # No percentage in the status, calculate based on progress
                current_percent = 15 + (current / total * 80) if total > 0 else 15
                logger.info(f"Calculated percentage: {current_percent}% from current/total: {current}/{total}")
            
            # Update the shared data object - ensure we don't go backwards in progress
            if current_percent > progress_data["percent"]:
                progress_data["percent"] = current_percent
            progress_data["status"] = step_info
            
            # Return for the pipeline's internal use
            return {"progress": current_percent, "status": step_info}
            
        # Set our callback
        pipeline.set_progress_callback(progress_callback)
            
        # Run segmentation
        try:
            logger.info("Starting segmentation process...")
            
            # Initial progress update
            yield {"progress": progress_data["percent"], "status": progress_data["status"]}
            
            # Process the image - this will update progress through the callback
            # Start processing in main thread - the pipeline will update our progress_data through the callback
            dist_map_result = [None]  # Use a list to store the result
            
            def process_image():
                # Process the image and store the result
                result = pipeline.pipeline.predict_on_full_img(image)
                dist_map_result[0] = result
                
            # Start processing in a separate thread
            dist_map_thread = threading.Thread(target=process_image)
            dist_map_thread.daemon = True
            dist_map_thread.start()
            
            # Poll for progress updates during processing
            last_percent = 0
            while dist_map_thread.is_alive():
                if progress_data["percent"] != last_percent:
                    # Only yield if progress has changed
                    last_percent = progress_data["percent"]
                    yield {"progress": progress_data["percent"], "status": progress_data["status"]}
                # Short sleep to prevent UI freeze
                time.sleep(0.1)
            
            # Wait for thread to complete
            dist_map_thread.join()
            
            # Get the result
            dist_map = dist_map_result[0]
            
            # Now that processing is complete, execute the watershed step
            # Extract cells from distance map - this will update progress to ~95%
            labels_result = [None]  # Use a list to store the result
            
            def process_labels():
                # Process the labels and store the result
                result = pipeline.pipeline.cells_from_dist_map(dist_map)
                labels_result[0] = result
                
            # Start processing in a separate thread
            labels_thread = threading.Thread(target=process_labels)
            labels_thread.daemon = True
            labels_thread.start()
            
            # Poll for progress updates during processing
            last_percent = 0
            while labels_thread.is_alive():
                if progress_data["percent"] != last_percent:
                    # Only yield if progress has changed
                    last_percent = progress_data["percent"]
                    yield {"progress": progress_data["percent"], "status": progress_data["status"]}
                # Short sleep to prevent UI freeze
                time.sleep(0.1)
            
            # Wait for thread to complete
            labels_thread.join()
            
            # Get the result
            labels = labels_result[0]
            
            # Progress update - should be around 95% now
            yield {"progress": progress_data["percent"], "status": progress_data["status"]}
            
            # Log results
            num_cells = len(np.unique(labels)) - 1  # -1 to exclude background
            logger.info(f"Segmentation complete. Found {num_cells} cells.")
            
            # Ensure outputs are numpy arrays and contiguous
            labels = np.ascontiguousarray(labels)
            dist_map = np.ascontiguousarray(dist_map)
            
            # Final progress update with the result data included
            # For generator workers, the last yielded value is used as the result
            yield {
                "progress": 100, 
                "status": f"Complete! Found {num_cells} cells.",
                "result": (labels, dist_map, num_cells)
            }
            
            # No return needed - the last yield contains the result
                
        except Exception as e:
            logger.error(f"Segmentation error: {str(e)}")
            logger.error(traceback.format_exc())
            # Create empty results instead of raising to avoid callback issues
            empty_labels = np.zeros(image.shape[:2], dtype=np.int32)
            empty_dist_map = np.zeros(image.shape[:2], dtype=np.float32)
            yield {
                "progress": 100, 
                "status": f"Error: {str(e)}",
                "result": (empty_labels, empty_dist_map, 0)
            }
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        # Return empty results to avoid callback issues
        empty_labels = np.zeros(image.shape[:2], dtype=np.int32)
        empty_dist_map = np.zeros(image.shape[:2], dtype=np.float32)
        yield {
            "progress": 100, 
            "status": f"Error: {str(e)}",
            "result": (empty_labels, empty_dist_map, 0)
        }

def check_dependencies():
    """Check if all required dependencies are installed correctly.
    
    Returns
    -------
    bool
        True if all dependencies are available, False otherwise
    str
        Error message if dependencies are missing, empty string otherwise
    """
    try:
        # Check for critical components
        import transformers
        from transformers.models.sam.modeling_sam import SamModel
        import torch
        import cv2
        import numpy as np
        import timm
        
        # Check transformers version
        if tuple(map(int, transformers.__version__.split('.'))) < (4, 26, 0):
            return False, f"Transformers version {transformers.__version__} is too old. Please install transformers>=4.26.0"
            
        # Check torch version
        if tuple(map(int, torch.__version__.split('.')[:2])) < (1, 9):
            return False, f"PyTorch version {torch.__version__} is too old. Please install torch>=1.9.0"
            
        return True, ""
    except ImportError as e:
        return False, f"Missing dependency: {str(e)}. Please check your installation."

@magic_factory(
    call_button="Run Segmentation",
    layout="vertical",
    model_path=dict(widget_type="FileEdit", 
                label="SAMCell model path:",
                value="",
                tooltip="Path to the SAMCell model weights (.pt, .bin, or .safetensors)"),
    threshold_max=dict(widget_type="FloatSlider", 
                   label="Cell peak threshold", 
                   min=0.1, max=0.9, step=0.05, value=0.47,
                   tooltip="Threshold for detecting cell centers (higher = fewer cells)"),
    threshold_fill=dict(widget_type="FloatSlider", 
                    label="Cell fill threshold", 
                    min=0.01, max=0.5, step=0.01, value=0.09,
                    tooltip="Threshold for cell boundaries (lower = larger cells)"),
    crop_size=dict(widget_type="SpinBox",
                label="Crop size", 
                min=128, max=1024, step=64, value=256,
                tooltip="Size of crops for sliding window (larger needs more memory)"),
    output_distance=dict(widget_type="CheckBox",
                     label="Output distance map",
                     value=True,
                     tooltip="Output the distance map as a separate layer"),
)
def samcell_widget(
    viewer: Viewer,
    image_layer: Image,
    model_path: str,
    threshold_max: float = 0.47,
    threshold_fill: float = 0.09,
    crop_size: int = 256,
    output_distance: bool = True,
):
    """
    SAMCell segmentation widget
    
    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance
    image_layer : napari.layers.Image
        Selected image layer
    model_path : str
        Path to SAMCell model
    threshold_max : float
        Threshold for cell centers
    threshold_fill : float
        Threshold for cell fill
    crop_size : int
        Size of crops for sliding window
    output_distance : bool
        Whether to output distance map
    """
    global progress_widget
    
    # Check that all dependencies are installed correctly
    deps_ok, error_msg = check_dependencies()
    if not deps_ok:
        show_error(f"Dependency error: {error_msg}")
        logger.error(f"Dependency check failed: {error_msg}")
        return
    
    # Image validation
    if image_layer is None:
        show_error("Please select an image layer.")
        logger.error("No image layer selected")
        return
    
    # Create progress widget if it doesn't exist
    if progress_widget is None:
        progress_widget = ProgressWidget()
        viewer.window.add_dock_widget(progress_widget, area="bottom", name="Segmentation Progress")
    
    # Reset progress bar
    progress_widget.reset()
    progress_widget.show()
    
    # Store the widget instance for later access
    widget_instance = samcell_widget
    
    # Disable button during processing - safely get call_button if available
    try:
        if hasattr(widget_instance, 'call_button') and widget_instance.call_button is not None:
            widget_instance.call_button.enabled = False
    except Exception as e:
        logger.warning(f"Could not disable call button: {str(e)}")
    
    show_info("Starting SAMCell segmentation...")
    
    # Start worker thread for segmentation
    worker = run_segmentation(
        image_layer.data, 
        model_path, 
        threshold_max, 
        threshold_fill, 
        crop_size, 
        output_distance
    )
    
    # Create a dict to store the final result
    result_container = {"final_result": None}
    
    # Handle progress updates
    def _update_progress(yield_result):
        progress = yield_result.get("progress", 0)
        status = yield_result.get("status", "")
        
        # Store the result if it's in the yielded data (will be in the final yield)
        if "result" in yield_result:
            result_container["final_result"] = yield_result["result"]
            logger.info(f"Stored result from yield: {type(result_container['final_result'])}")
        
        # Update progress bar
        progress_widget.set_progress(progress)
        
        # Update status text on the progress bar label instead of button
        if status:
            progress_text = f"Segmentation Progress: {int(progress)}%"
            if status:
                progress_text += f" - {status}"
            progress_widget.label.setText(progress_text)
    
    # Handle completion
    def _on_done(result=None):
        """
        Handle worker completion.
        
        For generator workers, the result is stored in the result_container from the last yield.
        """
        # Re-enable button - safely access call_button if available
        try:
            if hasattr(widget_instance, 'call_button') and widget_instance.call_button is not None:
                widget_instance.call_button.enabled = True
        except Exception as e:
            logger.warning(f"Could not re-enable call button: {str(e)}")
        
        # Get the result from our container
        result = result_container["final_result"]
        logger.info(f"Using stored result from final yield: {type(result)}")
        
        if result is None:
            show_error("Segmentation failed. Check logs for details.")
            progress_widget.label.setText("Segmentation Progress")
            progress_widget.hide()
            return
            
        try:
            # Get results - handle the tuple format (labels, dist_map, num_cells)
            if isinstance(result, tuple) and len(result) == 3:
                labels, dist_map, num_cells = result
            else:
                show_error(f"Unexpected result format: {type(result)}")
                progress_widget.hide()
                return
            
            # Add labels layer
            layer_name = f"{image_layer.name}_samcell_labels"
            if layer_name in viewer.layers:
                viewer.layers[layer_name].data = labels
            else:
                viewer.add_labels(labels, name=layer_name)
                
            # Add distance map layer if requested
            if output_distance and dist_map is not None:
                dist_name = f"{image_layer.name}_samcell_distmap"
                if dist_name in viewer.layers:
                    viewer.layers[dist_name].data = dist_map
                else:
                    viewer.add_image(dist_map, name=dist_name, colormap='magma')
                    
            # Show success message
            show_info(f"Segmentation complete. Found {num_cells} cells.")
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            logger.error(traceback.format_exc())
            show_error(f"Error processing results: {str(e)}")
        
        # Reset progress widget text and auto-hide
        progress_widget.label.setText("Segmentation Progress")
        progress_widget.hide()
    
    # Handle errors
    def _on_error(error):
        # Re-enable button - safely access call_button if available
        try:
            if hasattr(widget_instance, 'call_button') and widget_instance.call_button is not None:
                widget_instance.call_button.enabled = True
        except Exception as e:
            logger.warning(f"Could not re-enable call button: {str(e)}")
        
        # Show error
        show_error(f"Error: {str(error)}")
        logger.error(f"Worker error: {str(error)}")
        
        # Reset progress widget text and hide
        progress_widget.label.setText("Segmentation Progress")
        progress_widget.hide()
    
    # Connect worker signals
    try:
        worker.yielded.connect(_update_progress)
        worker.finished.connect(_on_done)
        worker.errored.connect(_on_error)
        logger.info("Worker signals connected successfully")
    except Exception as e:
        logger.error(f"Error connecting worker signals: {str(e)}")
        show_error(f"Error setting up worker: {str(e)}")
        return
    
    # Start worker
    worker.start()


# Napari uses this hook to find the widget
@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return samcell_widget