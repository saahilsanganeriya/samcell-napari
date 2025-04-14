import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import numpy as np
import logging
import traceback
import gc
import os
import sys
from pathlib import Path

# Check for required dependencies
required_packages = {
    'transformers': '4.26.0',
    'timm': '0.6.0', 
    'torch': '1.9.0'
}

missing_packages = []
for package, min_version in required_packages.items():
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(f"{package}>={min_version}")

if missing_packages:
    error_message = (
        f"Missing required dependencies: {', '.join(missing_packages)}. "
        f"Please install them using: pip install {' '.join(missing_packages)}"
    )
    print(error_message, file=sys.stderr)
    logging.error(error_message)

# Import SAM-related dependencies - wrapped in try/except for better error messages
try:
    from transformers import SamModel, SamConfig, SamMaskDecoderConfig
    from transformers.models.sam.modeling_sam import SamMaskDecoder, SamVisionConfig
except ImportError as e:
    error_message = (
        f"Error importing SAM modules from transformers: {e}. "
        f"This might indicate an incorrect transformers version. "
        f"Please install transformers>=4.26.0 using: pip install transformers>=4.26.0"
    )
    print(error_message, file=sys.stderr)
    logging.error(error_message)
    # Create dummy classes to prevent immediate failures
    # These will raise proper errors when actually used
    class MissingDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError(error_message)
        
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError(error_message)
    
    SamModel = SamConfig = SamMaskDecoderConfig = SamMaskDecoder = SamVisionConfig = MissingDependency

logger = logging.getLogger(__name__)

class FinetunedSAM:
    """Helper class to handle setting up SAM from the transformers library for finetuning.
    
    This class provides a simplified interface for loading the SAM model and its weights.
    
    Attributes
    ----------
    model : transformers.SamModel
        The loaded SAM model
    """
    
    def __init__(self, sam_model):
        """Initialize the FinetunedSAM model.
        
        Parameters
        ----------
        sam_model : str
            Path or identifier for the SAM model to load from transformers
        """
        try:
            logger.info(f"Loading SAM model from {sam_model}")
            self.model = SamModel.from_pretrained(sam_model)
            self.model.eval()
            logger.info("SAM model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading SAM model: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_model(self):
        """Get the underlying SAM model.
        
        Returns
        -------
        transformers.SamModel
            The loaded SAM model
        """
        return self.model
    
    def load_weights(self, weight_path, map_location=None):
        """Load weights from a file to the SAM model.
        
        Supports multiple file formats including .pt, .bin, and .safetensors.
        
        Parameters
        ----------
        weight_path : str
            Path to the weights file
        map_location : str or torch.device, optional
            Device to load the weights to, by default None
            
        Raises
        ------
        Exception
            If weights cannot be loaded
        """
        try:
            logger.info(f"Loading weights from {weight_path}")
            # Determine file format based on extension
            file_ext = Path(weight_path).suffix.lower()
            
            if file_ext == '.safetensors':
                # Use safetensors if available
                try:
                    import safetensors.torch
                    state_dict = safetensors.torch.load_file(weight_path)
                    logger.info("Loaded model using safetensors format")
                except ImportError:
                    logger.warning("safetensors is not installed, falling back to PyTorch loading")
                    state_dict = torch.load(weight_path, map_location=map_location)
            else:
                # For .pt, .bin, and any other format, use PyTorch's load
                state_dict = torch.load(weight_path, map_location=map_location)
                logger.info(f"Loaded model using PyTorch format ({file_ext})")
            
            # Load the state dictionary
            self.model.load_state_dict(state_dict)
            logger.info("Weights loaded successfully")
        except Exception as e:
            logger.error(f"Error loading weights: {str(e)}")
            logger.error(traceback.format_exc())
            raise


class SAMCellModel:
    """Adapter for FinetunedSAM to match the original implementation.
    
    This class provides a simplified interface for the SAMCell model,
    handling initialization, loading, and cleanup.
    
    Attributes
    ----------
    device : str
        Device to run the model on ('cuda' or 'cpu')
    model : FinetunedSAM or None
        The loaded SAM model or None if not initialized
    initialized : bool
        Whether the model has been successfully initialized
    model_type : str
        The SAM model type that was loaded successfully ('base' or 'large')
    """
    
    def __init__(self, model_path=None):
        """Initialize the SAMCell model.
        
        Parameters
        ----------
        model_path : str or Path, optional
            Path to the SAMCell model weights, by default None
        """
        logger.info("Initializing SAMCellModel")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.initialized = False
        self.model_type = None
        
        # Check for critical dependencies before proceeding
        self._check_dependencies()
        
        # Try to load model if path is provided
        if model_path:
            self.load_model(model_path)
    
    def _check_dependencies(self):
        """Check if all required dependencies are installed."""
        try:
            # Test imports of critical components
            from transformers.models.sam.modeling_sam import SamMaskDecoder
            import torch.nn as nn
            import cv2
            import numpy as np
            logger.info("All critical dependencies are available")
            return True
        except ImportError as e:
            error_msg = f"Missing critical dependency: {e}"
            logger.error(error_msg)
            self.initialized = False
            return False
            
    def load_model(self, model_path):
        """Load the SAMCell model from a given path.
        
        Parameters
        ----------
        model_path : str or Path
            Path to the SAMCell model weights (.pt, .bin, or .safetensors)
        
        Returns
        -------
        bool
            True if model loaded successfully, False otherwise
        """
        try:
            # Verify dependencies first
            if not self._check_dependencies():
                logger.error("Cannot load model: missing dependencies")
                return False
                
            # Verify that the file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
                
            logger.info(f"Loading model from {model_path}")
            
            # Get file extension for format detection
            file_ext = Path(model_path).suffix.lower()
            logger.info(f"Detected model format: {file_ext}")
            
            # Validate supported file formats
            if file_ext not in ['.pt', '.bin', '.safetensors']:
                logger.warning(f"Unsupported file format: {file_ext}. Attempting to load anyway.")
            
            # Try loading with sam-vit-base first
            try:
                logger.info("Attempting to load with sam-vit-base")
                self.model = FinetunedSAM('facebook/sam-vit-base')
                self.model.load_weights(model_path, map_location=self.device)
                self.model_type = 'base'
                logger.info("Successfully loaded model with sam-vit-base")
            except Exception as e:
                logger.warning(f"Failed to load with sam-vit-base: {str(e)}")
                logger.info("Attempting to load with sam-vit-large")
                
                # Try with sam-vit-large as fallback
                try:
                    # Clean up previous model attempt
                    if self.model is not None:
                        del self.model
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    self.model = FinetunedSAM('facebook/sam-vit-large')
                    self.model.load_weights(model_path, map_location=self.device)
                    self.model_type = 'large'
                    logger.info("Successfully loaded model with sam-vit-large")
                except Exception as e2:
                    logger.error(f"Failed to load with sam-vit-large: {str(e2)}")
                    logger.error(traceback.format_exc())
                    self.model = None
                    self.initialized = False
                    return False
            
            # Move model to the appropriate device and set to eval mode
            self.model.model.to(self.device)
            self.model.model.eval()
            self.initialized = True
            logger.info(f"Model successfully loaded using SAM {self.model_type} architecture")
            return True
            
        except Exception as e:
            logger.error(f"Error in load_model: {str(e)}")
            logger.error(traceback.format_exc())
            self.model = None
            self.initialized = False
            return False
    
    def is_initialized(self):
        """Check if model is initialized.
        
        Returns
        -------
        bool
            True if model is initialized, False otherwise
        """
        return self.initialized
    
    def get_model(self):
        """Get the underlying model.
        
        Returns
        -------
        FinetunedSAM
            The initialized model
            
        Raises
        ------
        RuntimeError
            If model is not initialized
        """
        if not self.initialized:
            logger.error("Model is not initialized. Call load_model first.")
            raise RuntimeError("Model is not initialized. Call load_model first.")
        return self.model
    
    def cleanup(self):
        """Clean up model resources.
        
        Releases memory used by the model and forces garbage collection.
        """
        if self.model is not None:
            try:
                logger.info("Cleaning up model resources")
                del self.model
                self.model = None
                self.initialized = False
                self.model_type = None
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info("Model resources cleaned up")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
                logger.error(traceback.format_exc()) 