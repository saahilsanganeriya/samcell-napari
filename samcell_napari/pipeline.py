import torch
import torch.nn as nn
import numpy as np
import cv2
from skimage.segmentation import watershed
from transformers import SamProcessor
from pathlib import Path
import math
import logging
import traceback
from typing import Any, Callable, Optional, Dict, Tuple, Union, List

from .slidingWindow import SlidingWindowHelper

logger = logging.getLogger(__name__)

class SlidingWindowPipeline:
    def __init__(self, model, device, crop_size=256):
        try:
            logger.info("Initializing SlidingWindowPipeline")
            self.model = model.get_model().get_model()
            
            self.device = device
            self.crop_size = crop_size
            self.sigmoid = nn.Sigmoid()
            self.processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
            self.sliding_window_helper = SlidingWindowHelper(crop_size, 32)
            
            # Default thresholds - will be updated from UI
            self.cells_max_threshold = 0.47
            self.cell_fill_threshold = 0.09
            
            # Progress callback
            self._progress_callback = None
            
            logger.info("SlidingWindowPipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set a callback function to report progress
        
        Parameters
        ----------
        callback : callable
            A function that takes (current, total, status) as arguments
        """
        self._progress_callback = callback
        
    def _report_progress(self, current: int, total: int, status: str):
        """Report progress through the callback if set
        
        Parameters
        ----------
        current : int
            Current progress value
        total : int
            Total expected value
        status : str
            Status message
        """
        if self._progress_callback is not None:
            try:
                result = self._progress_callback(current, total, status)
                # If the callback returns a dictionary with a progress value,
                # we can use that information for further processing if needed
                return result
            except Exception as e:
                logger.error(f"Error in progress callback: {str(e)}")
        return None

    def _preprocess(self, img):
        try:
            # Ensure image is not empty
            if img is None or img.size == 0:
                raise ValueError("Input image is empty")
                
            # Make sure image has the right type
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
                
            # Apply CLAHE for contrast enhancement
            img = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8)).apply(img)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Convert to color if necessary
            if len(img.shape) != 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Default SAM preprocessing
            inputs = self.processor(img, return_tensors="pt")
            return inputs['pixel_values'].to(self.device)
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _preprocess_sam(self, img):
        inputs = self.processor(img, return_tensors="pt")
        return inputs['pixel_values'].to(self.device)

    def get_model_prediction(self, image):
        try:
            image_orig = image.copy()
            image = self._preprocess(image_orig)
            self.model.eval()

            # Forward pass
            with torch.no_grad():
                outputs_finetuned = self.model(pixel_values=image, multimask_output=True)

            prob_finetuned = outputs_finetuned['pred_masks'].squeeze(1)

            # Sigmoid
            dist_map = self.sigmoid(prob_finetuned)[0][0]

            return dist_map
        except Exception as e:
            logger.error(f"Error in get_model_prediction: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_model_prediction_batched(self, images):
        try:
            images_preprocessed = []
            for image in images:
                image = self._preprocess(image)
                images_preprocessed.append(image)
            images = torch.stack(images_preprocessed, dim=0)
            images = images.squeeze(1)

            # Forward pass
            self.model.eval()
            with torch.no_grad():
                outputs_finetuned = self.model(pixel_values=images, multimask_output=True)
                prob_finetuned = outputs_finetuned['pred_masks'].squeeze(1)
                # Sigmoid
                dist_maps = self.sigmoid(prob_finetuned)[:,0]

            return dist_maps
        except Exception as e:
            logger.error(f"Error in get_model_prediction_batched: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def predict_on_full_img(self, image_orig):
        try:
            orig_shape = image_orig.shape
            
            # Ensure image is 2D
            if len(image_orig.shape) > 2:
                logger.info(f"Converting {image_orig.shape} to grayscale")
                if image_orig.shape[2] == 1:
                    # Single channel image in 3D format
                    image_orig = image_orig[:, :, 0]
                else:
                    # True color image - convert to grayscale
                    image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)

            # Split into crops
            crops, orig_regions, crop_unique_region = self.sliding_window_helper.seperate_into_crops(image_orig)
            logger.info(f"Split image into {len(crops)} crops")
            self._report_progress(0, len(crops), f"Processing image in {len(crops)} crops")
            
            # Debug log
            logger.info(f"First crop shape: {crops[0].shape}")
            
            # Process in small batches to avoid memory issues
            batches = []
            batch_size = 4  # Increased for faster processing
            for i in range(0, len(crops), batch_size):
                batches.append(np.array(crops[i:i+batch_size]))
            
            logger.info(f"Created {len(batches)} batches")

            dist_maps = []
            total_processed = 0
            for batch_idx, batch in enumerate(batches):
                try:
                    # Report progress before processing each batch
                    batch_progress = 15 + (batch_idx / len(batches) * 40)
                    self._report_progress(total_processed, len(crops), 
                                         f"Processing batch {batch_idx+1}/{len(batches)} ({batch_progress:.1f}%)")
                    
                    # Predict on crops
                    logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")
                    
                    pred_maps = self.get_model_prediction_batched(batch).cpu().detach().numpy()
                    logger.info(f"Got predictions with shape: {pred_maps.shape}")
                    
                    for dist_map in pred_maps:
                        dist_map = cv2.resize(dist_map, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
                        dist_maps.append(dist_map)
                        
                    # Update progress
                    total_processed += len(batch)
                    batch_progress = 15 + (batch_idx+1) / len(batches) * 40
                    self._report_progress(total_processed, len(crops), 
                                         f"Processed {total_processed}/{len(crops)} crops ({batch_progress:.1f}%)")
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

            if not dist_maps:
                raise ValueError("Failed to generate any valid distance maps")

            # Combine crops into full image - report progress at 60%
            self._report_progress(len(crops), len(crops), "Combining crops into full image (60%)")
            cell_dist_map = self.sliding_window_helper.combine_crops(orig_shape, crops, orig_regions, crop_unique_region, sam_outputs=dist_maps)
            
            # Debug log
            logger.info(f"Combined distance map shape: {cell_dist_map.shape}")
            logger.info(f"Distance map range: min={np.min(cell_dist_map)}, max={np.max(cell_dist_map)}")

            # Report progress at 70%
            self._report_progress(len(crops), len(crops), "Distance map complete (70%)")
            return cell_dist_map
        except Exception as e:
            logger.error(f"Error in predict_on_full_img: {str(e)}")
            logger.error(traceback.format_exc())
            # Return empty map on error
            return np.zeros(image_orig.shape[:2], dtype=np.float32)

    def cells_from_dist_map(self, dist_map, cells_max_threshold=None, cell_fill_threshold=None):
        try:
            # Use provided thresholds or defaults
            cells_max_threshold = cells_max_threshold if cells_max_threshold is not None else self.cells_max_threshold
            cell_fill_threshold = cell_fill_threshold if cell_fill_threshold is not None else self.cell_fill_threshold
            
            # Report progress
            self._report_progress(0, 3, "Finding cell centers (75%)")
            
            # Apply thresholds
            cells_max = dist_map > cells_max_threshold
            cell_fill = dist_map > cell_fill_threshold
            
            # Debug log
            logger.info(f"cells_max threshold: {cells_max_threshold}, sum: {np.sum(cells_max)}")
            logger.info(f"cell_fill threshold: {cell_fill_threshold}, sum: {np.sum(cell_fill)}")
            
            # Convert to binary masks
            cells_max = cells_max.astype(np.uint8)
            cell_fill = cell_fill.astype(np.uint8)
            
            # Find contours - handle different OpenCV versions
            try:
                contours, _ = cv2.findContours(cells_max, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except:
                _, contours, _ = cv2.findContours(cells_max, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
            logger.info(f"Found {len(contours)} contours")
            self._report_progress(1, 3, f"Processing {len(contours)} cell candidates (80%)")
                
            mask = np.zeros(dist_map.shape, dtype=np.int32)
            
            # Process each contour
            for i, contour in enumerate(contours):
                # Skip invalid contours
                if contour is None or len(contour) < 3:
                    continue
                    
                try:
                    M = cv2.moments(contour)

                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        # Handle zero division - use contour center
                        cX = int(np.mean([p[0][0] for p in contour]))
                        cY = int(np.mean([p[0][1] for p in contour]))

                    # Set closest pixel to centroid
                    cY = min(max(0, cY), mask.shape[0]-1)
                    cX = min(max(0, cX), mask.shape[1]-1)
                    mask[cY, cX] = i + 1
                except Exception as e:
                    logger.error(f"Error processing contour {i}: {str(e)}")
                    continue

            # Use watershed to create final segmentation
            self._report_progress(2, 3, "Running watershed segmentation (90%)")
            if np.max(mask) == 0:
                logger.warning("No centroids found - returning empty segmentation")
                return np.zeros(dist_map.shape, dtype=np.int32)
                
            labels = watershed(-dist_map, mask, mask=cell_fill).astype(np.int32)
            self._report_progress(3, 3, "Segmentation complete (95%)")

            return labels
        except Exception as e:
            logger.error(f"Error in cells_from_dist_map: {str(e)}")
            logger.error(traceback.format_exc())
            return np.zeros(dist_map.shape, dtype=np.int32)
    
    def run(self, image, return_dist_map=False, cells_max_threshold=None, cell_fill_threshold=None):
        try:
            # Make a copy to avoid modifying original
            image = image.copy()
            
            # Run prediction
            self._report_progress(0, 2, "Generating distance map")
            dist_map = self.predict_on_full_img(image)
            
            # Extract cells from distance map using provided or default thresholds
            self._report_progress(1, 2, "Extracting cells from distance map")
            labels = self.cells_from_dist_map(dist_map, cells_max_threshold, cell_fill_threshold)
            
            # Log results
            num_cells = len(np.unique(labels)) - 1  # -1 to exclude background
            logger.info(f"Segmentation complete. Found {num_cells} cells.")
            self._report_progress(2, 2, f"Found {num_cells} cells")
            
            if return_dist_map:
                return labels, dist_map
            else:
                return labels
        except Exception as e:
            logger.error(f"Error in run: {str(e)}")
            logger.error(traceback.format_exc())
            if return_dist_map:
                return np.zeros(image.shape[:2], dtype=np.int32), np.zeros(image.shape[:2], dtype=np.float32)
            else:
                return np.zeros(image.shape[:2], dtype=np.int32)
    
    def run_batch_thresholds(self, image, cells_max_values, cell_fill_values):
        """Run inference on a single image with multiple threshold values.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image
        cells_max_values : List[float]
            List of cells_max threshold values to try
        cell_fill_values : List[float]
            List of cell_fill threshold values to try
            
        Returns
        -------
        Dict[Tuple[float, float], numpy.ndarray]
            Dictionary mapping (cells_max, cell_fill) tuples to segmentation labels
        """
        try:
            # Make a copy to avoid modifying original
            image = image.copy()
            
            # Get the distance map once
            logger.info(f"Running batch threshold analysis with {len(cells_max_values)} x {len(cell_fill_values)} combinations")
            self._report_progress(0, 1, "Generating distance map for batch analysis")
            dist_map = self.predict_on_full_img(image)
            
            # Process all threshold combinations
            results = {}
            total_combinations = len(cells_max_values) * len(cell_fill_values)
            self._report_progress(1, total_combinations + 1, "Processing threshold combinations")
            
            combo_index = 0
            for cells_max in cells_max_values:
                # Recompute contours and markers for each cells_max value
                cells_max_mask = dist_map > cells_max
                
                # Convert to binary mask
                cells_max_mask = cells_max_mask.astype(np.uint8)
                
                # Find contours - handle different OpenCV versions
                try:
                    contours, _ = cv2.findContours(cells_max_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                except:
                    _, contours, _ = cv2.findContours(cells_max_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                logger.info(f"Found {len(contours)} contours with cells_max={cells_max}")
                
                # Create base mask with markers at centroids
                base_mask = np.zeros(dist_map.shape, dtype=np.int32)
                for i, contour in enumerate(contours):
                    if contour is None or len(contour) < 3:
                        continue
                        
                    try:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                        else:
                            # Handle zero division - use contour center
                            cX = int(np.mean([p[0][0] for p in contour]))
                            cY = int(np.mean([p[0][1] for p in contour]))
                            
                        # Set closest pixel to centroid
                        cY = min(max(0, cY), base_mask.shape[0]-1)
                        cX = min(max(0, cX), base_mask.shape[1]-1)
                        base_mask[cY, cX] = i + 1
                    except Exception as e:
                        logger.error(f"Error processing contour {i}: {str(e)}")
                        continue
                
                for cell_fill in cell_fill_values:
                    combo_index += 1
                    # Create cell fill mask for current threshold
                    cell_fill_mask = dist_map > cell_fill
                    cell_fill_mask = cell_fill_mask.astype(np.uint8)
                    
                    # Apply watershed with current thresholds
                    if np.max(base_mask) == 0:
                        logger.warning(f"No centroids found for cells_max={cells_max}, cell_fill={cell_fill}")
                        labels = np.zeros(dist_map.shape, dtype=np.int32)
                    else:
                        labels = watershed(-dist_map, base_mask, mask=cell_fill_mask).astype(np.int32)
                    
                    # Store results
                    results[(cells_max, cell_fill)] = labels
                    
                    # Log and report progress
                    num_cells = len(np.unique(labels)) - 1  # -1 to exclude background
                    logger.info(f"Combination cells_max={cells_max}, cell_fill={cell_fill}: found {num_cells} cells")
                    self._report_progress(combo_index + 1, total_combinations + 1, 
                                         f"Processed combination {combo_index}/{total_combinations}: found {num_cells} cells")
            
            return results
        except Exception as e:
            logger.error(f"Error in run_batch_thresholds: {str(e)}")
            logger.error(traceback.format_exc())
            return {}


class SAMCellPipeline:
    """Main pipeline class for SAMCell segmentation"""
    
    def __init__(self, model, device=None, crop_size=256):
        try:
            logger.info("Initializing SAMCellPipeline")
            # If no device specified, try to use CUDA
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"Using device: {device}")
                
            self.device = device
            self.pipeline = SlidingWindowPipeline(model, device=device, crop_size=crop_size)
            self.crop_size = crop_size
            self._progress_callback = None
            logger.info("SAMCellPipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SAMCellPipeline: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set a callback function to report progress
        
        Parameters
        ----------
        callback : callable
            A function that takes (current, total, status) as arguments
        """
        self._progress_callback = callback
        self.pipeline._progress_callback = callback

    @property
    def crop_size(self):
        return self.pipeline.crop_size
    
    @crop_size.setter
    def crop_size(self, value):
        self.pipeline.crop_size = value
        
    @property
    def cells_max_threshold(self):
        return self.pipeline.cells_max_threshold
    
    @cells_max_threshold.setter
    def cells_max_threshold(self, value):
        self.pipeline.cells_max_threshold = value
        
    @property
    def cell_fill_threshold(self):
        return self.pipeline.cell_fill_threshold
    
    @cell_fill_threshold.setter
    def cell_fill_threshold(self, value):
        self.pipeline.cell_fill_threshold = value
    
    def run(self, image, return_dist_map=False, cells_max_threshold=None, cell_fill_threshold=None):
        """Run the SAMCell pipeline on an input image
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image
        return_dist_map : bool, optional
            Whether to return the distance map as well, by default False
        cells_max_threshold : float, optional
            Override default cells_max threshold
        cell_fill_threshold : float, optional
            Override default cell_fill threshold
            
        Returns
        -------
        numpy.ndarray
            Segmentation labels
        numpy.ndarray, optional
            Distance map if return_dist_map is True
        """
        return self.pipeline.run(image, return_dist_map=return_dist_map, 
                                cells_max_threshold=cells_max_threshold, 
                                cell_fill_threshold=cell_fill_threshold)
                                
    def run_batch_thresholds(self, image, cells_max_values, cell_fill_values):
        """Run inference on a single image with multiple threshold values.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image
        cells_max_values : List[float]
            List of cells_max threshold values to try
        cell_fill_values : List[float]
            List of cell_fill threshold values to try
            
        Returns
        -------
        Dict[Tuple[float, float], numpy.ndarray]
            Dictionary mapping (cells_max, cell_fill) tuples to segmentation labels
        """
        return self.pipeline.run_batch_thresholds(image, cells_max_values, cell_fill_values) 