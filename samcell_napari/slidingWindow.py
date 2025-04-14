import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SlidingWindowHelper:
    def __init__(self, crop_size: int, overlap_size: int):
        self.crop_size = crop_size
        self.overlap_size = overlap_size
        # Create a blending mask that decreases towards the edges
        self._create_blending_mask()
        logger.info(f"SlidingWindowHelper initialized with crop_size={crop_size}, overlap_size={overlap_size}")
    
    def _create_blending_mask(self):
        """Create a blending mask that decreases towards the edges for smooth transitions."""
        try:
            mask = np.ones((self.crop_size, self.crop_size), dtype=np.float32)
            
            # Calculate a more gradual falloff for edges 
            # Use a cosine function for smoother transition
            for i in range(self.overlap_size):
                # Calculate weight factor (increases as we move from edge to center)
                # Using cosine falloff for smoother transition
                weight = 0.5 * (1 - np.cos(np.pi * i / self.overlap_size))
                
                # Apply to all four edges
                mask[i, :] *= weight  # Top edge
                mask[-i-1, :] *= weight  # Bottom edge
                mask[:, i] *= weight  # Left edge
                mask[:, -i-1] *= weight  # Right edge
            
            self.blending_mask = mask
            logger.info("Blending mask created successfully")
        except Exception as e:
            logger.error(f"Error in _create_blending_mask: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Create a simple fallback mask if the fancy one fails
            self.blending_mask = np.ones((self.crop_size, self.crop_size), dtype=np.float32)

    def seperate_into_crops(self, img):
        """
        Split image into overlapping crops
        Note: Method name has a typo but kept for compatibility with original code
        """
        try:
            if img is None or img.size == 0:
                logger.error("Empty image provided to seperate_into_crops")
                return [], [], (0, 0, 0, 0)
                
            # If image is smaller than crop size, adjust the crop size
            orig_height, orig_width = img.shape
            logger.info(f"Processing image of size {orig_height}x{orig_width}")
            
            if orig_height < self.crop_size or orig_width < self.crop_size:
                # Set crop size to the smaller dimension of the image
                new_crop_size = min(orig_height, orig_width)
                if new_crop_size < 2 * self.overlap_size:
                    # If even the smaller dimension is too small for proper overlap
                    # Just process the whole image at once
                    new_overlap_size = 0
                    logger.info("Image smaller than crop size, processing as single crop")
                    # Return the entire image as a single crop
                    return [img], [(0, 0, orig_width, orig_height)], (0, 0, orig_width, orig_height)
                else:
                    # Adjust crop size and keep the original overlap
                    new_overlap_size = self.overlap_size
                    logger.info(f"Adjusted crop size to {new_crop_size} due to small image")
            else:
                # Use the original crop size and overlap
                new_crop_size = self.crop_size
                new_overlap_size = self.overlap_size
            
            # Calculate the effective stride (distance between consecutive crop centers)
            stride = new_crop_size - 2 * new_overlap_size
            
            # Mirror the image such that the edges are repeated around the overlap region
            img_mirrored = cv2.copyMakeBorder(img, new_overlap_size, new_overlap_size, new_overlap_size, new_overlap_size, cv2.BORDER_REFLECT)
            logger.info(f"Mirrored image shape: {img_mirrored.shape}")

            # Get the image dimensions after mirroring
            mirrored_height, mirrored_width = img_mirrored.shape
            
            # Calculate exact number of crops needed to cover the entire image
            # Add 1 because we need at least one crop even if the image is smaller than the stride
            num_crops_y = max(1, int(np.ceil(orig_height / stride)))
            num_crops_x = max(1, int(np.ceil(orig_width / stride)))
            
            logger.info(f"Using crop size: {new_crop_size}, overlap: {new_overlap_size}, stride: {stride}")
            logger.info(f"Creating {num_crops_y}x{num_crops_x} crops")
            
            # Initialize a list to store cropped images
            cropped_images = []
            orig_regions = []
            crop_unique_region = (new_overlap_size, new_overlap_size, new_crop_size - 2 * new_overlap_size, new_crop_size - 2 * new_overlap_size)
            
            for y_idx in range(num_crops_y):
                for x_idx in range(num_crops_x):
                    # For all but the last tile in each dimension, use regular spacing
                    if y_idx < num_crops_y - 1:
                        y_start = y_idx * stride + new_overlap_size
                    else:
                        # Last row - ensure it aligns with the bottom edge
                        y_start = mirrored_height - new_crop_size
                    
                    if x_idx < num_crops_x - 1:
                        x_start = x_idx * stride + new_overlap_size
                    else:
                        # Last column - ensure it aligns with the right edge
                        x_start = mirrored_width - new_crop_size
                    
                    # Calculate crop boundaries in mirrored image
                    x_end = x_start + new_crop_size
                    y_end = y_start + new_crop_size
                    
                    # Extract the crop with mirrored edges
                    crop = img_mirrored[y_start:y_end, x_start:x_end]
                    
                    # Calculate the corresponding region in the original image
                    # The -new_overlap_size is to account for the mirroring we did earlier
                    orig_x = max(0, x_start - new_overlap_size)
                    orig_y = max(0, y_start - new_overlap_size)
                    
                    # Make sure we don't exceed the original image size
                    orig_w = min(new_crop_size, orig_width - orig_x)
                    orig_h = min(new_crop_size, orig_height - orig_y)
                    
                    # Ensure we're not trying to go outside the original image
                    orig_x = min(orig_x, orig_width)
                    orig_y = min(orig_y, orig_height)
                    
                    # Fix potential negative width/height
                    orig_w = max(0, min(orig_w, orig_width - orig_x))
                    orig_h = max(0, min(orig_h, orig_height - orig_y))
                    
                    orig_region = (orig_x, orig_y, orig_w, orig_h)
                    
                    # Don't add empty regions
                    if orig_w > 0 and orig_h > 0:
                        # Append the cropped image and region
                        cropped_images.append(crop)
                        orig_regions.append(orig_region)
            
            logger.info(f"Created {len(cropped_images)} crops")
            return cropped_images, orig_regions, crop_unique_region
            
        except Exception as e:
            logger.error(f"Error in seperate_into_crops: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return [], [], (0, 0, 0, 0)
    
    # Add alias with correct spelling
    separate_into_crops = seperate_into_crops

    def combine_crops(self, orig_size, cropped_images, orig_regions, crop_unique_region, sam_outputs=None):
        """
        Combine crops back into a full image with smooth blending.
        
        Args:
            orig_size: Original image size (H, W)
            cropped_images: List of image crops
            orig_regions: List of regions in the original image
            crop_unique_region: Region in each crop considered unique
            sam_outputs: Optional SAM model outputs to use instead of cropped_images
                        (SAM outputs are typically 256x256 regardless of input crop size)
        """
        try:
            if not cropped_images or not orig_regions:
                logger.error("No crops provided to combine_crops")
                return np.zeros(orig_size, dtype=np.float32)
                
            logger.info(f"Combining {len(cropped_images)} crops into image of size {orig_size}")
            
            # Initialize output image and weight accumulator for blending
            output_img = np.zeros(orig_size, dtype=np.float32)
            weight_map = np.zeros(orig_size, dtype=np.float32)
            
            # Get the actual crop size from the first crop
            crop_h, crop_w = cropped_images[0].shape
            
            # Create a blending mask for this specific crop size if needed
            if (crop_h != self.crop_size or crop_w != self.crop_size):
                temp_crop_size = self.crop_size
                self.crop_size = max(crop_h, crop_w)
                self._create_blending_mask()
                blending_mask = self.blending_mask
                self.crop_size = temp_crop_size
                logger.info(f"Created temporary blending mask for crop size {crop_h}x{crop_w}")
            else:
                blending_mask = self.blending_mask

            # Ensure blending mask matches crop size
            if blending_mask.shape != (crop_h, crop_w):
                blending_mask = cv2.resize(blending_mask, (crop_w, crop_h))
                logger.info(f"Resized blending mask to {crop_w}x{crop_h}")
                
            for i, (crop, region) in enumerate(zip(cropped_images, orig_regions)):
                try:
                    # Extract region coordinates
                    x, y, w, h = region
                    
                    # Skip if the region is invalid
                    if w <= 0 or h <= 0:
                        logger.warning(f"Skipping crop {i} with invalid region {region}")
                        continue
                    
                    # If we have SAM outputs, use them instead of the crop
                    if sam_outputs is not None:
                        sam_output = sam_outputs[i]
                        # Resize SAM output to match crop size if needed
                        if sam_output.shape != crop.shape:
                            sam_output = cv2.resize(sam_output, (crop.shape[1], crop.shape[0]), 
                                                  interpolation=cv2.INTER_LINEAR)
                        crop_to_use = sam_output
                    else:
                        crop_to_use = crop
                    
                    # Calculate which part of the blending mask to use
                    # This ensures we only use the portion of the mask that corresponds
                    # to the valid part of the crop
                    mask_h, mask_w = min(h, blending_mask.shape[0]), min(w, blending_mask.shape[1])
                    
                    # Ensure we're not exceeding the dimensions of the output image
                    y_end = min(y + mask_h, orig_size[0])
                    x_end = min(x + mask_w, orig_size[1])
                    mask_h = y_end - y
                    mask_w = x_end - x
                    
                    if mask_h <= 0 or mask_w <= 0:
                        logger.warning(f"Skipping crop {i} with invalid mask region")
                        continue
                    
                    # Get the region of the crop and mask to use
                    crop_region = crop_to_use[:mask_h, :mask_w]
                    mask_region = blending_mask[:mask_h, :mask_w]
                    
                    # Apply the blending mask to this region of the crop
                    weighted_crop = crop_region * mask_region
                    
                    # Add to the output image and weight map at the correct region
                    output_img[y:y_end, x:x_end] += weighted_crop
                    weight_map[y:y_end, x:x_end] += mask_region
                
                except Exception as e:
                    logger.error(f"Error processing crop {i}: {str(e)}")
            
            # Avoid division by zero and normalize
            mask = weight_map > 0.0001
            output_img[mask] /= weight_map[mask]
            
            # Handle any remaining zeros by filling with nearest neighbor
            if np.any(~mask):
                logger.info(f"Filling {np.sum(~mask)} pixels with nearest neighbor interpolation")
                # Create a mask for where we have values
                valid_mask = np.zeros(orig_size, dtype=np.uint8)
                valid_mask[mask] = 1
                
                # Use distance transform to find nearest valid pixel
                dist, indices = cv2.distanceTransformWithLabels(
                    1 - valid_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
                
                # Get coordinates of nearest valid pixels
                h, w = orig_size
                coords_y, coords_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                nearest_y = coords_y.flatten()[indices.flatten() - 1].reshape(h, w)
                nearest_x = coords_x.flatten()[indices.flatten() - 1].reshape(h, w)
                
                # Fill in missing values with nearest valid pixel
                for y in range(h):
                    for x in range(w):
                        if not mask[y, x]:
                            ny, nx = nearest_y[y, x], nearest_x[y, x]
                            if 0 <= ny < h and 0 <= nx < w:
                                output_img[y, x] = output_img[ny, nx]
            
            return output_img
            
        except Exception as e:
            logger.error(f"Error in combine_crops: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return np.zeros(orig_size, dtype=np.float32) 