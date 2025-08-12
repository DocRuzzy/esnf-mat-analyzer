#!/usr/bin/env python3
"""
Advanced Background Correction for ESNF Mat Analysis
This module implements sophisticated background correction methods specifically
designed for electrospun nanofiber mat images with ROI-aware processing.

Key improvements:
1. True RESTORE method implementation
2. ROI perimeter-based background estimation
3. Illumination gradient correction
4. Gel-square specific processing
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.optimize import minimize
from skimage import restoration, morphology
from skimage.filters import gaussian
import logging
from typing import Tuple, Optional

class EnhancedScipyBackgroundProcessor:
    """Enhanced background correction with ROI-aware processing using scipy optimization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_illumination_gradient(self, image: np.ndarray) -> np.ndarray:
        """
        Analyze and model illumination gradients in the image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Estimated illumination field
        """
        # Smooth the image heavily to get illumination field
        sigma = min(image.shape) // 20  # Adaptive smoothing based on image size
        illumination = gaussian(image, sigma=sigma, preserve_range=True)
        
        # Fit a polynomial surface to model the gradient
        h, w = image.shape
        y, x = np.ogrid[:h, :w]
        
        # Create coordinate matrices
        x_norm = (x - w/2) / (w/2)
        y_norm = (y - h/2) / (h/2)
        
        # Fit 2D polynomial (quadratic)
        def polynomial_surface(params):
            a, b, c, d, e, f = params
            return a + b*x_norm + c*y_norm + d*x_norm**2 + e*y_norm**2 + f*x_norm*y_norm
        
        # Optimize polynomial fit to illumination
        def objective(params):
            predicted = polynomial_surface(params)
            return np.mean((illumination - predicted)**2)
        
        # Initial guess
        initial_params = [np.mean(illumination), 0, 0, 0, 0, 0]
        result = minimize(objective, initial_params, method='BFGS')
        
        if result.success:
            fitted_illumination = polynomial_surface(result.x)
            self.logger.debug(f"Illumination gradient fitted with parameters: {result.x}")
            return fitted_illumination.astype(np.float32)
        else:
            self.logger.warning("Failed to fit illumination gradient, using Gaussian smoothing")
            return illumination.astype(np.float32)
    
    def estimate_roi_perimeter_background(self, image: np.ndarray, roi_mask: np.ndarray, 
                                        border_width: int = 20) -> float:
        """
        Estimate background intensity from ROI perimeter region.
        
        Args:
            image: Input image
            roi_mask: Boolean mask of ROI
            border_width: Width of border region to analyze
            
        Returns:
            Estimated background intensity
        """
        # Create perimeter mask - area just outside the ROI
        kernel = np.ones((border_width*2+1, border_width*2+1), np.uint8)
        dilated_roi = cv2.dilate(roi_mask.astype(np.uint8), kernel, iterations=1)
        perimeter_mask = (dilated_roi > 0) & (roi_mask == 0)
        
        if np.sum(perimeter_mask) == 0:
            self.logger.warning("No perimeter region found, using image edges")
            # Fall back to image edges
            edge_mask = np.zeros_like(image, dtype=bool)
            edge_mask[:border_width, :] = True  # Top edge
            edge_mask[-border_width:, :] = True  # Bottom edge
            edge_mask[:, :border_width] = True  # Left edge
            edge_mask[:, -border_width:] = True  # Right edge
            perimeter_mask = edge_mask
        
        # Get background values from perimeter
        background_values = image[perimeter_mask]
        
        if len(background_values) == 0:
            return np.mean(image)
        
        # Use robust statistics (median and IQR)
        background_median = np.median(background_values)
        q25, q75 = np.percentile(background_values, [25, 75])
        
        # Filter outliers and recalculate
        iqr = q75 - q25
        valid_mask = (background_values >= q25 - 1.5*iqr) & (background_values <= q75 + 1.5*iqr)
        
        if np.sum(valid_mask) > 0:
            background_estimate = np.mean(background_values[valid_mask])
        else:
            background_estimate = background_median
        
        self.logger.debug(f"ROI perimeter background estimate: {background_estimate:.2f}")
        return float(background_estimate)
    
    def gel_square_aware_correction(self, image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """
        Apply background correction specifically aware of gel square geometry.
        
        Args:
            image: Input image
            roi_mask: Boolean mask of the gel square region
            
        Returns:
            Background corrected image
        """
        # Estimate illumination gradient
        illumination_field = self.analyze_illumination_gradient(image)
        
        # Estimate perimeter background
        perimeter_bg = self.estimate_roi_perimeter_background(image, roi_mask)
        
        # Create correction field
        # Inside ROI: correct for both illumination and local background
        # Outside ROI: correct for illumination only
        
        corrected = image.astype(np.float32)
        
        # Apply illumination correction globally
        mean_illumination = np.mean(illumination_field)
        illumination_correction = mean_illumination - illumination_field
        corrected += illumination_correction
        
        # Apply local background correction within ROI
        roi_background = np.mean(image[roi_mask]) if np.any(roi_mask) else perimeter_bg
        local_correction = perimeter_bg - roi_background
        
        # Apply local correction with smooth transition at ROI borders
        if np.any(roi_mask):
            # Create smooth transition mask
            distance_transform = ndimage.distance_transform_edt(roi_mask)
            transition_width = 10  # pixels
            smooth_mask = np.clip(distance_transform / transition_width, 0, 1)
            
            corrected[roi_mask] += local_correction * smooth_mask[roi_mask]
        
        # Clip to valid range
        corrected = np.clip(corrected, 0, 255)
        
        self.logger.debug(f"Gel-square aware correction applied: "
                         f"illumination correction range: {illumination_correction.min():.2f} to {illumination_correction.max():.2f}, "
                         f"local correction: {local_correction:.2f}")
        
        return corrected.astype(np.uint8)
    
    def true_restore_method(self, image: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        True RESTORE method: Robust automatic background estimation.
        
        Based on the RESTORE algorithm for automatic background correction
        in fluorescence microscopy images.
        
        Args:
            image: Input image
            roi_mask: Optional ROI mask to focus correction
            
        Returns:
            RESTORE corrected image
        """
        try:
            # Convert to float for processing
            img_float = image.astype(np.float64)
            
            # If ROI mask provided, use gel-square aware method
            if roi_mask is not None:
                return self.gel_square_aware_correction(image, roi_mask)
            
            # Estimate noise level
            # Use Laplacian to estimate noise
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            noise_var = np.var(laplacian) / 4  # Theoretical relationship
            
            # Iterative background estimation
            background = np.copy(img_float)
            
            for iteration in range(10):  # Maximum iterations
                # Apply median filter for background estimation
                kernel_size = max(5, min(image.shape) // 50)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                    
                background_est = cv2.medianBlur(background.astype(np.uint8), kernel_size).astype(np.float64)
                
                # Identify foreground pixels (significantly above background)
                threshold = background_est + 2 * np.sqrt(noise_var)
                foreground_mask = img_float > threshold
                
                # Update background estimation only for background pixels
                background[~foreground_mask] = img_float[~foreground_mask]
                
                # Check convergence
                if iteration > 0:
                    change = np.mean(np.abs(background - prev_background))
                    if change < 0.1:  # Convergence threshold
                        break
                
                prev_background = np.copy(background)
            
            # Final background subtraction
            final_background = gaussian(background, sigma=kernel_size//3, preserve_range=True)
            corrected = img_float - final_background + np.median(final_background)
            
            # Clip to valid range
            corrected = np.clip(corrected, 0, 255)
            
            self.logger.debug(f"True RESTORE method completed in {iteration+1} iterations")
            return corrected.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Error in true RESTORE method: {e}")
            # Fallback to simple background subtraction
            return self.percentile_background_correction(image)
    
    def percentile_background_correction(self, image: np.ndarray, percentile: float = 5.0) -> np.ndarray:
        """
        Simple percentile-based background correction (renamed from misleading 'restore_method').
        
        Args:
            image: Input image
            percentile: Percentile for background estimation
            
        Returns:
            Corrected image
        """
        background_value = np.percentile(image, percentile)
        corrected = image.astype(np.float32) - background_value + 128
        corrected = np.clip(corrected, 0, 255)
        
        self.logger.debug(f"Percentile background correction: {percentile}th percentile = {background_value:.2f}")
        return corrected.astype(np.uint8)
    
    def adaptive_rolling_ball(self, image: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Improved rolling ball method with adaptive radius based on image content.
        
        Args:
            image: Input image
            roi_mask: Optional ROI mask
            
        Returns:
            Corrected image
        """
        # Adaptive radius based on image size and content
        base_radius = min(image.shape) // 40
        
        # Analyze image texture to adjust radius
        if roi_mask is not None and np.any(roi_mask):
            roi_std = np.std(image[roi_mask])
        else:
            roi_std = np.std(image)
        
        # Adjust radius based on texture
        if roi_std < 20:  # Low texture - smaller radius
            radius = max(10, base_radius // 2)
        elif roi_std > 50:  # High texture - larger radius
            radius = min(100, base_radius * 2)
        else:
            radius = base_radius
        
        # Rolling ball background estimation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Background subtraction with offset
        corrected = image.astype(np.float32) - background.astype(np.float32) + np.mean(background)
        corrected = np.clip(corrected, 0, 255)
        
        self.logger.debug(f"Adaptive rolling ball: radius={radius}, texture_std={roi_std:.2f}")
        return corrected.astype(np.uint8)
    
    def deconvolution_background_correction(self, image: np.ndarray, 
                                          roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Advanced deconvolution-based background correction.
        
        Uses Richardson-Lucy deconvolution to remove illumination artifacts.
        
        Args:
            image: Input image
            roi_mask: Optional ROI mask
            
        Returns:
            Deconvolved and background corrected image
        """
        try:
            # Estimate point spread function (PSF) for illumination
            # Use Gaussian PSF based on illumination scale
            sigma = min(image.shape) // 30
            psf = np.zeros((sigma*4+1, sigma*4+1))
            center = sigma*2
            y, x = np.ogrid[:psf.shape[0], :psf.shape[1]]
            psf = np.exp(-((x-center)**2 + (y-center)**2) / (2*sigma**2))
            psf /= psf.sum()
            
            # Apply Richardson-Lucy deconvolution
            img_float = image.astype(np.float64) / 255.0
            deconvolved = restoration.richardson_lucy(img_float, psf, num_iter=10, clip=False)
            
            # Enhance contrast and normalize
            deconvolved = np.clip(deconvolved * 255, 0, 255)
            
            # Additional background correction if ROI provided
            if roi_mask is not None:
                background_estimate = self.estimate_roi_perimeter_background(
                    deconvolved.astype(np.uint8), roi_mask)
                deconvolved += (128 - background_estimate)
                deconvolved = np.clip(deconvolved, 0, 255)
            
            self.logger.debug(f"Deconvolution correction applied with PSF sigma={sigma}")
            return deconvolved.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Error in deconvolution correction: {e}")
            # Fallback to gel-square aware correction
            if roi_mask is not None:
                return self.gel_square_aware_correction(image, roi_mask)
            else:
                return self.percentile_background_correction(image)

# Legacy compatibility functions (update existing methods)
def update_advanced_background_processor():
    """Update the existing AdvancedBackgroundProcessor with new methods."""
    
    # Import the existing class
    import esnf_mat_analyzer.processing.background_correction.advanced as abg_module
    
    # Add new methods to the existing class
    original_class = abg_module.AdvancedBackgroundProcessor
    enhanced_class = EnhancedScipyBackgroundProcessor
    
    # Update method mappings
    original_class.true_restore_method = enhanced_class.true_restore_method
    original_class.gel_square_aware_correction = enhanced_class.gel_square_aware_correction
    original_class.analyze_illumination_gradient = enhanced_class.analyze_illumination_gradient
    original_class.estimate_roi_perimeter_background = enhanced_class.estimate_roi_perimeter_background
    original_class.adaptive_rolling_ball = enhanced_class.adaptive_rolling_ball
    original_class.deconvolution_background_correction = enhanced_class.deconvolution_background_correction
    original_class.percentile_background_correction = enhanced_class.percentile_background_correction

if __name__ == "__main__":
    # Test the new background correction methods
    import cv2
    from pathlib import Path
    
    # Load test image
    test_image_path = Path("tests/Samples/square.png")
    if test_image_path.exists():
        image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
        
        # Create a test ROI mask (center square)
        h, w = image.shape
        roi_mask = np.zeros((h, w), dtype=bool)
        roi_mask[h//4:3*h//4, w//4:3*w//4] = True
        
        processor = EnhancedScipyBackgroundProcessor()
        
        # Test new methods
        print("Testing new background correction methods...")
        
        restore_corrected = processor.true_restore_method(image, roi_mask)
        gel_corrected = processor.gel_square_aware_correction(image, roi_mask)
        deconv_corrected = processor.deconvolution_background_correction(image, roi_mask)
        
        print("✓ All new methods executed successfully")
        print(f"Original range: {image.min()}-{image.max()}")
        print(f"RESTORE corrected range: {restore_corrected.min()}-{restore_corrected.max()}")
        print(f"Gel-aware corrected range: {gel_corrected.min()}-{gel_corrected.max()}")
        print(f"Deconvolution corrected range: {deconv_corrected.min()}-{deconv_corrected.max()}")
    else:
        print("Test image not found, skipping validation")
