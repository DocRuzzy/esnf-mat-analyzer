#!/usr/bin/env python3
"""
Enhanced Background Correction for ESNF Mat Analysis
ROI-aware background correction using only existing dependencies (OpenCV, NumPy).

Key improvements:
1. Proper RESTORE-style method implementation
2. ROI perimeter-based background estimation  
3. Illumination gradient correction
4. Gel-square specific processing
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Optional

class EnhancedOpenCVBackgroundProcessor:
    """Enhanced background correction with ROI-aware processing using only OpenCV and NumPy."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_illumination_gradient(self, image: np.ndarray) -> np.ndarray:
        """
        Analyze and model illumination gradients using polynomial fitting.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Estimated illumination field
        """
        # Smooth the image to get illumination field
        kernel_size = max(51, min(image.shape) // 20)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Use bilateral filter to preserve edges while smoothing illumination
        illumination = cv2.bilateralFilter(image, -1, 80, kernel_size)
        
        # Further smooth with Gaussian to get pure illumination
        gaussian_size = kernel_size // 2
        if gaussian_size % 2 == 0:
            gaussian_size += 1
        illumination = cv2.GaussianBlur(illumination, (gaussian_size, gaussian_size), 0)
        
        self.logger.debug(f"Illumination gradient analyzed with kernel size {kernel_size}")
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
            h, w = image.shape
            edge_mask = np.zeros_like(image, dtype=bool)
            edge_width = min(border_width, min(h, w) // 10)
            edge_mask[:edge_width, :] = True  # Top edge
            edge_mask[-edge_width:, :] = True  # Bottom edge
            edge_mask[:, :edge_width] = True  # Left edge
            edge_mask[:, -edge_width:] = True  # Right edge
            perimeter_mask = edge_mask
        
        # Get background values from perimeter
        background_values = image[perimeter_mask]
        
        if len(background_values) == 0:
            return float(np.mean(image))
        
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
        
        self.logger.debug(f"ROI perimeter background estimate: {background_estimate:.2f} "
                         f"(from {len(background_values)} perimeter pixels)")
        return float(background_estimate)
    
    def create_smooth_transition_mask(self, roi_mask: np.ndarray, transition_width: int = 10) -> np.ndarray:
        """
        Create a smooth transition mask for gradual corrections at ROI borders.
        
        Args:
            roi_mask: Boolean ROI mask
            transition_width: Width of transition zone in pixels
            
        Returns:
            Smooth transition mask (0 to 1)
        """
        # Distance transform from ROI edges
        roi_uint8 = roi_mask.astype(np.uint8)
        distance = cv2.distanceTransform(roi_uint8, cv2.DIST_L2, 5)
        
        # Create smooth transition
        smooth_mask = np.clip(distance / transition_width, 0, 1)
        
        return smooth_mask
    
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
        corrected = image.astype(np.float32)
        
        # Apply illumination correction globally
        mean_illumination = np.mean(illumination_field)
        illumination_correction = mean_illumination - illumination_field
        corrected += illumination_correction
        
        # Apply local background correction within ROI
        if np.any(roi_mask):
            roi_background = np.mean(image[roi_mask])
            local_correction = perimeter_bg - roi_background
            
            # Create smooth transition mask
            smooth_mask = self.create_smooth_transition_mask(roi_mask, transition_width=15)
            
            # Apply correction with smooth transition
            correction_field = np.zeros_like(corrected)
            correction_field[roi_mask] = local_correction * smooth_mask[roi_mask]
            corrected += correction_field
        
        # Clip to valid range
        corrected = np.clip(corrected, 0, 255)
        
        self.logger.debug(f"Gel-square aware correction applied: "
                         f"illumination range: [{illumination_correction.min():.2f}, {illumination_correction.max():.2f}], "
                         f"perimeter bg: {perimeter_bg:.2f}")
        
        return corrected.astype(np.uint8)
    
    def improved_restore_method(self, image: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Improved RESTORE-style method: Iterative background estimation.
        
        Args:
            image: Input image
            roi_mask: Optional ROI mask to focus correction
            
        Returns:
            RESTORE-style corrected image
        """
        # If ROI mask provided, use gel-square aware method
        if roi_mask is not None and np.any(roi_mask):
            return self.gel_square_aware_correction(image, roi_mask)
        
        # Convert to float for processing
        img_float = image.astype(np.float32)
        
        # Iterative background estimation
        background = np.copy(img_float)
        
        # Estimate noise level using edge detection
        edges = cv2.Canny(image, 50, 150)
        noise_estimate = np.std(image[edges == 0]) if np.any(edges == 0) else np.std(image) * 0.1
        
        for iteration in range(8):  # Maximum iterations
            # Apply median filter for background estimation
            kernel_size = max(5, min(image.shape) // 40)
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            background_est = cv2.medianBlur(background.astype(np.uint8), kernel_size).astype(np.float32)
            
            # Identify foreground pixels (significantly above background)
            threshold = background_est + 2.5 * noise_estimate
            foreground_mask = img_float > threshold
            
            # Update background estimation only for background pixels
            new_background = np.where(foreground_mask, background, img_float)
            
            # Check convergence
            change = np.mean(np.abs(new_background - background))
            background = new_background
            
            if change < 0.5:  # Convergence threshold
                self.logger.debug(f"RESTORE converged after {iteration+1} iterations")
                break
        
        # Final background subtraction with smooth correction
        final_background = cv2.GaussianBlur(background.astype(np.uint8), 
                                          (kernel_size//2*2+1, kernel_size//2*2+1), 0).astype(np.float32)
        
        corrected = img_float - final_background + np.median(final_background)
        
        # Clip to valid range
        corrected = np.clip(corrected, 0, 255)
        
        self.logger.debug(f"Improved RESTORE method completed in {iteration+1} iterations, "
                         f"noise estimate: {noise_estimate:.2f}")
        return corrected.astype(np.uint8)
    
    def adaptive_rolling_ball(self, image: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Improved rolling ball method with adaptive radius and ROI awareness.
        
        Args:
            image: Input image
            roi_mask: Optional ROI mask
            
        Returns:
            Corrected image
        """
        # Adaptive radius based on image size and content
        base_radius = max(10, min(image.shape) // 50)
        
        # Analyze image texture to adjust radius
        if roi_mask is not None and np.any(roi_mask):
            roi_std = np.std(image[roi_mask])
            # Use perimeter background estimate to improve correction
            perimeter_bg = self.estimate_roi_perimeter_background(image, roi_mask)
        else:
            roi_std = np.std(image)
            perimeter_bg = np.percentile(image, 10)  # Fallback
        
        # Adjust radius based on texture
        if roi_std < 15:  # Low texture - smaller radius
            radius = max(5, base_radius // 2)
        elif roi_std > 40:  # High texture - larger radius
            radius = min(80, base_radius * 2)
        else:
            radius = base_radius
        
        # Rolling ball background estimation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Background subtraction with perimeter-aware offset
        offset = perimeter_bg if roi_mask is not None else np.mean(background)
        corrected = image.astype(np.float32) - background.astype(np.float32) + offset
        corrected = np.clip(corrected, 0, 255)
        
        self.logger.debug(f"Adaptive rolling ball: radius={radius}, texture_std={roi_std:.2f}, "
                         f"offset={offset:.2f}")
        return corrected.astype(np.uint8)
    
    def percentile_background_correction(self, image: np.ndarray, 
                                       roi_mask: Optional[np.ndarray] = None,
                                       percentile: float = 5.0) -> np.ndarray:
        """
        Enhanced percentile-based background correction with ROI awareness.
        
        Args:
            image: Input image
            roi_mask: Optional ROI mask
            percentile: Percentile for background estimation
            
        Returns:
            Corrected image
        """
        if roi_mask is not None and np.any(roi_mask):
            # Use perimeter-based background estimation
            background_value = self.estimate_roi_perimeter_background(image, roi_mask)
            target_value = np.median(image[roi_mask])
        else:
            # Global percentile estimation
            background_value = np.percentile(image, percentile)
            target_value = np.median(image)
        
        # Apply correction
        corrected = image.astype(np.float32) - background_value + target_value
        corrected = np.clip(corrected, 0, 255)
        
        self.logger.debug(f"Enhanced percentile correction: bg={background_value:.2f}, "
                         f"target={target_value:.2f}")
        return corrected.astype(np.uint8)
    
    def homomorphic_filter_enhanced(self, image: np.ndarray, 
                                  roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Enhanced homomorphic filtering with ROI awareness.
        
        Args:
            image: Input image
            roi_mask: Optional ROI mask
            
        Returns:
            Filtered image
        """
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-6
        img_float = image.astype(np.float64) + epsilon
        
        # Take logarithm
        log_img = np.log(img_float)
        
        # Apply DFT
        dft = cv2.dft(log_img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft, axes=[0, 1])
        
        # Create high-pass filter
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Adaptive filter parameters based on ROI
        if roi_mask is not None and np.any(roi_mask):
            roi_std = np.std(image[roi_mask])
            cutoff = max(20, min(60, rows // 20))  # Adaptive cutoff
            gamma_low = 0.3 if roi_std < 20 else 0.5
            gamma_high = 1.8 if roi_std < 20 else 2.2
        else:
            cutoff = 30
            gamma_low = 0.5
            gamma_high = 2.0
        
        # Create meshgrid
        u = np.arange(cols) - ccol
        v = np.arange(rows) - crow
        u, v = np.meshgrid(u, v)
        d = np.sqrt(u**2 + v**2)
        
        # Butterworth high-pass filter
        h = (gamma_high - gamma_low) * (1 - np.exp(-(d**2) / (2 * cutoff**2))) + gamma_low
        
        # Apply filter
        filtered_dft = dft_shift.copy()
        filtered_dft[:, :, 0] *= h
        filtered_dft[:, :, 1] *= h
        
        # Inverse DFT
        idft_shift = np.fft.ifftshift(filtered_dft, axes=[0, 1])
        idft = cv2.idft(idft_shift)
        result = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
        
        # Take exponential with overflow protection
        result = np.clip(result, -10, 10)  # Prevent overflow
        corrected = np.exp(result) - epsilon
        
        # Normalize to 0-255 range
        corrected = np.clip(corrected, 0, None)
        if corrected.max() > corrected.min():
            corrected = 255 * (corrected - corrected.min()) / (corrected.max() - corrected.min())
        else:
            corrected = np.full_like(corrected, 128)  # Fallback if no variation
        
        self.logger.debug(f"Enhanced homomorphic filter: cutoff={cutoff}, "
                         f"gamma_low={gamma_low}, gamma_high={gamma_high}")
        
        return corrected.astype(np.uint8)


# Test and validation functions
def test_enhanced_background_correction():
    """Test the enhanced background correction methods."""
    from pathlib import Path
    
    # Load test image
    test_image_path = Path("tests/Samples/square.png")
    if not test_image_path.exists():
        print("Test image not found, creating synthetic test...")
        # Create synthetic test image with gradient
        h, w = 400, 600
        image = np.ones((h, w), dtype=np.uint8) * 128
        
        # Add illumination gradient (lighter top, darker bottom)
        y, x = np.ogrid[:h, :w]
        gradient = 50 * (1 - y / h)  # Linear gradient
        image = np.clip(image.astype(np.float32) + gradient, 0, 255).astype(np.uint8)
        
        # Add some texture
        noise = np.random.normal(0, 10, (h, w))
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        print("Using synthetic test image")
    else:
        image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
        print("Using real test image")
    
    # Create test ROI mask (center region)
    h, w = image.shape
    roi_mask = np.zeros((h, w), dtype=bool)
    roi_mask[h//4:3*h//4, w//4:3*w//4] = True
    
    processor = EnhancedOpenCVBackgroundProcessor()
    
    # Test all new methods
    print("\nTesting enhanced background correction methods...")
    
    methods = [
        ("Gel-Square Aware", lambda: processor.gel_square_aware_correction(image, roi_mask)),
        ("Improved RESTORE", lambda: processor.improved_restore_method(image, roi_mask)),
        ("Adaptive Rolling Ball", lambda: processor.adaptive_rolling_ball(image, roi_mask)),
        ("Enhanced Percentile", lambda: processor.percentile_background_correction(image, roi_mask)),
        ("Enhanced Homomorphic", lambda: processor.homomorphic_filter_enhanced(image, roi_mask)),
    ]
    
    results = {}
    for name, method in methods:
        try:
            corrected = method()
            results[name] = corrected
            print(f"✓ {name}: {corrected.min()}-{corrected.max()} range")
        except Exception as e:
            print(f"✗ {name}: Error - {e}")
    
    # Summary
    print(f"\nOriginal image range: {image.min()}-{image.max()}")
    print(f"ROI covers {np.sum(roi_mask)} pixels ({np.sum(roi_mask)/roi_mask.size*100:.1f}%)")
    print(f"Successfully tested {len(results)}/{len(methods)} methods")
    
    return results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    
    test_enhanced_background_correction()
