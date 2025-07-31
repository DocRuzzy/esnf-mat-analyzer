import numpy as np
import cv2
from typing import Optional
import logging

class AdvancedBackgroundProcessor:
    """Advanced background correction using state-of-the-art methods."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def basic_correction(self, image: np.ndarray, n_components: int = 1) -> np.ndarray:
        """
        BaSiC-inspired background correction using robust background estimation.
        This is a simplified version that uses percentile-based background correction.
        """
        # Convert to float for processing
        image_float = image.astype(np.float32)
        
        # Use robust background estimation instead of NMF for stability
        # Estimate background using a large median filter
        background = cv2.medianBlur(image, 51)  # Large kernel for background
        
        # Convert background to float
        background_float = background.astype(np.float32)
        
        # Subtract background and add offset to maintain positive values
        corrected = image_float - background_float + 128.0
        
        # Clip to valid range
        corrected_image = np.clip(corrected, 0, 255)
        
        return corrected_image.astype(np.uint8)

    def rolling_ball_3d(self, image: np.ndarray, radius: int) -> np.ndarray:
        """
        Rolling ball background correction using morphological operations.
        This is a simplified version using available OpenCV operations.
        """
        # Create a circular structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
        
        # Perform morphological opening (erosion followed by dilation)
        # This simulates the rolling ball effect
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Subtract the background from the original image
        corrected_image = cv2.subtract(image, background)
        
        # Clip the values to the original range [0, 255]
        corrected_image = np.clip(corrected_image, 0, 255)

        return corrected_image.astype(np.uint8)
    
    def roi_aware_background_correction(self, image: np.ndarray, 
                                       roi_mask: Optional[np.ndarray] = None,
                                       method: str = "gel_square") -> np.ndarray:
        """
        ROI-aware background correction for gel square analysis.
        
        Args:
            image: Input image
            roi_mask: Boolean mask of ROI region
            method: Method to use ('gel_square', 'restore', 'adaptive_rolling', 'enhanced_percentile')
            
        Returns:
            Background corrected image
        """
        if method == "gel_square" and roi_mask is not None:
            return self._gel_square_aware_correction(image, roi_mask)
        elif method == "restore":
            return self._improved_restore_method(image, roi_mask)
        elif method == "adaptive_rolling":
            return self._adaptive_rolling_ball(image, roi_mask)
        elif method == "enhanced_percentile":
            return self._enhanced_percentile_correction(image, roi_mask)
        else:
            # Fallback to improved restore
            return self._improved_restore_method(image, roi_mask)
    
    def _estimate_roi_perimeter_background(self, image: np.ndarray, roi_mask: np.ndarray, 
                                         border_width: int = 20) -> float:
        """Estimate background intensity from ROI perimeter region."""
        # Create perimeter mask - area just outside the ROI
        kernel = np.ones((border_width*2+1, border_width*2+1), np.uint8)
        dilated_roi = cv2.dilate(roi_mask.astype(np.uint8), kernel, iterations=1)
        perimeter_mask = (dilated_roi > 0) & (roi_mask == 0)
        
        if np.sum(perimeter_mask) == 0:
            # Fall back to image edges
            h, w = image.shape
            edge_mask = np.zeros_like(image, dtype=bool)
            edge_width = min(border_width, min(h, w) // 10)
            edge_mask[:edge_width, :] = True
            edge_mask[-edge_width:, :] = True
            edge_mask[:, :edge_width] = True
            edge_mask[:, -edge_width:] = True
            perimeter_mask = edge_mask
        
        # Get background values from perimeter
        background_values = image[perimeter_mask]
        
        if len(background_values) == 0:
            return float(np.mean(image))
        
        # Use robust statistics
        background_median = np.median(background_values)
        q25, q75 = np.percentile(background_values, [25, 75])
        iqr = q75 - q25
        valid_mask = (background_values >= q25 - 1.5*iqr) & (background_values <= q75 + 1.5*iqr)
        
        if np.sum(valid_mask) > 0:
            background_estimate = np.mean(background_values[valid_mask])
        else:
            background_estimate = background_median
        
        self.logger.debug(f"ROI perimeter background estimate: {background_estimate:.2f}")
        return float(background_estimate)
    
    def _analyze_illumination_gradient(self, image: np.ndarray) -> np.ndarray:
        """Analyze and model illumination gradients using bilateral filtering."""
        kernel_size = max(51, min(image.shape) // 20)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        illumination = cv2.bilateralFilter(image, -1, 80, kernel_size)
        gaussian_size = kernel_size // 2
        if gaussian_size % 2 == 0:
            gaussian_size += 1
        illumination = cv2.GaussianBlur(illumination, (gaussian_size, gaussian_size), 0)
        
        return illumination.astype(np.float32)
    
    def _create_smooth_transition_mask(self, roi_mask: np.ndarray, transition_width: int = 10) -> np.ndarray:
        """Create a smooth transition mask for gradual corrections at ROI borders."""
        roi_uint8 = roi_mask.astype(np.uint8)
        distance = cv2.distanceTransform(roi_uint8, cv2.DIST_L2, 5)
        smooth_mask = np.clip(distance / transition_width, 0, 1)
        return smooth_mask
    
    def _gel_square_aware_correction(self, image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """Apply background correction specifically aware of gel square geometry."""
        # Estimate illumination gradient
        illumination_field = self._analyze_illumination_gradient(image)
        
        # Estimate perimeter background
        perimeter_bg = self._estimate_roi_perimeter_background(image, roi_mask)
        
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
            smooth_mask = self._create_smooth_transition_mask(roi_mask, transition_width=15)
            
            # Apply correction with smooth transition
            correction_field = np.zeros_like(corrected)
            correction_field[roi_mask] = local_correction * smooth_mask[roi_mask]
            corrected += correction_field
        
        # Clip to valid range
        corrected = np.clip(corrected, 0, 255)
        
        self.logger.debug("Gel-square aware correction applied")
        return corrected.astype(np.uint8)
    
    def _improved_restore_method(self, image: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Improved RESTORE-style method with iterative background estimation."""
        if roi_mask is not None and np.any(roi_mask):
            return self._gel_square_aware_correction(image, roi_mask)
        
        # Convert to float for processing
        img_float = image.astype(np.float32)
        background = np.copy(img_float)
        
        # Estimate noise level
        edges = cv2.Canny(image, 50, 150)
        noise_estimate = np.std(image[edges == 0]) if np.any(edges == 0) else np.std(image) * 0.1
        
        for iteration in range(8):
            kernel_size = max(5, min(image.shape) // 40)
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            background_est = cv2.medianBlur(background.astype(np.uint8), kernel_size).astype(np.float32)
            threshold = background_est + 2.5 * noise_estimate
            foreground_mask = img_float > threshold
            
            new_background = np.where(foreground_mask, background, img_float)
            change = np.mean(np.abs(new_background - background))
            background = new_background
            
            if change < 0.5:
                break
        
        # Final background subtraction
        final_background = cv2.GaussianBlur(background.astype(np.uint8), 
                                          (kernel_size//2*2+1, kernel_size//2*2+1), 0).astype(np.float32)
        
        corrected = img_float - final_background + np.median(final_background)
        corrected = np.clip(corrected, 0, 255)
        
        return corrected.astype(np.uint8)
    
    def _adaptive_rolling_ball(self, image: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Improved rolling ball method with adaptive radius and ROI awareness."""
        base_radius = max(10, min(image.shape) // 50)
        
        if roi_mask is not None and np.any(roi_mask):
            roi_std = np.std(image[roi_mask])
            perimeter_bg = self._estimate_roi_perimeter_background(image, roi_mask)
        else:
            roi_std = np.std(image)
            perimeter_bg = np.percentile(image, 10)
        
        # Adjust radius based on texture
        if roi_std < 15:
            radius = max(5, base_radius // 2)
        elif roi_std > 40:
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
        
        return corrected.astype(np.uint8)
    
    def _enhanced_percentile_correction(self, image: np.ndarray, 
                                      roi_mask: Optional[np.ndarray] = None,
                                      percentile: float = 5.0) -> np.ndarray:
        """Enhanced percentile-based background correction with ROI awareness."""
        if roi_mask is not None and np.any(roi_mask):
            background_value = self._estimate_roi_perimeter_background(image, roi_mask)
            target_value = np.median(image[roi_mask])
        else:
            background_value = np.percentile(image, percentile)
            target_value = np.median(image)
        
        corrected = image.astype(np.float32) - background_value + target_value
        corrected = np.clip(corrected, 0, 255)
        
        return corrected.astype(np.uint8)

    def restore_method(self, image: np.ndarray, percentile: float = 5.0) -> np.ndarray:
        """
        RESTORE: automatic negative control region identification.
        This is a simplified version. It identifies dark regions as background.
        """
        # Identify the background based on a low percentile threshold
        background_threshold = np.percentile(image, percentile)

        # Create a background mask
        background_mask = image <= background_threshold

        # Calculate the average background value
        background_value = image[background_mask].mean()

        # Subtract the background value from the image
        corrected_image = image - background_value

        # Clip the values to the original range [0, 255]
        corrected_image = np.clip(corrected_image, 0, 255)

        return corrected_image.astype(np.uint8)


    def gaussian_low_pass(self, image: np.ndarray, sigma: int = 21) -> np.ndarray:
        """
        Applies a Gaussian low-pass filter to estimate the background and subtracts it.
        """
        # Kernel size is often chosen as a multiple of sigma. 3*sigma is common.
        kernel_size = int(3 * sigma)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Apply a Gaussian blur to get the low-frequency background
        background = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        # Subtract the background from the original image
        # Use floating point arithmetic for the subtraction
        corrected_image = image.astype(np.float32) - background.astype(np.float32)

        # Normalize the image to use the full dynamic range
        corrected_image = cv2.normalize(corrected_image, None, 0, 255, cv2.NORM_MINMAX)

        return corrected_image.astype(np.uint8)

    def homomorphic_filter(self, image: np.ndarray, cutoff: float = 30, g_low: float = 0.5, g_high: float = 2.0) -> np.ndarray:
        """
        Homomorphic filtering for multiplicative illumination.
        This is a simplified version.
        """
        # Take the logarithm of the image
        log_image = np.log1p(image.astype(np.float64))

        # Perform a Fourier transform
        fft_image = np.fft.fft2(log_image)
        fft_shifted = np.fft.fftshift(fft_image)

        # Create a high-pass filter (Butterworth)
        rows, cols = image.shape
        crow, ccol = rows // 2 , cols // 2

        # Create a meshgrid
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))

        # Calculate the distance from the center
        d = np.sqrt((u - ccol)**2 + (v - crow)**2)

        # Create the filter
        h = (g_high - g_low) * (1 - np.exp(-(d**2) / (2 * (cutoff**2)))) + g_low

        # Apply the filter
        filtered_fft = fft_shifted * h

        # Inverse Fourier transform
        ifft_shifted = np.fft.ifftshift(filtered_fft)
        ifft_image = np.fft.ifft2(ifft_shifted)

        # Take the exponential to get the corrected image
        corrected_image = np.expm1(np.real(ifft_image))

        # Clip the values to the original range [0, 255]
        corrected_image = np.clip(corrected_image, 0, 255)

        return corrected_image.astype(np.uint8)
