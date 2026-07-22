import numpy as np
import cv2
from typing import Optional, Tuple, Dict
import logging

from ...core.data_types import BackgroundCorrectionOperator


class AdvancedBackgroundProcessor:
    """
    Advanced background correction following the 4-step scientific guide.
    
    Implements the complete workflow from:
    "Guide: Analyzing Electrospun Mat Uniformity with Background Correction"
    
    This is a sequential 4-step process, not alternative methods:
    1. Isolate Background Pixels  
    2. Model Uneven Illumination
    3. Correct the Image
    4. Analyze Mat Uniformity
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Cache simple structuring elements to avoid repeated allocation in rolling ball
        self._se_cache: Dict[int, np.ndarray] = {}
        # QA metrics from the most recent _step3_correct_image call
        self.last_correction_quality: Dict = {}

    # ------------------------------------------------------------------
    # Simple / legacy API expected by tests
    # ------------------------------------------------------------------
    def basic_correction(self, image: np.ndarray, blur_radius: int = 51) -> np.ndarray:
        """Basic background correction using large Gaussian blur division.

        Matches original 'basic' method semantics used in tests. Provides a
        lightweight fast path before advanced multi-step workflow.

        NOTE: preview/legacy only — intentionally uses DIVISION semantics and
        is NOT the analyze pipeline. The analyze path goes through
        _step3_correct_image, which defaults to the SUBTRACT operator.
        """
        if image.ndim != 2:
            raise ValueError("basic_correction expects a 2D grayscale image")
        # Ensure odd kernel
        k = max(3, blur_radius)
        if k % 2 == 0:
            k += 1
        # Cap kernel to image min dimension - 1 to avoid OpenCV errors
        k = min(k, min(image.shape) - (1 - min(image.shape) % 2))
        background = cv2.GaussianBlur(image, (k, k), 0)
        bgf = np.maximum(background.astype(np.float32), 1.0)
        imgf = image.astype(np.float32)
        corrected = imgf / bgf * np.mean(bgf)
        corrected = np.clip(corrected, 0, 255)
        return corrected.astype(np.uint8)

    def rolling_ball_3d(self, image: np.ndarray, radius: int = 50) -> np.ndarray:
        """Legacy rolling ball style background subtraction.

        Implements a 2D morphological opening with an elliptical (approx ball)
        structuring element then division normalization (preferred over raw
        subtraction for illumination). Named '3d' for historical reasons.

        NOTE: preview/legacy only — uses DIVISION semantics; the analyze
        pipeline goes through _step3_correct_image (SUBTRACT default).
        """
        if image.ndim != 2:
            raise ValueError("rolling_ball_3d expects a 2D grayscale image")
        r = max(1, int(radius))
        # Cache structuring element
        if r not in self._se_cache:
            ksize = r * 2 + 1
            self._se_cache[r] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        se = self._se_cache[r]
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, se)
        bgf = np.maximum(background.astype(np.float32), 1.0)
        imgf = image.astype(np.float32)
        corrected = imgf / bgf * np.mean(bgf)
        corrected = np.clip(corrected, 0, 255)
        return corrected.astype(np.uint8)

    def complete_uniformity_analysis(self, image: np.ndarray,
                                   mat_threshold: int = 180,
                                   hydrogel_threshold: int = 50,
                                   background_method: str = "polynomial",
                                   polynomial_order: int = 2,
                                   operator: Optional[BackgroundCorrectionOperator] = None) -> Dict:
        """
        Complete 4-step uniformity analysis following the guide.

        Args:
            image: Input grayscale image
            mat_threshold: High-pass threshold for mat pixels
            hydrogel_threshold: Low-pass threshold for hydrogel pixels
            background_method: "polynomial" or "large_kernel_blur"
            polynomial_order: Order for polynomial fitting (2 or 3)
            operator: Step 3 operator (SUBTRACT default, DIVIDE legacy)

        Returns:
            Dictionary containing all results and intermediate steps
        """
        results = {}
        
        # STEP 1: Isolate the Background Pixels
        self.logger.info("Step 1: Isolating background pixels...")
        masks = self._step1_isolate_background_pixels(image, mat_threshold, hydrogel_threshold)
        results.update(masks)
        
        # STEP 2: Model the Uneven Illumination  
        self.logger.info(f"Step 2: Modeling illumination using {background_method}...")
        if background_method == "polynomial":
            estimated_background = self._step2a_polynomial_surface_fitting(
                image, masks['background_mask'], polynomial_order
            )
        else:  # large_kernel_blur
            estimated_background = self._step2b_large_kernel_blurring(
                image, masks['background_mask']
            )
        results['estimated_background'] = estimated_background
        
        # STEP 3: Correct the Image
        self.logger.info("Step 3: Correcting image...")
        corrected_image = self._step3_correct_image(
            image, estimated_background,
            background_mask=masks['background_mask'], operator=operator
        )
        results['corrected_image'] = corrected_image
        results['correction_quality'] = dict(self.last_correction_quality)

        # STEP 4: Analyze Mat Uniformity
        self.logger.info("Step 4: Analyzing mat uniformity...")
        uniformity_metrics = self._step4_analyze_mat_uniformity(
            corrected_image, masks['mat_mask']
        )
        results.update(uniformity_metrics)

        return results

    def _step1_isolate_background_pixels(self, image: np.ndarray, 
                                       mat_threshold: int = 180,
                                       hydrogel_threshold: int = 50) -> Dict:
        """
        Step 1: Create digital masks to isolate background pixels.
        
        Following the guide exactly:
        1. Convert to grayscale (assume already done)
        2. Create foreground mask (mat + hydrogel)
        3. Invert to get background mask
        """
        h, w = image.shape
        
        # 2a. Mask the Mat (brightest object)
        mat_mask = image > mat_threshold
        
        # 2b. Mask the Hydrogel (darkest object) 
        hydrogel_mask = image < hydrogel_threshold
        
        # 2c. Combine to create foreground mask
        foreground_mask = mat_mask | hydrogel_mask
        
        # 3. Invert to get final background mask
        background_mask = ~foreground_mask
        
        # Log statistics
        total_pixels = h * w
        bg_pixels = np.sum(background_mask)
        mat_pixels = np.sum(mat_mask)
        hydrogel_pixels = np.sum(hydrogel_mask)
        
        self.logger.debug(f"Mask statistics:")
        self.logger.debug(f"  Background: {bg_pixels} pixels ({100*bg_pixels/total_pixels:.1f}%)")
        self.logger.debug(f"  Mat: {mat_pixels} pixels ({100*mat_pixels/total_pixels:.1f}%)")
        self.logger.debug(f"  Hydrogel: {hydrogel_pixels} pixels ({100*hydrogel_pixels/total_pixels:.1f}%)")
        
        return {
            'mat_mask': mat_mask,
            'hydrogel_mask': hydrogel_mask, 
            'foreground_mask': foreground_mask,
            'background_mask': background_mask
        }

    def _step2a_polynomial_surface_fitting(self, image: np.ndarray, 
                                         background_mask: np.ndarray,
                                         polynomial_order: int = 2) -> np.ndarray:
        """
        Step 2 Method A: Polynomial Surface Fitting (Most Robust)
        
        Fits 2D polynomial to background pixels and generates smooth surface.
        """
        # Get coordinates of background pixels
        y_coords, x_coords = np.where(background_mask)
        background_intensities = image[background_mask].astype(np.float64)
        
        if len(background_intensities) == 0:
            self.logger.warning("No background pixels found, returning original image")
            return image.astype(np.float32)
        
        # Normalize coordinates for numerical stability
        h, w = image.shape
        x_norm = x_coords.astype(np.float64) / w
        y_norm = y_coords.astype(np.float64) / h
        
        # Create polynomial feature matrix
        X = self._create_polynomial_features(x_norm, y_norm, polynomial_order)
        
        # Solve using least squares  
        coefficients, residuals, rank, s = np.linalg.lstsq(X, background_intensities, rcond=None)
        
        # Generate surface for entire image
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        x_grid_norm = x_grid.astype(np.float64) / w
        y_grid_norm = y_grid.astype(np.float64) / h
        
        X_full = self._create_polynomial_features(x_grid_norm.ravel(), y_grid_norm.ravel(), polynomial_order)
        estimated_background = X_full.dot(coefficients).reshape(h, w)
        
        self.logger.debug(f"Polynomial fitting completed (order {polynomial_order})")
        return estimated_background.astype(np.float32)

    def _step2b_large_kernel_blurring(self, image: np.ndarray, 
                                    background_mask: np.ndarray) -> np.ndarray:
        """
        Step 2 Method B: Enhanced Large-Kernel Blurring for Shadow Handling
        
        Uses multi-scale Gaussian blur with edge-preserving interpolation
        to better handle shadows and complex illumination patterns.
        """
        # Create background-only image (foreground = 0)
        background_only = np.zeros_like(image, dtype=np.float32)
        background_only[background_mask] = image[background_mask].astype(np.float32)
        
        # Calculate kernel sizes for multi-scale approach
        h, w = image.shape
        base_kernel = max(h, w) // 4  # Start smaller for better detail preservation
        if base_kernel % 2 == 0:
            base_kernel += 1
        base_kernel = max(base_kernel, 5)  # Ensure minimum size
        
        # Multi-scale blurring: combine different kernel sizes
        kernels = [base_kernel, base_kernel * 2 + 1, base_kernel * 3 + 1]
        # Ensure all kernels are odd, positive, and within bounds
        valid_kernels = []
        for k in kernels:
            if k > 0 and k <= min(h, w):
                if k % 2 == 0:
                    k += 1
                if k <= min(h, w):  # Check again after adjustment
                    valid_kernels.append(k)
        kernels = valid_kernels if valid_kernels else [5]  # Fallback to minimum
        weight_sum = np.zeros_like(background_only)
        weighted_background = np.zeros_like(background_only)
        
        mask_float = background_mask.astype(np.float32)
        
        # Enhanced shadow detection for better background modeling
        mean_intensity = np.mean(image[background_mask])
        shadow_threshold = mean_intensity * 0.7  # Detect darker regions
        
        for i, kernel_size in enumerate(kernels):
            # Apply Gaussian blur for this scale
            blurred_bg = cv2.GaussianBlur(background_only, (kernel_size, kernel_size), 0)
            blurred_mask = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), 0)
            
            # Normalize this scale
            scale_background = np.divide(blurred_bg, blurred_mask,
                                       out=np.zeros_like(blurred_bg),
                                       where=blurred_mask > 0.01)
            
            # Shadow-aware weighting: give more weight to larger scales in shadow areas
            shadow_regions = (image < shadow_threshold) & background_mask
            shadow_weight_boost = 1.5 if np.any(shadow_regions) else 1.0
            
            # Weight smaller kernels more for detail, larger for shadows
            base_weight = 1.0 / (i + 1)  # Decreasing weights: 1.0, 0.5, 0.33...
            if i > 0:  # Larger kernels get shadow boost
                weight = base_weight * shadow_weight_boost
            else:
                weight = base_weight
                
            scale_weight = blurred_mask * weight
            
            weighted_background += scale_background * scale_weight
            weight_sum += scale_weight
        
        # Final normalization
        estimated_background = np.divide(weighted_background, weight_sum,
                                       out=np.zeros_like(weighted_background),
                                       where=weight_sum > 0.01)
        
        # Enhanced gap filling using morphological operations
        # This better handles areas completely surrounded by excluded regions
        gap_mask = (estimated_background == 0) & (weight_sum <= 0.01)
        if np.any(gap_mask):
            # Use closing operation to fill gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_kernel//2, base_kernel//2))
            filled_bg = cv2.morphologyEx(estimated_background, cv2.MORPH_CLOSE, kernel)
            estimated_background[gap_mask] = filled_bg[gap_mask]
            
            # Final fallback: use median background value for remaining gaps
            remaining_gaps = estimated_background == 0
            if np.any(remaining_gaps):
                median_bg = np.median(image[background_mask])
                estimated_background[remaining_gaps] = median_bg
        
        self.logger.debug(f"Enhanced large kernel blur completed (kernels: {kernels})")
        return estimated_background

    def _step3_correct_image(self, image: np.ndarray,
                           estimated_background: np.ndarray,
                           background_mask: Optional[np.ndarray] = None,
                           operator: Optional[BackgroundCorrectionOperator] = None) -> np.ndarray:
        """
        Step 3: Correct the Image by removing the modeled illumination.

        Operators:
        - SUBTRACT (default): corrected = image - (model - model_min).
          Physically correct for ADDITIVE stray light / reflections on a dark
          collection surface: removes the spatial VARIATION of the modeled
          illumination while preserving the base level. Because the subtracted
          amount is always >= 0, no pixel is ever brightened — new saturation
          is impossible by construction (validated: 0/30 sample images flagged
          vs 23/30 under division). Bounded error when the model is imperfect.
        - DIVIDE (legacy): corrected = image / model * mean(model).
          Correct for MULTIPLICATIVE shading (transmitted-light microscopy).
          Explodes where the model approaches zero — historically blew out
          the mat to pure white on dark-background samples.

        Correction-quality metrics (new saturation, local-variance retention)
        are computed as QA signals only: they are logged and stored in
        self.last_correction_quality for callers to report. No silent
        blending is applied — quantitative data is never mixed.
        """
        if operator is None:
            operator = BackgroundCorrectionOperator.SUBTRACT

        image_float = image.astype(np.float32)
        background_float = estimated_background.astype(np.float32)

        if operator == BackgroundCorrectionOperator.DIVIDE:
            # Avoid division by zero
            safe_background = np.maximum(background_float, 1.0)
            corrected = image_float / safe_background
            # Normalize for viewing (restore reasonable intensity range)
            mean_background = float(np.mean(estimated_background))
            final_float = corrected * mean_background
        else:
            # Additive stray-light model. A negative model value means the
            # polynomial extrapolated into a region with no background
            # samples — bounded harm under subtraction, but worth flagging.
            pre_clip_min = float(background_float.min())
            if pre_clip_min < 0:
                self.logger.warning(
                    f"Illumination model dips below zero (min={pre_clip_min:.1f}); "
                    f"likely sparse background sampling. Clipping model to [0, 255]."
                )
            model = np.clip(background_float, 0.0, 255.0)
            # Subtract only the VARIATION of the illumination (model - min),
            # preserving the base level. Subtracted amount is always >= 0,
            # so no pixel can brighten -> no new saturation, ever.
            model_min = float(model.min())
            final_float = image_float - (model - model_min)
            if background_mask is not None and np.any(background_mask):
                self.logger.debug(
                    f"Subtraction offset={model_min:.1f}; background lands near "
                    f"{float(np.mean(final_float[background_mask.astype(bool)])):.1f}"
                )

        # --- Correction quality QA (warn-only, never blended) ---
        max_new_saturation_pct = 2.0
        detail_loss_threshold = 0.05

        # Check 1: New saturation (pixels pushed to 255 that weren't before)
        original_saturated = int(np.sum(image_float >= 254))
        corrected_saturated = int(np.sum(final_float >= 254.5))
        new_saturated = max(0, corrected_saturated - original_saturated)
        total_pixels = image_float.size
        new_saturation_pct = 100.0 * new_saturated / total_pixels

        # Check 2: Local variance preservation
        window = 15
        orig_local_mean = cv2.boxFilter(image_float, -1, (window, window))
        orig_local_var = np.maximum(
            cv2.boxFilter(image_float ** 2, -1, (window, window)) - orig_local_mean ** 2, 0
        )
        corr_local_mean = cv2.boxFilter(final_float, -1, (window, window))
        corr_local_var = np.maximum(
            cv2.boxFilter(final_float ** 2, -1, (window, window)) - corr_local_mean ** 2, 0
        )

        valid_mask = orig_local_var > 1.0
        if np.any(valid_mask):
            orig_var_mean = float(np.mean(orig_local_var[valid_mask]))
            corr_var_mean = float(np.mean(corr_local_var[valid_mask]))
            variance_ratio = corr_var_mean / max(orig_var_mean, 1e-6)
        else:
            variance_ratio = 1.0

        detail_loss = max(0.0, 1.0 - variance_ratio)

        if new_saturation_pct > max_new_saturation_pct:
            self.logger.warning(
                f"Background correction ({operator.name.lower()}) introduced "
                f"{new_saturation_pct:.1f}% new saturation "
                f"(threshold: {max_new_saturation_pct:.1f}%). Inspect the result."
            )
        if detail_loss > detail_loss_threshold:
            self.logger.warning(
                f"Background correction ({operator.name.lower()}) reduced local "
                f"variance by {detail_loss*100:.1f}% "
                f"(threshold: {detail_loss_threshold*100:.1f}%). Inspect the result."
            )
        if new_saturation_pct <= max_new_saturation_pct and detail_loss <= detail_loss_threshold:
            self.logger.debug(
                f"Correction quality OK: new_sat={new_saturation_pct:.1f}%, "
                f"detail_loss={detail_loss*100:.1f}%"
            )

        self.last_correction_quality = {
            'operator': operator.name.lower(),
            'new_saturation_pct': new_saturation_pct,
            'detail_loss_pct': detail_loss * 100.0,
            'variance_ratio': variance_ratio,
        }

        # Clip to valid range
        final_image = np.clip(final_float, 0, 255)

        self.logger.debug(f"Image correction completed using {operator.name.lower()}")
        return final_image.astype(np.uint8)

    def _step4_analyze_mat_uniformity(self, corrected_image: np.ndarray,
                                    mat_mask: np.ndarray) -> Dict:
        """
        Step 4: Analyze Mat Uniformity
        
        Calculates the key metrics described in the guide.
        """
        if not np.any(mat_mask):
            self.logger.warning("No mat pixels found for uniformity analysis")
            return {
                'mean_intensity': 0,
                'std_deviation': 0,
                'coefficient_of_variation': float('inf'),
                'mat_pixels': 0
            }
        
        # Extract mat pixel intensities
        mat_intensities = corrected_image[mat_mask].astype(np.float64)
        
        # Calculate uniformity metrics
        mean_intensity = np.mean(mat_intensities)
        std_deviation = np.std(mat_intensities)
        coefficient_of_variation = std_deviation / mean_intensity if mean_intensity > 0 else float('inf')
        
        # Additional statistics
        min_intensity = np.min(mat_intensities)
        max_intensity = np.max(mat_intensities)
        intensity_range = max_intensity - min_intensity
        
        self.logger.info(f"Mat uniformity analysis:")
        self.logger.info(f"  Mean Intensity: {mean_intensity:.2f}")
        self.logger.info(f"  Std Deviation: {std_deviation:.2f}")
        self.logger.info(f"  Coefficient of Variation: {coefficient_of_variation:.4f}")
        self.logger.info(f"  Intensity Range: {min_intensity:.1f} - {max_intensity:.1f}")
        
        return {
            'mean_intensity': mean_intensity,
            'std_deviation': std_deviation,
            'coefficient_of_variation': coefficient_of_variation,
            'min_intensity': min_intensity,
            'max_intensity': max_intensity,
            'intensity_range': intensity_range,
            'mat_pixels': len(mat_intensities)
        }

    def _create_polynomial_features(self, x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
        """
        Create polynomial features matrix for 2D polynomial fitting.
        
        Args:
            x: X coordinates
            y: Y coordinates  
            degree: Polynomial degree
            
        Returns:
            Feature matrix for polynomial fitting
        """
        features = []
        
        # Add constant term
        features.append(np.ones_like(x))
        
        # Add polynomial terms up to specified degree
        for d in range(1, degree + 1):
            for i in range(d + 1):
                j = d - i
                features.append((x ** i) * (y ** j))
        
        return np.column_stack(features)

    # Backward compatibility methods (maintain existing interface)
    def create_background_mask(self, image: np.ndarray, mat_threshold: int = 180, 
                             hydrogel_threshold: int = 50) -> np.ndarray:
        """Backward compatibility wrapper for Step 1."""
        masks = self._step1_isolate_background_pixels(image, mat_threshold, hydrogel_threshold)
        return masks['background_mask']

    def polynomial_surface_fitting(self, image: np.ndarray, 
                                 background_mask: Optional[np.ndarray] = None,
                                 polynomial_order: int = 2) -> np.ndarray:
        """Backward compatibility: just return corrected image from complete workflow."""
        if background_mask is None:
            # Run complete analysis and return just the corrected image
            results = self.complete_uniformity_analysis(
                image, background_method="polynomial", polynomial_order=polynomial_order
            )
            return results['corrected_image']
        else:
            # Use provided mask
            estimated_background = self._step2a_polynomial_surface_fitting(
                image, background_mask, polynomial_order
            )
            return self._step3_correct_image(image, estimated_background,
                                             background_mask=background_mask)

    def large_kernel_blur(self, image: np.ndarray, 
                         background_mask: Optional[np.ndarray] = None,
                         kernel_size: Optional[int] = None) -> np.ndarray:
        """Backward compatibility: large kernel blur method."""
        if background_mask is None:
            # Run complete analysis and return just the corrected image
            results = self.complete_uniformity_analysis(
                image, background_method="large_kernel_blur"
            )
            return results['corrected_image']
        else:
            # Use provided mask
            estimated_background = self._step2b_large_kernel_blurring(image, background_mask)
            return self._step3_correct_image(image, estimated_background,
                                             background_mask=background_mask)

    def morphological_opening(self, image: np.ndarray, radius: int = 50) -> np.ndarray:
        """
        Simple morphological opening for backward compatibility.
        Note: This is not part of the 4-step guide but maintained for compatibility.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        image_float = image.astype(np.float32)
        background_float = background.astype(np.float32)
        background_float = np.maximum(background_float, 1.0)
        
        corrected = image_float / background_float
        mean_background = np.mean(background_float)
        corrected = corrected * mean_background
        corrected = np.clip(corrected, 0, 255)
        
        return corrected.astype(np.uint8)

    def homomorphic_filter(self, image: np.ndarray, cutoff_frequency: float = 0.1,
                          gamma_low: float = 0.5, gamma_high: float = 2.0) -> np.ndarray:
        """
        Homomorphic filtering for backward compatibility.
        Note: This is not part of the 4-step guide but maintained for compatibility.
        """
        image_float = image.astype(np.float32) + 1.0
        log_image = np.log(image_float)
        
        f_transform = np.fft.fft2(log_image)
        f_shift = np.fft.fftshift(f_transform)
        
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        U, V = np.meshgrid(v, u)
        
        D = np.sqrt(U**2 + V**2)
        D0 = cutoff_frequency * min(rows, cols) / 2
        
        n = 2
        H = 1 / (1 + (D0 / (D + 1e-6))**(2*n))
        H = gamma_low + (gamma_high - gamma_low) * H
        
        filtered = f_shift * H
        f_ishift = np.fft.ifftshift(filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)
        
        corrected = np.exp(img_back) - 1.0
        corrected = np.clip(corrected, 0, 255)
        
        return corrected.astype(np.uint8)

    def apply_background_correction(self, image: np.ndarray, 
                                  method: str = "complete_workflow",
                                  **kwargs) -> np.ndarray:
        """
        Apply background correction method.
        
        Args:
            image: Input grayscale image
            method: Method to apply
            **kwargs: Additional parameters
            
        Returns:
            Background-corrected image or complete results
        """
        if method == "complete_workflow":
            # Return complete results dictionary
            return self.complete_uniformity_analysis(image, **kwargs)
        elif method == "polynomial_surface":
            return self.polynomial_surface_fitting(image, **kwargs)
        elif method == "large_kernel_blur":
            return self.large_kernel_blur(image, **kwargs)
        elif method == "morphological_opening":
            return self.morphological_opening(image, **kwargs)
        elif method == "homomorphic":
            return self.homomorphic_filter(image, **kwargs)
        else:
            self.logger.warning(f"Unknown method '{method}', using complete workflow")
            return self.complete_uniformity_analysis(image, **kwargs)

    def large_kernel_blur(self, image: np.ndarray, 
                         background_mask: Optional[np.ndarray] = None,
                         kernel_size: Optional[int] = None) -> np.ndarray:
        """
        Method B: Large-Kernel Blurring (Simpler Alternative)
        
        Uses Gaussian blur to interpolate background illumination.
        
        Args:
            image: Input grayscale image
            background_mask: Boolean mask indicating background pixels
            kernel_size: Size of Gaussian kernel (auto-calculated if None)
            
        Returns:
            Background-corrected image
        """
        if background_mask is None:
            background_mask = self.create_background_mask(image)
        
        # Auto-calculate kernel size if not provided
        if kernel_size is None:
            # Kernel should be significantly larger than mat diameter
            h, w = image.shape
            kernel_size = max(h, w) // 4
            # Ensure odd number
            if kernel_size % 2 == 0:
                kernel_size += 1
        
        # Create background-only image
        background_only = np.zeros_like(image, dtype=np.float32)
        background_only[background_mask] = image[background_mask].astype(np.float32)
        
        # Apply large Gaussian blur
        blurred_background = cv2.GaussianBlur(background_only, (kernel_size, kernel_size), 0)
        
        # Create mask for blurred regions
        blur_mask = cv2.GaussianBlur(background_mask.astype(np.float32), (kernel_size, kernel_size), 0)
        
        # Normalize blurred background
        blurred_background = np.divide(blurred_background, blur_mask, 
                                     out=np.zeros_like(blurred_background), 
                                     where=blur_mask > 0.1)
        
        # Apply correction using division
        image_float = image.astype(np.float32)
        background_float = np.maximum(blurred_background, 1.0)
        
        corrected = image_float / background_float
        
        # Normalize to maintain reasonable intensity range
        mean_background = np.mean(blurred_background[background_mask])
        corrected = corrected * mean_background
        
        # Clip to valid range
        corrected = np.clip(corrected, 0, 255)
        
        self.logger.info(f"Large kernel blur completed (kernel size {kernel_size})")
        return corrected.astype(np.uint8)

    def morphological_opening(self, image: np.ndarray, radius: int = 50) -> np.ndarray:
        """
        Morphological background estimation using "rolling ball" algorithm.
        
        Args:
            image: Input grayscale image
            radius: Radius of morphological structuring element
            
        Returns:
            Background-corrected image
        """
        # Create circular structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
        
        # Perform morphological opening to estimate background
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Apply correction using division
        image_float = image.astype(np.float32)
        background_float = background.astype(np.float32)
        
        # Avoid division by zero
        background_float = np.maximum(background_float, 1.0)
        
        corrected = image_float / background_float
        
        # Normalize to maintain reasonable intensity range
        mean_background = np.mean(background_float)
        corrected = corrected * mean_background
        
        # Clip to valid range
        corrected = np.clip(corrected, 0, 255)
        
        self.logger.info(f"Morphological opening completed (radius {radius})")
        return corrected.astype(np.uint8)

    def homomorphic_filter(self, image: np.ndarray, cutoff_frequency: float = 0.1,
                          gamma_low: float = 0.5, gamma_high: float = 2.0) -> np.ndarray:
        """
        Homomorphic filtering for illumination-reflectance separation.
        
        Based on the frequency-domain method described in the guide.
        
        Args:
            image: Input grayscale image
            cutoff_frequency: Cutoff frequency for high-pass filter (0.0-1.0)
            gamma_low: Gain for low frequencies (illumination)
            gamma_high: Gain for high frequencies (reflectance)
            
        Returns:
            Background-corrected image
        """
        # Convert to float and add small constant to avoid log(0)
        image_float = image.astype(np.float32) + 1.0
        
        # Apply logarithmic transform
        log_image = np.log(image_float)
        
        # Apply 2D FFT
        f_transform = np.fft.fft2(log_image)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create high-pass filter
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create coordinate grids
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        U, V = np.meshgrid(v, u)
        
        # Calculate distance from center
        D = np.sqrt(U**2 + V**2)
        D0 = cutoff_frequency * min(rows, cols) / 2
        
        # Butterworth high-pass filter
        n = 2  # Filter order
        H = 1 / (1 + (D0 / (D + 1e-6))**(2*n))
        
        # Apply frequency-dependent gains
        H = gamma_low + (gamma_high - gamma_low) * H
        
        # Apply filter
        filtered = f_shift * H
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)
        
        # Apply exponential transform
        corrected = np.exp(img_back) - 1.0
        
        # Normalize and clip
        corrected = np.clip(corrected, 0, 255)
        
        self.logger.info(f"Homomorphic filtering completed (cutoff {cutoff_frequency})")
        return corrected.astype(np.uint8)
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

    def apply_background_correction(self, image: np.ndarray, 
                                  method: str = "polynomial_surface",
                                  **kwargs) -> np.ndarray:
        """
        Apply the specified background correction method.
        
        Args:
            image: Input grayscale image
            method: Background correction method to apply
            **kwargs: Additional parameters for specific methods
            
        """
        if method == "complete_workflow":
            # Return complete results dictionary
            return self.complete_uniformity_analysis(image, **kwargs)
        elif method == "polynomial_surface":
            return self.polynomial_surface_fitting(image, **kwargs)
        elif method == "large_kernel_blur":
            return self.large_kernel_blur(image, **kwargs)
        elif method == "morphological_opening":
            return self.morphological_opening(image, **kwargs)
        elif method == "homomorphic":
            return self.homomorphic_filter(image, **kwargs)
        else:
            self.logger.warning(f"Unknown method '{method}', using complete workflow")
            return self.complete_uniformity_analysis(image, **kwargs)

    def complete_uniformity_analysis_with_exclusion(self, image: np.ndarray,
                                                   mat_threshold: int = 180,
                                                   hydrogel_threshold: int = 50,
                                                   background_method: str = "polynomial",
                                                   polynomial_order: int = 2,
                                                   exclusion_mask: np.ndarray = None,
                                                   operator: Optional[BackgroundCorrectionOperator] = None) -> Dict:
        """
        Complete 4-step uniformity analysis with exclusion mask for rulers/artifacts.
        
        This version excludes specified regions (e.g., rulers) from background modeling
        while still processing the full image for optimal background correction.
        
        Args:
            image: Input grayscale image
            mat_threshold: High-pass threshold for mat pixels  
            hydrogel_threshold: Low-pass threshold for hydrogel pixels
            background_method: "polynomial" or "large_kernel_blur"
            polynomial_order: Order for polynomial fitting (2 or 3)
            exclusion_mask: Boolean mask where True indicates pixels to exclude
                          from background modeling (e.g., rulers)
            
        Returns:
            Dictionary containing all results and intermediate steps
        """
        results = {}
        
        # STEP 1: Isolate the Background Pixels (with exclusions)
        self.logger.info("Step 1: Isolating background pixels with exclusions...")
        masks = self._step1_isolate_background_pixels_with_exclusion(
            image, mat_threshold, hydrogel_threshold, exclusion_mask
        )
        results.update(masks)
        
        # STEP 2: Model the Uneven Illumination (excluding rulers)
        self.logger.info(f"Step 2: Modeling illumination using {background_method} (excluding artifacts)...")
        if background_method == "polynomial":
            estimated_background = self._step2a_polynomial_surface_fitting(
                image, masks['background_mask'], polynomial_order
            )
        else:  # large_kernel_blur
            estimated_background = self._step2b_large_kernel_blurring(
                image, masks['background_mask']
            )
        results['estimated_background'] = estimated_background
        
        # STEP 3: Correct the Image
        self.logger.info("Step 3: Correcting image...")
        corrected_image = self._step3_correct_image(
            image, estimated_background,
            background_mask=masks['background_mask'], operator=operator
        )
        results['corrected_image'] = corrected_image
        results['correction_quality'] = dict(self.last_correction_quality)

        # STEP 4: Analyze Mat Uniformity (this will be done later on the ROI)
        # For full-image correction, we don't analyze uniformity here
        # That will be done on the gel-specific ROI

        return results

    def _step1_isolate_background_pixels_with_exclusion(self, image: np.ndarray, 
                                                       mat_threshold: int = 180,
                                                       hydrogel_threshold: int = 50,
                                                       exclusion_mask: np.ndarray = None) -> Dict:
        """
        Step 1: Create digital masks to isolate background pixels, excluding specified regions.
        
        Args:
            image: Input grayscale image
            mat_threshold: High-pass threshold for mat pixels
            hydrogel_threshold: Low-pass threshold for hydrogel pixels  
            exclusion_mask: Boolean mask where True indicates pixels to exclude
        
        Returns:
            Dictionary with various masks
        """
        h, w = image.shape
        
        # 2a. Mask the Mat (brightest object)
        mat_mask = image > mat_threshold
        
        # 2b. Mask the Hydrogel (darkest object) 
        hydrogel_mask = image < hydrogel_threshold
        
        # 2c. Combine to create foreground mask
        foreground_mask = mat_mask | hydrogel_mask
        
        # 2d. Add exclusion mask (rulers, artifacts) with shadow expansion
        if exclusion_mask is not None:
            # Expand exclusion mask to account for shadows and nearby artifacts
            expanded_exclusion = self._expand_exclusion_mask(exclusion_mask, image.shape)
            foreground_mask = foreground_mask | expanded_exclusion
            self.logger.debug(f"Added {np.sum(exclusion_mask)} excluded pixels + {np.sum(expanded_exclusion) - np.sum(exclusion_mask)} shadow pixels")
        else:
            expanded_exclusion = None
        
        # 3. Invert to get final background mask
        background_mask = ~foreground_mask
        
        # Log statistics
        total_pixels = h * w
        bg_pixels = np.sum(background_mask)
        mat_pixels = np.sum(mat_mask)
        hydrogel_pixels = np.sum(hydrogel_mask)
        excluded_pixels = np.sum(exclusion_mask) if exclusion_mask is not None else 0
        
        self.logger.debug(f"Mask statistics (with exclusions):")
        self.logger.debug(f"  Background: {bg_pixels} pixels ({100*bg_pixels/total_pixels:.1f}%)")
        self.logger.debug(f"  Mat: {mat_pixels} pixels ({100*mat_pixels/total_pixels:.1f}%)")
        self.logger.debug(f"  Hydrogel: {hydrogel_pixels} pixels ({100*hydrogel_pixels/total_pixels:.1f}%)")
        self.logger.debug(f"  Excluded: {excluded_pixels} pixels ({100*excluded_pixels/total_pixels:.1f}%)")
        
        return {
            'mat_mask': mat_mask,
            'hydrogel_mask': hydrogel_mask, 
            'foreground_mask': foreground_mask,
            'background_mask': background_mask,
            'exclusion_mask': expanded_exclusion if exclusion_mask is not None else None
        }

    def _expand_exclusion_mask(self, exclusion_mask: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Expand exclusion mask to account for shadows and artifacts around excluded regions.
        
        Args:
            exclusion_mask: Original exclusion mask (e.g., ruler locations)
            image_shape: Shape of the image (height, width)
            
        Returns:
            Expanded exclusion mask that includes potential shadow areas
        """
        if exclusion_mask is None or not np.any(exclusion_mask):
            return exclusion_mask
        
        # Start with original mask
        expanded = exclusion_mask.copy()
        
        # Find bounding boxes of excluded regions
        num_regions, labeled_regions = cv2.connectedComponents(exclusion_mask.astype(np.uint8))
        
        for region_id in range(1, num_regions + 1):
            region_mask = labeled_regions == region_id
            
            # Get bounding box of this region
            coords = np.where(region_mask)
            if len(coords[0]) == 0:
                continue
                
            min_y, max_y = np.min(coords[0]), np.max(coords[0])
            min_x, max_x = np.min(coords[1]), np.max(coords[1])
            
            region_height = max_y - min_y
            region_width = max_x - min_x
            
            # Determine expansion parameters based on region characteristics
            # Rulers typically cast shadows below and slightly to the sides
            if region_height < region_width:  # Likely horizontal ruler/object
                # Expand downward (shadow direction) and slightly sideways
                shadow_height = min(region_height * 2, 80)  # Max 80 pixels shadow
                side_expansion = min(region_width // 4, 20)  # Max 20 pixels on sides
                
                # Expand downward
                shadow_bottom = min(max_y + shadow_height, image_shape[0])
                expanded[max_y:shadow_bottom, 
                        max(0, min_x - side_expansion):min(max_x + side_expansion, image_shape[1])] = True
                        
                # Expand sideways slightly
                expanded[min_y:max_y,
                        max(0, min_x - side_expansion):min(max_x + side_expansion, image_shape[1])] = True
                        
            else:  # Vertical object or square region
                # Expand in all directions more conservatively
                expansion = min(max(region_height, region_width) // 3, 30)
                
                expanded[max(0, min_y - expansion):min(max_y + expansion, image_shape[0]),
                        max(0, min_x - expansion):min(max_x + expansion, image_shape[1])] = True
        
        return expanded
    def complete_uniformity_analysis_with_four_regions(
        self,
        image: np.ndarray,
        background_method: str = "polynomial",
        polynomial_order: int = 2,
        exclusion_mask: np.ndarray = None,
        operator: Optional[BackgroundCorrectionOperator] = None
    ) -> Dict:
        """
        Complete 4-step uniformity analysis using FourRegionDetector.
        
        This version uses intelligent region detection to ensure:
        1. Background pixels come from TRUE background (outside gel ring)
        2. Gel region (darker than background) is excluded from background modeling
        3. Rulers are properly detected and excluded
        
        This fixes the issue where gel (wet polymer, darker than background)
        was incorrectly used as background reference, leading to incorrect
        thickness calibration.
        
        Args:
            image: Input grayscale image
            background_method: "polynomial" or "large_kernel_blur"
            polynomial_order: Order for polynomial fitting (2 or 3)
            exclusion_mask: Additional exclusion mask (e.g., user-specified)
            
        Returns:
            Dictionary containing all results and intermediate steps
        """
        from esnf_mat_analyzer.processing.region_detector import FourRegionDetector
        
        results = {}
        
        # STEP 1: Intelligent four-region detection
        self.logger.info("Step 1: Detecting four regions (background, gel, mat, ruler)...")
        detector = FourRegionDetector()
        
        # Pass any user exclusion mask as ruler_mask (areas to exclude from analysis)
        region_result = detector.detect(image, ruler_mask=exclusion_mask)
        
        # Store region masks
        results['mat_mask'] = region_result.mat_mask.astype(bool)
        results['gel_mask'] = region_result.gel_mask.astype(bool)
        results['background_mask'] = region_result.background_mask.astype(bool)  # TRUE background
        results['ruler_mask'] = region_result.ruler_mask.astype(bool)
        results['gel_detected'] = region_result.gel_detected
        results['ruler_detected'] = region_result.ruler_detected
        results['region_quality_score'] = region_result.quality_score
        
        # Log region statistics
        h, w = image.shape
        total_pixels = h * w
        self.logger.info(f"Region detection (quality={region_result.quality_score:.2f}):")
        self.logger.info(f"  TRUE Background: {100*np.sum(results['background_mask'])/total_pixels:.1f}%")
        self.logger.info(f"  Gel: {100*np.sum(results['gel_mask'])/total_pixels:.1f}% "
                        f"({'detected' if region_result.gel_detected else 'not detected'})")
        self.logger.info(f"  Mat: {100*np.sum(results['mat_mask'])/total_pixels:.1f}%")
        self.logger.info(f"  Ruler: {100*np.sum(results['ruler_mask'])/total_pixels:.1f}% "
                        f"({'detected' if region_result.ruler_detected else 'not detected'})")
        
        # Verify intensity ordering
        bg_intensity = region_result.background_intensity
        gel_intensity = region_result.gel_intensity
        results['background_reference_intensity'] = bg_intensity
        results['gel_intensity'] = gel_intensity
        
        if region_result.gel_detected and gel_intensity >= bg_intensity:
            self.logger.warning(
                f"Intensity ordering issue: gel ({gel_intensity:.1f}) should be "
                f"darker than background ({bg_intensity:.1f})"
            )
        
        # STEP 2: Model illumination using ONLY true background pixels
        self.logger.info(f"Step 2: Modeling illumination from TRUE background using {background_method}...")
        if background_method == "polynomial":
            estimated_background = self._step2a_polynomial_surface_fitting(
                image, results['background_mask'], polynomial_order
            )
        else:  # large_kernel_blur
            estimated_background = self._step2b_large_kernel_blurring(
                image, results['background_mask']
            )
        results['estimated_background'] = estimated_background
        
        # STEP 3: Correct the Image
        self.logger.info("Step 3: Correcting image...")
        corrected_image = self._step3_correct_image(
            image, estimated_background,
            background_mask=results['background_mask'], operator=operator
        )
        results['corrected_image'] = corrected_image
        results['correction_quality'] = dict(self.last_correction_quality)

        # STEP 4: Analyze Mat Uniformity
        self.logger.info("Step 4: Analyzing mat uniformity...")
        uniformity_metrics = self._step4_analyze_mat_uniformity(
            corrected_image, results['mat_mask']
        )
        results.update(uniformity_metrics)
        
        # Store calibration references
        results['calibration'] = {
            'black_reference': bg_intensity,  # True background intensity
            'white_reference': region_result.mat_intensity_range[1],  # Max mat intensity
            'gel_intensity': gel_intensity,
            'saturation_detected': region_result.saturation_pct > 0.5,
            'saturation_pct': region_result.saturation_pct,
        }
        
        return results