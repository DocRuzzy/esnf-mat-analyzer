import numpy as np
import cv2
from typing import Optional, Tuple, Dict
import logging
from scipy import ndimage
from scipy.signal import medfilt2d
from skimage import morphology, filters
import warnings

class MatLevelBackgroundProcessor:
    """
    Background correction methods specifically designed for ESNF mat analysis
    where individual fibers are NOT visible - mat appears as continuous regions.
    
    Key Focus:
    - Preserve continuous mat regions
    - Maintain thickness gradients within mat
    - Remove non-uniform background illumination
    - Handle saturation in thick regions
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        warnings.filterwarnings('ignore')
    
    def adaptive_polynomial_surface_fitting(self, 
                                          image: np.ndarray,
                                          roi_mask: Optional[np.ndarray] = None,
                                          polynomial_order: int = 3,
                                          sample_density: float = 0.05) -> np.ndarray:
        """
        Fits a polynomial surface to the background regions and subtracts it.
        Excellent for smooth, gradual illumination variations.
        
        This method:
        1. Samples background points (dark regions outside mat)
        2. Fits a polynomial surface to these points
        3. Subtracts the fitted surface from the image
        
        Args:
            image: Input grayscale image
            roi_mask: Optional mask of mat region (True where mat is)
            polynomial_order: Order of polynomial (2-4 recommended)
            sample_density: Fraction of background points to use (0.01-0.1)
            
        Returns:
            Background-corrected image
        """
        h, w = image.shape
        img_float = image.astype(np.float32)
        
        # Create coordinate grids
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        
        # Identify background regions
        if roi_mask is not None:
            # Use provided mask
            background_mask = ~roi_mask
        else:
            # Auto-detect using Otsu's threshold
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Background is dark regions
            background_mask = binary == 0
            
            # Erode to ensure we're sampling true background
            kernel = np.ones((5, 5), np.uint8)
            background_mask = cv2.erode(background_mask.astype(np.uint8), kernel, iterations=2)
            background_mask = background_mask.astype(bool)
        
        # Sample background points
        bg_points = np.where(background_mask)
        n_samples = int(len(bg_points[0]) * sample_density)
        n_samples = max(n_samples, 100)  # Ensure minimum samples
        
        if len(bg_points[0]) > n_samples:
            # Random sampling
            indices = np.random.choice(len(bg_points[0]), n_samples, replace=False)
            sample_y = bg_points[0][indices]
            sample_x = bg_points[1][indices]
        else:
            sample_y = bg_points[0]
            sample_x = bg_points[1]
        
        # Get intensity values at sample points
        sample_z = img_float[sample_y, sample_x]
        
        # Create polynomial features
        features = []
        for i in range(polynomial_order + 1):
            for j in range(polynomial_order + 1 - i):
                features.append((sample_x ** i) * (sample_y ** j))
        
        # Stack features
        A = np.column_stack(features)
        
        # Solve least squares
        coeffs, _, _, _ = np.linalg.lstsq(A, sample_z, rcond=None)
        
        # Evaluate polynomial on full grid
        features_full = []
        for i in range(polynomial_order + 1):
            for j in range(polynomial_order + 1 - i):
                features_full.append((xx ** i) * (yy ** j))
        
        # Calculate background surface
        background_surface = np.zeros_like(img_float)
        for k, feature in enumerate(features_full):
            background_surface += coeffs[k] * feature
        
        # Subtract background
        corrected = img_float - background_surface + np.mean(sample_z)
        
        # Clip to valid range
        corrected = np.clip(corrected, 0, 255)
        
        self.logger.debug(f"Polynomial surface fitting applied: order={polynomial_order}")
        
        return corrected.astype(np.uint8)
    
    def region_based_leveling(self,
                            image: np.ndarray,
                            mat_mask: Optional[np.ndarray] = None,
                            reference_percentile: int = 10) -> np.ndarray:
        """
        Level the image based on reference regions outside the mat.
        Preserves relative intensities within the mat while correcting background.
        
        Args:
            image: Input grayscale image
            mat_mask: Mask indicating mat regions (True where mat is)
            reference_percentile: Percentile of background to use as reference
            
        Returns:
            Leveled image
        """
        img_float = image.astype(np.float32)
        
        # Auto-detect mat if mask not provided
        if mat_mask is None:
            # Use adaptive thresholding to find mat regions
            mat_mask = self._detect_mat_regions(image)
        
        # Divide image into grid cells
        grid_size = 32  # Size of each grid cell
        h, w = image.shape
        
        # Initialize correction map
        correction_map = np.zeros_like(img_float)
        
        # Process each grid cell
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                # Define cell boundaries
                y_end = min(y + grid_size, h)
                x_end = min(x + grid_size, w)
                
                # Extract cell
                cell = img_float[y:y_end, x:x_end]
                cell_mask = mat_mask[y:y_end, x:x_end]

                # Calculate background reference in this cell
                if np.any(~cell_mask):  # If there's background in this cell
                    bg_values = cell[~cell_mask]
                    reference_value = np.percentile(bg_values, reference_percentile)
                else:
                    # No background in cell, use neighboring cells
                    reference_value = self._get_neighbor_reference(
                        img_float, mat_mask, x, y, grid_size, reference_percentile)

                # Store reference value
                correction_map[y:y_end, x:x_end] = reference_value

        # Smooth the correction map
        correction_map = cv2.GaussianBlur(correction_map, (31, 31), 0)

        # Calculate global reference
        global_reference = np.percentile(img_float[~mat_mask], reference_percentile) \
                          if np.any(~mat_mask) else np.mean(img_float)

        # Apply correction
        corrected = img_float - correction_map + global_reference

        # Preserve mat brightness relationships
        if np.any(mat_mask):
            # Scale mat values to maintain contrast
            mat_values_original = img_float[mat_mask]
            mat_values_corrected = corrected[mat_mask]

            if len(mat_values_original) > 0 and np.std(mat_values_original) > 0:
                # Maintain contrast ratio
                scale = np.std(mat_values_original) / (np.std(mat_values_corrected) + 1e-6)
                offset = np.mean(mat_values_original) - scale * np.mean(mat_values_corrected)
                corrected[mat_mask] = scale * corrected[mat_mask] + offset

        # Clip to valid range
        corrected = np.clip(corrected, 0, 255)

        return corrected.astype(np.uint8)

    def selective_illumination_correction(self,
                                        image: np.ndarray,
                                        preserve_threshold: float = 0.7,
                                        blur_size: int = 101) -> np.ndarray:
        """
        Corrects illumination while selectively preserving bright regions (mat).
        Uses confidence mapping to determine where to apply correction.
        
        Args:
            image: Input grayscale image
            preserve_threshold: Threshold for mat preservation (0-1)
            blur_size: Size of blur kernel for illumination estimation
            
        Returns:
            Selectively corrected image
        """
        img_float = image.astype(np.float32)
        
        # Estimate illumination using large Gaussian blur
        if blur_size % 2 == 0:
            blur_size += 1
        illumination = cv2.GaussianBlur(img_float, (blur_size, blur_size), 0)
        
        # Create mat confidence map based on local brightness and variance
        local_mean = cv2.GaussianBlur(img_float, (21, 21), 0)
        local_var = cv2.GaussianBlur(np.square(img_float - local_mean), (21, 21), 0)
        
        # Normalize to create confidence map
        brightness_score = local_mean / 255.0
        variance_score = np.sqrt(local_var) / 128.0
        
        # Combined confidence (high for mat regions)
        mat_confidence = np.clip(brightness_score * variance_score, 0, 1)
        mat_confidence = np.power(mat_confidence, 0.5)  # Adjust sensitivity
        
        # Create preservation mask
        preserve_mask = mat_confidence > preserve_threshold
        
        # Smooth preservation boundaries
        preserve_weight = cv2.GaussianBlur(
            preserve_mask.astype(np.float32), (31, 31), 0)
        
        # Calculate correction
        mean_illumination = np.mean(illumination)
        correction = mean_illumination - illumination
        
        # Apply correction with preservation
        corrected = img_float + correction * (1 - preserve_weight)
        
        # Clip to valid range
        corrected = np.clip(corrected, 0, 255)
        
        return corrected.astype(np.uint8)
    
    def enhanced_percentile_with_gradient_preservation(self,
                                                     image: np.ndarray,
                                                     background_percentile: float = 5.0,
                                                     gradient_weight: float = 0.5) -> np.ndarray:
        """
        Enhanced percentile correction that preserves intensity gradients within mat.

        Args:
            image: Input grayscale image
            background_percentile: Percentile for background estimation
            gradient_weight: Weight for gradient preservation (0-1)

        Returns:
            Corrected image with preserved gradients
        """
        img_float = image.astype(np.float32)

        # Estimate background value
        background_value = np.percentile(img_float, background_percentile)

        # Simple correction
        simple_corrected = img_float - background_value

        # Calculate gradients in original image
        grad_x = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)

        # Reconstruct from gradients (Poisson reconstruction simplified)
        # This preserves relative intensities
        grad_corrected = self._simple_poisson_reconstruction(
            grad_x, grad_y, boundary=simple_corrected)

        # Blend based on gradient weight
        corrected = (1 - gradient_weight) * simple_corrected + gradient_weight * grad_corrected

        # Add offset to maintain positive values
        corrected = corrected + np.abs(np.min(corrected)) + 10

        # Normalize to use full range
        corrected = 255 * (corrected - np.min(corrected)) / (np.max(corrected) - np.min(corrected))

        return corrected.astype(np.uint8)

    def two_stage_correction(self,
                           image: np.ndarray,
                           roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Two-stage correction optimized for mat preservation:
        Stage 1: Global illumination correction
        Stage 2: Local refinement with mat preservation
        
        Args:
            image: Input grayscale image
            roi_mask: Optional ROI mask
            
        Returns:
            Two-stage corrected image
        """
        # Stage 1: Global polynomial surface correction
        stage1 = self.adaptive_polynomial_surface_fitting(
            image, roi_mask, polynomial_order=2)
        
        # Stage 2: Local refinement with mat preservation
        stage2 = self.selective_illumination_correction(
            stage1, preserve_threshold=0.6, blur_size=51)
        
        return stage2
    
    def _detect_mat_regions(self, image: np.ndarray) -> np.ndarray:
        """Automatically detect mat regions using adaptive thresholding."""
        # Use adaptive threshold to handle non-uniform illumination
        adaptive_thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, -5)
        
        # Clean up with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned > 0

    def _get_neighbor_reference(self, image, mask, x, y, grid_size, percentile):
        """Get reference value from neighboring cells."""
        h, w = image.shape
        references = []

        # Check 8 neighboring cells
        for dy in [-grid_size, 0, grid_size]:
            for dx in [-grid_size, 0, grid_size]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    nx_end = min(nx + grid_size, w)
                    ny_end = min(ny + grid_size, h)

                    cell = image[ny:ny_end, nx:nx_end]
                    cell_mask = mask[ny:ny_end, nx:nx_end]

                    if np.any(~cell_mask):
                        bg_values = cell[~cell_mask]
                        references.append(np.percentile(bg_values, percentile))

        return np.median(references) if references else np.mean(image)

    def _simple_poisson_reconstruction(self, grad_x, grad_y, boundary):
        """Simplified Poisson reconstruction from gradients."""
        h, w = grad_x.shape

        # Use iterative approach (simplified)
        result = boundary.copy()

        for _ in range(10):  # Fixed iterations for simplicity
            # Update interior points based on gradients
            result[1:-1, 1:-1] = 0.25 * (
                result[:-2, 1:-1] + result[2:, 1:-1] +
                result[1:-1, :-2] + result[1:-1, 2:] +
                grad_x[1:-1, 1:-1] - grad_x[1:-1, :-2] +
                grad_y[1:-1, 1:-1] - grad_y[:-2, 1:-1]
            )

        return result

    def compare_methods_for_mat_preservation(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compare all methods specifically for mat preservation quality.

        Returns:
            Dictionary of method names to corrected images
        """
        results = {
            'Original': image,
            'Polynomial Surface (Order 2)': self.adaptive_polynomial_surface_fitting(image, polynomial_order=2),
            'Polynomial Surface (Order 3)': self.adaptive_polynomial_surface_fitting(image, polynomial_order=3),
            'Region-Based Leveling': self.region_based_leveling(image),
            'Selective Illumination': self.selective_illumination_correction(image),
            'Enhanced Percentile': self.enhanced_percentile_with_gradient_preservation(image),
            'Two-Stage Correction': self.two_stage_correction(image)
        }

        return results
