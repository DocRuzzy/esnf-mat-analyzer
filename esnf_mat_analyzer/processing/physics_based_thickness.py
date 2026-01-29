"""
Physics-based thickness estimation using Beer-Lambert law and other optical models.
Specifically designed for electrospun fiber mat analysis from single images.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BeerLambertEstimator:
    """
    Beer-Lambert law implementation for fiber mat thickness estimation.
    
    The Beer-Lambert law: I = I₀ × e^(-α×t)
    Rearranged for thickness: t = -ln(I/I₀) / α
    
    Where:
    - I = transmitted intensity (observed pixel value)
    - I₀ = incident intensity (background reference)
    - α = attenuation coefficient (material-dependent)
    - t = optical thickness (fiber density × physical thickness)
    """
    
    def __init__(self, 
                 attenuation_coefficient: float = 0.04778,
                 reference_intensity: Optional[float] = None,
                 min_transmittance: float = 0.001,
                 max_thickness_um: float = 1000.0):
        """
        Initialize Beer-Lambert estimator.
        
        Args:
            attenuation_coefficient: Material attenuation coefficient (1/μm)
                Default 0.04778 achieves 18.84% average relative error per literature
            reference_intensity: Background intensity (I₀). If None, auto-detected
            min_transmittance: Minimum transmittance to prevent log(0) errors
            max_thickness_um: Maximum reasonable thickness in micrometers
        """
        self.alpha = attenuation_coefficient
        self.I0 = reference_intensity
        self.min_transmittance = min_transmittance
        self.max_thickness = max_thickness_um
        
    def estimate_thickness(self, 
                          image: np.ndarray,
                          background_corrected: bool = True,
                          spatial_scale_um_per_pixel: Optional[float] = None,
                          mask: Optional[np.ndarray] = None,
                          invert_intensity: bool = True) -> np.ndarray:
        """
        Estimate thickness using Beer-Lambert law.
        
        Args:
            image: Input grayscale image (0-255)
            background_corrected: Whether image is already background corrected
            spatial_scale_um_per_pixel: Spatial scale for absolute thickness
            mask: Optional ROI mask to focus on sample region
            invert_intensity: If True, higher intensity = thicker (for scattering/reflectance images)
                            If False, higher intensity = thinner (standard transmission)
            
        Returns:
            Thickness map in micrometers (if scale provided) or relative units
        """
        # Convert to float and normalize to [0,1]
        image_float = image.astype(np.float64) / 255.0
        
        # Log original intensity statistics
        if mask is not None:
            valid_orig = image_float[mask > 0]
            logger.info(f"Original intensity - Min: {np.min(valid_orig):.4f}, Max: {np.max(valid_orig):.4f}, Mean: {np.mean(valid_orig):.4f}, Std: {np.std(valid_orig):.4f}")
        
        # Invert intensity if needed (for reflectance/scattering-based images)
        # where bright regions = thick material
        if invert_intensity:
            logger.info("Inverting intensity: bright = thick, dark = thin")
            image_float = 1.0 - image_float
            
            # Log inverted intensity statistics
            if mask is not None:
                valid_inv = image_float[mask > 0]
                logger.info(f"After inversion - Min: {np.min(valid_inv):.4f}, Max: {np.max(valid_inv):.4f}, Mean: {np.mean(valid_inv):.4f}, Std: {np.std(valid_inv):.4f}")
        
        # Auto-detect background intensity if not provided
        if self.I0 is None:
            if background_corrected:
                # For inverted images: after inversion, MINIMUM = thinnest (highest original intensity)
                # We want I0 to represent the reference "no fiber" intensity
                if mask is not None:
                    # Focus on the actual sample region
                    valid_pixels = image_float[mask > 0]
                    if len(valid_pixels) > 0:
                        # After inversion: low values = thin, high values = thick
                        # Use low percentile as reference (thinnest regions)
                        min_intensity = np.min(valid_pixels)
                        p1 = np.percentile(valid_pixels, 1)
                        p5 = np.percentile(valid_pixels, 5)
                        
                        # Use a low percentile to represent thinnest regions
                        self.I0 = np.percentile(valid_pixels, 2)
                        
                        logger.info(f"Intensity distribution: Min={min_intensity:.4f}, P1={p1:.4f}, P5={p5:.4f}")
                        logger.info(f"Reference I0 (2nd percentile): {self.I0:.4f}")
                    else:
                        self.I0 = np.percentile(image_float, 5)
                else:
                    # No mask - use global low percentile
                    self.I0 = np.percentile(image_float, 5)
                    
                # Ensure I0 is reasonable
                self.I0 = np.clip(self.I0, 0.001, 0.9)
                logger.info(f"Final I0: {self.I0:.4f}")
            else:
                # For raw images, use background regions
                self.I0 = self._estimate_background_intensity(image_float)
        
        # Calculate transmittance T = I/I₀
        transmittance = image_float / self.I0
        
        # Log transmittance statistics before clamping
        if mask is not None:
            valid_trans = transmittance[mask > 0]
            if len(valid_trans) > 0:
                logger.info(f"Transmittance before clamp - Min: {np.min(valid_trans):.4f}, Max: {np.max(valid_trans):.4f}, Mean: {np.mean(valid_trans):.4f}")
        
        # Clamp transmittance to prevent numerical issues
        # Lower bound prevents log(0), upper bound prevents negative thickness
        transmittance = np.clip(transmittance, self.min_transmittance, 1.0)
        
        # Apply Beer-Lambert law: t = -ln(T) / α
        # Lower transmittance (darker) = higher thickness
        optical_thickness = -np.log(transmittance) / self.alpha
        
        # Convert to physical thickness if spatial scale provided
        if spatial_scale_um_per_pixel:
            # Optical thickness is related to physical thickness by fiber density
            # For uniform mats, assume linear relationship
            physical_thickness = optical_thickness * spatial_scale_um_per_pixel
        else:
            physical_thickness = optical_thickness
            
        # Clamp to reasonable range
        thickness_map = np.clip(physical_thickness, 0, self.max_thickness)
        
        return thickness_map.astype(np.float32)
    
    def _estimate_background_intensity(self, image: np.ndarray) -> float:
        """
        Estimate background intensity from image regions with minimal fiber coverage.
        
        Args:
            image: Normalized image [0,1]
            
        Returns:
            Estimated background intensity
        """
        # Use morphological opening to identify background regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        opened = cv2.morphologyEx((image * 255).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        opened_norm = opened.astype(np.float64) / 255.0
        
        # Background regions should have high intensity (low fiber density)
        background_mask = opened_norm > np.percentile(opened_norm, 80)
        
        if np.sum(background_mask) > 0:
            background_intensity = np.mean(image[background_mask])
        else:
            # Fallback to high percentile
            background_intensity = np.percentile(image, 90)
            
        logger.info(f"Estimated background intensity: {background_intensity:.3f}")
        return background_intensity
    
    def get_thickness_statistics(self, thickness_map: np.ndarray, 
                               mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate thickness statistics for validation.
        
        Args:
            thickness_map: Estimated thickness map
            mask: Optional mask for ROI analysis
            
        Returns:
            Dictionary of thickness statistics
        """
        if mask is not None:
            valid_thickness = thickness_map[mask > 0]
        else:
            valid_thickness = thickness_map.flatten()
            
        if len(valid_thickness) == 0:
            return {}
            
        stats = {
            'mean_thickness': float(np.mean(valid_thickness)),
            'std_thickness': float(np.std(valid_thickness)),
            'min_thickness': float(np.min(valid_thickness)),
            'max_thickness': float(np.max(valid_thickness)),
            'median_thickness': float(np.median(valid_thickness)),
            'thickness_range': float(np.max(valid_thickness) - np.min(valid_thickness)),
            'coefficient_of_variation': float(np.std(valid_thickness) / np.mean(valid_thickness))
        }
        
        return stats


class MultiModelThicknessEstimator:
    """
    Advanced thickness estimator supporting multiple physical models.
    """
    
    def __init__(self):
        self.beer_lambert = BeerLambertEstimator()
        
    def estimate_with_multiple_models(self, 
                                    image: np.ndarray,
                                    models: list = None) -> Dict[str, np.ndarray]:
        """
        Estimate thickness using multiple models for comparison.
        
        Args:
            image: Input grayscale image
            models: List of models to use ['beer_lambert', 'linear', 'logarithmic']
            
        Returns:
            Dictionary mapping model names to thickness maps
        """
        if models is None:
            models = ['beer_lambert', 'linear', 'logarithmic']
            
        results = {}
        
        # Normalize image
        image_norm = image.astype(np.float32) / 255.0
        
        if 'beer_lambert' in models:
            results['beer_lambert'] = self.beer_lambert.estimate_thickness(image)
            
        if 'linear' in models:
            # Simple linear inversion (legacy method)
            results['linear'] = (1.0 - image_norm) * 100.0
            
        if 'logarithmic' in models:
            # Logarithmic relationship
            epsilon = 0.01
            thickness = -np.log(image_norm + epsilon) * 50.0
            results['logarithmic'] = np.clip(thickness, 0, 500)
            
        return results
    
    def validate_model_performance(self, 
                                 image: np.ndarray,
                                 reference_thickness: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """
        Validate different model performances.
        
        Args:
            image: Input image
            reference_thickness: Ground truth thickness (if available)
            
        Returns:
            Performance metrics for each model
        """
        thickness_maps = self.estimate_with_multiple_models(image)
        performance = {}
        
        for model_name, thickness_map in thickness_maps.items():
            stats = self.beer_lambert.get_thickness_statistics(thickness_map)
            
            # Add model-specific metrics
            stats['model_name'] = model_name
            stats['dynamic_range'] = stats['max_thickness'] - stats['min_thickness']
            stats['signal_to_noise'] = stats['mean_thickness'] / stats['std_thickness']
            
            if reference_thickness is not None:
                # Calculate accuracy metrics if reference is available
                mse = np.mean((thickness_map - reference_thickness) ** 2)
                mae = np.mean(np.abs(thickness_map - reference_thickness))
                stats['mse'] = float(mse)
                stats['mae'] = float(mae)
                stats['relative_error'] = float(mae / np.mean(reference_thickness) * 100)
                
            performance[model_name] = stats
            
        return performance


def calibrate_attenuation_coefficient(images: list, 
                                    known_thicknesses: list,
                                    initial_alpha: float = 0.05) -> float:
    """
    Calibrate attenuation coefficient from images with known thicknesses.
    
    Args:
        images: List of calibration images
        known_thicknesses: List of corresponding known thicknesses
        initial_alpha: Initial guess for attenuation coefficient
        
    Returns:
        Optimized attenuation coefficient
    """
    try:
        from scipy.optimize import minimize_scalar
    except ImportError:
        logger.warning("SciPy not available. Using simple grid search for calibration.")
        return _simple_calibration(images, known_thicknesses, initial_alpha)
    
    def objective(alpha):
        total_error = 0
        estimator = BeerLambertEstimator(attenuation_coefficient=alpha)
        
        for image, true_thickness in zip(images, known_thicknesses):
            estimated = estimator.estimate_thickness(image)
            mean_estimated = np.mean(estimated)
            error = abs(mean_estimated - true_thickness) / true_thickness
            total_error += error
            
        return total_error / len(images)
    
    result = minimize_scalar(objective, bounds=(0.001, 0.2), method='bounded')
    optimized_alpha = result.x
    
    logger.info(f"Calibrated attenuation coefficient: {optimized_alpha:.6f}")
    return optimized_alpha


def _simple_calibration(images: list, known_thicknesses: list, initial_alpha: float) -> float:
    """Simple grid search calibration when SciPy is not available."""
    best_alpha = initial_alpha
    best_error = float('inf')
    
    # Test range of alpha values
    alpha_range = np.linspace(0.001, 0.2, 50)
    
    for alpha in alpha_range:
        total_error = 0
        estimator = BeerLambertEstimator(attenuation_coefficient=alpha)
        
        for image, true_thickness in zip(images, known_thicknesses):
            estimated = estimator.estimate_thickness(image)
            mean_estimated = np.mean(estimated)
            error = abs(mean_estimated - true_thickness) / true_thickness
            total_error += error
            
        avg_error = total_error / len(images)
        if avg_error < best_error:
            best_error = avg_error
            best_alpha = alpha
    
    logger.info(f"Calibrated attenuation coefficient (grid search): {best_alpha:.6f}")
    return best_alpha
