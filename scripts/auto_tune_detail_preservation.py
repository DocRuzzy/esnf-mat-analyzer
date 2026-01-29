"""
Auto-Tuning Script for Detail Preservation in Thickness Maps

This script systematically tests different parameter combinations to find
settings that preserve fine detail in the thickness heatmap.

The problem: Current pipeline is "flattening" the image, losing subtle
thickness variations that are visible in the original.

Metrics used to measure detail preservation:
1. Local variance ratio - preserved local texture
2. Entropy ratio - information content preserved  
3. Frequency content ratio - high-frequency features preserved
4. Gradient magnitude ratio - edge/detail sharpness preserved

Author: ESNF Mat Analyzer Team
Date: 2026-01-29
"""

import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import entropy as scipy_entropy


@dataclass
class DetailMetrics:
    """Metrics measuring detail preservation."""
    local_variance: float = 0.0  # Mean local variance (texture)
    entropy: float = 0.0  # Shannon entropy (information content)
    high_freq_power: float = 0.0  # Power in high frequencies
    gradient_magnitude: float = 0.0  # Mean gradient (edge strength)
    dynamic_range: float = 0.0  # Max - min value
    percentile_range: float = 0.0  # 98th - 2nd percentile
    global_gradient_strength: float = 0.0  # NEW: Measures residual illumination gradient
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'local_variance': self.local_variance,
            'entropy': self.entropy,
            'high_freq_power': self.high_freq_power,
            'gradient_magnitude': self.gradient_magnitude,
            'dynamic_range': self.dynamic_range,
            'percentile_range': self.percentile_range,
            'global_gradient_strength': self.global_gradient_strength,
        }


@dataclass
class TuningResult:
    """Result from one parameter combination test."""
    params: Dict[str, Any]
    input_metrics: DetailMetrics
    output_metrics: DetailMetrics
    preservation_scores: Dict[str, float]  # Ratio of output/input for each metric
    overall_score: float = 0.0
    notes: List[str] = field(default_factory=list)


class DetailPreservationAnalyzer:
    """Analyzes how well detail is preserved through processing."""
    
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
    
    def compute_metrics(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> DetailMetrics:
        """
        Compute detail metrics for an image.
        
        Args:
            image: Grayscale image (any dtype, will be normalized)
            mask: Optional mask for region of interest
        """
        # Normalize to 0-1 float
        img = image.astype(np.float64)
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        
        if mask is not None:
            mask = mask.astype(bool)
        else:
            mask = np.ones_like(img, dtype=bool)
        
        metrics = DetailMetrics()
        
        # 1. Local variance (texture measure)
        local_mean = ndimage.uniform_filter(img, size=self.window_size)
        local_sqr_mean = ndimage.uniform_filter(img**2, size=self.window_size)
        local_var = local_sqr_mean - local_mean**2
        local_var = np.clip(local_var, 0, None)  # Numerical stability
        metrics.local_variance = float(np.mean(local_var[mask]))
        
        # 2. Entropy (information content)
        # Use histogram-based entropy
        masked_values = img[mask]
        hist, _ = np.histogram(masked_values, bins=256, range=(0, 1))
        hist = hist / hist.sum()  # Normalize to probability
        hist = hist[hist > 0]  # Remove zeros for log
        metrics.entropy = float(-np.sum(hist * np.log2(hist)))
        
        # 3. High frequency power (FFT-based)
        # Pad to avoid edge effects
        padded = np.pad(img, self.window_size, mode='reflect')
        fft = np.fft.fft2(padded)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Create high-pass mask (exclude low frequencies)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        low_freq_radius = min(h, w) // 8  # Low frequencies = inner 1/8
        high_pass_mask = np.sqrt((x - cx)**2 + (y - cy)**2) > low_freq_radius
        
        total_power = np.sum(magnitude**2)
        high_freq_power = np.sum((magnitude * high_pass_mask)**2)
        metrics.high_freq_power = float(high_freq_power / (total_power + 1e-10))
        
        # 4. Gradient magnitude (edge strength)
        gy, gx = np.gradient(img)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        metrics.gradient_magnitude = float(np.mean(gradient_mag[mask]))
        
        # 5. Dynamic range
        metrics.dynamic_range = float(img[mask].max() - img[mask].min())
        
        # 6. Percentile range (robust to outliers)
        metrics.percentile_range = float(
            np.percentile(img[mask], 98) - np.percentile(img[mask], 2)
        )
        
        # 7. Global gradient strength (measures illumination gradient)
        # Fit linear plane to the masked region and measure slope
        h_img, w_img = img.shape
        y_grid, x_grid = np.mgrid[:h_img, :w_img]
        y_pts = y_grid[mask].flatten()
        x_pts = x_grid[mask].flatten()
        z_pts = img[mask].flatten()
        
        # Fit: z = a*x + b*y + c
        A = np.column_stack([x_pts, y_pts, np.ones_like(x_pts)])
        coeffs, _, _, _ = np.linalg.lstsq(A, z_pts, rcond=None)
        
        # Gradient strength = magnitude of (a, b) normalized by image dimensions
        gradient_strength = np.sqrt(coeffs[0]**2 + coeffs[1]**2)
        # Normalize by typical scale
        metrics.global_gradient_strength = float(gradient_strength * min(h_img, w_img))
        
        return metrics
    
    def compute_preservation_scores(
        self, 
        input_metrics: DetailMetrics, 
        output_metrics: DetailMetrics
    ) -> Dict[str, float]:
        """Compute preservation ratio for each metric (1.0 = perfect preservation)."""
        scores = {}
        
        for field_name in ['local_variance', 'entropy', 'high_freq_power', 
                          'gradient_magnitude', 'percentile_range']:
            input_val = getattr(input_metrics, field_name)
            output_val = getattr(output_metrics, field_name)
            
            if input_val > 1e-10:
                # Ratio, capped at 1.0 (can't have more detail than input)
                scores[field_name] = min(1.0, output_val / input_val)
            else:
                scores[field_name] = 1.0 if output_val < 1e-10 else 0.0
        
        # Special handling for gradient reduction - we WANT this to decrease
        input_gradient = input_metrics.global_gradient_strength
        output_gradient = output_metrics.global_gradient_strength
        if input_gradient > 1e-10:
            # Score is higher when gradient is reduced
            # 1.0 = gradient completely removed, 0.0 = gradient unchanged or worse
            gradient_reduction = 1.0 - min(1.0, output_gradient / input_gradient)
            scores['gradient_reduction'] = gradient_reduction
        else:
            scores['gradient_reduction'] = 1.0  # No gradient to reduce
        
        return scores
    
    def compute_overall_score(
        self, 
        preservation_scores: Dict[str, float],
        input_metrics: Optional[DetailMetrics] = None,
        output_metrics: Optional[DetailMetrics] = None,
    ) -> float:
        """
        Compute weighted overall preservation score.
        
        Weights emphasize:
        - local_variance: texture preservation (important)
        - entropy: information preservation (important)  
        - high_freq_power: fine detail (very important)
        - gradient_magnitude: edge sharpness (important)
        - percentile_range: dynamic range (moderate)
        - gradient_reduction: removing illumination gradient (important!)
        """
        weights = {
            'local_variance': 0.20,
            'entropy': 0.15,
            'high_freq_power': 0.25,
            'gradient_magnitude': 0.10,
            'percentile_range': 0.10,
            'gradient_reduction': 0.20,  # NEW: reward gradient removal
        }
        
        score = 0.0
        for metric, weight in weights.items():
            score += weight * preservation_scores.get(metric, 0.0)
        
        return score


class PipelineParameterTuner:
    """
    Automatic tuning system for finding optimal parameters.
    
    Tests combinations of:
    - Background correction method and strength
    - Thickness model parameters
    - Colormap gamma/scaling
    """
    
    def __init__(self, image_path: Optional[Path] = None):
        self.image_path = image_path
        self.original_image: Optional[np.ndarray] = None
        self.mat_mask: Optional[np.ndarray] = None
        self.analyzer = DetailPreservationAnalyzer()
        self.results: List[TuningResult] = []
        
    def load_image(self, path: Path) -> np.ndarray:
        """Load and prepare image for testing."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        self.original_image = img
        self.image_path = path
        return img
    
    def detect_mat_region(self, image: np.ndarray) -> np.ndarray:
        """Simple mat detection for testing purposes."""
        # Use Otsu's threshold to find bright mat region
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find largest contour (mat)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [largest], -1, 255, -1)
            self.mat_mask = mask > 0
        else:
            self.mat_mask = np.ones_like(image, dtype=bool)
        
        return self.mat_mask
    
    def test_parameter_set(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        params: Dict[str, Any]
    ) -> TuningResult:
        """
        Test one set of parameters and measure detail preservation.
        
        Parameters tested:
        - bg_correction: 'none', 'polynomial', 'blur', 'morphological'
        - bg_strength: 0.0 - 1.0 (blend with original)
        - thickness_gamma: 0.5 - 2.0 (nonlinear mapping)
        - local_contrast_enhance: True/False
        - contrast_clip_limit: 1.0 - 4.0 (for CLAHE)
        """
        # Compute input metrics
        input_metrics = self.analyzer.compute_metrics(image, mask)
        
        # Apply processing based on params
        processed = self._apply_processing(image, mask, params)
        
        # Compute output metrics
        output_metrics = self.analyzer.compute_metrics(processed, mask)
        
        # Compute preservation scores
        preservation_scores = self.analyzer.compute_preservation_scores(
            input_metrics, output_metrics
        )
        
        # Overall score
        overall_score = self.analyzer.compute_overall_score(preservation_scores)
        
        result = TuningResult(
            params=params.copy(),
            input_metrics=input_metrics,
            output_metrics=output_metrics,
            preservation_scores=preservation_scores,
            overall_score=overall_score,
        )
        
        return result
    
    def _apply_processing(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply processing pipeline with given parameters.
        
        IMPORTANT: Background correction methods now fit ONLY to background
        pixels (outside mat) and extrapolate into mat region. This preserves
        real thickness variations while removing illumination gradients.
        """
        img = image.astype(np.float64)
        
        # 1. Background correction
        bg_method = params.get('bg_correction', 'none')
        bg_strength = params.get('bg_strength', 1.0)
        
        if bg_method == 'bg_linear':
            # LINEAR gradient removal - fit plane to background, extrapolate into mat
            corrected = self._background_only_gradient_correction(img, mask, degree=1)
        elif bg_method == 'bg_quadratic':
            # QUADRATIC gradient removal - fit surface to background, extrapolate
            corrected = self._background_only_gradient_correction(img, mask, degree=2)
        elif bg_method == 'polynomial':
            # Legacy: polynomial fit (may include mat pixels)
            corrected = self._polynomial_bg_correction(img, mask, degree=2)
        elif bg_method == 'rolling_ball':
            # Rolling ball background subtraction
            radius = params.get('rolling_ball_radius', 50)
            corrected = self._rolling_ball_correction(img, radius)
        elif bg_method == 'blur':
            kernel_size = params.get('blur_kernel', 51)
            bg = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            corrected = img - bg + np.mean(img[mask])
        elif bg_method == 'morphological':
            kernel_size = params.get('morph_kernel', 51)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            bg = cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            corrected = img - bg.astype(np.float64) + np.mean(img[mask])
        else:  # 'none'
            corrected = img.copy()
        
        # Blend with original based on strength
        if bg_strength < 1.0:
            corrected = bg_strength * corrected + (1 - bg_strength) * img
        
        # 2. Local contrast enhancement (optional)
        if params.get('local_contrast_enhance', False):
            clip_limit = params.get('contrast_clip_limit', 2.0)
            tile_size = params.get('contrast_tile_size', 8)
            
            # Normalize to 0-255 for CLAHE
            norm = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            enhanced = clahe.apply(norm.astype(np.uint8))
            corrected = enhanced.astype(np.float64)
        
        # 3. Gamma correction for thickness mapping
        gamma = params.get('thickness_gamma', 1.0)
        if gamma != 1.0:
            # Normalize to 0-1, apply gamma, scale back
            norm_min, norm_max = corrected.min(), corrected.max()
            if norm_max > norm_min:
                normalized = (corrected - norm_min) / (norm_max - norm_min)
                normalized = np.power(normalized, gamma)
                corrected = normalized * (norm_max - norm_min) + norm_min
        
        return corrected
    
    def _polynomial_bg_correction(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        degree: int = 2
    ) -> np.ndarray:
        """Fit polynomial surface to background and subtract."""
        h, w = image.shape
        y, x = np.mgrid[:h, :w]
        
        # Use points outside mat for fitting
        bg_mask = ~mask
        if np.sum(bg_mask) < 100:
            # Not enough background, use edges
            bg_mask = np.zeros_like(mask)
            bg_mask[:50, :] = True
            bg_mask[-50:, :] = True
            bg_mask[:, :50] = True
            bg_mask[:, -50:] = True
        
        # Sample points for fitting (subsample for speed)
        sample_rate = max(1, np.sum(bg_mask) // 5000)
        y_pts = y[bg_mask][::sample_rate]
        x_pts = x[bg_mask][::sample_rate]
        z_pts = image[bg_mask][::sample_rate]
        
        # Build polynomial terms
        terms = []
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                terms.append((x_pts ** i) * (y_pts ** j))
        
        A = np.column_stack(terms)
        
        # Solve least squares
        coeffs, _, _, _ = np.linalg.lstsq(A, z_pts, rcond=None)
        
        # Evaluate polynomial over full image (this EXTRAPOLATES into mat region)
        bg_surface = np.zeros_like(image)
        idx = 0
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                bg_surface += coeffs[idx] * (x ** i) * (y ** j)
                idx += 1
        
        # Subtract the EXTRAPOLATED background from mat region
        # Key: we're subtracting what we PREDICT the background would be under the mat
        corrected = image - bg_surface + np.mean(image[mask])
        
        return corrected
    
    def _background_only_gradient_correction(
        self, 
        image: np.ndarray, 
        mat_mask: np.ndarray,
        degree: int = 1
    ) -> np.ndarray:
        """
        CORRECT APPROACH: Estimate illumination gradient from background only,
        then extrapolate and subtract from entire image.
        
        This preserves real thickness variations in the mat while removing
        the illumination gradient that affects the background.
        
        Args:
            image: Input grayscale image
            mat_mask: Boolean mask where True = mat region
            degree: Polynomial degree (1=linear, 2=quadratic)
        
        Returns:
            Corrected image with background gradient removed but mat detail preserved
        """
        h, w = image.shape
        y, x = np.mgrid[:h, :w]
        
        # Use ONLY background pixels for fitting
        bg_mask = ~mat_mask
        
        # Ensure we have enough background pixels
        if np.sum(bg_mask) < 500:
            print("Warning: Not enough background pixels for gradient estimation")
            return image.copy()
        
        # Sample background points
        y_pts = y[bg_mask].flatten()
        x_pts = x[bg_mask].flatten()
        z_pts = image[bg_mask].flatten()
        
        # Subsample if too many points (for speed)
        if len(z_pts) > 10000:
            indices = np.random.choice(len(z_pts), 10000, replace=False)
            y_pts = y_pts[indices]
            x_pts = x_pts[indices]
            z_pts = z_pts[indices]
        
        # Build polynomial design matrix
        # degree=1: z = a + bx + cy (linear plane)
        # degree=2: z = a + bx + cy + dx² + ey² + fxy (quadratic surface)
        terms = [np.ones_like(x_pts)]  # Constant term
        if degree >= 1:
            terms.extend([x_pts, y_pts])
        if degree >= 2:
            terms.extend([x_pts**2, y_pts**2, x_pts * y_pts])
        
        A = np.column_stack(terms)
        
        # Solve least squares fit
        coeffs, residuals, rank, s = np.linalg.lstsq(A, z_pts, rcond=None)
        
        # Evaluate the fitted surface over the ENTIRE image
        # This extrapolates the background gradient into the mat region
        eval_terms = [np.ones_like(x, dtype=float)]
        if degree >= 1:
            eval_terms.extend([x.astype(float), y.astype(float)])
        if degree >= 2:
            eval_terms.extend([
                (x**2).astype(float), 
                (y**2).astype(float), 
                (x * y).astype(float)
            ])
        
        # Compute predicted background at each pixel
        predicted_bg = np.zeros_like(image, dtype=float)
        for i, term in enumerate(eval_terms):
            predicted_bg += coeffs[i] * term
        
        # Subtract predicted background, add back mean to preserve brightness
        mean_mat_intensity = np.mean(image[mat_mask])
        corrected = image.astype(float) - predicted_bg
        corrected = corrected - np.mean(corrected[mat_mask]) + mean_mat_intensity
        
        return corrected
    
    def _gradient_removal(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Remove linear illumination gradient (top-bottom, left-right).
        
        Uses the new background-only fitting approach.
        """
        return self._background_only_gradient_correction(image, mask, degree=1)
    
    def _quadratic_bg_correction(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Remove quadratic illumination gradient using background-only fitting.
        """
        return self._background_only_gradient_correction(image, mask, degree=2)
        A = np.column_stack([x_pts, y_pts, np.ones_like(x_pts)])
        
        # Solve least squares
        coeffs, _, _, _ = np.linalg.lstsq(A, z_pts, rcond=None)
        
        # Create gradient surface
        gradient_surface = coeffs[0] * x + coeffs[1] * y + coeffs[2]
        
        # Subtract gradient, preserve mean
        corrected = image - gradient_surface + np.mean(image[mask])
        
        return corrected
    
    def _rolling_ball_correction(self, image: np.ndarray, radius: int = 50) -> np.ndarray:
        """
        Rolling ball background subtraction.
        
        Better at preserving local features while removing large-scale gradients.
        """
        from scipy import ndimage as ndi
        
        # Create structuring element (approximating a ball)
        kernel_size = 2 * radius + 1
        y, x = np.ogrid[:kernel_size, :kernel_size]
        center = radius
        dist_sq = (x - center)**2 + (y - center)**2
        
        # Ball shape (inverted paraboloid approximation for speed)
        ball = np.zeros((kernel_size, kernel_size))
        ball_mask = dist_sq <= radius**2
        ball[ball_mask] = np.sqrt(radius**2 - dist_sq[ball_mask])
        ball = ball.max() - ball  # Invert so it's a "bowl"
        
        # Rolling ball = erosion followed by dilation
        # This finds the "background" under the signal
        eroded = ndi.grey_erosion(image, footprint=ball_mask, structure=-ball)
        background = ndi.grey_dilation(eroded, footprint=ball_mask, structure=ball)
        
        # Subtract background
        corrected = image - background + np.mean(background)
        
        return corrected
    
    def run_parameter_sweep(
        self,
        image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> List[TuningResult]:
        """
        Run systematic parameter sweep to find optimal settings.
        
        Returns list of results sorted by overall score (best first).
        """
        if image is None:
            image = self.original_image
        if mask is None:
            mask = self.mat_mask
        
        if image is None or mask is None:
            raise ValueError("No image/mask loaded. Call load_image() first.")
        
        # Parameter grid - focused on CORRECT background correction
        # Key insight: fit gradient ONLY to background, extrapolate into mat
        param_grid = {
            'bg_correction': [
                'none',              # No correction (baseline)
                'bg_linear',         # LINEAR gradient from background only (RECOMMENDED)
                'bg_quadratic',      # QUADRATIC gradient from background only
                'rolling_ball',      # Rolling ball (local, may remove mat features)
            ],
            'bg_strength': [0.7, 0.85, 1.0],      # How much correction to apply
            'thickness_gamma': [0.9, 1.0, 1.1],   # Mild gamma options
            'local_contrast_enhance': [False, True],
            'contrast_clip_limit': [2.0, 3.0],    
            'rolling_ball_radius': [50, 80],      # Only for rolling_ball method
        }
        
        # Generate combinations (skip irrelevant ones)
        self.results = []
        total_tests = 0
        
        # Count total combinations for progress
        estimated_total = 0
        for bg_method in param_grid['bg_correction']:
            radii_count = len(param_grid['rolling_ball_radius']) if bg_method == 'rolling_ball' else 1
            strength_count = 1 if bg_method == 'none' else len(param_grid['bg_strength'])
            enhance_combos = len(param_grid['thickness_gamma']) * (1 + len(param_grid['contrast_clip_limit']))
            estimated_total += radii_count * strength_count * enhance_combos
        
        print(f"Starting parameter sweep (~{estimated_total} combinations)...")
        print(f"NOTE: bg_linear and bg_quadratic fit ONLY to background pixels,")
        print(f"      then extrapolate into mat region (preserves mat detail)")
        print("-" * 50)
        
        for bg_method in param_grid['bg_correction']:
            print(f"  Testing background method: {bg_method}...")
            
            # For rolling ball, test different radii
            if bg_method == 'rolling_ball':
                radii_to_test = param_grid['rolling_ball_radius']
            else:
                radii_to_test = [None]  # Not applicable
            
            for rb_radius in radii_to_test:
                for bg_strength in param_grid['bg_strength']:
                    # Skip bg_strength variations for 'none' method
                    if bg_method == 'none' and bg_strength != 1.0:
                        continue
                    
                    for gamma in param_grid['thickness_gamma']:
                        for enhance in param_grid['local_contrast_enhance']:
                            if enhance:
                                for clip in param_grid['contrast_clip_limit']:
                                    params = {
                                        'bg_correction': bg_method,
                                        'bg_strength': bg_strength,
                                        'thickness_gamma': gamma,
                                        'local_contrast_enhance': True,
                                        'contrast_clip_limit': clip,
                                    }
                                    if rb_radius is not None:
                                        params['rolling_ball_radius'] = rb_radius
                                    result = self.test_parameter_set(image, mask, params)
                                    self.results.append(result)
                                    total_tests += 1
                                    
                                    # Progress update every 20 tests
                                    if total_tests % 20 == 0:
                                        print(f"    Progress: {total_tests} tests completed...")
                            else:
                                params = {
                                    'bg_correction': bg_method,
                                    'bg_strength': bg_strength,
                                    'thickness_gamma': gamma,
                                    'local_contrast_enhance': False,
                                }
                                if rb_radius is not None:
                                    params['rolling_ball_radius'] = rb_radius
                                result = self.test_parameter_set(image, mask, params)
                                self.results.append(result)
                                total_tests += 1
            
            # Report best so far for this method
            method_results = [r for r in self.results if r.params.get('bg_correction') == bg_method]
            if method_results:
                best = max(method_results, key=lambda r: r.overall_score)
                print(f"    Best for {bg_method}: score={best.overall_score:.4f}")
        
        print("-" * 50)
        
        # Sort by overall score (descending)
        self.results.sort(key=lambda r: r.overall_score, reverse=True)
        
        print(f"Completed {total_tests} parameter combinations")
        print(f"Best overall score: {self.results[0].overall_score:.4f}")
        print(f"Best params: {self.results[0].params}")
        
        return self.results
    
    def get_top_results(self, n: int = 5) -> List[TuningResult]:
        """Get top N results by overall score."""
        return self.results[:n]
    
    def visualize_comparison(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        results: List[TuningResult],
        save_path: Optional[Path] = None,
    ):
        """Visualize original vs processed for top results."""
        n = min(len(results), 4)
        
        fig, axes = plt.subplots(3, n + 1, figsize=(4 * (n + 1), 10))
        
        # Original (grayscale for direct comparison)
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Original histogram
        masked_values = image[mask]
        axes[1, 0].hist(masked_values.flatten(), bins=100, color='blue', alpha=0.7)
        axes[1, 0].set_title('Original Histogram')
        axes[1, 0].set_xlabel('Intensity')
        
        # Placeholder for difference row
        axes[2, 0].axis('off')
        axes[2, 0].set_title('(Difference maps below)')
        
        # Top results
        for i, result in enumerate(results[:n]):
            processed = self._apply_processing(image, mask, result.params)
            
            # Use grayscale for direct comparison with original
            axes[0, i + 1].imshow(processed, cmap='gray')
            axes[0, i + 1].set_title(f'#{i+1}: Score={result.overall_score:.3f}')
            axes[0, i + 1].axis('off')
            
            # Processed histogram
            proc_values = processed[mask]
            axes[1, i + 1].hist(proc_values.flatten(), bins=100, color='green', alpha=0.7)
            axes[1, i + 1].set_title(f'Params: {self._short_params(result.params)}')
            axes[1, i + 1].set_xlabel('Intensity')
            
            # Difference map (what was removed/changed)
            # Normalize both to 0-1 for comparison
            img_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
            proc_norm = (processed - processed.min()) / (processed.max() - processed.min() + 1e-10)
            diff = img_norm - proc_norm
            
            # Red = removed (was bright, now dark), Blue = added (was dark, now bright)
            im = axes[2, i + 1].imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
            axes[2, i + 1].set_title(f'Diff: Red=removed, Blue=added')
            axes[2, i + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to {save_path}")
        
        # Close to prevent hanging
        plt.close(fig)
        print("Visualization complete (saved to file, window closed)")
    
    def _short_params(self, params: Dict[str, Any]) -> str:
        """Create short parameter string for labels."""
        parts = []
        if params.get('bg_correction') != 'none':
            parts.append(f"bg={params['bg_correction'][:4]}")
            parts.append(f"str={params.get('bg_strength', 1.0):.1f}")
        if params.get('thickness_gamma', 1.0) != 1.0:
            parts.append(f"γ={params['thickness_gamma']:.2f}")
        if params.get('local_contrast_enhance'):
            parts.append(f"CLAHE={params.get('contrast_clip_limit', 2.0):.1f}")
        return ', '.join(parts) if parts else 'raw'
    
    def generate_report(self, output_path: Path):
        """Generate detailed tuning report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'image_path': str(self.image_path) if self.image_path else None,
            'total_tests': len(self.results),
            'top_results': [],
        }
        
        for i, result in enumerate(self.results[:10]):
            report['top_results'].append({
                'rank': i + 1,
                'overall_score': result.overall_score,
                'params': result.params,
                'preservation_scores': result.preservation_scores,
                'input_metrics': result.input_metrics.to_dict(),
                'output_metrics': result.output_metrics.to_dict(),
            })
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Saved report to {output_path}")
        return report


def run_tuning_on_real_image(image_path: Path):
    """Run tuning on a real sample image."""
    print(f"\n{'='*60}")
    print(f"AUTO-TUNING: Detail Preservation Analysis")
    print(f"Image: {image_path}")
    print(f"{'='*60}\n")
    
    tuner = PipelineParameterTuner()
    
    # Load image
    print("Loading image...")
    image = tuner.load_image(image_path)
    print(f"  Image size: {image.shape}")
    
    # Detect mat region
    print("Detecting mat region...")
    mask = tuner.detect_mat_region(image)
    mat_pct = 100 * np.sum(mask) / mask.size
    print(f"  Mat region: {mat_pct:.1f}% of image")
    
    # Run parameter sweep
    print("\nRunning parameter sweep...")
    results = tuner.run_parameter_sweep(image, mask)
    
    # Show top results
    print(f"\n{'='*60}")
    print("TOP 5 PARAMETER COMBINATIONS")
    print(f"{'='*60}")
    
    for i, result in enumerate(results[:5]):
        print(f"\n#{i+1} - Overall Score: {result.overall_score:.4f}")
        print(f"  Parameters:")
        for k, v in result.params.items():
            print(f"    {k}: {v}")
        print(f"  Preservation Scores:")
        for k, v in result.preservation_scores.items():
            print(f"    {k}: {v:.3f}")
    
    # Visualize
    print("\nGenerating visualization...")
    output_dir = image_path.parent / "tuning_results"
    output_dir.mkdir(exist_ok=True)
    
    tuner.visualize_comparison(
        image, mask, results,
        save_path=output_dir / f"{image_path.stem}_comparison.png"
    )
    
    # Save report
    tuner.generate_report(output_dir / f"{image_path.stem}_tuning_report.json")
    
    return tuner, results


def run_tuning_on_synthetic():
    """Run tuning on synthetic data with known features."""
    print(f"\n{'='*60}")
    print(f"AUTO-TUNING: Synthetic Data Validation")
    print(f"{'='*60}\n")
    
    # Import synthetic generator
    from esnf_mat_analyzer.data import SyntheticMatGenerator, SyntheticMatConfig, PatternType
    
    # Generate test image with known features
    config = SyntheticMatConfig(
        pattern_type=PatternType.MULTI_FREQUENCY,
        width=600,
        height=600,
        gel_enabled=True,
    )
    
    gen = SyntheticMatGenerator(config)
    result = gen.generate()
    
    print(f"Generated synthetic image: {result.image.shape}")
    print(f"Ground truth range: {result.ground_truth_thickness.min():.1f} - {result.ground_truth_thickness.max():.1f}")
    
    # Run tuning
    tuner = PipelineParameterTuner()
    tuner.original_image = result.image.astype(np.uint8)
    tuner.mat_mask = result.mat_mask
    
    print("\nRunning parameter sweep on synthetic data...")
    results = tuner.run_parameter_sweep()
    
    # Show results
    print(f"\n{'='*60}")
    print("TOP 5 PARAMETER COMBINATIONS (Synthetic)")
    print(f"{'='*60}")
    
    for i, r in enumerate(results[:5]):
        print(f"\n#{i+1} - Score: {r.overall_score:.4f}")
        print(f"  {tuner._short_params(r.params)}")
    
    return tuner, results


if __name__ == "__main__":
    import sys
    
    # Check for command line argument
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        if image_path.exists():
            run_tuning_on_real_image(image_path)
        else:
            print(f"File not found: {image_path}")
            sys.exit(1)
    else:
        # Default: try the sample image, fall back to synthetic
        sample_path = Path("tests/Samples/TCD4-GPED4.0.png")
        if sample_path.exists():
            run_tuning_on_real_image(sample_path)
        else:
            print("No sample image found, running on synthetic data...")
            run_tuning_on_synthetic()
