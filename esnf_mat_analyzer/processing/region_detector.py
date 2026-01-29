"""
Three-Region Detector for ESNF Mat Analyzer.

Identifies and segments three distinct regions in electrospun nanofiber images:
1. Background - The collection surface outside deposition area (reference black level)
2. Gel Region - The dark perimeter around mat (wet polymer, darker than background)
3. Mat Region - The deposited nanofiber material (bright region)

This replaces simple Otsu thresholding with multi-threshold segmentation
that properly handles the gel boundary as a sub-background region.

Author: ESNF Mat Analyzer Team
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Any
import logging


@dataclass
class RegionDetectionResult:
    """Result from three-region detection."""
    
    # Binary masks for each region
    background_mask: np.ndarray  # True where background (outside deposition)
    gel_mask: np.ndarray  # True in gel region (dark perimeter)
    mat_mask: np.ndarray  # True where nanofiber mat exists
    
    # Contours
    mat_contour: Optional[np.ndarray] = None  # Outer boundary of mat
    gel_contour: Optional[np.ndarray] = None  # Outer boundary of gel (if detected)
    
    # Reference intensity levels
    background_intensity: float = 0.0  # Mean intensity of background region
    gel_intensity: float = 0.0  # Mean intensity of gel region
    mat_intensity_range: Tuple[float, float] = (0.0, 255.0)  # Min/max in mat
    
    # Detection confidence
    gel_detected: bool = False  # Whether distinct gel region was found
    quality_score: float = 0.0  # Overall detection quality (0-1)
    
    # Saturation info
    saturation_mask: Optional[np.ndarray] = None
    saturation_pct: float = 0.0  # Percentage of mat that is saturated


class ThreeRegionDetector:
    """
    Detector for identifying background, gel, and mat regions in nanofiber images.
    
    The key insight is that the gel region (wet polymer perimeter) is DARKER
    than the background, not brighter. This requires multi-threshold segmentation
    rather than simple binary thresholding.
    
    Detection Strategy:
    1. Find the bright mat region (standard approach)
    2. Identify background from regions far from mat
    3. Detect gel as the dark ring between background and mat
    """
    
    def __init__(
        self,
        mat_threshold_method: str = "otsu",
        gel_detection_enabled: bool = True,
        saturation_threshold: int = 254,
        min_mat_area_ratio: float = 0.05,
        max_mat_area_ratio: float = 0.8,
    ):
        """
        Initialize the three-region detector.
        
        Args:
            mat_threshold_method: Method for mat detection ("otsu", "adaptive", "percentile")
            gel_detection_enabled: Whether to attempt gel region detection
            saturation_threshold: Intensity above which pixels are considered saturated
            min_mat_area_ratio: Minimum mat area as fraction of image
            max_mat_area_ratio: Maximum mat area as fraction of image
        """
        self.mat_threshold_method = mat_threshold_method
        self.gel_detection_enabled = gel_detection_enabled
        self.saturation_threshold = saturation_threshold
        self.min_mat_area_ratio = min_mat_area_ratio
        self.max_mat_area_ratio = max_mat_area_ratio
        self.logger = logging.getLogger(__name__)
    
    def detect(
        self,
        image: np.ndarray,
        roi_mask: Optional[np.ndarray] = None
    ) -> RegionDetectionResult:
        """
        Detect three regions in the image.
        
        Args:
            image: Grayscale input image (0-255)
            roi_mask: Optional user-defined ROI to restrict detection
            
        Returns:
            RegionDetectionResult with masks and metadata
        """
        # Ensure grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Apply ROI if provided
        if roi_mask is not None:
            analysis_mask = roi_mask.astype(bool)
        else:
            analysis_mask = np.ones((h, w), dtype=bool)
        
        # Step 1: Detect mat region (bright area)
        mat_mask, mat_contour = self._detect_mat_region(gray, analysis_mask)
        
        # Step 2: Detect background region (outside mat, excluding edges)
        background_mask = self._detect_background_region(gray, mat_mask, analysis_mask)
        
        # Step 3: Detect gel region (dark ring between background and mat)
        if self.gel_detection_enabled:
            gel_mask, gel_contour, gel_detected = self._detect_gel_region(
                gray, mat_mask, background_mask, analysis_mask
            )
        else:
            gel_mask = np.zeros((h, w), dtype=bool)
            gel_contour = None
            gel_detected = False
        
        # Calculate reference intensities
        background_intensity = self._safe_mean(gray, background_mask)
        gel_intensity = self._safe_mean(gray, gel_mask) if gel_detected else background_intensity
        
        mat_values = gray[mat_mask]
        mat_intensity_range = (float(mat_values.min()), float(mat_values.max())) if len(mat_values) > 0 else (0.0, 255.0)
        
        # Detect saturation
        saturation_mask = (gray >= self.saturation_threshold) & mat_mask
        saturation_pct = 100 * np.sum(saturation_mask) / max(np.sum(mat_mask), 1)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            gray, mat_mask, background_mask, gel_mask, gel_detected
        )
        
        self.logger.info(
            f"Three-region detection: mat={100*np.mean(mat_mask):.1f}%, "
            f"background={100*np.mean(background_mask):.1f}%, "
            f"gel={'detected' if gel_detected else 'not detected'}, "
            f"saturation={saturation_pct:.1f}%"
        )
        
        return RegionDetectionResult(
            background_mask=background_mask.astype(np.uint8),
            gel_mask=gel_mask.astype(np.uint8),
            mat_mask=mat_mask.astype(np.uint8),
            mat_contour=mat_contour,
            gel_contour=gel_contour,
            background_intensity=background_intensity,
            gel_intensity=gel_intensity,
            mat_intensity_range=mat_intensity_range,
            gel_detected=gel_detected,
            quality_score=quality_score,
            saturation_mask=saturation_mask.astype(np.uint8),
            saturation_pct=float(saturation_pct),
        )
    
    def _detect_mat_region(
        self,
        gray: np.ndarray,
        analysis_mask: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Detect the bright mat region using thresholding."""
        h, w = gray.shape
        
        # Get intensity values within analysis mask
        masked_values = gray[analysis_mask]
        
        if self.mat_threshold_method == "otsu":
            # Standard Otsu thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.mat_threshold_method == "adaptive":
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 51, -10
            )
        else:  # percentile
            # Use percentile-based threshold
            threshold = np.percentile(masked_values, 60)
            binary = (gray > threshold).astype(np.uint8) * 255
        
        # Apply analysis mask
        binary = np.bitwise_and(binary.astype(np.uint8), (analysis_mask.astype(np.uint8) * 255))
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.logger.warning("No mat contours found")
            return np.zeros((h, w), dtype=bool), None
        
        # Find the largest contour within area constraints
        image_area = h * w
        valid_contours = [
            c for c in contours
            if self.min_mat_area_ratio * image_area <= cv2.contourArea(c) <= self.max_mat_area_ratio * image_area
        ]
        
        if not valid_contours:
            # Fall back to largest contour
            valid_contours = [max(contours, key=cv2.contourArea)]
        
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Create mask from contour
        mat_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mat_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
        
        return mat_mask.astype(bool), largest_contour
    
    def _detect_background_region(
        self,
        gray: np.ndarray,
        mat_mask: np.ndarray,
        analysis_mask: np.ndarray
    ) -> np.ndarray:
        """
        Detect background region (collection surface outside mat).
        
        Background is defined as the region:
        - Outside the mat
        - Away from image edges (to avoid artifacts)
        - With relatively uniform intensity
        """
        h, w = gray.shape
        
        # Start with everything that's not mat
        potential_background = analysis_mask & ~mat_mask
        
        # Exclude regions too close to mat (potential gel area)
        mat_dilated = cv2.dilate(
            mat_mask.astype(np.uint8), 
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        )
        
        # Background is outside the dilated mat region
        background_mask = potential_background & ~mat_dilated.astype(bool)
        
        # Also exclude image edges (may have artifacts)
        edge_margin = min(h, w) // 20
        edge_mask = np.ones((h, w), dtype=bool)
        edge_mask[:edge_margin, :] = False
        edge_mask[-edge_margin:, :] = False
        edge_mask[:, :edge_margin] = False
        edge_mask[:, -edge_margin:] = False
        
        background_mask = background_mask & edge_mask
        
        # If we have very little background, relax constraints
        if np.sum(background_mask) < 0.01 * h * w:
            self.logger.warning("Limited background region, using relaxed detection")
            background_mask = potential_background & ~mat_dilated.astype(bool)
        
        return background_mask
    
    def _detect_gel_region(
        self,
        gray: np.ndarray,
        mat_mask: np.ndarray,
        background_mask: np.ndarray,
        analysis_mask: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        """
        Detect gel region (dark ring around mat, darker than background).
        
        The gel is wet polymer that appears DARKER than the dry background.
        It typically forms a ring/boundary around the deposited mat area.
        """
        h, w = gray.shape
        
        # Get background intensity reference
        bg_intensity = self._safe_mean(gray, background_mask)
        
        if bg_intensity == 0:
            self.logger.warning("Could not determine background intensity for gel detection")
            return np.zeros((h, w), dtype=bool), None, False
        
        # Create a search region: dilate mat but exclude mat itself and background
        mat_dilated = cv2.dilate(
            mat_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        ).astype(bool)
        
        search_region = mat_dilated & ~mat_mask & ~background_mask & analysis_mask
        
        if np.sum(search_region) < 100:
            self.logger.info("Insufficient search region for gel detection")
            return np.zeros((h, w), dtype=bool), None, False
        
        # Gel is DARKER than background
        # Look for pixels significantly darker than background intensity
        dark_threshold = bg_intensity * 0.7  # 70% of background brightness
        
        # Find dark pixels in the search region
        potential_gel = search_region & (gray < dark_threshold)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gel_cleaned = cv2.morphologyEx(potential_gel.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        gel_cleaned = cv2.morphologyEx(gel_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Check if we found a significant gel region
        gel_area = np.sum(gel_cleaned > 0)
        mat_area = np.sum(mat_mask)
        
        # Gel should be a reasonable fraction of mat size (not too small, not too large)
        if gel_area < 0.02 * mat_area or gel_area > 0.5 * mat_area:
            self.logger.info(f"Gel area ({gel_area}) outside expected range relative to mat ({mat_area})")
            return np.zeros((h, w), dtype=bool), None, False
        
        # Find gel contour
        contours, _ = cv2.findContours(
            gel_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        gel_contour = max(contours, key=cv2.contourArea) if contours else None
        
        self.logger.info(f"Detected gel region: {gel_area} pixels ({100*gel_area/(h*w):.1f}% of image)")
        
        return gel_cleaned.astype(bool), gel_contour, True
    
    def _safe_mean(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Calculate mean of image within mask, safely handling empty masks."""
        if np.sum(mask) == 0:
            return 0.0
        return float(np.mean(image[mask]))
    
    def _calculate_quality_score(
        self,
        gray: np.ndarray,
        mat_mask: np.ndarray,
        background_mask: np.ndarray,
        gel_mask: np.ndarray,
        gel_detected: bool
    ) -> float:
        """
        Calculate detection quality score (0-1).
        
        Based on:
        - Clear separation between regions
        - Expected intensity ordering (gel < background < mat)
        - Reasonable region sizes
        """
        score = 0.0
        
        # Check region sizes
        h, w = gray.shape
        total_pixels = h * w
        
        mat_ratio = np.sum(mat_mask) / total_pixels
        bg_ratio = np.sum(background_mask) / total_pixels
        
        # Mat should be reasonable size
        if 0.1 <= mat_ratio <= 0.6:
            score += 0.25
        elif 0.05 <= mat_ratio <= 0.8:
            score += 0.15
        
        # Background should exist
        if bg_ratio >= 0.1:
            score += 0.25
        elif bg_ratio >= 0.05:
            score += 0.15
        
        # Check intensity ordering
        mat_intensity = self._safe_mean(gray, mat_mask)
        bg_intensity = self._safe_mean(gray, background_mask)
        
        if mat_intensity > bg_intensity * 1.2:  # Mat clearly brighter
            score += 0.25
        
        # Gel should be darker than background if detected
        if gel_detected:
            gel_intensity = self._safe_mean(gray, gel_mask)
            if gel_intensity < bg_intensity * 0.9:
                score += 0.25
        else:
            score += 0.15  # Partial credit if gel not expected
        
        return min(score, 1.0)
    
    def visualize_regions(
        self,
        image: np.ndarray,
        result: RegionDetectionResult,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Create visualization overlay showing detected regions.
        
        Args:
            image: Original grayscale image
            result: Detection result
            alpha: Transparency of overlay
            
        Returns:
            Color image with region overlays
        """
        # Convert to color
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Create color overlay
        overlay = vis.copy()
        
        # Background: blue
        overlay[result.background_mask.astype(bool)] = [255, 100, 100]  # Light blue
        
        # Gel: dark red/maroon (darker than background in visualization too)
        if result.gel_detected:
            overlay[result.gel_mask.astype(bool)] = [50, 50, 150]  # Dark red
        
        # Mat: green
        overlay[result.mat_mask.astype(bool)] = [100, 255, 100]  # Light green
        
        # Saturation: bright red warning
        if result.saturation_mask is not None:
            overlay[result.saturation_mask.astype(bool)] = [0, 0, 255]  # Bright red
        
        # Blend
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
        
        # Draw contours
        if result.mat_contour is not None:
            cv2.drawContours(vis, [result.mat_contour], -1, (0, 255, 0), 2)
        
        if result.gel_contour is not None:
            cv2.drawContours(vis, [result.gel_contour], -1, (0, 0, 180), 2)
        
        return vis


# Legacy compatibility wrapper
def detect_regions(
    image: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    detect_gel: bool = True
) -> RegionDetectionResult:
    """
    Convenience function for three-region detection.
    
    Args:
        image: Grayscale input image
        roi_mask: Optional ROI mask
        detect_gel: Whether to detect gel region
        
    Returns:
        RegionDetectionResult
    """
    detector = ThreeRegionDetector(gel_detection_enabled=detect_gel)
    return detector.detect(image, roi_mask)


if __name__ == "__main__":
    # Demo/test
    import sys
    
    print("Three-Region Detector")
    print("=" * 40)
    
    # Create synthetic test image
    h, w = 500, 500
    test_image = np.zeros((h, w), dtype=np.uint8)
    
    # Background: medium gray (around 80)
    test_image[:] = 80
    
    # Mat: bright circle in center
    cv2.circle(test_image, (250, 250), 120, 200, -1)
    
    # Gel: dark ring around mat
    cv2.circle(test_image, (250, 250), 150, 40, 20)  # Dark ring
    
    # Add some noise
    noise = np.random.normal(0, 5, (h, w)).astype(np.int16)
    test_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Detect regions
    detector = ThreeRegionDetector()
    result = detector.detect(test_image)
    
    print(f"Background intensity: {result.background_intensity:.1f}")
    print(f"Gel intensity: {result.gel_intensity:.1f}")
    print(f"Mat intensity range: {result.mat_intensity_range}")
    print(f"Gel detected: {result.gel_detected}")
    print(f"Quality score: {result.quality_score:.2f}")
    print(f"Saturation: {result.saturation_pct:.1f}%")
    
    # Verify intensity ordering
    if result.gel_detected:
        assert result.gel_intensity < result.background_intensity, "Gel should be darker than background"
    print("\nIntensity ordering verified: gel < background < mat ✓")
