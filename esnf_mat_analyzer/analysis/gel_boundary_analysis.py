"""
Gel Boundary Shape Analysis for ESNF Mat Analyzer.

Analyzes the shape characteristics of the gel boundary (wet polymer perimeter)
and investigates its influence on nanofiber mat uniformity.

Metrics Include:
- Circularity (how circular is the gel boundary)
- Irregularity (boundary roughness/waviness)
- Area and perimeter
- Center offset (is gel centered in image/ROI)
- Aspect ratio (elongation)

These metrics help researchers understand how gel boundary shape
correlates with mat deposition uniformity.

Author: ESNF Mat Analyzer Team
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Any
import logging


@dataclass
class GelBoundaryMetrics:
    """Metrics describing gel boundary shape characteristics."""
    
    # Basic geometry
    area_pixels: float = 0.0  # Gel region area in pixels
    perimeter_pixels: float = 0.0  # Gel boundary length in pixels
    
    # Shape metrics
    circularity: float = 0.0  # 4π × area / perimeter² (1.0 = perfect circle)
    irregularity: float = 0.0  # Boundary roughness (0 = smooth)
    aspect_ratio: float = 1.0  # Major/minor axis ratio of fitted ellipse
    solidity: float = 0.0  # Area / convex hull area
    
    # Position metrics
    centroid: Tuple[float, float] = (0.0, 0.0)  # (x, y) center of gel
    center_offset: float = 0.0  # Distance from image center (normalized)
    
    # Derived metrics
    equivalent_diameter: float = 0.0  # Diameter of circle with same area
    
    # Quality flags
    valid: bool = False
    notes: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.notes is None:
            self.notes = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "area_pixels": self.area_pixels,
            "perimeter_pixels": self.perimeter_pixels,
            "circularity": self.circularity,
            "irregularity": self.irregularity,
            "aspect_ratio": self.aspect_ratio,
            "solidity": self.solidity,
            "centroid_x": self.centroid[0],
            "centroid_y": self.centroid[1],
            "center_offset": self.center_offset,
            "equivalent_diameter": self.equivalent_diameter,
            "valid": self.valid,
        }


@dataclass
class GelUniformityCorrelation:
    """Correlation between gel shape and mat uniformity."""
    
    # Gel metrics
    gel_metrics: Optional[GelBoundaryMetrics] = None
    
    # Uniformity metrics within gel-adjacent region
    edge_uniformity_cv: float = 0.0  # CV of thickness near gel boundary
    edge_thickness_mean: float = 0.0  # Mean thickness near gel
    
    # Uniformity gradient (edge vs center)
    edge_to_center_ratio: float = 0.0  # Edge thickness / center thickness
    
    # Correlation indicators
    boundary_effect_strength: float = 0.0  # How much gel boundary affects uniformity
    
    # Notes
    interpretation: str = ""


class GelBoundaryAnalyzer:
    """
    Analyzer for gel boundary shape and its influence on mat uniformity.
    
    The gel boundary is the perimeter of wet polymer that defines the
    collection zone. Its shape may influence fiber deposition patterns.
    """
    
    def __init__(
        self,
        edge_region_width: int = 30,
    ):
        """
        Initialize analyzer.
        
        Args:
            edge_region_width: Width in pixels of the region near gel boundary
                               to analyze for edge effects
        """
        self.edge_region_width = edge_region_width
        self.logger = logging.getLogger(__name__)
    
    def analyze_gel_shape(
        self,
        gel_mask: np.ndarray,
        image_shape: Tuple[int, int],
        gel_contour: Optional[np.ndarray] = None
    ) -> GelBoundaryMetrics:
        """
        Analyze shape characteristics of gel boundary.
        
        Args:
            gel_mask: Binary mask of gel region
            image_shape: (height, width) of original image
            gel_contour: Optional pre-computed contour
            
        Returns:
            GelBoundaryMetrics with shape analysis
        """
        h, w = image_shape
        image_center = (w / 2, h / 2)
        
        # Validate input
        if gel_mask is None or not np.any(gel_mask):
            self.logger.warning("Empty or invalid gel mask")
            return GelBoundaryMetrics(valid=False, notes=["No gel region detected"])
        
        # Get contour if not provided
        if gel_contour is None:
            contours, _ = cv2.findContours(
                gel_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return GelBoundaryMetrics(valid=False, notes=["Could not extract gel contour"])
            gel_contour = max(contours, key=cv2.contourArea)
        
        # Basic measurements
        area = cv2.contourArea(gel_contour)
        perimeter = cv2.arcLength(gel_contour, closed=True)
        
        if area <= 0 or perimeter <= 0:
            return GelBoundaryMetrics(valid=False, notes=["Invalid area or perimeter"])
        
        # Circularity: 4π × area / perimeter² (perfect circle = 1.0)
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        circularity = min(circularity, 1.0)  # Cap at 1.0 due to discretization
        
        # Equivalent diameter
        equivalent_diameter = np.sqrt(4 * area / np.pi)
        
        # Fit ellipse for aspect ratio (need at least 5 points)
        if len(gel_contour) >= 5:
            ellipse = cv2.fitEllipse(gel_contour)
            (cx, cy), (minor_axis, major_axis), angle = ellipse
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
            centroid = (cx, cy)
        else:
            M = cv2.moments(gel_contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                centroid = (cx, cy)
            else:
                centroid = image_center
            aspect_ratio = 1.0
        
        # Solidity: area / convex hull area
        hull = cv2.convexHull(gel_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        
        # Irregularity: measure boundary roughness
        irregularity = self._calculate_irregularity(gel_contour, equivalent_diameter)
        
        # Center offset (normalized by image diagonal)
        diagonal = np.sqrt(h**2 + w**2)
        center_offset = np.sqrt(
            (centroid[0] - image_center[0])**2 + 
            (centroid[1] - image_center[1])**2
        ) / diagonal
        
        metrics = GelBoundaryMetrics(
            area_pixels=area,
            perimeter_pixels=perimeter,
            circularity=circularity,
            irregularity=irregularity,
            aspect_ratio=aspect_ratio,
            solidity=solidity,
            centroid=centroid,
            center_offset=center_offset,
            equivalent_diameter=equivalent_diameter,
            valid=True,
            notes=[],
        )
        
        # Add interpretive notes (notes is guaranteed to be a list here)
        notes_list = metrics.notes if metrics.notes is not None else []
        if circularity < 0.7:
            notes_list.append("Low circularity - gel boundary is irregular")
        if aspect_ratio > 1.3:
            notes_list.append(f"Elongated gel shape (aspect ratio: {aspect_ratio:.2f})")
        if center_offset > 0.1:
            notes_list.append(f"Gel offset from image center by {100*center_offset:.1f}%")
        metrics.notes = notes_list
        
        self.logger.info(
            f"Gel shape analysis: circularity={circularity:.3f}, "
            f"irregularity={irregularity:.3f}, aspect_ratio={aspect_ratio:.2f}"
        )
        
        return metrics
    
    def analyze_uniformity_correlation(
        self,
        thickness_map: np.ndarray,
        mat_mask: np.ndarray,
        gel_mask: np.ndarray,
        gel_metrics: GelBoundaryMetrics
    ) -> GelUniformityCorrelation:
        """
        Analyze correlation between gel boundary and mat uniformity.
        
        Args:
            thickness_map: Thickness values within mat
            mat_mask: Mat region mask
            gel_mask: Gel region mask
            gel_metrics: Pre-computed gel shape metrics
            
        Returns:
            GelUniformityCorrelation with analysis results
        """
        result = GelUniformityCorrelation(gel_metrics=gel_metrics)
        
        if not gel_metrics.valid or not np.any(mat_mask):
            result.interpretation = "Unable to analyze - invalid gel or mat region"
            return result
        
        # Create edge region (mat pixels near gel boundary)
        gel_dilated = cv2.dilate(
            gel_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                      (self.edge_region_width, self.edge_region_width))
        )
        edge_region = (gel_dilated > 0) & mat_mask.astype(bool) & ~gel_mask.astype(bool)
        
        # Create center region (mat pixels far from gel)
        mat_eroded = cv2.erode(
            mat_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        )
        center_region = (mat_eroded > 0) & mat_mask.astype(bool)
        
        # Calculate edge uniformity
        if np.any(edge_region):
            edge_values = thickness_map[edge_region]
            result.edge_thickness_mean = float(np.mean(edge_values))
            result.edge_uniformity_cv = float(np.std(edge_values) / (np.mean(edge_values) + 1e-8))
        
        # Calculate edge-to-center ratio
        if np.any(center_region) and np.any(edge_region):
            center_mean = float(np.mean(thickness_map[center_region]))
            result.edge_to_center_ratio = result.edge_thickness_mean / (center_mean + 1e-8)
        
        # Estimate boundary effect strength
        # Based on how much edge region differs from center
        if result.edge_to_center_ratio > 0:
            deviation = abs(result.edge_to_center_ratio - 1.0)
            result.boundary_effect_strength = min(deviation, 1.0)
        
        # Generate interpretation
        interpretations = []
        
        if result.edge_to_center_ratio < 0.7:
            interpretations.append("Mat is significantly thinner near gel boundary")
        elif result.edge_to_center_ratio > 1.3:
            interpretations.append("Mat is thicker near gel boundary (edge accumulation)")
        else:
            interpretations.append("Relatively uniform thickness from edge to center")
        
        if result.edge_uniformity_cv > 0.3:
            interpretations.append("High thickness variation near gel boundary")
        
        if gel_metrics.circularity < 0.8 and result.boundary_effect_strength > 0.2:
            interpretations.append(
                "Irregular gel boundary may be contributing to non-uniform deposition"
            )
        
        result.interpretation = "; ".join(interpretations)
        
        self.logger.info(f"Gel-uniformity correlation: {result.interpretation}")
        
        return result
    
    def _calculate_irregularity(
        self,
        contour: np.ndarray,
        equivalent_diameter: float
    ) -> float:
        """
        Calculate boundary irregularity (roughness).
        
        Compares actual perimeter to ideal circle perimeter.
        Higher values = more irregular/rough boundary.
        """
        actual_perimeter = cv2.arcLength(contour, closed=True)
        ideal_perimeter = np.pi * equivalent_diameter
        
        if ideal_perimeter <= 0:
            return 0.0
        
        # Irregularity = (actual - ideal) / ideal
        # A perfect circle has irregularity ~0
        irregularity = (actual_perimeter - ideal_perimeter) / ideal_perimeter
        
        return max(0.0, irregularity)
    
    def visualize_analysis(
        self,
        image: np.ndarray,
        mat_mask: np.ndarray,
        gel_mask: np.ndarray,
        metrics: GelBoundaryMetrics
    ) -> np.ndarray:
        """
        Create visualization of gel boundary analysis.
        
        Args:
            image: Original grayscale image
            mat_mask: Mat region mask
            gel_mask: Gel region mask
            metrics: Computed gel metrics
            
        Returns:
            Color visualization image
        """
        # Convert to color
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Draw gel region outline
        contours, _ = cv2.findContours(
            gel_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)  # Red
        
        # Draw mat boundary
        mat_contours, _ = cv2.findContours(
            mat_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, mat_contours, -1, (0, 255, 0), 2)  # Green
        
        # Mark centroid
        if metrics.valid:
            cx, cy = int(metrics.centroid[0]), int(metrics.centroid[1])
            cv2.circle(vis, (cx, cy), 5, (255, 255, 0), -1)  # Cyan dot
            cv2.circle(vis, (cx, cy), int(metrics.equivalent_diameter / 2), (255, 255, 0), 1)
        
        # Add metrics text
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        
        if metrics.valid:
            texts = [
                f"Circularity: {metrics.circularity:.3f}",
                f"Irregularity: {metrics.irregularity:.3f}",
                f"Aspect Ratio: {metrics.aspect_ratio:.2f}",
                f"Solidity: {metrics.solidity:.3f}",
            ]
            for text in texts:
                cv2.putText(vis, text, (10, y_offset), font, font_scale, color, 1)
                y_offset += 20
        
        return vis


def analyze_gel_influence(
    image: np.ndarray,
    thickness_map: np.ndarray,
    mat_mask: np.ndarray,
    gel_mask: np.ndarray
) -> Dict[str, Any]:
    """
    Convenience function for complete gel influence analysis.
    
    Args:
        image: Original grayscale image
        thickness_map: Computed thickness values
        mat_mask: Mat region mask
        gel_mask: Gel region mask (can be empty if not detected)
        
    Returns:
        Dictionary with all analysis results
    """
    analyzer = GelBoundaryAnalyzer()
    
    # Analyze gel shape
    gel_metrics = analyzer.analyze_gel_shape(
        gel_mask, image.shape[:2]
    )
    
    # Analyze uniformity correlation
    correlation = analyzer.analyze_uniformity_correlation(
        thickness_map, mat_mask, gel_mask, gel_metrics
    )
    
    # Create visualization
    visualization = analyzer.visualize_analysis(
        image, mat_mask, gel_mask, gel_metrics
    )
    
    return {
        "gel_metrics": gel_metrics,
        "uniformity_correlation": correlation,
        "visualization": visualization,
        "summary": {
            "gel_detected": gel_metrics.valid,
            "circularity": gel_metrics.circularity,
            "boundary_effect_strength": correlation.boundary_effect_strength,
            "interpretation": correlation.interpretation,
        }
    }


if __name__ == "__main__":
    # Demo with synthetic data
    print("Gel Boundary Shape Analysis")
    print("=" * 40)
    
    # Create synthetic gel + mat image
    h, w = 500, 500
    
    # Mat mask (bright circle)
    mat_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mat_mask, (250, 250), 120, 1, -1)
    
    # Gel mask (irregular dark ring)
    gel_mask = np.zeros((h, w), dtype=np.uint8)
    # Create slightly irregular boundary
    pts = []
    for angle in np.linspace(0, 2*np.pi, 60):
        r = 145 + 10 * np.sin(5 * angle) + 5 * np.cos(3 * angle)
        x = int(250 + r * np.cos(angle))
        y = int(250 + r * np.sin(angle))
        pts.append([x, y])
    pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(gel_mask, [pts], 1)
    
    # Remove inner circle (mat area)
    inner = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(inner, (250, 250), 125, 1, -1)
    gel_mask = gel_mask & ~inner
    
    # Create synthetic thickness map
    thickness_map = np.zeros((h, w), dtype=np.float32)
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - 250)**2 + (y - 250)**2)
    thickness_map = np.clip(100 - dist * 0.5, 0, 100)
    thickness_map[~mat_mask.astype(bool)] = 0
    
    # Analyze
    analyzer = GelBoundaryAnalyzer()
    
    print("\nGel Shape Metrics:")
    metrics = analyzer.analyze_gel_shape(gel_mask, (h, w))
    print(f"  Circularity: {metrics.circularity:.3f}")
    print(f"  Irregularity: {metrics.irregularity:.3f}")
    print(f"  Aspect Ratio: {metrics.aspect_ratio:.2f}")
    print(f"  Solidity: {metrics.solidity:.3f}")
    print(f"  Centroid: ({metrics.centroid[0]:.1f}, {metrics.centroid[1]:.1f})")
    
    print("\nUniformity Correlation:")
    correlation = analyzer.analyze_uniformity_correlation(
        thickness_map, mat_mask, gel_mask, metrics
    )
    print(f"  Edge-to-center ratio: {correlation.edge_to_center_ratio:.2f}")
    print(f"  Boundary effect strength: {correlation.boundary_effect_strength:.2f}")
    print(f"  Interpretation: {correlation.interpretation}")
    
    if metrics.notes:
        print("\nNotes:")
        for note in metrics.notes:
            print(f"  - {note}")
