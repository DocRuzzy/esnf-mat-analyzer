"""
Core data types and configuration structures for the ESNF Mat Analyzer.

This module defines comprehensive data structures used throughout the electrospun
nanofiber mat analysis system, including configuration settings, analysis results,
and intermediate data containers. All structures implement validation and type
safety to ensure robust operation in scientific analysis workflows.

The module provides:
- Configuration classes for all analysis components
- Enumeration types for method selection
- Result containers with validation
- Type hints for enhanced code safety

Classes:
    Config: Main configuration container
    AnalysisResult: Container for analysis outputs
    RadialProfile: Radial analysis data structure
    Various *Config classes: Component-specific configurations

Enums:
    GrayscaleConversionMethod: RGB to grayscale conversion methods
    BackgroundCorrectionMethod: Background correction algorithms
    ThicknessModelType: Thickness estimation models

Author: ESNF Mat Analyzer Team
License: MIT
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import numpy as np
from enum import Enum, auto


class GrayscaleConversionMethod(Enum):
    """
    Methods for converting RGB images to grayscale.
    
    Each method balances different aspects of the original color information:
    - WEIGHTED: Standard perceptually-weighted conversion (0.299R + 0.587G + 0.114B)
    - AVERAGE: Simple arithmetic mean of RGB channels
    - LUMINANCE: CIE luminance formula for accurate brightness perception
    """

    WEIGHTED = auto()  # Standard weighted conversion (default in OpenCV)
    AVERAGE = auto()  # Simple average of channels
    LUMINANCE = auto()  # Perceptual luminance-preserving conversion


class BackgroundCorrectionMethod(Enum):
    """
    Methods for Step 2 of the scientific background correction workflow.
    
    This enum represents the choice of illumination modeling method within
    the complete 4-step process described in the guide:
    
    Step 1: Isolate Background Pixels (automatic)
    Step 2: Model Uneven Illumination (choice of method below)  
    Step 3: Correct Image (automatic division)
    Step 4: Analyze Mat Uniformity (automatic)

    - NONE: Skip background correction entirely
    - POLYNOMIAL_SURFACE: 2D polynomial surface fitting (Method A - Most Robust)
    - LARGE_KERNEL_BLUR: Large-kernel blurring (Method B - Simpler Alternative)
    - COMPLETE_WORKFLOW: Use the full 4-step process with polynomial fitting
    """
    NONE = auto()
    POLYNOMIAL_SURFACE = auto()
    LARGE_KERNEL_BLUR = auto()
    COMPLETE_WORKFLOW = auto()


class ThicknessModelType(Enum):
    """
    Models for converting image brightness/transmittance to physical thickness.
    
    Each model represents different physical assumptions about light-matter interaction:
    - LINEAR: Direct linear relationship (thickness ∝ intensity)
    - LOGARITHMIC: Logarithmic relationship for nonlinear opacity effects
    - EXPONENTIAL: Exponential decay model for high absorption materials
    - BEER_LAMBERT: Physics-based Beer-Lambert law for accurate thickness estimation
    """

    LINEAR = auto()  # Linear model: thickness = a * brightness + b
    LOGARITHMIC = auto()  # Log model: thickness = a * log(1 + brightness) + b
    EXPONENTIAL = auto()  # Exp model: thickness = a * (exp(brightness / 255) - 1) + b
    BEER_LAMBERT = auto()  # Beer-Lambert law: I = I₀ * exp(-α * t)


@dataclass
class LevelingConfig:
    """Configuration for background leveling."""

    enabled: bool = True
    """Enable or disable background leveling."""

    kernel_size: int = 25
    """Size of the kernel for morphological operations."""


@dataclass
class ProcessingConfig:
    """Configuration parameters for image preprocessing."""

    leveling: LevelingConfig = field(default_factory=LevelingConfig)
    """Background leveling configuration."""

    background_correction_method: BackgroundCorrectionMethod = BackgroundCorrectionMethod.NONE
    """Method for background correction."""

    blur_kernel_size: int = 5
    """Size of Gaussian blur kernel. Should be odd. Set to 0 to disable blurring."""

    contrast_alpha: float = 1.5
    """Contrast adjustment factor. Values > 1.0 increase contrast."""

    contrast_beta: int = 0
    """Brightness adjustment factor. Range: -255 to 255."""

    grayscale_conversion: GrayscaleConversionMethod = GrayscaleConversionMethod.WEIGHTED
    """Method used for converting RGB images to grayscale."""

    basic_correction_n_components: int = 1
    """Number of components for BaSiC correction."""

    rolling_ball_radius: int = 50
    """Radius for rolling ball background correction."""

    restore_percentile: float = 5.0
    """Percentile for RESTORE background correction."""

    homomorphic_cutoff: float = 30
    """Cutoff frequency for homomorphic filter."""

    homomorphic_g_low: float = 0.5
    """Low gain for homomorphic filter."""

    homomorphic_g_high: float = 2.0
    """High gain for homomorphic filter."""

    saturation_recovery: bool = False
    """Enable or disable saturation recovery."""


@dataclass
class ShapeDetectionConfig:
    """Configuration for shape detection."""

    min_area: int = 100
    """Minimum area for shape detection."""

    max_area: int = 50000
    """Maximum area for shape detection."""

    detection_method: str = "contour"
    """Shape detection method: 'contour', 'hough', or 'adaptive'."""

    approx_epsilon_ratio: float = 0.02
    """Epsilon ratio for contour approximation."""

    min_vertices: int = 3
    """Minimum number of vertices for polygon detection."""

    max_vertices: int = 20
    """Maximum number of vertices for polygon detection."""


@dataclass
class ThicknessConfig:
    """Configuration parameters for thickness estimation."""

    model_type: ThicknessModelType = ThicknessModelType.BEER_LAMBERT
    """Model for converting brightness to thickness."""

    a: float = 1.0
    """Scaling factor for thickness model."""

    b: float = 0.0
    """Offset for thickness model."""

    saturation_threshold: int = 240
    """Pixel value threshold for saturation (0-255). Values at or above this are considered saturated."""

    normalization: bool = False
    """Whether to normalize thickness values to [0, 1] range."""

    calibration_factor: Optional[float] = None
    """Optional calibration factor for converting to absolute units (e.g., nm)."""

    # Beer-Lambert specific parameters
    attenuation_coefficient: float = 0.04778
    """Attenuation coefficient for Beer-Lambert law (literature-validated value)."""
    
    reference_intensity: Optional[float] = None
    """Background intensity (I₀). If None, auto-detected from image."""
    
    min_transmittance: float = 0.01
    """Minimum transmittance to prevent log(0) errors."""
    
    thickness_range_um: Tuple[float, float] = (0.0, 1000.0)
    """Valid thickness range in micrometers."""
    
    spatial_scale_um_per_pixel: Optional[float] = None
    """Spatial scale for absolute thickness measurements."""


@dataclass

@dataclass
class UniformityConfig:
    """Configuration parameters for uniformity analysis."""

    num_radial_lines: int = 36
    bin_count: int = 50
    smoothing_factor: float = 0.5
    min_thickness_percentile: float = 5.0
    max_thickness_percentile: float = 95.0
    ignore_saturated: bool = True
    lbp_radius: int = 1
    lbp_points: int = 8
    max_coefficients: int = 100000
    """Maximum number of coefficients to use per scale in multiscale uniformity (prevents OOM)."""


@dataclass
class VisualizationConfig:
    """Configuration parameters for data visualization."""

    colormap: str = "viridis"
    """Colormap for heatmaps (e.g., 'viridis', 'plasma', 'inferno')."""

    dpi: int = 300
    """Output image resolution (DPI)."""

    figure_size: Tuple[int, int] = (10, 8)
    """Figure size in inches (width, height)."""

    show_saturated: bool = True
    """Whether to highlight saturated regions in visualizations."""

    heatmap_percentile_range: Tuple[float, float] = (2.0, 98.0)
    """Percentile range for heatmap color scaling (min_percentile, max_percentile)."""

    auto_range_heatmap: bool = True
    """Whether to automatically adjust heatmap color range based on data percentiles."""

    radial_avg_line_color: str = "blue"
    """Color for the average line in radial profiles."""

    radial_std_fill_color: str = "skyblue"
    """Color for the standard deviation fill in radial profiles."""

    histogram_color: str = "skyblue"
    """Color for histogram bars."""

    histogram_edge_color: str = "black"
    """Color for histogram bar edges."""


@dataclass
class RulerScoringConfig:
    """Parameters for the scoring-based ruler body detection model."""
    enabled: bool = True
    """Enable or disable the scoring-based model in favor of the legacy model."""
    min_thickness_px: int = 5
    """Minimum plausible thickness for a ruler body in pixels."""
    max_thickness_px: int = 50
    """Maximum plausible thickness for a ruler body in pixels."""
    length_weight: float = 1.0
    """Weight for the combined length of the ruler lines in the score."""
    overlap_weight: float = 1.5
    """Weight for the horizontal overlap of the ruler lines in the score."""
    tick_count_weight: float = 3.0
    """Weight for the number of detected ticks between the lines in the score."""
    parallelism_penalty: float = 2.0
    """Penalty multiplier for lines that are not parallel."""

@dataclass
class RulerDetectionConfig:
    """Configuration parameters for ruler detection and scale calibration."""

    enabled: bool = True
    """Enable or disable ruler detection."""

    min_line_length: int = 50
    """Minimum length of lines to be considered part of a ruler (in pixels)."""

    max_line_gap: int = 10
    """Maximum allowed gap between line segments to treat them as a single line."""

    expected_tick_distance_mm: float = 1.0
    """The expected real-world distance between major ticks being searched for (e.g., 1mm, 5mm, 10mm)."""

    canny_threshold1: int = 50
    """First threshold for the Canny edge detector."""

    canny_threshold2: int = 150
    """Second threshold for the Canny edge detector."""

    hough_threshold: int = 20
    """Accumulator threshold parameter for Hough Line Transform."""

    min_tick_separation_px: int = 5
    """Minimum pixel distance between two adjacent ticks to be considered distinct."""

    tick_y_tolerance_factor: float = 1.5
    """Factor to extend the search for ticks vertically beyond the ruler body (e.g., 1.5 means 150% of ruler thickness)."""

    min_tick_length_factor: float = 0.3
    """Minimum length of a tick as a factor of the ruler's thickness."""

    abs_min_tick_length_px: int = 5
    """Absolute minimum length of a tick in pixels, overrides the factor if larger."""

    max_tick_length_factor: float = 3.0
    """Maximum length of a tick as a factor of the ruler's thickness."""

    scoring: RulerScoringConfig = field(default_factory=RulerScoringConfig)
    """Configuration for the scoring-based ruler detection model."""


@dataclass
class Tick:
    """Represents a single detected tick mark on a ruler."""
    x_position: float
    """The x-coordinate of the tick mark in the image."""
    line: np.ndarray
    """The (x1, y1, x2, y2) coordinates of the line segment forming the tick."""

@dataclass
class Ruler:
    """
    Represents a detected ruler, its components, and the calculated scale.

    This data structure holds all information related to a ruler found in an
    image, making it easier to pass this information between different parts of
    the analysis pipeline.
    """
    body_lines: Tuple[np.ndarray, np.ndarray]
    """A tuple containing the two main parallel lines that form the ruler's body."""

    ticks: List[Tick]
    """A list of `Tick` objects detected on the ruler."""

    scale_px_per_mm: Optional[float]
    """The calculated scale in pixels per millimeter. None if calculation failed."""

    mask: np.ndarray
    """A boolean numpy array with the same dimensions as the image, where True
    indicates the area occupied by the detected ruler."""

    roi: Tuple[int, int, int, int]
    """The bounding box (x_start, y_start, x_end, y_end) of the region where the
    ruler was detected."""


@dataclass
class ExportConfig:
    """Configuration parameters for data export."""

    csv_delimiter: str = ","
    """Delimiter for CSV files."""

    export_formats: List[str] = field(default_factory=lambda: ["csv", "json", "txt"])
    """Formats to export data in."""

    compress_outputs: bool = False
    """Whether to compress output files."""

    include_metadata: bool = True
    """Whether to include metadata in exports."""


@dataclass
class Config:
    """Main configuration container."""
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    shape_detection: ShapeDetectionConfig = field(default_factory=ShapeDetectionConfig)
    thickness: ThicknessConfig = field(default_factory=ThicknessConfig)
    uniformity: UniformityConfig = field(default_factory=UniformityConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    ruler_detection: RulerDetectionConfig = field(default_factory=RulerDetectionConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    output_dir: str = "output"
    log_level: str = "INFO"


@dataclass
class RadialProfile:
    """Data structure for storing radial profile information."""

    angles: np.ndarray
    """Angles of radial lines (in radians)."""

    distances: List[np.ndarray]
    """List of distances along each radial line."""

    thickness_values: List[np.ndarray]
    """List of thickness values along each radial line."""

    avg_thickness: np.ndarray
    """Average thickness at each distance from center."""

    std_thickness: np.ndarray
    """Standard deviation of thickness at each distance from center."""

    distance_bins: np.ndarray
    """Distance bins for averaged profile."""

    @property
    def coefficient_of_variation(self) -> float:
        """
        Calculate the overall coefficient of variation.

        Returns:
            Coefficient of variation as a measure of non-uniformity
        """
        if np.mean(self.avg_thickness) == 0:
            return float("inf")
        return float(np.std(self.avg_thickness) / np.mean(self.avg_thickness))


@dataclass
class AnalysisResult:
    """
    Comprehensive container for nanofiber mat analysis results.
    
    This class encapsulates all outputs from the analysis pipeline including
    quantitative metrics, intermediate data products, and metadata necessary
    for result interpretation and reproducibility.
    
    The container includes validation to ensure data consistency and provides
    utility methods for result interpretation and export preparation.
    
    Attributes:
        image_path: Path to the original input image
        contour: Detected boundary contour of the nanofiber mat
        thickness_map: 2D array of estimated thickness values
        mask: Binary mask defining the region of interest
        saturation_mask: Binary mask identifying saturated pixels
        metrics: Dictionary of computed uniformity and quality metrics
        radial_profile: Radial analysis data if computed
        spatial_scale_pixels_per_mm: Spatial calibration factor
        results_dir: Directory containing exported result files
        processing_time: Total analysis time in seconds
        metadata: Additional analysis metadata and parameters
        
    Examples:
        >>> result = analyzer.process_image("sample.tif")
        >>> print(f"Overall uniformity: {result.get_metric('overall_mat_uniformity'):.3f}")
        >>> print(f"Processing took {result.processing_time:.2f} seconds")
        >>> 
        >>> # Check for saturated regions
        >>> if result.has_saturation():
        ...     print(f"Warning: {result.saturation_percentage():.1f}% of pixels are saturated")
        >>>
        >>> # Export summary
        >>> summary = result.get_summary()
        >>> print(summary)
    """

    image_path: Path
    """Path to the original image file."""

    contour: np.ndarray
    """Detected contour of the nanofiber mat as (N, 1, 2) array."""

    thickness_map: np.ndarray
    """2D array of estimated thickness values in configured units."""

    mask: np.ndarray
    """Binary mask of the region of interest (True = foreground)."""

    saturation_mask: Optional[np.ndarray] = None
    """Binary mask identifying saturated pixels (True = saturated)."""

    metrics: Dict[str, float] = field(default_factory=dict)
    """Dictionary of computed uniformity and analysis metrics."""

    radial_profile: Optional[RadialProfile] = None
    """Radial profile analysis data if computed."""

    spatial_scale_pixels_per_mm: Optional[float] = None
    """Spatial calibration factor (pixels per millimeter)."""

    results_dir: Optional[Path] = None
    """Directory containing exported result files."""

    processing_time: float = 0.0
    """Total analysis time in seconds."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional analysis metadata and configuration parameters."""

    def __post_init__(self) -> None:
        """
        Validate the analysis result data consistency.
        
        Raises:
            ValueError: If data shapes are inconsistent or values are invalid
            TypeError: If data types are incorrect
        """
        # Validate required attributes exist and have correct types
        if not isinstance(self.image_path, Path):
            raise TypeError("image_path must be a Path object")
            
        if not isinstance(self.thickness_map, np.ndarray):
            raise TypeError("thickness_map must be a numpy array")
            
        if not isinstance(self.mask, np.ndarray):
            raise TypeError("mask must be a numpy array")
            
        # Validate shape consistency
        if self.thickness_map.shape != self.mask.shape:
            raise ValueError(
                f"Thickness map shape {self.thickness_map.shape} must match "
                f"mask shape {self.mask.shape}"
            )
            
        # Validate saturation mask if present
        if self.saturation_mask is not None:
            if self.saturation_mask.shape != self.mask.shape:
                raise ValueError(
                    f"Saturation mask shape {self.saturation_mask.shape} must match "
                    f"mask shape {self.mask.shape}"
                )
                
        # Validate processing time
        if self.processing_time < 0:
            raise ValueError("processing_time cannot be negative")
            
        # Validate spatial scale if present
        if self.spatial_scale_pixels_per_mm is not None:
            if self.spatial_scale_pixels_per_mm <= 0:
                raise ValueError("spatial_scale_pixels_per_mm must be positive")
    
    def get_metric(self, metric_name: str, default: Optional[float] = None) -> Optional[float]:
        """
        Safely retrieve a metric value by name.
        
        Args:
            metric_name: Name of the metric to retrieve
            default: Default value if metric not found
            
        Returns:
            Metric value or default if not found
            
        Examples:
            >>> uniformity = result.get_metric('overall_mat_uniformity', 0.0)
            >>> anisotropy = result.get_metric('anisotropy_index')
        """
        return self.metrics.get(metric_name, default)
    
    def has_saturation(self) -> bool:
        """
        Check if the analysis detected any saturated pixels.
        
        Returns:
            True if saturated pixels were detected, False otherwise
        """
        return (self.saturation_mask is not None and 
                np.any(self.saturation_mask))
    
    def saturation_percentage(self) -> float:
        """
        Calculate the percentage of pixels that are saturated within the ROI.
        
        Returns:
            Percentage of saturated pixels (0-100)
        """
        if not self.has_saturation():
            return 0.0
            
        roi_pixels = np.sum(self.mask)
        saturated_pixels = np.sum(self.saturation_mask & self.mask)
        
        return (saturated_pixels / roi_pixels) * 100.0 if roi_pixels > 0 else 0.0
    
    def get_physical_scale(self) -> Optional[str]:
        """
        Get a string representation of the physical scale if available.
        
        Returns:
            Scale description string or None if no scale available
            
        Examples:
            >>> scale = result.get_physical_scale()
            >>> print(f"Image scale: {scale}")
            Image scale: 50.2 pixels/mm (0.020 mm/pixel)
        """
        if self.spatial_scale_pixels_per_mm is None:
            return None
            
        mm_per_pixel = 1.0 / self.spatial_scale_pixels_per_mm
        return (f"{self.spatial_scale_pixels_per_mm:.1f} pixels/mm "
                f"({mm_per_pixel:.3f} mm/pixel)")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the analysis results.
        
        Returns:
            Dictionary containing key result information
            
        Examples:
            >>> summary = result.get_summary()
            >>> for key, value in summary.items():
            ...     print(f"{key}: {value}")
        """
        summary = {
            "image_file": self.image_path.name,
            "processing_time_sec": round(self.processing_time, 2),
            "roi_area_pixels": int(np.sum(self.mask)),
            "has_spatial_scale": self.spatial_scale_pixels_per_mm is not None,
            "has_saturation": self.has_saturation(),
            "metric_count": len(self.metrics),
        }
        
        if self.spatial_scale_pixels_per_mm is not None:
            summary["spatial_scale"] = self.get_physical_scale()
            
        if self.has_saturation():
            summary["saturation_percentage"] = round(self.saturation_percentage(), 1)
            
        # Add key metrics if available
        key_metrics = [
            'overall_mat_uniformity',
            'anisotropy_index', 
            'texture_energy',
            'radial_uniformity_index',
            'total_relative_roughness'
        ]
        
        for metric in key_metrics:
            value = self.get_metric(metric)
            if value is not None:
                summary[metric] = round(value, 4)
        
        return summary
    
    def validate_completeness(self) -> List[str]:
        """
        Validate result completeness and return any warnings.
        
        Returns:
            List of warning messages for incomplete or problematic results
        """
        warnings = []
        
        if not self.metrics:
            warnings.append("No metrics calculated")
            
        if self.saturation_percentage() > 10.0:
            warnings.append(f"High saturation: {self.saturation_percentage():.1f}%")
            
        if self.spatial_scale_pixels_per_mm is None:
            warnings.append("No spatial scale detected - results are in pixel units")
            
        roi_area = np.sum(self.mask)
        if roi_area < 1000:  # Minimum reasonable ROI size
            warnings.append(f"Small ROI detected: {roi_area} pixels")
            
        return warnings
