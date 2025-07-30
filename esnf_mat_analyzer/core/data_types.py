"""
Core data types for the ESNF Mat Analyzer.

This module defines data structures used throughout the system, including
configuration settings and analysis results. Most data structures are
implemented as dataclasses for simplicity and type safety.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import numpy as np
from enum import Enum, auto


class GrayscaleConversionMethod(Enum):
    """Methods for converting RGB images to grayscale."""

    WEIGHTED = auto()  # Standard weighted conversion (default in OpenCV)
    AVERAGE = auto()  # Simple average of channels
    LUMINANCE = auto()  # Perceptual luminance-preserving conversion


class ThicknessModelType(Enum):
    """Models for converting brightness to thickness."""

    LINEAR = auto()  # Linear model: thickness = a * brightness + b
    LOGARITHMIC = auto()  # Log model: thickness = a * log(1 + brightness) + b
    EXPONENTIAL = auto()  # Exp model: thickness = a * (exp(brightness / 255) - 1) + b


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

    blur_kernel_size: int = 5
    """Size of Gaussian blur kernel. Should be odd. Set to 0 to disable blurring."""

    contrast_alpha: float = 1.5
    """Contrast adjustment factor. Values > 1.0 increase contrast."""

    contrast_beta: int = 0
    """Brightness adjustment factor. Range: -255 to 255."""

    grayscale_conversion: GrayscaleConversionMethod = GrayscaleConversionMethod.WEIGHTED
    """Method used for converting RGB images to grayscale."""


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

    model_type: ThicknessModelType = ThicknessModelType.LINEAR
    """Model for converting brightness to thickness."""

    a: float = 1.0
    """Scaling factor for thickness model."""

    b: float = 0.0
    """Offset for thickness model."""

    saturation_threshold: int = 250
    """Pixel value threshold for saturation (0-255)."""

    normalization: bool = False
    """Whether to normalize thickness values to [0, 1] range."""

    calibration_factor: Optional[float] = None
    """Optional calibration factor for converting to absolute units (e.g., nm)."""


@dataclass
class UniformityConfig:
    """Configuration parameters for uniformity analysis."""

    num_radial_lines: int = 36
    """Number of radial lines for radial uniformity analysis."""

    bin_count: int = 50
    """Number of bins for histograms."""

    smoothing_factor: float = 0.5
    """Factor for smoothing radial profiles (0.0 to 1.0)."""

    min_thickness_percentile: float = 5.0
    """Percentile to use for minimum thickness in range calculations."""

    max_thickness_percentile: float = 95.0
    """Percentile to use for maximum thickness in range calculations."""

    ignore_saturated: bool = True
    """Whether to exclude saturated pixels from uniformity calculations."""


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

    radial_avg_line_color: str = "blue"
    """Color for the average line in radial profiles."""

    radial_std_fill_color: str = "skyblue"
    """Color for the standard deviation fill in radial profiles."""

    histogram_color: str = "skyblue"
    """Color for histogram bars."""

    histogram_edge_color: str = "black"
    """Color for histogram bar edges."""


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
    """Image processing configuration."""

    shape_detection: ShapeDetectionConfig = field(default_factory=ShapeDetectionConfig)
    """Shape detection configuration."""

    thickness: ThicknessConfig = field(default_factory=ThicknessConfig)
    """Thickness estimation configuration."""

    uniformity: UniformityConfig = field(default_factory=UniformityConfig)
    """Uniformity analysis configuration."""

    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    """Visualization configuration."""

    ruler_detection: RulerDetectionConfig = field(default_factory=RulerDetectionConfig)
    """Ruler detection configuration."""

    export: ExportConfig = field(default_factory=ExportConfig)
    """Export configuration."""

    output_dir: Path = Path("./output")
    """Directory for saving results."""

    log_level: int = logging.INFO
    """Logging level."""

    cache_intermediates: bool = True
    """Whether to cache intermediate results."""


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
    """Container for analysis results."""

    image_path: Path
    """Path to the original image."""

    contour: np.ndarray
    """Detected contour of the mat."""

    thickness_map: np.ndarray
    """Estimated thickness map."""

    mask: np.ndarray
    """Binary mask of the region of interest."""

    saturation_mask: Optional[np.ndarray] = None
    """Binary mask of saturated pixels."""

    metrics: Dict[str, float] = field(default_factory=dict)
    """Dictionary of uniformity metrics."""

    radial_profile: Optional[RadialProfile] = None
    """Radial profile data if calculated."""

    spatial_scale_pixels_per_mm: Optional[float] = None # <<< NEW FIELD

    results_dir: Optional[Path] = None
    """Directory containing result files."""

    processing_time: float = 0.0
    """Time taken for analysis in seconds."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    def __post_init__(self):
        """Validate the analysis result."""
        if hasattr(self, 'thickness_map') and hasattr(self, 'mask') and \
               self.thickness_map.shape != self.mask.shape:
                # Check existence of attributes because of potential partial mock objects in tests
            raise ValueError("Thickness map and mask must have the same shape")
