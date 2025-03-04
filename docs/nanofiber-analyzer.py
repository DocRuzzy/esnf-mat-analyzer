"""
Nanofiber Thickness Uniformity Analysis System

This program converts color images of electrospun nanofiber (ESNF) mats into 2D thickness 
maps and quantifies their uniformity through multiple metrics.

The system follows SOLID design principles:
- Single Responsibility: Each class has one specific purpose
- Open/Closed: Components can be extended without modification
- Liskov Substitution: Derived classes are substitutable for base classes
- Interface Segregation: Clients only depend on methods they use
- Dependency Inversion: High-level modules depend on abstractions

Directory structure:
nanofiber_analyzer/
├── __init__.py
├── main.py                  # Entry point
├── config/                  # Configuration management
│   ├── __init__.py
│   └── config_manager.py    # Configuration handling
├── core/                    # Core interfaces and base classes
│   ├── __init__.py
│   ├── interfaces.py        # Define all interfaces
│   └── data_types.py        # Define data structures
├── processing/              # Image processing components
│   ├── __init__.py
│   ├── image_processor.py   # Image loading and preprocessing
│   ├── circle_detector.py   # Circle ROI detection
│   └── thickness_estimator.py # Conversion of brightness to thickness
├── analysis/                # Uniformity analysis components
│   ├── __init__.py
│   ├── radial_uniformity.py # Radial uniformity calculation
│   ├── gini_coefficient.py  # Gini coefficient calculation
│   └── thickness_ratio.py   # Thickness range ratio calculation
├── visualization/           # Visualization components
│   ├── __init__.py
│   ├── thickness_map.py     # Thickness map visualization
│   └── uniformity_plots.py  # Uniformity metrics visualization
├── utils/                   # Utility functions and classes
│   ├── __init__.py
│   ├── logger.py            # Logging utilities
│   └── exporters.py         # Data export functionality
└── tests/                   # Unit and integration tests
    ├── __init__.py
    ├── test_processing.py
    ├── test_analysis.py
    └── test_visualization.py
"""

# Core Interfaces

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Protocol, Any
import numpy as np
from pathlib import Path
import logging

# Data classes for configuration

@dataclass
class ProcessingConfig:
    """Configuration parameters for image preprocessing."""
    blur_kernel_size: int = 5
    contrast_alpha: float = 1.5  # Contrast adjustment factor
    contrast_beta: int = 0       # Brightness adjustment
    grayscale_conversion: str = "weighted"  # Method for RGB to grayscale conversion

@dataclass
class CircleDetectionConfig:
    """Configuration parameters for circle detection."""
    min_radius: int = 50
    max_radius: int = 500
    detection_method: str = "hough"  # "hough" or "contour"
    param1: int = 50  # For Hough transform
    param2: int = 30  # For Hough transform

@dataclass
class ThicknessConfig:
    """Configuration parameters for thickness estimation."""
    model_type: str = "linear"  # "linear", "logarithmic", or "exponential"
    a: float = 1.0             # Scaling factor
    b: float = 0.0             # Offset
    saturation_threshold: int = 250  # Pixel value threshold for saturation

@dataclass
class UniformityConfig:
    """Configuration parameters for uniformity analysis."""
    num_radial_lines: int = 36  # Number of radial lines for analysis
    bin_count: int = 50         # Number of bins for histograms
    smoothing_factor: float = 0.5  # For smoothing radial profiles

@dataclass
class VisualizationConfig:
    """Configuration parameters for data visualization."""
    colormap: str = "viridis"
    dpi: int = 300
    figure_size: Tuple[int, int] = (10, 8)
    show_saturated: bool = True  # Whether to highlight saturated regions

@dataclass
class Config:
    """Main configuration container."""
    processing: ProcessingConfig = ProcessingConfig()
    circle_detection: CircleDetectionConfig = CircleDetectionConfig()
    thickness: ThicknessConfig = ThicknessConfig()
    uniformity: UniformityConfig = UniformityConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    output_dir: Path = Path("./output")
    log_level: int = logging.INFO

# Core interfaces

class ImageProcessorInterface(Protocol):
    """Interface for image loading and preprocessing."""
    
    def load_image(self, path: Path) -> np.ndarray:
        """
        Load an image from the specified path.
        
        Args:
            path: Path to the image file
            
        Returns:
            Loaded image as a numpy array
        """
        ...
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for analysis.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image ready for analysis
        """
        ...

class CircleDetectorInterface(Protocol):
    """Interface for detecting circular regions in images."""
    
    def detect(self, image: np.ndarray) -> Tuple[Tuple[int, int], int]:
        """
        Detect the circular region in the image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Tuple of ((center_x, center_y), radius)
        """
        ...
    
    def create_mask(self, image_shape: Tuple[int, int], center: Tuple[int, int], radius: int) -> np.ndarray:
        """
        Create a binary mask for the circular region.
        
        Args:
            image_shape: Shape of the image (height, width)
            center: Center coordinates (x, y)
            radius: Radius of the circle
            
        Returns:
            Binary mask where 1 indicates the circular region
        """
        ...

class ThicknessEstimatorInterface(Protocol):
    """Interface for estimating thickness from image brightness."""
    
    def estimate(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Estimate thickness from image brightness.
        
        Args:
            image: Preprocessed grayscale image
            mask: Binary mask indicating the region of interest
            
        Returns:
            2D thickness map
        """
        ...
    
    def get_saturation_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Create a mask of saturated pixels.
        
        Args:
            image: Preprocessed grayscale image
            mask: Binary mask indicating the region of interest
            
        Returns:
            Binary mask where 1 indicates saturated pixels
        """
        ...

class UniformityMetricInterface(ABC):
    """Interface for uniformity metrics."""
    
    @abstractmethod
    def calculate(self, thickness_map: np.ndarray, mask: np.ndarray, center: Tuple[int, int]) -> float:
        """
        Calculate the uniformity metric.
        
        Args:
            thickness_map: 2D thickness map
            mask: Binary mask indicating the region of interest
            center: Center coordinates (x, y) of the circular region
            
        Returns:
            Uniformity metric value
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the metric."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Get a description of what the metric measures."""
        pass

class VisualizerInterface(Protocol):
    """Interface for data visualization."""
    
    def create_thickness_heatmap(self, thickness_map: np.ndarray, mask: np.ndarray, 
                                saturation_mask: Optional[np.ndarray] = None) -> Any:
        """
        Create a heatmap visualization of the thickness map.
        
        Args:
            thickness_map: 2D thickness map
            mask: Binary mask indicating the region of interest
            saturation_mask: Optional mask of saturated pixels
            
        Returns:
            Figure object
        """
        ...
    
    def create_radial_profile(self, thickness_map: np.ndarray, mask: np.ndarray, 
                             center: Tuple[int, int]) -> Any:
        """
        Create a visualization of thickness along radial lines.
        
        Args:
            thickness_map: 2D thickness map
            mask: Binary mask indicating the region of interest
            center: Center coordinates (x, y) of the circular region
            
        Returns:
            Figure object
        """
        ...
    
    def create_uniformity_visualization(self, thickness_map: np.ndarray, mask: np.ndarray,
                                      metrics: Dict[str, float]) -> Any:
        """
        Create a visualization of uniformity metrics.
        
        Args:
            thickness_map: 2D thickness map
            mask: Binary mask indicating the region of interest
            metrics: Dictionary of metric names and values
            
        Returns:
            Figure object
        """
        ...

class DataExporterInterface(Protocol):
    """Interface for exporting data."""
    
    def export_thickness_map(self, thickness_map: np.ndarray, path: Path) -> None:
        """
        Export the thickness map to a file.
        
        Args:
            thickness_map: 2D thickness map
            path: Output file path
        """
        ...
    
    def export_metrics(self, metrics: Dict[str, float], path: Path) -> None:
        """
        Export uniformity metrics to a file.
        
        Args:
            metrics: Dictionary of metric names and values
            path: Output file path
        """
        ...
    
    def export_summary(self, image_path: Path, center: Tuple[int, int], radius: int,
                      metrics: Dict[str, float], path: Path) -> None:
        """
        Export a summary of the analysis results.
        
        Args:
            image_path: Path to the original image
            center: Center coordinates (x, y) of the circular region
            radius: Radius of the circular region
            metrics: Dictionary of metric names and values
            path: Output file path
        """
        ...

# Main analyzer class that coordinates all components

class NanoFiberAnalyzer:
    """Main class for analyzing nanofiber thickness and uniformity."""
    
    def __init__(self, 
                 config: Config,
                 image_processor: ImageProcessorInterface,
                 circle_detector: CircleDetectorInterface,
                 thickness_estimator: ThicknessEstimatorInterface,
                 uniformity_metrics: List[UniformityMetricInterface],
                 visualizer: VisualizerInterface,
                 data_exporter: DataExporterInterface):
        """
        Initialize the analyzer with its components.
        
        Args:
            config: Configuration parameters
            image_processor: Component for image loading and preprocessing
            circle_detector: Component for detecting circular regions
            thickness_estimator: Component for estimating thickness
            uniformity_metrics: List of uniformity metrics to calculate
            visualizer: Component for creating visualizations
            data_exporter: Component for exporting data
        """
        self.config = config
        self.image_processor = image_processor
        self.circle_detector = circle_detector
        self.thickness_estimator = thickness_estimator
        self.uniformity_metrics = uniformity_metrics
        self.visualizer = visualizer
        self.data_exporter = data_exporter
        
        # Set up logging
        logging.basicConfig(level=config.log_level,
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Process a single image and calculate uniformity metrics.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info(f"Processing image: {image_path}")
        
        # Load and preprocess the image
        image = self.image_processor.load_image(image_path)
        preprocessed = self.image_processor.preprocess(image)
        
        # Detect the circular region
        center, radius = self.circle_detector.detect(preprocessed)
        mask = self.circle_detector.create_mask(preprocessed.shape[:2], center, radius)
        
        # Estimate thickness
        thickness_map = self.thickness_estimator.estimate(preprocessed, mask)
        saturation_mask = self.thickness_estimator.get_saturation_mask(preprocessed, mask)
        
        # Calculate uniformity metrics
        metrics = {}
        for metric in self.uniformity_metrics:
            value = metric.calculate(thickness_map, mask, center)
            metrics[metric.name] = value
            self.logger.info(f"{metric.name}: {value:.4f}")
        
        # Create visualizations
        results_dir = self.config.output_dir / image_path.stem
        results_dir.mkdir(exist_ok=True, parents=True)
        
        thickness_heatmap = self.visualizer.create_thickness_heatmap(
            thickness_map, mask, saturation_mask)
        thickness_heatmap.savefig(
            results_dir / "thickness_heatmap.png", 
            dpi=self.config.visualization.dpi)
        
        radial_profile = self.visualizer.create_radial_profile(
            thickness_map, mask, center)
        radial_profile.savefig(
            results_dir / "radial_profile.png", 
            dpi=self.config.visualization.dpi)
        
        uniformity_viz = self.visualizer.create_uniformity_visualization(
            thickness_map, mask, metrics)
        uniformity_viz.savefig(
            results_dir / "uniformity_metrics.png", 
            dpi=self.config.visualization.dpi)
        
        # Export data
        self.data_exporter.export_thickness_map(
            thickness_map, results_dir / "thickness_map.csv")
        self.data_exporter.export_metrics(
            metrics, results_dir / "metrics.json")
        self.data_exporter.export_summary(
            image_path, center, radius, metrics, 
            results_dir / "summary.txt")
        
        self.logger.info(f"Analysis complete for {image_path}")
        
        return {
            "image_path": str(image_path),
            "center": center,
            "radius": radius,
            "metrics": metrics,
            "results_dir": str(results_dir)
        }
    
    def batch_process(self, image_dir: Path, pattern: str = "*.jpg") -> List[Dict[str, Any]]:
        """
        Process multiple images in a directory.
        
        Args:
            image_dir: Directory containing images
            pattern: File pattern for matching images
            
        Returns:
            List of results dictionaries for each processed image
        """
        self.logger.info(f"Batch processing images in {image_dir} matching {pattern}")
        image_paths = list(image_dir.glob(pattern))
        self.logger.info(f"Found {len(image_paths)} images to process")
        
        results = []
        for path in image_paths:
            try:
                result = self.process_image(path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {path}: {e}")
        
        return results
