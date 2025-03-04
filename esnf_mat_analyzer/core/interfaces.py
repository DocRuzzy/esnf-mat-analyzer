"""
Core interfaces for the ESNF Mat Analyzer.

This module defines the core interfaces and protocols for the nanofiber thickness
uniformity analysis system, following SOLID design principles. These interfaces
establish the contract between different components of the system, enabling
loose coupling and dependency injection.
"""

from abc import ABC, abstractmethod
from typing import (
    Protocol,
    Tuple,
    Dict,
    List,
    Optional,
    Any,
    TypeVar,
    runtime_checkable,
)
import numpy as np
from pathlib import Path

# Type definitions
Image = np.ndarray
Mask = np.ndarray
ThicknessMap = np.ndarray
Point = Tuple[int, int]
Metrics = Dict[str, float]
Figure = Any  # Matplotlib figure or similar


@runtime_checkable
class ImageProcessorInterface(Protocol):
    """
    Interface for image loading and preprocessing.

    Responsible for loading images from files and performing preprocessing
    operations such as noise reduction, contrast enhancement, and grayscale
    conversion.
    """

    def load_image(self, path: Path) -> Image:
        """
        Load an image from the specified path.

        Args:
            path: Path to the image file

        Returns:
            Loaded image as a numpy array

        Raises:
            FileNotFoundError: If the image file does not exist
            ValueError: If the image cannot be loaded
        """
        ...

    def preprocess(self, image: Image) -> Image:
        """
        Preprocess the image for analysis.

        Applies operations like noise reduction, contrast enhancement,
        and grayscale conversion.

        Args:
            image: Input image

        Returns:
            Preprocessed image ready for analysis
        """
        ...


@runtime_checkable
class CircleDetectorInterface(Protocol):
    """
    Interface for detecting circular regions in images.

    Responsible for identifying and extracting the circular region of interest
    in nanofiber mat images.
    """

    def detect(self, image: Image) -> Tuple[Point, int]:
        """
        Detect the circular region in the image.

        Args:
            image: Preprocessed image

        Returns:
            Tuple of ((center_x, center_y), radius)

        Raises:
            ValueError: If no circular region can be detected
        """
        ...

    def create_mask(
        self, image_shape: Tuple[int, int], center: Point, radius: int
    ) -> Mask:
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


@runtime_checkable
class ThicknessEstimatorInterface(Protocol):
    """
    Interface for estimating thickness from image brightness.

    Responsible for converting brightness values to estimated thickness
    using various models (linear, logarithmic, exponential).
    """

    def estimate(self, image: Image, mask: Mask) -> ThicknessMap:
        """
        Estimate thickness from image brightness.

        Args:
            image: Preprocessed grayscale image
            mask: Binary mask indicating the region of interest

        Returns:
            2D thickness map
        """
        ...

    def get_saturation_mask(self, image: Image, mask: Mask) -> Mask:
        """
        Create a mask of saturated pixels.

        Identifies pixels that have reached brightness saturation,
        which may affect thickness estimation accuracy.

        Args:
            image: Preprocessed grayscale image
            mask: Binary mask indicating the region of interest

        Returns:
            Binary mask where 1 indicates saturated pixels
        """
        ...


class UniformityMetricInterface(ABC):
    """
    Abstract base class for uniformity metrics.

    Provides a common interface for different metrics that quantify
    the uniformity of nanofiber thickness.
    """

    @abstractmethod
    def calculate(
        self, thickness_map: ThicknessMap, mask: Mask, center: Point
    ) -> float:
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

    @property
    def higher_is_better(self) -> bool:
        """
        Indicates whether higher values of this metric indicate better uniformity.

        Returns:
            True if higher values indicate better uniformity, False otherwise
        """
        return True  # Default implementation


@runtime_checkable
class VisualizerInterface(Protocol):
    """
    Interface for data visualization.

    Responsible for creating visual representations of thickness maps
    and uniformity metrics.
    """

    def create_thickness_heatmap(
        self,
        thickness_map: ThicknessMap,
        mask: Mask,
        saturation_mask: Optional[Mask] = None,
    ) -> Figure:
        """
        Create a heatmap visualization of the thickness map.

        Args:
            thickness_map: 2D thickness map
            mask: Binary mask indicating the region of interest
            saturation_mask: Optional mask of saturated pixels

        Returns:
            Figure object (e.g., matplotlib figure)
        """
        ...

    def create_radial_profile(
        self, thickness_map: ThicknessMap, mask: Mask, center: Point
    ) -> Figure:
        """
        Create a visualization of thickness along radial lines.

        Args:
            thickness_map: 2D thickness map
            mask: Binary mask indicating the region of interest
            center: Center coordinates (x, y) of the circular region

        Returns:
            Figure object (e.g., matplotlib figure)
        """
        ...

    def create_uniformity_visualization(
        self, thickness_map: ThicknessMap, mask: Mask, metrics: Metrics
    ) -> Figure:
        """
        Create a visualization of uniformity metrics.

        Args:
            thickness_map: 2D thickness map
            mask: Binary mask indicating the region of interest
            metrics: Dictionary of metric names and values

        Returns:
            Figure object (e.g., matplotlib figure)
        """
        ...


@runtime_checkable
class DataExporterInterface(Protocol):
    """
    Interface for exporting data.

    Responsible for saving analysis results to various file formats.
    """

    def export_thickness_map(self, thickness_map: ThicknessMap, path: Path) -> None:
        """
        Export the thickness map to a file.

        Args:
            thickness_map: 2D thickness map
            path: Output file path

        Raises:
            IOError: If the file cannot be written
        """
        ...

    def export_metrics(self, metrics: Metrics, path: Path) -> None:
        """
        Export uniformity metrics to a file.

        Args:
            metrics: Dictionary of metric names and values
            path: Output file path

        Raises:
            IOError: If the file cannot be written
        """
        ...

    def export_summary(
        self, image_path: Path, center: Point, radius: int, metrics: Metrics, path: Path
    ) -> None:
        """
        Export a summary of the analysis results.

        Args:
            image_path: Path to the original image
            center: Center coordinates (x, y) of the circular region
            radius: Radius of the circular region
            metrics: Dictionary of metric names and values
            path: Output file path

        Raises:
            IOError: If the file cannot be written
        """
        ...


class AnalyzerInterface(ABC):
    """
    Interface for the main analyzer class.

    Coordinates the overall analysis process, from image loading to
    result generation.
    """

    @abstractmethod
    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Process a single image and calculate uniformity metrics.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing analysis results including:
            - image_path: Original image path
            - center: Detected center coordinates
            - radius: Detected radius
            - metrics: Dictionary of uniformity metrics
            - results_dir: Directory containing result files

        Raises:
            FileNotFoundError: If the image file does not exist
            ValueError: If the image cannot be processed
        """
        pass

    @abstractmethod
    def batch_process(
        self, image_dir: Path, pattern: str = "*.jpg"
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images in a directory.

        Args:
            image_dir: Directory containing images
            pattern: File pattern for matching images

        Returns:
            List of results dictionaries for each processed image
        """
        pass
