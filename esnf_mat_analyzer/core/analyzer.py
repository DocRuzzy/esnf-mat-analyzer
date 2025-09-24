"""
Advanced electrospun nanofiber mat uniformity analyzer.

This module provides the core analysis pipeline for quantitative assessment
of nanofiber mat uniformity using multiple complementary metrics including
traditional statistical measures, FFT-based anisotropy analysis, texture
analysis via GLCM, and power spectral density characterization.

The analyzer implements methods described in:

Author: ESNF Mat Analyzer Team
License: GNU General Public License v3.0 or later (GPLv3)
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from .interfaces import (
    AnalyzerInterface, ImageProcessorInterface,
    ThicknessEstimatorInterface, UniformityMetricInterface,
    VisualizerInterface, DataExporterInterface
)
from .data_types import Config, AnalysisResult
from ..processing.shape_detector import ShapeDetector
from ..processing.ruler_detector import RulerDetector
from ..analysis.mat_anisotropy import MatAnisotropyAnalyzer, MatTextureAnalyzer, PowerSpectralDensityAnalyzer
from ..analysis.multiscale_uniformity import MultiScaleUniformityAnalyzer

# Setup module logger
logger = logging.getLogger(__name__)

class NanoFiberAnalyzer(AnalyzerInterface):
    """
    Advanced electrospun nanofiber mat uniformity analyzer.
    
    This class provides comprehensive analysis of nanofiber mat uniformity using
    multiple complementary metrics including traditional statistical measures,
    FFT-based anisotropy analysis, texture analysis via GLCM, and power spectral
    density characterization.
    
    The analyzer implements a complete pipeline from image preprocessing through
    uniformity quantification, providing both individual metric values and
    composite uniformity scores suitable for comparative analysis and quality
    control applications.
    
    Attributes:
        config: Analysis configuration parameters
        image_processor: Image preprocessing and enhancement pipeline
        shape_detector: ROI detection and validation system
        thickness_estimator: Thickness map generation from intensity data
        ruler_detector: Spatial scale calibration system
        uniformity_metrics: Collection of uniformity calculation methods
        visualizer: Results visualization and plotting system
        data_exporter: Data export and reporting system
        
    Examples:
        >>> from esnf_mat_analyzer import NanoFiberAnalyzer, get_default_config
        >>> config = get_default_config()
        >>> analyzer = NanoFiberAnalyzer.from_config(config)
        >>> result = analyzer.process_image("nanofiber_mat.tif")
        >>> print(f"Overall uniformity: {result.metrics['overall_mat_uniformity']:.3f}")
        Overall uniformity: 0.782
        
        >>> # Batch processing
        >>> results = analyzer.batch_process("data/", pattern="*.tif")
        >>> avg_uniformity = np.mean([r.metrics['overall_mat_uniformity'] for r in results])
    """
    
    def __init__(
        self,
        config: Config,
        image_processor: ImageProcessorInterface,
        shape_detector: ShapeDetector,
        thickness_estimator: ThicknessEstimatorInterface,
        ruler_detector: RulerDetector,
        uniformity_metrics: List[UniformityMetricInterface],
        visualizer: VisualizerInterface,
        data_exporter: DataExporterInterface,
    ) -> None:
        """
        Initialize the ESNF analyzer with all required components.
        
        Args:
            config: Analysis configuration containing parameters for uniformity
                   metrics, image processing, and output options
            image_processor: Component for image preprocessing and enhancement
            shape_detector: Component for ROI detection and validation
            thickness_estimator: Component for thickness map generation
            ruler_detector: Component for spatial scale calibration
            uniformity_metrics: List of uniformity metric calculators
            visualizer: Component for results visualization
            data_exporter: Component for data export and reporting
                   
        Raises:
            ValueError: If any required component is None or invalid
            TypeError: If component types don't match expected interfaces
        """
        # Validate inputs
        if config is None:
            raise ValueError("Configuration cannot be None")
        if not uniformity_metrics:
            raise ValueError("At least one uniformity metric must be provided")
        
        self.config = config
        self.image_processor = image_processor
        self.shape_detector = shape_detector
        self.thickness_estimator = thickness_estimator
        self.ruler_detector = ruler_detector
        self.uniformity_metrics = uniformity_metrics
        self.visualizer = visualizer
        self.data_exporter = data_exporter
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(
            f"NanoFiberAnalyzer initialized with {len(uniformity_metrics)} "
            f"uniformity metrics and configuration: {config.__class__.__name__}"
        )

    def process_image(
        self, 
        image_path: Union[str, Path], 
        roi: Optional[Tuple[int, int, int, int]] = None,
        spatial_scale_pixels_per_mm: Optional[float] = None
    ) -> AnalysisResult:
        """
        Process a single nanofiber mat image for comprehensive uniformity analysis.
        
        This method applies the complete analysis pipeline including image
        preprocessing, ROI detection, thickness estimation, and uniformity
        quantification using multiple complementary metrics.
        
        Args:
            image_path: Path to the input image file (TIFF, PNG, JPEG supported).
                       Supports both string paths and Path objects.
            roi: Optional region of interest as (x, y, width, height).
                If None, attempts automatic ROI detection. Coordinates are
                in image pixel space with origin at top-left.
            spatial_scale_pixels_per_mm: Optional spatial calibration factor.
                If None, attempts automatic scale detection via ruler analysis.
                Used for metric normalization and real-world measurements.
                
        Returns:
            AnalysisResult containing:
                - Computed uniformity metrics dictionary
                - Processing metadata and timing information
                - Spatial calibration data
                - ROI coordinates and validation results
                - Error flags and quality indicators
                
        Raises:
            FileNotFoundError: If image_path does not exist
            ValueError: If image format is unsupported or ROI is invalid
            RuntimeError: If analysis fails due to computational errors
            
        Examples:
            >>> analyzer = NanoFiberAnalyzer.from_config(config)
            >>> result = analyzer.process_image("sample.tif")
            >>> uniformity = result.metrics['overall_mat_uniformity']
            >>> anisotropy = result.metrics['anisotropy_index']
            >>> print(f"Uniformity: {uniformity:.3f}, Anisotropy: {anisotropy:.3f}")
            
            >>> # With manual ROI
            >>> roi = (100, 100, 500, 500)  # x, y, width, height
            >>> result = analyzer.process_image("sample.tif", roi=roi)
            
            >>> # With known spatial scale
            >>> result = analyzer.process_image(
            ...     "sample.tif", 
            ...     spatial_scale_pixels_per_mm=50.0
            ... )
        """
        # Convert to Path object for consistent handling
        image_path = Path(image_path)
        
        # Validate input path
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not image_path.is_file():
            raise ValueError(f"Path is not a file: {image_path}")
        
        # Log analysis start
        self.logger.info(f"Processing image: {image_path}")
        processing_start_time = time.time()

        try:
            # 1. Load and validate image
            raw_image = self.image_processor.load_image(image_path)
            full_image_shape = raw_image.shape[:2]  # Store original dimensions
        except FileNotFoundError:
            self.logger.error(f"Image not found: {image_path}")
            raise
        except ValueError as ve:
            self.logger.error(f"Failed to load image {image_path}: {ve}")
            raise

        # 2. Ruler Detection (on FULL image, OUTSIDE user ROI if provided)
        ruler_mask = None
        if spatial_scale_pixels_per_mm:
            self.logger.info(f"Using provided spatial scale: {spatial_scale_pixels_per_mm:.2f} pixels/mm.")
        elif self.config.ruler_detection.enabled and self.ruler_detector:
            self.logger.info("Attempting scale detection on full image...")
            try:
                # Create exclusion mask for user ROI during ruler detection
                ruler_search_mask = None
                if roi:
                    ruler_search_mask = np.ones(full_image_shape, dtype=bool)
                    x, y, w, h = roi
                    ruler_search_mask[y:y+h, x:x+w] = False  # Exclude user ROI
                
                spatial_scale_pixels_per_mm, ruler_mask = self.ruler_detector.detect_scale_with_mask(
                    raw_image, exclusion_mask=ruler_search_mask
                )
                
                if spatial_scale_pixels_per_mm:
                    self.logger.info(f"Detected spatial scale: {spatial_scale_pixels_per_mm:.2f} pixels/mm.")
                else:
                    self.logger.warning("Scale detection did not yield a result.")
            except Exception as e:
                self.logger.error(f"Error during scale detection: {e}", exc_info=True)

        # 3. Background Correction (on FULL image, excluding rulers)
        full_preprocessed_image = self.image_processor.preprocess(raw_image)
        
        # Create comprehensive background correction exclusion mask
        bg_exclusion_mask = None
        if ruler_mask is not None:
            bg_exclusion_mask = ruler_mask
            self.logger.info("Excluding detected rulers from background correction")
        
        # Apply background correction to full image
        corrected_full_image = self.image_processor.apply_background_correction_with_exclusion(
            full_preprocessed_image, exclusion_mask=bg_exclusion_mask
        )

        # 4. Extract user ROI from corrected image
        if roi:
            x, y, w, h = roi
            roi_corrected_image = corrected_full_image[y:y+h, x:x+w]
            roi_raw_image = raw_image[y:y+h, x:x+w] if len(raw_image.shape) == 3 else raw_image[y:y+h, x:x+w]
            self.logger.info(f"Extracted user ROI: {roi} from corrected image")
        else:
            roi_corrected_image = corrected_full_image
            roi_raw_image = raw_image
            self.logger.info("Using full corrected image (no user ROI specified)")

        # 5. Detect Gel Shape (optimal ROI within user ROI)
        try:
            # Use the ROI-corrected image for gel detection
            gel_contour = self.shape_detector.detect(roi_corrected_image)
            if gel_contour is None:
                self.logger.warning("Gel shape detection failed, using full ROI")
                # Fallback: create mask for entire ROI
                gel_mask = np.ones(roi_corrected_image.shape, dtype=bool)
            else:
                gel_mask = self.shape_detector.create_mask(roi_corrected_image.shape, gel_contour)
                self.logger.info("Successfully detected gel boundary for optimal ROI")
        except Exception as e:
            self.logger.error(f"Error during gel shape detection: {e}", exc_info=True)
            # Fallback: use full ROI
            gel_contour = None
            gel_mask = np.ones(roi_corrected_image.shape, dtype=bool)
            self.logger.warning("Using full ROI as fallback for gel detection failure")

        # 6. Estimate Thickness (on corrected ROI image with gel mask)
        # ThicknessConfig (self.config.thickness) might have its own calibration_factor
        # set by user for converting model units to physical thickness (e.g., nm).
        # This is separate from the ruler's spatial_scale_pixels_per_mm.
        thickness_map = self.thickness_estimator.estimate(roi_corrected_image, gel_mask)
        saturation_mask = self.thickness_estimator.get_saturation_mask(roi_corrected_image, gel_mask)

        # 7. Calculate Uniformity Metrics (using gel mask for optimal accuracy)
        metrics: Dict[str, float] = {}
        
        # Traditional uniformity metrics
        for metric_calculator in self.uniformity_metrics:
            try:
                metric_value = metric_calculator.calculate(thickness_map, gel_mask, None)
                metrics[metric_calculator.name] = metric_value
                self.logger.debug(f"Calculated {metric_calculator.name}: {metric_value}")
            except Exception as e:
                self.logger.error(f"Error calculating metric {metric_calculator.name}: {e}", exc_info=True)
                metrics[metric_calculator.name] = float('nan')
        
        # Mat-scale uniformity analysis
        try:
            anisotropy_analyzer = MatAnisotropyAnalyzer()
            anisotropy_metrics = anisotropy_analyzer.analyze_mat_anisotropy(thickness_map, gel_mask)
            metrics.update(anisotropy_metrics)
            self.logger.debug(f"Anisotropy metrics calculated: {len(anisotropy_metrics)} metrics")

            texture_analyzer = MatTextureAnalyzer()
            texture_metrics = texture_analyzer.analyze_mat_texture(roi_corrected_image, gel_mask)
            metrics.update(texture_metrics)
            self.logger.debug(f"Texture metrics calculated: {len(texture_metrics)} metrics")

            psd_analyzer = PowerSpectralDensityAnalyzer()
            psd_metrics = psd_analyzer.analyze_psd(thickness_map, gel_mask)
            metrics.update(psd_metrics)
            self.logger.debug(f"PSD metrics calculated: {len(psd_metrics)} metrics")

        except Exception as e:
            self.logger.error(f"Error calculating advanced mat-scale uniformity metrics: {e}", exc_info=True)

        try:
            mat_analyzer = MultiScaleUniformityAnalyzer()
            mat_metrics = mat_analyzer.analyze_multiscale_uniformity(thickness_map, gel_mask)
            metrics.update(mat_metrics)
            self.logger.debug(f"Mat-scale uniformity metrics calculated: {len(mat_metrics)} metrics")
        except Exception as e:
            self.logger.error(f"Error calculating mat-scale uniformity metrics: {e}", exc_info=True)

        processing_duration = time.time() - processing_start_time
        self.logger.info(f"Image processing completed in {processing_duration:.2f} seconds.")

        # 8. Store and Return Results
        # analysis_metadata can be used for other metadata if needed in the future
        analysis_metadata: Dict[str, Any] = {
            'user_roi': roi,
            'gel_detection_successful': gel_contour is not None,
            'ruler_detected': ruler_mask is not None,
            'background_correction_applied': True,
            'full_image_shape': full_image_shape,
            'roi_shape': roi_corrected_image.shape
        }

        # Create results directory (example, could be done by data_exporter or visualizer)
        results_output_dir = Path(self.config.output_dir) / image_path.stem
        # results_output_dir.mkdir(parents=True, exist_ok=True) # Defer to exporter/visualizer

        if spatial_scale_pixels_per_mm:
            self.data_exporter.write_scale_to_metadata(image_path, spatial_scale_pixels_per_mm)

        result = AnalysisResult(
            image_path=image_path,
            contour=gel_contour,  # This is now the detected gel boundary
            thickness_map=thickness_map, # This is the (possibly) calibrated thickness map
            mask=gel_mask,  # This is now the optimal gel mask
            saturation_mask=saturation_mask,
            metrics=metrics,
            spatial_scale_pixels_per_mm=spatial_scale_pixels_per_mm, # <<< Use new field
            # radial_profile=None, # To be implemented if RadialProfile analysis is added
            results_dir=results_output_dir,
            processing_time=processing_duration,
            metadata=analysis_metadata # Pass other metadata here
        )

        # Placeholder for actual visualization and data export calls
        # if self.visualizer:
        #    self.visualizer.visualize_all(result, raw_image) # Example method
        # if self.data_exporter:
        #    self.data_exporter.export_all(result) # Example method

        return result

    def batch_process(
        self, 
        image_dir: Union[str, Path], 
        pattern: str = "*.jpg",
        max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple nanofiber mat images in batch for comparative uniformity analysis.
        
        This method applies the complete analysis pipeline to all images matching
        the specified pattern in the given directory. Results are returned as a
        summary list suitable for statistical analysis and comparative studies.
        
        Args:
            image_dir: Directory containing images to process. Supports both
                      string paths and Path objects.
            pattern: Glob pattern for image file selection. Defaults to "*.jpg".
                    Common patterns: "*.tif", "*.png", "*.jpg", "*.bmp"
            max_workers: Maximum number of parallel workers for processing.
                        If None, uses single-threaded processing for deterministic
                        results and memory management.
                        
        Returns:
            List of dictionaries containing:
                - image_path: Path to processed image
                - status: "success" or "error"
                - metrics: Computed uniformity metrics (if successful)
                - processing_time: Time taken for analysis
                - spatial_scale_pixels_per_mm: Detected or provided scale
                - results_dir: Output directory for results
                - error_message: Error details (if failed)
                
        Raises:
            ValueError: If image_dir does not exist or is not a directory
            PermissionError: If insufficient permissions to read directory
            
        Examples:
            >>> analyzer = NanoFiberAnalyzer.from_config(config)
            >>> results = analyzer.batch_process("data/samples/", "*.tif")
            >>> success_count = sum(1 for r in results if r['status'] == 'success')
            >>> print(f"Successfully processed {success_count}/{len(results)} images")
            
            >>> # Calculate batch statistics
            >>> uniformities = [r['metrics']['overall_mat_uniformity'] 
            ...                for r in results if r['status'] == 'success']
            >>> mean_uniformity = np.mean(uniformities)
            >>> std_uniformity = np.std(uniformities)
        """
        # Convert to Path object for consistent handling
        image_dir = Path(image_dir)
        
        # Validate input directory
        if not image_dir.exists():
            raise ValueError(f"Directory does not exist: {image_dir}")
        if not image_dir.is_dir():
            raise ValueError(f"Path is not a directory: {image_dir}")
        
        self.logger.info(
            f"Starting batch processing for directory: {image_dir} "
            f"with pattern: {pattern}"
        )
        
        batch_results_summary: List[Dict[str, Any]] = []
        image_files = sorted(list(image_dir.glob(pattern)))  # Sort for consistent order

        if not image_files:
            self.logger.warning(
                f"No images found in {image_dir} matching pattern {pattern}"
            )
            return []

        self.logger.info(f"Found {len(image_files)} images to process")
        
        for image_path in image_files:
            self.logger.info(f"Batch processing image: {image_path.name}")
            try:
                result_obj = self.process_image(image_path)
                # Convert AnalysisResult object to a dictionary for the summary list
                summary_item = {
                    "image_path": str(result_obj.image_path),
                    "status": "success",
                    "metrics": result_obj.metrics,
                    "processing_time": result_obj.processing_time,
                    "spatial_scale_pixels_per_mm": result_obj.spatial_scale_pixels_per_mm,
                    "results_dir": str(result_obj.results_dir)
                }
                batch_results_summary.append(summary_item)
                
            except Exception as e:
                self.logger.error(
                    f"Failed to process {image_path.name} in batch: {e}", 
                    exc_info=True
                )
                batch_results_summary.append({
                    "image_path": str(image_path),
                    "status": "error",
                    "error_message": str(e),
                    "processing_time": 0.0
                })

        success_count = sum(1 for r in batch_results_summary if r['status'] == 'success')
        self.logger.info(
            f"Batch processing completed. Successfully processed "
            f"{success_count}/{len(image_files)} images."
        )
        
        return batch_results_summary
    
    @classmethod
    def from_config(cls, config: Config) -> 'NanoFiberAnalyzer':
        """
        Create a NanoFiberAnalyzer instance from configuration using dependency injection.
        
        This factory method instantiates all required components based on the provided
        configuration and returns a fully initialized analyzer ready for processing.
        
        Args:
            config: Complete analysis configuration containing parameters for all
                   components including uniformity metrics, image processing, and
                   output options.
                   
        Returns:
            Fully initialized NanoFiberAnalyzer instance with all dependencies
            injected according to configuration.
            
        Raises:
            ValueError: If configuration is invalid or incomplete
            ImportError: If required dependencies are not available
            RuntimeError: If component initialization fails
            
        Examples:
            >>> from esnf_mat_analyzer import get_default_config
            >>> config = get_default_config()
            >>> analyzer = NanoFiberAnalyzer.from_config(config)
            >>> result = analyzer.process_image("sample.tif")
        """
        from ..main import setup_dependencies
        
        if config is None:
            raise ValueError("Configuration cannot be None")
            
        try:
            return setup_dependencies(config)
        except Exception as e:
            logger.error(f"Failed to create analyzer from config: {e}")
            raise RuntimeError(f"Analyzer initialization failed: {e}") from e
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported image formats for analysis.
        
        Returns:
            List of supported file extensions (with dots)
            
        Examples:
            >>> analyzer = NanoFiberAnalyzer.from_config(config)
            >>> formats = analyzer.get_supported_formats()
            >>> print(f"Supported formats: {', '.join(formats)}")
            Supported formats: .tif, .tiff, .png, .jpg, .jpeg, .bmp
        """
        return self.image_processor.get_supported_formats()
    
    def validate_image(self, image_path: Union[str, Path]) -> bool:
        """
        Validate if an image can be processed by this analyzer.
        
        Args:
            image_path: Path to image file to validate
            
        Returns:
            True if image can be processed, False otherwise
            
        Examples:
            >>> analyzer = NanoFiberAnalyzer.from_config(config)
            >>> if analyzer.validate_image("sample.tif"):
            ...     result = analyzer.process_image("sample.tif")
        """
        try:
            image_path = Path(image_path)
            
            # Check file existence
            if not image_path.exists() or not image_path.is_file():
                return False
                
            # Check format support
            if image_path.suffix.lower() not in self.get_supported_formats():
                return False
                
            # Try to load image headers
            return self.image_processor.validate_image(image_path)
            
        except Exception:
            return False
    
    def get_analysis_info(self) -> Dict[str, Any]:
        """
        Get information about the current analyzer configuration and capabilities.
        
        Returns:
            Dictionary containing analyzer configuration details, available
            metrics, and component information.
            
        Examples:
            >>> analyzer = NanoFiberAnalyzer.from_config(config)
            >>> info = analyzer.get_analysis_info()
            >>> print(f"Available metrics: {info['uniformity_metrics']}")
        """
        return {
            "version": "1.0.0",  # Could be read from package metadata
            "uniformity_metrics": [metric.name for metric in self.uniformity_metrics],
            "supported_formats": self.get_supported_formats(),
            "config_type": self.config.__class__.__name__,
            "ruler_detection_enabled": getattr(self.config, 'ruler_detection', {}).get('enabled', False),
            "thickness_estimation_method": getattr(self.config, 'thickness', {}).get('method', 'unknown'),
            "output_directory": str(getattr(self.config, 'output_dir', 'not_configured')),
        }
