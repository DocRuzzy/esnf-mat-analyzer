import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from .interfaces import (
    AnalyzerInterface, ImageProcessorInterface, CircleDetectorInterface,
    ThicknessEstimatorInterface, UniformityMetricInterface,
    VisualizerInterface, DataExporterInterface
)
from .data_types import Config, AnalysisResult, Point
from ..processing.ruler_detector import RulerDetector
# Ensure other necessary types like Image, Mask, etc. are available if used directly
# For now, they are mostly encapsulated in other processors' return types.

class NanoFiberAnalyzer(AnalyzerInterface):
    """
    Main orchestrator for the nanofiber analysis pipeline.
    """
    def __init__(
        self,
        config: Config,
        image_processor: ImageProcessorInterface,
        circle_detector: CircleDetectorInterface,
        thickness_estimator: ThicknessEstimatorInterface,
        ruler_detector: RulerDetector, # Concrete class, not interface here
        uniformity_metrics: List[UniformityMetricInterface],
        visualizer: VisualizerInterface,
        data_exporter: DataExporterInterface,
    ):
        self.config = config
        self.image_processor = image_processor
        self.circle_detector = circle_detector
        self.thickness_estimator = thickness_estimator
        self.ruler_detector = ruler_detector
        self.uniformity_metrics = uniformity_metrics
        self.visualizer = visualizer # Stored, but not used in this phase
        self.data_exporter = data_exporter # Stored, but not used in this phase
        self.logger = logging.getLogger(__name__)
        self.logger.info("NanoFiberAnalyzer initialized.")

    def process_image(self, image_path: Path) -> AnalysisResult: # Changed return to AnalysisResult
        """
        Processes a single image to analyze nanofiber mat properties.
        """
        self.logger.info(f"Processing image: {image_path}")
        processing_start_time = time.time()

        # 1. Load Image (raw image for ruler detection and primary processing)
        try:
            raw_image = self.image_processor.load_image(image_path)
        except FileNotFoundError:
            self.logger.error(f"Image not found: {image_path}")
            raise
        except ValueError as ve:
            self.logger.error(f"Failed to load image {image_path}: {ve}")
            raise

        # 2. Ruler Detection (on raw image, if enabled)
        spatial_scale_pixels_per_mm: Optional[float] = None
        if self.config.ruler_detection.enabled and self.ruler_detector:
            self.logger.info("Attempting scale detection...")
            try:
                spatial_scale_pixels_per_mm = self.ruler_detector.detect_scale(raw_image)
                if spatial_scale_pixels_per_mm:
                    self.logger.info(f"Detected spatial scale: {spatial_scale_pixels_per_mm:.2f} pixels/mm.")
                else:
                    self.logger.warning("Scale detection did not yield a result.")
            except Exception as e:
                self.logger.error(f"Error during scale detection: {e}", exc_info=True)

        # 3. Preprocess Image (for circle detection and thickness estimation)
        # It's important that image_processor.preprocess takes the raw_image (RGB)
        # and returns a grayscale image suitable for downstream tasks.
        preprocessed_image = self.image_processor.preprocess(raw_image)

        # 4. Detect Circle
        try:
            center, radius = self.circle_detector.detect(preprocessed_image)
            mask = self.circle_detector.create_mask(preprocessed_image.shape, center, radius)
        except Exception as e:
            self.logger.error(f"Error during circle detection: {e}", exc_info=True)
            # Depending on desired robustness, could raise or return a partial/error result
            raise ValueError(f"Circle detection failed for {image_path}") from e


        # 5. Estimate Thickness
        # ThicknessConfig (self.config.thickness) might have its own calibration_factor
        # set by user for converting model units to physical thickness (e.g., nm).
        # This is separate from the ruler's spatial_scale_pixels_per_mm.
        thickness_map = self.thickness_estimator.estimate(preprocessed_image, mask)
        saturation_mask = self.thickness_estimator.get_saturation_mask(preprocessed_image, mask)

        # 6. Calculate Uniformity Metrics
        metrics: Dict[str, float] = {}
        for metric_calculator in self.uniformity_metrics:
            try:
                metric_value = metric_calculator.calculate(thickness_map, mask, center)
                metrics[metric_calculator.name] = metric_value
                self.logger.debug(f"Calculated {metric_calculator.name}: {metric_value}")
            except Exception as e:
                self.logger.error(f"Error calculating metric {metric_calculator.name}: {e}", exc_info=True)
                metrics[metric_calculator.name] = float('nan')

        processing_duration = time.time() - processing_start_time
        self.logger.info(f"Image processing completed in {processing_duration:.2f} seconds.")

        # 7. Store and Return Results
        # analysis_metadata can be used for other metadata if needed in the future
        analysis_metadata: Dict[str, Any] = {}
        # Add other useful config details to metadata if needed, e.g.:
        # analysis_metadata['ruler_config_used'] = self.config.ruler_detection

        # Create results directory (example, could be done by data_exporter or visualizer)
        results_output_dir = self.config.output_dir / image_path.stem
        # results_output_dir.mkdir(parents=True, exist_ok=True) # Defer to exporter/visualizer

        result = AnalysisResult(
            image_path=image_path,
            center=center,
            radius=radius,
            thickness_map=thickness_map, # This is the (possibly) calibrated thickness map
            mask=mask,
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

    def batch_process(self, image_dir: Path, pattern: str = "*.jpg") -> List[Dict[str, Any]]:
        """
        Processes multiple images in a directory.
        Returns a list of dictionaries for summary purposes, compatible with main.py.
        """
        self.logger.info(f"Starting batch processing for directory: {image_dir} with pattern: {pattern}")
        batch_results_summary: List[Dict[str, Any]] = []
        image_files = sorted(list(image_dir.glob(pattern))) # Sort for consistent order

        if not image_files:
            self.logger.warning(f"No images found in {image_dir} matching pattern {pattern}")
            return []

        for image_path in image_files:
            self.logger.info(f"Batch processing image: {image_path.name}")
            try:
                result_obj = self.process_image(image_path)
                # Convert AnalysisResult object to a dictionary for the summary list
                # This can be expanded to include more details if needed.
                summary_item = {
                    "image_path": str(result_obj.image_path),
                    "status": "success",
                    "center": result_obj.center,
                    "radius": result_obj.radius,
                    "metrics": result_obj.metrics,
                    "processing_time": result_obj.processing_time,
                    "spatial_scale_pixels_per_mm": result_obj.spatial_scale_pixels_per_mm, # <<< Access new field
                    "results_dir": str(result_obj.results_dir)
                }
                batch_results_summary.append(summary_item)
            except Exception as e:
                self.logger.error(f"Failed to process {image_path.name} in batch: {e}", exc_info=True)
                batch_results_summary.append({
                    "image_path": str(image_path),
                    "status": "error",
                    "error_message": str(e)
                })

        self.logger.info(f"Batch processing finished. Processed {len(image_files)} images.")
        return batch_results_summary
