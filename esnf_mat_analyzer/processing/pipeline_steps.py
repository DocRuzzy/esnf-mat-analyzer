"""
Step-by-step processing pipeline for ESNF Mat Analyzer.

Provides individual processing steps that can be run and visualized
independently, enabling step-by-step verification of the analysis pipeline.

Each step returns both results and a visualization image for display in the GUI.

Author: ESNF Mat Analyzer Team
"""

import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from ..processing.region_detector import FourRegionDetector, FourRegionDetectionResult
from ..processing.image_processor import ImageProcessor
from ..core.data_types import ProcessingConfig, BackgroundCorrectionMethod


logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result from a single pipeline step."""
    success: bool = True
    error_message: str = ""
    # Visualization image (BGR, for display in GUI canvas)
    display_image: Optional[np.ndarray] = None
    # Any data produced by this step
    data: Dict[str, Any] = field(default_factory=dict)


def load_and_convert_grayscale(image_path: str) -> StepResult:
    """
    Step 1: Load image and convert to grayscale.

    Args:
        image_path: Path to the image file

    Returns:
        StepResult with:
            data['raw_image']: Original RGB image
            data['grayscale']: Grayscale image
            display_image: Grayscale image for display
    """
    try:
        path = Path(image_path)
        if not path.exists():
            return StepResult(success=False, error_message=f"File not found: {image_path}")

        raw = cv2.imread(str(path))
        if raw is None:
            return StepResult(success=False, error_message=f"Could not load image: {image_path}")

        # Convert to RGB for internal use
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        # Convert to grayscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        # Display image: grayscale converted back to BGR for canvas
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        logger.info(f"Loaded image: {gray.shape}, range [{gray.min()}-{gray.max()}]")

        return StepResult(
            success=True,
            display_image=display,
            data={
                'raw_image': rgb,
                'grayscale': gray,
                'image_path': str(image_path),
                'shape': gray.shape,
                'intensity_range': (int(gray.min()), int(gray.max())),
                'mean_intensity': float(np.mean(gray)),
            }
        )
    except Exception as e:
        logger.error(f"Error loading image: {e}", exc_info=True)
        return StepResult(success=False, error_message=str(e))


def detect_regions(
    grayscale: np.ndarray,
    roi_mask: Optional[np.ndarray] = None
) -> StepResult:
    """
    Step 2: Detect four regions (background, gel, mat, ruler).

    Args:
        grayscale: Grayscale image (from step 1)
        roi_mask: Optional ROI mask to restrict detection

    Returns:
        StepResult with:
            data['detection_result']: FourRegionDetectionResult
            data['region_stats']: Dict with region statistics
            display_image: Color overlay showing detected regions
    """
    try:
        detector = FourRegionDetector(
            mat_threshold_method="otsu",
            gel_detection_enabled=True,
            ruler_detection_enabled=True,
        )

        result = detector.detect(grayscale, roi_mask=roi_mask)

        # Create visualization overlay
        display = detector.visualize_regions(grayscale, result, alpha=0.4)

        # Add legend text
        h, w = grayscale.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(h, w) / 1500)
        thickness = max(1, int(min(h, w) / 500))
        y_pos = int(h * 0.03)
        line_height = int(30 * font_scale / 0.4)

        legend_items = [
            ("Background", (255, 150, 100), f"intensity={result.background_intensity:.1f}"),
            ("Mat", (100, 255, 100), f"range={result.mat_intensity_range[0]:.0f}-{result.mat_intensity_range[1]:.0f}"),
        ]
        if result.gel_detected:
            legend_items.insert(1, ("Gel", (150, 50, 100), f"intensity={result.gel_intensity:.1f}"))
        if result.ruler_detected:
            legend_items.append(("Ruler", (128, 128, 128), "detected"))

        for label, color, info in legend_items:
            y_pos += line_height
            cv2.putText(display, f"{label}: {info}", (10, y_pos),
                       font, font_scale, color, thickness)

        # Quality score
        y_pos += line_height
        cv2.putText(display, f"Quality: {result.quality_score:.2f}",
                   (10, y_pos), font, font_scale, (255, 255, 255), thickness)

        # Region statistics
        total_pixels = h * w
        stats = {
            'mat_pct': float(100 * np.sum(result.mat_mask > 0) / total_pixels),
            'background_pct': float(100 * np.sum(result.background_mask > 0) / total_pixels),
            'gel_pct': float(100 * np.sum(result.gel_mask > 0) / total_pixels) if result.gel_detected else 0.0,
            'ruler_pct': float(100 * np.sum(result.ruler_mask > 0) / total_pixels) if result.ruler_detected else 0.0,
            'background_intensity': result.background_intensity,
            'gel_intensity': result.gel_intensity,
            'mat_intensity_range': result.mat_intensity_range,
            'gel_detected': result.gel_detected,
            'ruler_detected': result.ruler_detected,
            'quality_score': result.quality_score,
            'saturation_pct': result.saturation_pct,
        }

        logger.info(f"Region detection: mat={stats['mat_pct']:.1f}%, bg={stats['background_pct']:.1f}%, "
                    f"gel={'yes' if result.gel_detected else 'no'}, quality={result.quality_score:.2f}")

        return StepResult(
            success=True,
            display_image=display,
            data={
                'detection_result': result,
                'region_stats': stats,
            }
        )
    except Exception as e:
        logger.error(f"Error in region detection: {e}", exc_info=True)
        return StepResult(success=False, error_message=str(e))


def correct_background(
    grayscale: np.ndarray,
    detection_result: FourRegionDetectionResult,
    method: str = "polynomial_surface",
    processing_config: Optional[ProcessingConfig] = None,
) -> StepResult:
    """
    Step 3: Apply background correction using detected regions.

    Args:
        grayscale: Grayscale image
        detection_result: Region detection result (from step 2)
        method: Correction method name
        processing_config: Optional processing config. If None, uses defaults.

    Returns:
        StepResult with:
            data['corrected']: Background-corrected image
            data['correction_stats']: Before/after statistics
            display_image: Side-by-side original vs. corrected
    """
    try:
        if processing_config is None:
            processing_config = ProcessingConfig()
            processing_config.leveling.enabled = True

        # Map method string to enum
        method_map = {
            'none': BackgroundCorrectionMethod.NONE,
            'polynomial_surface': BackgroundCorrectionMethod.POLYNOMIAL_SURFACE,
            'large_kernel_blur': BackgroundCorrectionMethod.LARGE_KERNEL_BLUR,
            'complete_workflow': BackgroundCorrectionMethod.COMPLETE_WORKFLOW,
        }
        processing_config.background_correction_method = method_map.get(
            method.lower(), BackgroundCorrectionMethod.POLYNOMIAL_SURFACE
        )

        # Create processor and run correction
        processor = ImageProcessor(processing_config)

        # Build exclusion mask from ruler
        exclusion_mask = None
        if detection_result.ruler_detected:
            exclusion_mask = detection_result.ruler_mask.astype(bool)

        if processing_config.background_correction_method == BackgroundCorrectionMethod.NONE:
            corrected = grayscale.copy()
        else:
            # Need to give it the grayscale image as if preprocessed
            processing_config.leveling.enabled = True
            corrected = processor.apply_background_correction_with_exclusion(
                grayscale, exclusion_mask=exclusion_mask
            )

        # Statistics
        mat_mask = detection_result.mat_mask.astype(bool)
        bg_mask = detection_result.background_mask.astype(bool)

        before_stats = {
            'mat_mean': float(np.mean(grayscale[mat_mask])) if np.any(mat_mask) else 0,
            'mat_std': float(np.std(grayscale[mat_mask])) if np.any(mat_mask) else 0,
            'bg_mean': float(np.mean(grayscale[bg_mask])) if np.any(bg_mask) else 0,
            'bg_std': float(np.std(grayscale[bg_mask])) if np.any(bg_mask) else 0,
        }
        after_stats = {
            'mat_mean': float(np.mean(corrected[mat_mask])) if np.any(mat_mask) else 0,
            'mat_std': float(np.std(corrected[mat_mask])) if np.any(mat_mask) else 0,
            'bg_mean': float(np.mean(corrected[bg_mask])) if np.any(bg_mask) else 0,
            'bg_std': float(np.std(corrected[bg_mask])) if np.any(bg_mask) else 0,
        }

        # Assess correction quality
        from ..processing.image_processor import assess_correction_quality
        quality_metrics = {}
        if np.any(mat_mask):
            try:
                quality_metrics = assess_correction_quality(
                    grayscale, corrected, mat_mask
                )
                logger.info(
                    f"Correction quality: bg_cv={quality_metrics.get('background_cv', 'N/A'):.3f}, "
                    f"mat_preservation={quality_metrics.get('mat_preservation', 'N/A'):.3f}, "
                    f"gradient_corr={quality_metrics.get('gradient_correlation', 'N/A'):.3f}"
                )
            except Exception as e:
                logger.warning(f"Could not assess correction quality: {e}")

        # Create side-by-side display
        h, w = grayscale.shape
        orig_bgr = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        corr_bgr = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(h, w) / 1500)
        thickness = max(1, int(min(h, w) / 500))
        cv2.putText(orig_bgr, "Original", (10, 25), font, font_scale, (0, 255, 255), thickness)
        cv2.putText(corr_bgr, "Corrected", (10, 25), font, font_scale, (0, 255, 255), thickness)

        # Add quality info to corrected image
        if quality_metrics:
            import math
            y_info = 50
            line_h = int(22 * font_scale / 0.4)
            mp = quality_metrics.get('mat_preservation', float('nan'))
            gc = quality_metrics.get('gradient_correlation', float('nan'))
            if not math.isnan(mp):
                cv2.putText(corr_bgr, f"Mat preservation: {mp:.3f}",
                           (10, y_info), font, font_scale * 0.8, (0, 255, 255), thickness)
                y_info += line_h
            if not math.isnan(gc):
                cv2.putText(corr_bgr, f"Gradient corr: {gc:.3f}",
                           (10, y_info), font, font_scale * 0.8, (0, 255, 255), thickness)

        # Draw a thin divider line
        divider = np.full((h, 3, 3), 128, dtype=np.uint8)
        display = np.hstack([orig_bgr, divider, corr_bgr])

        logger.info(f"Background correction ({method}): "
                    f"mat mean {before_stats['mat_mean']:.1f}->{after_stats['mat_mean']:.1f}, "
                    f"bg std {before_stats['bg_std']:.1f}->{after_stats['bg_std']:.1f}")

        return StepResult(
            success=True,
            display_image=display,
            data={
                'corrected': corrected,
                'before_stats': before_stats,
                'after_stats': after_stats,
                'method': method,
                'quality_metrics': quality_metrics,
            }
        )
    except Exception as e:
        logger.error(f"Error in background correction: {e}", exc_info=True)
        return StepResult(success=False, error_message=str(e))


def create_intensity_map(
    image: np.ndarray,
    mat_mask: np.ndarray,
    saturation_threshold: int = 254,
) -> StepResult:
    """
    Step 4: Create intensity heat map of mat region.

    Maps raw pixel intensities within the mat to a colormap for visualization.
    No thickness model transformation - this is what the camera captured.

    Args:
        image: Grayscale image (original or corrected)
        mat_mask: Binary mask of mat region
        saturation_threshold: Threshold for marking saturated pixels

    Returns:
        StepResult with:
            data['intensity_map']: Raw intensity values within mat mask
            data['intensity_stats']: Statistics of mat intensities
            data['saturation_mask']: Mask of saturated pixels
            display_image: Colormapped intensity heat map (BGR)
    """
    try:
        bool_mask = mat_mask.astype(bool)

        # Raw intensity map (float)
        intensity_map = image.astype(np.float32)

        # Saturation detection
        saturation_mask = (image >= saturation_threshold) & bool_mask

        # Stats
        mat_values = image[bool_mask]
        if len(mat_values) == 0:
            return StepResult(success=False, error_message="No mat pixels found")

        stats = {
            'mean': float(np.mean(mat_values)),
            'std': float(np.std(mat_values)),
            'min': float(np.min(mat_values)),
            'max': float(np.max(mat_values)),
            'cv': float(np.std(mat_values) / np.mean(mat_values) * 100) if np.mean(mat_values) > 0 else 0,
            'saturation_pct': float(100 * np.sum(saturation_mask) / np.sum(bool_mask)),
            'pixel_count': int(np.sum(bool_mask)),
        }

        # Create colormap display
        # Normalize to data range for colormapping
        vmin = float(np.percentile(mat_values, 0.5))
        vmax = float(np.percentile(mat_values, 99.5))

        # Normalize and apply colormap
        normalized = np.zeros_like(intensity_map)
        if vmax > vmin:
            normalized = (intensity_map - vmin) / (vmax - vmin)
            normalized = np.clip(normalized, 0, 1)

        # Apply viridis colormap
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

        # Mask out non-mat regions (make them dark gray)
        colored[~bool_mask] = [40, 40, 40]

        # Highlight saturated pixels in red
        if np.any(saturation_mask):
            colored[saturation_mask] = [0, 0, 255]  # Red in BGR

        # Add stats text
        h, w = image.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.35, min(h, w) / 2000)
        thickness = max(1, int(min(h, w) / 700))
        y = int(h * 0.03)
        line_h = int(22 * font_scale / 0.35)

        for text in [
            f"Mean: {stats['mean']:.1f}",
            f"Std: {stats['std']:.1f}",
            f"CV: {stats['cv']:.1f}%",
            f"Range: {stats['min']:.0f}-{stats['max']:.0f}",
            f"Saturated: {stats['saturation_pct']:.1f}%",
        ]:
            y += line_h
            cv2.putText(colored, text, (10, y), font, font_scale, (255, 255, 255), thickness)

        logger.info(f"Intensity map: mean={stats['mean']:.1f}, CV={stats['cv']:.1f}%, "
                    f"saturated={stats['saturation_pct']:.1f}%")

        return StepResult(
            success=True,
            display_image=colored,
            data={
                'intensity_map': intensity_map,
                'intensity_stats': stats,
                'saturation_mask': saturation_mask,
                'mat_mask': mat_mask,
            }
        )
    except Exception as e:
        logger.error(f"Error creating intensity map: {e}", exc_info=True)
        return StepResult(success=False, error_message=str(e))


def compute_uniformity_metrics(
    intensity_map: np.ndarray,
    mat_mask: np.ndarray,
    window_size: int = 51,
) -> StepResult:
    """
    Step 5: Compute key uniformity metrics and spatial CV map.

    Produces a small set of interpretable metrics plus a spatial map
    showing where non-uniformity exists.

    Args:
        intensity_map: 2D intensity/thickness map
        mat_mask: Binary mask of mat region
        window_size: Window size for local CV computation

    Returns:
        StepResult with:
            data['metrics']: Dict of key uniformity metrics
            data['local_cv_map']: 2D spatial CV map
            data['sector_analysis']: Per-sector metrics
            display_image: Spatial CV map visualization (BGR)
    """
    try:
        bool_mask = mat_mask.astype(bool)
        data = intensity_map.astype(np.float32)
        mat_values = data[bool_mask]

        if len(mat_values) == 0:
            return StepResult(success=False, error_message="No mat pixels found")

        # --- Key metrics ---
        overall_mean = float(np.mean(mat_values))
        overall_std = float(np.std(mat_values))
        overall_cv = (overall_std / overall_mean) if overall_mean > 0 else 0

        # Gini coefficient
        sorted_vals = np.sort(mat_values)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        gini = float((2 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n)
        gini = max(0.0, min(1.0, gini))

        # --- Spatial CV map ---
        valid_float = bool_mask.astype(np.float32)
        count_map = cv2.boxFilter(valid_float, -1, (window_size, window_size), normalize=False)
        count_map = np.maximum(count_map, 1)

        data_masked = np.where(bool_mask, data, 0).astype(np.float32)
        local_sum = cv2.boxFilter(data_masked, -1, (window_size, window_size), normalize=False)
        local_sum_sq = cv2.boxFilter(data_masked ** 2, -1, (window_size, window_size), normalize=False)

        local_mean = local_sum / count_map
        local_var = np.maximum((local_sum_sq / count_map) - (local_mean ** 2), 0)
        local_std = np.sqrt(local_var)
        local_cv = np.where(local_mean > 1e-6, local_std / local_mean, 0)

        # --- Radial gradient (center vs edge) ---
        ys, xs = np.where(bool_mask)
        if len(ys) > 0:
            cy, cx = float(np.mean(ys)), float(np.mean(xs))
            distances = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
            max_dist = float(np.max(distances))

            if max_dist > 0:
                # Inner 30% vs outer 30%
                inner_mask_vals = mat_values[distances < 0.3 * max_dist]
                outer_mask_vals = mat_values[distances > 0.7 * max_dist]

                inner_mean = float(np.mean(inner_mask_vals)) if len(inner_mask_vals) > 0 else 0
                outer_mean = float(np.mean(outer_mask_vals)) if len(outer_mask_vals) > 0 else 0
                radial_ratio = (inner_mean / outer_mean) if outer_mean > 0 else 1.0
            else:
                radial_ratio = 1.0
                inner_mean = outer_mean = overall_mean
        else:
            radial_ratio = 1.0
            inner_mean = outer_mean = overall_mean

        # --- Sector analysis (8 sectors) ---
        sector_stats = []
        if len(ys) > 0:
            angles = np.arctan2(ys - cy, xs - cx)  # -pi to pi
            for i in range(8):
                angle_min = -np.pi + i * np.pi / 4
                angle_max = angle_min + np.pi / 4
                sector_mask = (angles >= angle_min) & (angles < angle_max)
                sector_vals = mat_values[sector_mask]
                if len(sector_vals) > 10:
                    s_mean = float(np.mean(sector_vals))
                    s_std = float(np.std(sector_vals))
                    s_cv = s_std / s_mean if s_mean > 0 else 0
                    sector_stats.append({
                        'sector': i,
                        'angle_deg': float((angle_min + angle_max) / 2 * 180 / np.pi),
                        'mean': s_mean,
                        'std': s_std,
                        'cv': s_cv,
                        'pixel_count': int(len(sector_vals)),
                    })

        # Coverage: what fraction of mask-bounded area has significant intensity?
        coverage = float(np.sum(bool_mask)) / float(bool_mask.size)

        metrics = {
            'overall_cv': overall_cv,
            'gini_coefficient': gini,
            'radial_ratio': radial_ratio,
            'center_mean': inner_mean,
            'edge_mean': outer_mean,
            'coverage': coverage,
            'mean_intensity': overall_mean,
            'std_intensity': overall_std,
            'mean_local_cv': float(np.mean(local_cv[bool_mask])),
        }

        # --- Create display image: spatial CV map ---
        cv_display = local_cv.copy()
        valid_cv = local_cv[bool_mask]
        if len(valid_cv) > 0:
            cv_vmax = float(np.percentile(valid_cv, 97))
        else:
            cv_vmax = 1.0
        cv_vmax = max(cv_vmax, 0.01)

        # Normalize CV to 0-255 for colormap
        cv_normalized = np.clip(cv_display / cv_vmax, 0, 1)
        cv_uint8 = (cv_normalized * 255).astype(np.uint8)

        # Use a diverging colormap: green=uniform, red=non-uniform
        # OpenCV doesn't have RdYlGn_r, so use COLORMAP_JET reversed (blue=low, red=high)
        colored = cv2.applyColorMap(cv_uint8, cv2.COLORMAP_JET)

        # Mask non-mat regions
        colored[~bool_mask] = [40, 40, 40]

        # Add legend text
        h, w = intensity_map.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.35, min(h, w) / 2000)
        thickness = max(1, int(min(h, w) / 700))
        y_pos = int(h * 0.03)
        line_h = int(22 * font_scale / 0.35)

        for text in [
            f"Overall CV: {overall_cv:.3f} ({overall_cv*100:.1f}%)",
            f"Gini: {gini:.3f}",
            f"Radial ratio (center/edge): {radial_ratio:.2f}",
            f"Mean local CV: {metrics['mean_local_cv']:.3f}",
            f"Coverage: {coverage*100:.1f}%",
        ]:
            y_pos += line_h
            cv2.putText(colored, text, (10, y_pos), font, font_scale, (255, 255, 255), thickness)

        logger.info(f"Uniformity: CV={overall_cv:.3f}, Gini={gini:.3f}, "
                    f"radial={radial_ratio:.2f}, mean_local_cv={metrics['mean_local_cv']:.3f}")

        return StepResult(
            success=True,
            display_image=colored,
            data={
                'metrics': metrics,
                'local_cv_map': local_cv,
                'sector_analysis': sector_stats,
            }
        )
    except Exception as e:
        logger.error(f"Error computing uniformity metrics: {e}", exc_info=True)
        return StepResult(success=False, error_message=str(e))
