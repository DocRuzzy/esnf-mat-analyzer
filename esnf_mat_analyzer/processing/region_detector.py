"""
Four-Region Detector for ESNF Mat Analyzer.

Identifies and segments four distinct regions in electrospun nanofiber images:
1. Background - The collection surface outside deposition area (TRUE black reference)
2. Gel Region - The dark perimeter around mat (wet polymer, DARKER than background)
3. Mat Region - The deposited nanofiber material (bright region)
4. Ruler Region - Scale bar/ruler at edge of image (exclude from all analysis)

CRITICAL: The gel region is DARKER than the true background because wet polymer
absorbs light. The true background reference must come from the collection surface
OUTSIDE the gel ring, not from the gel itself.

Author: ESNF Mat Analyzer Team
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List, Any
import logging


@dataclass
class FourRegionDetectionResult:
    """Result from four-region detection."""
    
    # Binary masks for each region
    background_mask: np.ndarray  # True where TRUE background (collection surface outside gel)
    gel_mask: np.ndarray  # True in gel region (dark perimeter, DARKER than background)
    mat_mask: np.ndarray  # True where nanofiber mat exists
    ruler_mask: np.ndarray  # True where ruler/scale bar detected
    
    # Contours
    mat_contour: Optional[np.ndarray] = None  # Outer boundary of mat
    gel_contour: Optional[np.ndarray] = None  # Outer boundary of gel (if detected)
    
    # Reference intensity levels (CRITICAL for proper calibration)
    background_intensity: float = 0.0  # Mean intensity of TRUE background (outside gel)
    gel_intensity: float = 0.0  # Mean intensity of gel region (should be < background)
    mat_intensity_range: Tuple[float, float] = (0.0, 255.0)  # Min/max in mat
    
    # Detection confidence
    gel_detected: bool = False  # Whether distinct gel region was found
    ruler_detected: bool = False  # Whether ruler region was found
    quality_score: float = 0.0  # Overall detection quality (0-1)
    
    # Saturation info
    saturation_mask: Optional[np.ndarray] = None
    saturation_pct: float = 0.0  # Percentage of mat that is saturated
    
    # Analysis mask (mat region only, excluding gel and ruler)
    analysis_mask: Optional[np.ndarray] = None


# Keep legacy class for backward compatibility
@dataclass
class RegionDetectionResult:
    """Result from three-region detection (legacy compatibility)."""
    
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


class FourRegionDetector:
    """
    Detector for identifying background, gel, mat, and ruler regions.
    
    CRITICAL INSIGHT: The gel region (wet polymer perimeter) is DARKER than
    the true background. The background reference for calibration must come
    from the collection surface OUTSIDE the gel ring.
    
    Detection Strategy (order matters):
    1. Detect ruler region first (typically at image edges with high contrast ticks)
    2. Find the bright mat region (standard Otsu on non-ruler area)
    3. Detect gel as the dark ring immediately around mat
    4. Background is everything else outside gel (true reference)
    """
    
    def __init__(
        self,
        mat_threshold_method: str = "otsu",
        gel_detection_enabled: bool = True,
        ruler_detection_enabled: bool = True,
        saturation_threshold: int = 254,
        min_mat_area_ratio: float = 0.05,
        max_mat_area_ratio: float = 0.8,
        ruler_edge_margin: int = 50,  # Pixels from edge to search for ruler
    ):
        """
        Initialize the four-region detector.
        
        Args:
            mat_threshold_method: Method for mat detection ("otsu", "adaptive", "percentile")
            gel_detection_enabled: Whether to attempt gel region detection
            ruler_detection_enabled: Whether to detect and exclude ruler regions
            saturation_threshold: Intensity above which pixels are considered saturated
            min_mat_area_ratio: Minimum mat area as fraction of image
            max_mat_area_ratio: Maximum mat area as fraction of image
            ruler_edge_margin: Pixels from edge to search for ruler
        """
        self.mat_threshold_method = mat_threshold_method
        self.gel_detection_enabled = gel_detection_enabled
        self.ruler_detection_enabled = ruler_detection_enabled
        self.saturation_threshold = saturation_threshold
        self.min_mat_area_ratio = min_mat_area_ratio
        self.max_mat_area_ratio = max_mat_area_ratio
        self.ruler_edge_margin = ruler_edge_margin
        self.logger = logging.getLogger(__name__)
    
    def detect(
        self,
        image: np.ndarray,
        roi_mask: Optional[np.ndarray] = None,
        ruler_mask: Optional[np.ndarray] = None
    ) -> FourRegionDetectionResult:
        """
        Detect four regions in the image.
        
        Args:
            image: Grayscale input image (0-255)
            roi_mask: Optional user-defined ROI to restrict detection
            ruler_mask: Optional pre-computed ruler mask (from RulerDetector)
            
        Returns:
            FourRegionDetectionResult with masks and metadata
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
        
        # Step 1: Detect/use ruler mask
        if ruler_mask is not None:
            detected_ruler_mask = ruler_mask.astype(bool)
            ruler_detected = np.any(detected_ruler_mask)
        elif self.ruler_detection_enabled:
            detected_ruler_mask = self._detect_ruler_region(gray)
            ruler_detected = np.any(detected_ruler_mask)
        else:
            detected_ruler_mask = np.zeros((h, w), dtype=bool)
            ruler_detected = False
        
        # Exclude ruler from analysis
        analysis_mask = analysis_mask & ~detected_ruler_mask
        
        # Step 2: Detect mat region (bright area)
        mat_mask, mat_contour = self._detect_mat_region(gray, analysis_mask)
        
        # Step 3: Detect gel region (dark ring around mat, DARKER than background)
        if self.gel_detection_enabled:
            gel_mask, gel_contour, gel_detected = self._detect_gel_region_improved(
                gray, mat_mask, analysis_mask
            )
        else:
            gel_mask = np.zeros((h, w), dtype=bool)
            gel_contour = None
            gel_detected = False
        
        # Step 4: TRUE background is everything outside mat AND gel (and ruler)
        background_mask = analysis_mask & ~mat_mask & ~gel_mask & ~detected_ruler_mask

        # Safety valve: if the HEURISTIC ruler mask ate so much of the image
        # that background coverage collapsed (<5%), the mask is almost
        # certainly over-detected — drop it rather than starve the
        # background fit. Caller-provided ruler masks are trusted as-is.
        heuristic_ruler = ruler_mask is None and self.ruler_detection_enabled
        if heuristic_ruler and np.sum(background_mask) < 0.05 * h * w:
            self.logger.warning(
                f"Background coverage {100 * np.sum(background_mask) / (h * w):.1f}% "
                f"after heuristic ruler masking (<5%): dropping heuristic ruler "
                f"mask to avoid starving the background fit."
            )
            detected_ruler_mask = np.zeros((h, w), dtype=bool)
            ruler_detected = False
            analysis_mask_no_ruler = (roi_mask.astype(bool) if roi_mask is not None
                                      else np.ones((h, w), dtype=bool))
            background_mask = analysis_mask_no_ruler & ~mat_mask & ~gel_mask

        # Ensure we have some background
        if np.sum(background_mask) < 100:
            self.logger.warning("Very limited true background detected")
            # Fall back to areas far from mat
            background_mask = self._detect_background_far_from_mat(
                gray, mat_mask, gel_mask, detected_ruler_mask, analysis_mask
            )
        
        # Calculate reference intensities
        background_intensity = self._safe_mean(gray, background_mask)
        gel_intensity = self._safe_mean(gray, gel_mask) if gel_detected else 0.0
        
        mat_values = gray[mat_mask]
        mat_intensity_range = (float(mat_values.min()), float(mat_values.max())) if len(mat_values) > 0 else (0.0, 255.0)
        
        # CRITICAL CHECK: Verify gel is darker than background
        if gel_detected and gel_intensity >= background_intensity:
            self.logger.warning(
                f"Gel intensity ({gel_intensity:.1f}) >= background ({background_intensity:.1f}). "
                "This suggests incorrect region detection. Gel should be DARKER than background."
            )
        
        # Detect saturation
        saturation_mask = (gray >= self.saturation_threshold) & mat_mask
        saturation_pct = 100 * np.sum(saturation_mask) / max(np.sum(mat_mask), 1)
        
        # Create analysis mask (mat only, for thickness analysis)
        final_analysis_mask = mat_mask.copy()
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            gray, mat_mask, background_mask, gel_mask, gel_detected, 
            background_intensity, gel_intensity
        )
        
        self.logger.info(
            f"Four-region detection: mat={100*np.mean(mat_mask):.1f}%, "
            f"background={100*np.mean(background_mask):.1f}%, "
            f"gel={'detected' if gel_detected else 'not detected'}, "
            f"ruler={'detected' if ruler_detected else 'not detected'}, "
            f"saturation={saturation_pct:.1f}%"
        )
        self.logger.info(
            f"Intensity levels: background={background_intensity:.1f}, "
            f"gel={gel_intensity:.1f}, mat={mat_intensity_range}"
        )
        
        return FourRegionDetectionResult(
            background_mask=background_mask.astype(np.uint8),
            gel_mask=gel_mask.astype(np.uint8),
            mat_mask=mat_mask.astype(np.uint8),
            ruler_mask=detected_ruler_mask.astype(np.uint8),
            mat_contour=mat_contour,
            gel_contour=gel_contour,
            background_intensity=background_intensity,
            gel_intensity=gel_intensity,
            mat_intensity_range=mat_intensity_range,
            gel_detected=gel_detected,
            ruler_detected=ruler_detected,
            quality_score=quality_score,
            saturation_mask=saturation_mask.astype(np.uint8),
            saturation_pct=float(saturation_pct),
            analysis_mask=final_analysis_mask.astype(np.uint8),
        )
    
    def _detect_ruler_region(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect ruler/scale bar region at image edges.

        Rulers typically have high-contrast tick marks and appear at edges.

        Historical bug: a fixed 50 px margin covered ~22% of each dimension
        on small (~230 px) images, and the bright mat rim tripped the
        contrast test in ALL four edge strips — the whole perimeter (~56%
        of the image) was marked as ruler, starving the background fit.
        Fixes: margin is scaled to the image, and mat-class (bright blob)
        pixels are excluded from the contrast statistics.
        """
        h, w = gray.shape
        ruler_mask = np.zeros((h, w), dtype=bool)

        # Scale margin to image size: never more than 8% of the short side.
        margin = min(self.ruler_edge_margin, max(5, int(0.08 * min(h, w))))

        # Check each edge for ruler-like patterns
        edges = [
            ('bottom', gray[-margin:, :]),
            ('top', gray[:margin, :]),
            ('left', gray[:, :margin]),
            ('right', gray[:, -margin:]),
        ]

        for edge_name, edge_region in edges:
            if self._is_ruler_region(edge_region):
                if edge_name == 'bottom':
                    ruler_mask[-margin:, :] = True
                elif edge_name == 'top':
                    ruler_mask[:margin, :] = True
                elif edge_name == 'left':
                    ruler_mask[:, :margin] = True
                elif edge_name == 'right':
                    ruler_mask[:, -margin:] = True
                self.logger.info(f"Detected ruler at {edge_name} edge")

        return ruler_mask

    def _is_ruler_region(self, region: np.ndarray) -> bool:
        """Check if a region contains ruler-like patterns (regular tick marks).

        Discriminates by STRUCTURE, not raw contrast: a ruler strip has many
        bright/dark transitions per line along its long axis (tick marks),
        while a mat rim crossing the strip is a single contiguous blob
        (~2 transitions). The old global-std>40 test could not tell them
        apart, which caused all four edge strips to fire on small images
        (the whole perimeter was masked as ruler).
        """
        if region.size == 0 or min(region.shape) == 0:
            return False

        u8 = np.clip(region, 0, 255).astype(np.uint8)
        _, binary = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = binary > 0

        # Nearly uniform strip (all background or all mat): not a ruler.
        # Lower bound is small because a thin tick comb at the extreme edge
        # occupies only a few rows of the strip.
        bright_frac = float(bright.mean())
        if not (0.02 < bright_frac < 0.98):
            return False

        # Require real intensity separation between the two classes,
        # otherwise Otsu is just amplifying background noise.
        contrast = float(u8[bright].mean()) - float(u8[~bright].mean())
        if contrast < 40:
            return False

        # Count bright/dark transitions per line along the strip's long axis.
        # 90th percentile (not median): a thin tick comb fills only a few
        # lines of the strip, but those lines carry dozens of transitions.
        axis = 1 if region.shape[1] >= region.shape[0] else 0
        transitions = np.abs(np.diff(bright.astype(np.int8), axis=axis)).sum(axis=axis)
        p90_transitions = float(np.percentile(transitions, 90))

        # Measured separation on the 30-image sample set: real rulers score
        # 22-108, mat rims / labels / glare score 0-16.
        return p90_transitions >= 20.0
    
    def _detect_mat_region(
        self,
        gray: np.ndarray,
        analysis_mask: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Detect the bright mat region using thresholding."""
        h, w = gray.shape
        
        # Get intensity values within analysis mask
        masked_values = gray[analysis_mask]
        
        if len(masked_values) == 0:
            return np.zeros((h, w), dtype=bool), None
        
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
        
        # Find the best mat candidate within area constraints.
        # Pure largest-area selection fails when a bright glare band in the
        # background outgrows the mat disc (e.g. TCD10-GPED3.0-1: full-width
        # top band = 9134 px vs disc = 8677 px). The mat is compact, roughly
        # circular, and near the image center, so score candidates by
        # area x circularity x centrality instead of area alone.
        image_area = h * w
        valid_contours = [
            c for c in contours
            if self.min_mat_area_ratio * image_area <= cv2.contourArea(c) <= self.max_mat_area_ratio * image_area
        ]

        if not valid_contours:
            # Fall back to largest contour
            valid_contours = [max(contours, key=cv2.contourArea)]

        def _mat_score(contour):
            area = cv2.contourArea(contour)
            if area <= 0:
                return 0.0
            perimeter = max(cv2.arcLength(contour, True), 1.0)
            circularity = min(4.0 * np.pi * area / (perimeter ** 2), 1.0)
            m = cv2.moments(contour)
            if m['m00'] > 0:
                cx_c, cy_c = m['m10'] / m['m00'], m['m01'] / m['m00']
            else:
                cx_c, cy_c = w / 2.0, h / 2.0
            dist = np.hypot(cx_c - w / 2.0, cy_c - h / 2.0)
            centrality = max(0.0, 1.0 - dist / (0.5 * np.hypot(h, w)))
            return area * circularity * centrality

        largest_contour = max(valid_contours, key=_mat_score)

        # Create mask from contour
        mat_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mat_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)

        # Filling the contour also fills dark gel bays enclosed between the
        # mat body and attached bright spill arcs — those enclosed pixels are
        # gel, not mat. Restrict the fill to pixels that are actually bright,
        # then reseal genuine interior speckle.
        mat_mask = mat_mask & (binary > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mat_mask = cv2.morphologyEx(mat_mask, cv2.MORPH_CLOSE, kernel)

        return mat_mask.astype(bool), largest_contour

    def _detect_gel_region_improved(
        self,
        gray: np.ndarray,
        mat_mask: np.ndarray,
        analysis_mask: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        """
        Detect gel region - the dark ring around mat that is DARKER than background.

        Primary strategy: radial ray-casting (_detect_gel_region_radial) —
        the gel is geometrically an annulus hugging the mat, so walk outward
        from the mat edge per angle until intensity returns to background
        level. This includes thin gel that is not much darker than the
        background AND stops before diffuse shadows further out — both
        failure modes of the pure darkness-threshold approach below, which
        is kept as fallback.
        """
        radial = self._detect_gel_region_radial(gray, mat_mask, analysis_mask)
        if radial is not None:
            return radial
        self.logger.info("Radial gel detection unavailable; falling back to threshold method")
        return self._detect_gel_region_threshold(gray, mat_mask, analysis_mask)

    def _detect_gel_region_radial(
        self,
        gray: np.ndarray,
        mat_mask: np.ndarray,
        analysis_mask: np.ndarray
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], bool]]:
        """
        Radial gel detection: cast rays from the mat centroid.

        For each angle, start at the mat's outer edge and walk outward.
        Gel continues while pixels stay below an adaptive threshold placed
        midway between measured gel darkness and background brightness;
        the ray stops once intensity returns to background level. The
        per-angle outer radii are median-smoothed (bridges specular glints
        on the wet gel) and filled into a closed annulus.

        Returns None if the geometry is unusable (no mat, degenerate rays),
        signalling the caller to fall back to the threshold method.
        """
        h, w = gray.shape
        if not np.any(mat_mask):
            return None

        ys, xs = np.nonzero(mat_mask)
        cy, cx = float(ys.mean()), float(xs.mean())
        r_mat_typ = float(np.sqrt(mat_mask.sum() / np.pi))
        max_ext = max(10, int(0.55 * r_mat_typ))

        # Sanity: there must be usable area beyond the possible gel band
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * max_ext + 1, 2 * max_ext + 1)
        )
        mat_dilated = cv2.dilate(mat_mask.astype(np.uint8), dilate_kernel).astype(bool)
        if np.sum(analysis_mask & ~mat_dilated) < 100:
            return None

        n_angles = 720
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        max_r = int(np.hypot(h, w))
        gray_f = gray.astype(np.float32)

        # Illumination varies strongly across these photos (shadowed side can
        # be 20+ DN darker than the global background mean), and the wet gel
        # carries bright specular glints right at the mat edge. So each ray
        # uses its own LOCAL background (sampled beyond the maximum gel
        # extent) and marks gel out to the LAST deep-dark sample — glints
        # are bridged instead of terminating the walk.
        min_contrast = 15.0  # DN; below this a ray has no discernible gel

        r_outer = np.zeros(n_angles, dtype=np.float64)
        r_mat_edge = np.zeros(n_angles, dtype=np.float64)
        bg_locals = []
        for i in range(n_angles):
            samples = np.arange(0, max_r)
            px = (cx + samples * cos_a[i]).astype(int)
            py = (cy + samples * sin_a[i]).astype(int)
            valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            px, py, samples = px[valid], py[valid], samples[valid]
            on_mat = mat_mask[py, px]
            if not on_mat.any():
                r_mat_edge[i] = 0.0
                r_outer[i] = 0.0
                continue
            # Walk from the end of the CONTIGUOUS mat body, not the
            # outermost mat pixel: spilled mat outside the gel ring would
            # otherwise start the walk beyond the gel and hide it. The
            # spill stays classified as mat because the mat mask is
            # subtracted from the annulus at the end. Small gaps (<=3 px,
            # antialiasing) do not break the run.
            on_idx = np.nonzero(on_mat)[0]
            gap_pos = np.nonzero(np.diff(on_idx) > 3)[0]
            edge_idx = int(on_idx[gap_pos[0]]) if gap_pos.size else int(on_idx.max())
            r_mat_edge[i] = samples[edge_idx]
            r_outer[i] = samples[edge_idx]

            walk_lo = edge_idx + 1
            walk_hi = min(edge_idx + max_ext, len(samples) - 1)
            far_hi = min(edge_idx + 2 * max_ext, len(samples) - 1)
            if walk_hi <= walk_lo or far_hi <= walk_hi + 2:
                continue

            walk_vals = gray_f[py[walk_lo:walk_hi + 1], px[walk_lo:walk_hi + 1]]
            far_vals = gray_f[py[walk_hi + 1:far_hi + 1], px[walk_hi + 1:far_hi + 1]]
            bg_local = float(np.median(far_vals))
            bg_locals.append(bg_local)
            gel_dark = float(np.percentile(walk_vals, 5))
            if bg_local - gel_dark < min_contrast:
                # Gel and background are indistinguishable by intensity in
                # this direction (e.g. gel over deep shadow). Mark unknown;
                # the radius is interpolated from angular neighbours below —
                # the gel annulus is spatially continuous.
                r_outer[i] = np.nan
                continue

            # Gel cut: meaningfully below the local background. The outer
            # half of the gel ring can sit only 10-20 DN below background
            # (thin translucent gel), so the cut is placed at 65% of the
            # gel-to-background contrast — still strictly below bg_local,
            # which keeps sustained background/shadow runs excluded.
            deep_thr = gel_dark + 0.65 * (bg_local - gel_dark)
            deep_idx = np.nonzero(walk_vals <= deep_thr)[0]
            if deep_idx.size:
                r_outer[i] = samples[walk_lo + int(deep_idx.max())]

        if not np.any(r_mat_edge > 0):
            return None

        # Interpolate unknown (NaN) radii from angular neighbours with
        # circular wrap-around: directions where gel and shadow were
        # indistinguishable inherit the boundary from adjacent directions.
        nan_mask = np.isnan(r_outer)
        if nan_mask.all():
            return None
        # Angular coverage of INDEPENDENT gel findings (pre-interpolation)
        with np.errstate(invalid='ignore'):
            independent_found = (~nan_mask) & (r_outer > r_mat_edge + 1.0) & (r_mat_edge > 0)
        coverage = float(np.mean(independent_found))

        # Ring-thickness cap: the gel ring's thickness varies smoothly, but
        # deep-dark smudges/shadow beyond the true ring can drag single
        # rays far out (over-detection reported on the brightly lit side).
        # Cap each ray's thickness at 1.6x the median found thickness.
        if independent_found.any():
            thickness = r_outer - r_mat_edge
            t_med = float(np.median(thickness[independent_found]))
            t_cap = 1.6 * t_med + 2.0
            with np.errstate(invalid='ignore'):
                over = independent_found & (thickness > t_cap)
            r_outer[over] = r_mat_edge[over] + t_cap
        if nan_mask.any():
            idx = np.arange(n_angles)
            valid = ~nan_mask
            r_outer = np.interp(
                idx, idx[valid], r_outer[valid],
                period=n_angles
            )

        # Circular smoothing bridges glints and single-ray spikes. The 30th
        # percentile (not the median) biases toward the SMALLER radius, so
        # lumpy outward protrusions from background smudges are shaved off
        # while consistent ring stretches are preserved.
        k = 18
        padded = np.concatenate([r_outer[-k:], r_outer, r_outer[:k]])
        smoothed = np.array([
            np.percentile(padded[i:i + 2 * k + 1], 30) for i in range(n_angles)
        ])
        # Never let smoothing pull the boundary inside the mat edge
        smoothed = np.maximum(smoothed, r_mat_edge)

        # Fill the outer boundary polygon, then remove the mat -> annulus
        pts = np.stack([
            cx + smoothed * cos_a, cy + smoothed * sin_a
        ], axis=1).astype(np.int32)
        filled = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(filled, [pts], 1)
        gel_mask = filled.astype(bool) & ~mat_mask & analysis_mask

        # Light cleanup only (geometry already constrains the shape)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gel_mask = cv2.morphologyEx(
            gel_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool) & ~mat_mask & analysis_mask

        gel_area = int(gel_mask.sum())
        if gel_area < 0.01 * mat_mask.sum():
            self.logger.info(f"Radial gel area too small ({gel_area} px)")
            return np.zeros((h, w), dtype=bool), None, False

        gel_intensity = float(np.mean(gray[gel_mask]))
        # Sanity check by ANGULAR COVERAGE, not by comparing the gel mean
        # against a background average: every walked-in pixel was already
        # required to sit below its own ray's local background, and on
        # unevenly lit photos a global mean/median comparison mixes lit
        # and shadowed sides and falsely rejects legitimate gel.
        if coverage < 0.25:
            self.logger.info(
                f"Radial gel coverage only {coverage:.0%} of directions; "
                f"treating as no gel"
            )
            return np.zeros((h, w), dtype=bool), None, False

        contours, _ = cv2.findContours(
            gel_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        gel_contour = max(contours, key=cv2.contourArea) if contours else None

        self.logger.info(
            f"Radial gel detection: {gel_area} px, intensity={gel_intensity:.1f}, "
            f"angular coverage={coverage:.0%}"
        )
        return gel_mask, gel_contour, True

    def _detect_gel_region_threshold(
        self,
        gray: np.ndarray,
        mat_mask: np.ndarray,
        analysis_mask: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        """
        Fallback gel detection - dark-pixel threshold in a dilated ring.

        Known failure modes (why radial is preferred): includes diffuse
        shadows that are darker than background but are not gel, and
        misses thin gel that is not far enough below the threshold.
        """
        h, w = gray.shape
        
        # Create search ring around mat
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
        mat_dilated = cv2.dilate(mat_mask.astype(np.uint8), dilate_kernel).astype(bool)
        
        # Search region is the ring around mat (dilated - original mat)
        ring_region = mat_dilated & ~mat_mask & analysis_mask
        
        if np.sum(ring_region) < 100:
            return np.zeros((h, w), dtype=bool), None, False
        
        # Get intensity of the outer background (far from mat)
        outer_background = analysis_mask & ~mat_dilated
        
        # Exclude edges which might have artifacts
        edge_margin = min(h, w) // 15
        edge_mask = np.ones((h, w), dtype=bool)
        edge_mask[:edge_margin, :] = False
        edge_mask[-edge_margin:, :] = False
        edge_mask[:, :edge_margin] = False
        edge_mask[:, -edge_margin:] = False
        outer_background = outer_background & edge_mask
        
        if np.sum(outer_background) < 100:
            # Not enough outer background, use percentile-based approach
            self.logger.info("Limited outer background, using percentile-based gel detection")
            outer_bg_intensity = np.percentile(gray[analysis_mask & ~mat_mask], 70)
        else:
            outer_bg_intensity = np.mean(gray[outer_background])
        
        # Gel is DARKER than outer background
        # Threshold: pixels significantly darker than outer background
        gel_threshold = outer_bg_intensity * 0.75  # 75% of background brightness
        
        # Find dark pixels in the ring region
        potential_gel = ring_region & (gray < gel_threshold)
        
        # Also require gel to be darker than a minimum (not just noise)
        min_dark_threshold = outer_bg_intensity * 0.5
        potential_gel = potential_gel & (gray < outer_bg_intensity * 0.85)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        gel_cleaned = cv2.morphologyEx(potential_gel.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        gel_cleaned = cv2.morphologyEx(gel_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Check if we found a significant gel region
        gel_area = np.sum(gel_cleaned > 0)
        mat_area = np.sum(mat_mask)
        
        if gel_area < 0.01 * mat_area:
            self.logger.info(f"Gel area ({gel_area}) too small relative to mat ({mat_area})")
            return np.zeros((h, w), dtype=bool), None, False
        
        # Find gel contour
        contours, _ = cv2.findContours(
            gel_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        gel_contour = max(contours, key=cv2.contourArea) if contours else None
        
        # Verify: gel intensity should be darker than outer background
        gel_intensity = np.mean(gray[gel_cleaned > 0]) if np.any(gel_cleaned) else 0
        if gel_intensity >= outer_bg_intensity:
            self.logger.warning(
                f"Detected 'gel' ({gel_intensity:.1f}) is not darker than background ({outer_bg_intensity:.1f}). "
                "Rejecting gel detection."
            )
            return np.zeros((h, w), dtype=bool), None, False
        
        self.logger.info(
            f"Detected gel region: {gel_area} pixels, intensity={gel_intensity:.1f} "
            f"(background={outer_bg_intensity:.1f})"
        )
        
        return gel_cleaned.astype(bool), gel_contour, True
    
    def _detect_background_far_from_mat(
        self,
        gray: np.ndarray,
        mat_mask: np.ndarray,
        gel_mask: np.ndarray,
        ruler_mask: np.ndarray,
        analysis_mask: np.ndarray
    ) -> np.ndarray:
        """Find background region far from mat when standard detection fails."""
        h, w = gray.shape
        
        # Dilate mat significantly
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
        mat_dilated = cv2.dilate(mat_mask.astype(np.uint8), dilate_kernel).astype(bool)
        
        # Background is far from mat, not gel, not ruler
        background = analysis_mask & ~mat_dilated & ~gel_mask & ~ruler_mask
        
        # Exclude edges
        edge_margin = min(h, w) // 10
        edge_mask = np.ones((h, w), dtype=bool)
        edge_mask[:edge_margin, :] = False
        edge_mask[-edge_margin:, :] = False
        edge_mask[:, :edge_margin] = False
        edge_mask[:, -edge_margin:] = False
        
        background = background & edge_mask
        
        return background
    
    def _safe_mean(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Calculate mean of image within mask, safely handling empty masks."""
        if np.sum(mask) == 0:
            return 0.0
        return float(np.mean(image[mask.astype(bool)]))
    
    def _calculate_quality_score(
        self,
        gray: np.ndarray,
        mat_mask: np.ndarray,
        background_mask: np.ndarray,
        gel_mask: np.ndarray,
        gel_detected: bool,
        background_intensity: float,
        gel_intensity: float
    ) -> float:
        """Calculate detection quality score (0-1)."""
        score = 0.0
        h, w = gray.shape
        total_pixels = h * w
        
        mat_ratio = np.sum(mat_mask) / total_pixels
        bg_ratio = np.sum(background_mask) / total_pixels
        
        # Mat should be reasonable size
        if 0.1 <= mat_ratio <= 0.6:
            score += 0.2
        elif 0.05 <= mat_ratio <= 0.8:
            score += 0.1
        
        # Background should exist
        if bg_ratio >= 0.1:
            score += 0.2
        elif bg_ratio >= 0.05:
            score += 0.1
        
        # Check intensity ordering: gel < background < mat
        mat_intensity = self._safe_mean(gray, mat_mask)
        
        if mat_intensity > background_intensity * 1.2:
            score += 0.2
        
        if gel_detected:
            # CRITICAL: Gel should be darker than its LOCAL background.
            # A global mean comparison fails on unevenly lit photos (a
            # shadowed side drags the global background mean to or below
            # the gel mean even when the gel is locally darker everywhere),
            # so compare per angular sector against the adjacent background.
            frac = self._gel_locally_darker_fraction(gray, gel_mask, background_mask)
            if frac >= 0.7:
                score += 0.3  # High score for correct local ordering
            elif frac >= 0.5:
                score += 0.15
            # Bonus for clear separation in nearly all directions
            if frac >= 0.9:
                score += 0.1
        else:
            score += 0.15  # Partial credit if gel not expected

        return min(score, 1.0)

    def _gel_locally_darker_fraction(
        self,
        gray: np.ndarray,
        gel_mask: np.ndarray,
        background_mask: np.ndarray,
        n_sectors: int = 12
    ) -> float:
        """Fraction of angular sectors where gel is darker than the
        background immediately around it (median vs median)."""
        gel = gel_mask.astype(bool)
        if not gel.any():
            return 0.0
        band = cv2.dilate(
            gel.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        ).astype(bool) & background_mask.astype(bool) & ~gel

        gys, gxs = np.nonzero(gel)
        cy, cx = float(gys.mean()), float(gxs.mean())

        def sector_of(ys, xs):
            ang = np.arctan2(ys - cy, xs - cx)
            return ((ang + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors

        gel_sec = sector_of(gys, gxs)
        bys, bxs = np.nonzero(band)
        if len(bys) == 0:
            return 0.0
        bg_sec = sector_of(bys, bxs)

        gel_vals = gray[gys, gxs].astype(np.float64)
        bg_vals = gray[bys, bxs].astype(np.float64)

        ok = total = 0
        for s in range(n_sectors):
            gv = gel_vals[gel_sec == s]
            bv = bg_vals[bg_sec == s]
            if len(gv) < 20 or len(bv) < 20:
                continue
            total += 1
            # Compare the gel's DARK QUARTILE against the background median:
            # translucent gel carries bright glints that pull its median up
            # to background level, but its dark tail stays distinctly below.
            if np.percentile(gv, 25) < np.median(bv) - 5.0:
                ok += 1
        return ok / total if total else 0.0
    
    def visualize_regions(
        self,
        image: np.ndarray,
        result: FourRegionDetectionResult,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Create visualization overlay showing detected four regions.
        
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
        
        # Background: blue (TRUE background outside gel)
        overlay[result.background_mask.astype(bool)] = [255, 150, 100]  # Light blue
        
        # Gel: dark purple (darker than background)
        if result.gel_detected:
            overlay[result.gel_mask.astype(bool)] = [150, 50, 100]  # Dark purple
        
        # Mat: green
        overlay[result.mat_mask.astype(bool)] = [100, 255, 100]  # Light green
        
        # Ruler: gray
        if result.ruler_detected:
            overlay[result.ruler_mask.astype(bool)] = [128, 128, 128]  # Gray
        
        # Saturation: bright red warning
        if result.saturation_mask is not None:
            overlay[result.saturation_mask.astype(bool)] = [0, 0, 255]  # Bright red
        
        # Blend
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
        
        # Draw contours
        if result.mat_contour is not None:
            cv2.drawContours(vis, [result.mat_contour], -1, (0, 255, 0), 2)
        
        if result.gel_contour is not None:
            cv2.drawContours(vis, [result.gel_contour], -1, (150, 50, 100), 2)
        
        return vis


class ThreeRegionDetector:
    """
    Legacy three-region detector for backward compatibility.
    
    For new code, use FourRegionDetector which properly handles ruler exclusion
    and ensures background reference comes from outside the gel ring.
    """
    
    def __init__(
        self,
        mat_threshold_method: str = "otsu",
        gel_detection_enabled: bool = True,
        saturation_threshold: int = 254,
        min_mat_area_ratio: float = 0.05,
        max_mat_area_ratio: float = 0.8,
    ):
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
        Detect three regions: background, gel, mat.
        
        Args:
            image: Grayscale input image
            roi_mask: Optional ROI mask
            
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
