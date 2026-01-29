import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict
from esnf_mat_analyzer.core.data_types import RulerDetectionConfig, Ruler, Tick
import math

# Optional heavy dependency (PyTorch) for experimental DeepGP functionality.
# Import errors (including OSError from mismatched binaries) should not break core features.
try:  # pragma: no cover - defensive import handling
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
    _TORCH_IMPORT_ERROR = None
except Exception as _e:  # Broad except to catch WinError 193 and others
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _e

def _lazy_import_deep_gp():  # pragma: no cover - exercised only when feature enabled
    """Safely import DeepGPModule only when torch is available and feature enabled."""
    if not _TORCH_AVAILABLE:
        return None
    try:
        from .deep_gp_module import DeepGPModule  # local import to avoid unconditional torch import
        return DeepGPModule
    except Exception:
        return None

class RulerDetector:
    """
    Detects a ruler in an image to establish a physical scale (pixels per mm).

    This class encapsulates the full pipeline for ruler detection, which is
    essential for converting pixel-based measurements into meaningful physical
    units. The process involves several key steps:
    1.  Image preprocessing (grayscale conversion, blurring).
    2.  Edge detection using the Canny algorithm.
    3.  Line segment detection using the Probabilistic Hough Transform.
    4.  A robust, scoring-based model to find the main parallel lines of the
        ruler body from all detected horizontal lines.
    5.  Filtering and validation of vertical lines to identify tick marks.
    6.  Statistical analysis of tick spacing to calculate a reliable scale.

    The final output is a `Ruler` object containing the detected components
    and the calculated scale.
    """

    def __init__(self, config: RulerDetectionConfig):
        """
        Initializes the RulerDetector with a given configuration.

        Args:
            config: A `RulerDetectionConfig` object containing all parameters
                    for the detection process.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"RulerDetector initialized with config: {self.config}")
        self.deep_gp_model = None
        if self.config.use_deep_gp:
            if not _TORCH_AVAILABLE:
                self.logger.warning(
                    "PyTorch unavailable ({}). Disabling DeepGP scale refinement.".format(
                        _TORCH_IMPORT_ERROR.__class__.__name__ if _TORCH_IMPORT_ERROR else "import error"
                    )
                )
                self.config.use_deep_gp = False
            else:
                DeepGPModule = _lazy_import_deep_gp()
                if DeepGPModule is None:
                    self.logger.warning("DeepGP module import failed; disabling DeepGP feature.")
                    self.config.use_deep_gp = False
                elif not self.config.deep_gp_model_path:
                    self.logger.warning("`use_deep_gp` set without model path; disabling DeepGP.")
                    self.config.use_deep_gp = False
                else:
                    try:
                        self.deep_gp_model = DeepGPModule()
                        # torch is guaranteed non-None here
                        self.deep_gp_model.load_state_dict(torch.load(self.config.deep_gp_model_path, map_location="cpu"))  # type: ignore
                        self.deep_gp_model.eval()
                        self.logger.info(
                            f"Loaded DeepGP model from {self.config.deep_gp_model_path}"
                        )
                    except FileNotFoundError:
                        self.logger.error(
                            f"DeepGP model file not found at {self.config.deep_gp_model_path}. Disabling DeepGP."
                        )
                        self.config.use_deep_gp = False
                    except Exception as e:
                        self.logger.error(
                            f"Failed to load DeepGP model: {e}. Disabling DeepGP.",
                            exc_info=True,
                        )
                        self.config.use_deep_gp = False

    def _filter_and_group_lines(self, lines: np.ndarray, max_angle_diff_deg: float = 5.0) -> Dict[str, List[np.ndarray]]:
        """
        Filters detected line segments into horizontal and vertical groups.

        This function categorizes lines based on their angle, which is a crucial
        first step to distinguish between the ruler's body (horizontal) and its
        tick marks (vertical).

        Args:
            lines: An array of line segments from `cv2.HoughLinesP`.
            max_angle_diff_deg: The tolerance in degrees to classify a line as
                                horizontal or vertical.

        Returns:
            A dictionary with "horizontal" and "vertical" line lists.
        """
        horizontal_lines = []
        vertical_lines = []
        if lines is None:
            return {"horizontal": horizontal_lines, "vertical": vertical_lines}
        for line_segment in lines:
            x1, y1, x2, y2 = line_segment[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            if angle < 0: angle += 180
            if (abs(angle) < max_angle_diff_deg) or \
               (abs(angle - 180.0) < max_angle_diff_deg):
                horizontal_lines.append(line_segment)
            elif abs(angle - 90.0) < max_angle_diff_deg:
                vertical_lines.append(line_segment)
        return {"horizontal": horizontal_lines, "vertical": vertical_lines}

    def _get_main_ruler_body_lines(self, horizontal_lines: List[np.ndarray], vertical_lines: List[np.ndarray],
                                   min_length: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Identifies the best pair of ruler body lines using a scoring model.

        This method moves beyond simple heuristics (e.g., longest lines) and
        evaluates every plausible pair of horizontal lines based on a weighted
        score. The score considers line length, horizontal overlap, parallelism,
        and, crucially, the presence of perpendicular ticks between the lines.
        This makes the detection more robust against spurious long lines in the
        image that are not part of the ruler.

        Args:
            horizontal_lines: A list of detected horizontal line segments.
            vertical_lines: A list of detected vertical line segments.
            min_length: The minimum length for a line to be considered.

        Returns:
            A tuple containing the two best-scoring line segments, ordered with
            the upper line first, or None if no suitable pair is found.
        """
        if len(horizontal_lines) < 2:
            return None

        best_pair = None
        max_score = -np.inf

        line_props = []
        for line in horizontal_lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < min_length:
                continue
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            line_props.append({
                'line': line, 'length': length, 'angle': angle,
                'y_avg': (y1 + y2) / 2, 'x_coords': (min(x1, x2), max(x1, x2))
            })

        for i in range(len(line_props)):
            for j in range(i + 1, len(line_props)):
                props1 = line_props[i]
                props2 = line_props[j]

                thickness = abs(props1['y_avg'] - props2['y_avg'])
                if not (self.config.scoring['min_thickness_px'] <= thickness <= self.config.scoring['max_thickness_px']):
                    continue

                # Calculate score components
                length_score = (props1['length'] + props2['length']) * self.config.scoring['length_weight']

                overlap_start = max(props1['x_coords'][0], props2['x_coords'][0])
                overlap_end = min(props1['x_coords'][1], props2['x_coords'][1])
                overlap = max(0, overlap_end - overlap_start)
                overlap_score = overlap * self.config.scoring['overlap_weight']

                parallelism_penalty = abs(props1['angle'] - props2['angle']) * self.config.scoring['parallelism_penalty']

                # Simplified tick count for scoring
                num_ticks = self._count_ticks_between_lines(props1, props2, vertical_lines)
                tick_score = num_ticks * self.config.scoring['tick_count_weight']

                total_score = length_score + overlap_score - parallelism_penalty + tick_score

                if total_score > max_score:
                    max_score = total_score
                    # Ensure consistent ordering (upper, lower)
                    if props1['y_avg'] < props2['y_avg']:
                        best_pair = (props1['line'], props2['line'])
                    else:
                        best_pair = (props2['line'], props1['line'])

        if best_pair:
            self.logger.info(f"Found best ruler body pair with score: {max_score:.2f}")
            return best_pair

        self.logger.warning("Could not identify a suitable pair of ruler body lines.")
        return None

    def _count_ticks_between_lines(self, line1_props: Dict, line2_props: Dict, vertical_lines: List[np.ndarray]) -> int:
        """
        Performs a simplified count of vertical lines between two horizontal lines.

        This helper function is used exclusively for scoring potential ruler
        body pairs. It provides a quick estimate of how many tick-like lines
        exist in the region defined by the two horizontal lines, which is a
        strong indicator of a ruler.

        Args:
            line1_props: Properties dictionary for the first horizontal line.
            line2_props: Properties dictionary for the second horizontal line.
            vertical_lines: A list of all detected vertical lines.

        Returns:
            The number of vertical lines found between the two horizontal lines.
        """
        count = 0
        y_upper = min(line1_props['y_avg'], line2_props['y_avg'])
        y_lower = max(line1_props['y_avg'], line2_props['y_avg'])
        x_min_overlap = max(line1_props['x_coords'][0], line2_props['x_coords'][0])
        x_max_overlap = min(line1_props['x_coords'][1], line2_props['x_coords'][1])

        for v_line in vertical_lines:
            vx1, vy1, vx2, vy2 = v_line[0]
            vx_center = (vx1 + vx2) / 2
            # Check if tick is horizontally within the overlap region
            if not (x_min_overlap < vx_center < x_max_overlap):
                continue
            # Check if tick is vertically between the ruler lines
            if (min(vy1, vy2) < y_lower and max(vy1, vy2) > y_upper):
                count += 1
        return count

    def _filter_ticks_in_roi(self, vertical_lines: List[np.ndarray], roi: Tuple[int, int, int, int],
                             line_upper_y: float, line_lower_y: float) -> List[Tick]:
        """
        Filters and validates vertical lines to identify true tick marks.

        Once the ruler body is found, this method performs a more rigorous
        filtering of vertical lines within the ruler's region of interest (ROI).
        It checks for plausible tick length relative to the ruler's thickness
        and deduplicates tick marks that are too close together, using
        configurable parameters.

        Args:
            vertical_lines: A list of all detected vertical line segments.
            roi: The region of interest defined by the ruler body.
            line_upper_y: The average y-coordinate of the upper ruler line.
            line_lower_y: The average y-coordinate of the lower ruler line.

        Returns:
            A sorted and deduplicated list of validated `Tick` objects.
        """
        tick_candidates = []
        roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi
        ruler_thickness = line_lower_y - line_upper_y

        for line_seg in vertical_lines:
            x1, y1, x2, y2 = line_seg[0]
            tick_x_center = (x1 + x2) / 2
            tick_y_center = (y1 + y2) / 2
            tick_length = abs(y2 - y1)

            if not (roi_x_start <= tick_x_center <= roi_x_end and \
                    roi_y_start <= tick_y_center <= roi_y_end) :
                continue

            y_check_tolerance = ruler_thickness * self.config.tick_y_tolerance_factor
            if not ( (min(y1,y2) > line_upper_y - y_check_tolerance) and \
                       (max(y1,y2) < line_lower_y + y_check_tolerance) ):
                continue

            min_tick_len = max(self.config.abs_min_tick_length_px, ruler_thickness * self.config.min_tick_length_factor)
            max_tick_len = ruler_thickness * self.config.max_tick_length_factor
            if not (min_tick_len <= tick_length <= max_tick_len):
                continue
            tick_candidates.append({'x': tick_x_center, 'line': line_seg})

        sorted_ticks = sorted(tick_candidates, key=lambda t: t['x'])
        if not sorted_ticks: return []

        deduplicated_ticks = [Tick(x_position=sorted_ticks[0]['x'], line=sorted_ticks[0]['line'])]
        for i in range(1, len(sorted_ticks)):
            if (sorted_ticks[i]['x'] - deduplicated_ticks[-1].x_position) >= self.config.min_tick_separation_px:
                deduplicated_ticks.append(Tick(x_position=sorted_ticks[i]['x'], line=sorted_ticks[i]['line']))
        return deduplicated_ticks

    def _calculate_scale_from_ticks(self, ticks: List[Tick]) -> Optional[float]:
        """
        Calculates the spatial scale using the distances between tick marks.

        This method uses a robust statistical approach to determine the scale.
        It calculates all distances between adjacent ticks and uses the median
        distance as the most probable representation of the primary tick spacing.
        This is robust to outliers caused by missed or spurious ticks. It also
        calculates the coefficient of variation to warn if the tick spacing is
        highly irregular.

        Args:
            ticks: A list of validated and sorted `Tick` objects.

        Returns:
            The calculated scale in pixels per millimeter, or None if the scale
            could not be reliably determined.
        """
        if len(ticks) < 2:
            self.logger.warning("Not enough tick candidates (<2) to calculate distances.")
            return None

        distances = [ticks[i+1].x_position - ticks[i].x_position for i in range(len(ticks) - 1)]

        if not distances:
            self.logger.warning("No distances calculated between ticks.")
            return None

        self.logger.debug(f"Calculated distances between ticks: {distances}")

        median_spacing_px = float(np.median(distances))
        self.logger.info(f"Median tick spacing: {median_spacing_px:.2f} pixels.")

        if median_spacing_px <= 0:
            self.logger.error("Median spacing is zero or negative, cannot calculate scale.")
            return None

        if len(distances) > 1:
            std_dev_spacing = np.std(distances)
            coeff_of_variation = std_dev_spacing / median_spacing_px
            self.logger.debug(f"Tick spacing StdDev: {std_dev_spacing:.2f}, Coeff of Variation: {coeff_of_variation:.2f}")
            max_spacing_variation_coeff = 0.2
            if coeff_of_variation > max_spacing_variation_coeff:
                self.logger.warning(f"High variability in tick spacing (CoV = {coeff_of_variation:.2f}). Scale may be unreliable.")

        min_reliable_intervals = 2
        if len(distances) < min_reliable_intervals:
            self.logger.warning(f"Too few tick intervals ({len(distances)}) to reliably determine scale.")
            return None

        calculated_scale = median_spacing_px / self.config.expected_tick_distance_mm
        self.logger.info(f"Calculated scale: {calculated_scale:.2f} pixels / {self.config.expected_tick_distance_mm} mm.")
        return calculated_scale

    def _calculate_scale_with_deep_gp(self, ticks: List[Tick], image_width: int) -> Optional[float]:
        """
        Calculates the scale using the DeepGP model.
        """
        if len(ticks) < 3: # DeepGP might need a few ticks to find a pattern
            self.logger.warning("Not enough ticks for DeepGP model, falling back to statistical method.")
            return self._calculate_scale_from_ticks(ticks)

        # 1. Create a 1D signal from the tick positions
        signal = torch.zeros(1, 1, image_width)
        for tick in ticks:
            pos = int(round(tick.x_position))
            if 0 <= pos < image_width:
                signal[0, 0, pos] = 1.0

        # 2. Predict the GP parameters
        try:
            gp_params = self.deep_gp_model.predict_gp_params(signal)
            m0, m1, r = gp_params[0].tolist()
            self.logger.info(f"DeepGP predicted params (m0, m1, r): ({m0:.2f}, {m1:.2f}, {r:.4f})")

            # 3. Placeholder logic
            self.logger.warning("DeepGP integration is a placeholder. Returning statistical scale.")
            return self._calculate_scale_from_ticks(ticks)

        except Exception as e:
            self.logger.error(f"Error during DeepGP prediction: {e}")
            return self._calculate_scale_from_ticks(ticks)

    def detect(self, image: np.ndarray, exclusion_mask: Optional[np.ndarray] = None) -> Optional[Ruler]:
        """
        Executes the full ruler detection and scale calculation pipeline.

        This is the main public method of the class. It orchestrates the entire
        process from image preprocessing to returning a final `Ruler` object.

        Args:
            image: The input image as a NumPy array (can be BGR or grayscale).
            exclusion_mask: An optional boolean mask where `True` indicates
                            regions of the image to ignore during detection.
                            This is useful for masking out the primary sample
                            area to prevent it from interfering with ruler
                            detection.

        Returns:
            A `Ruler` object containing all information about the detected
            ruler (body lines, ticks, scale, mask, ROI), or `None` if no
            ruler could be reliably detected.
        """
        if not self.config.enabled:
            self.logger.info("Ruler detection is disabled. Skipping.")
            return None

        self.logger.debug(f"Starting ruler detection on image shape {image.shape}")
        original_height, original_width = image.shape[:2]

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        if exclusion_mask is not None:
            gray_image[exclusion_mask] = 0
            self.logger.debug(f"Applied exclusion mask covering {np.sum(exclusion_mask)} pixels")

        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, self.config.canny_threshold1, self.config.canny_threshold2)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, self.config.hough_threshold,
                                minLineLength=self.config.min_line_length,
                                maxLineGap=self.config.max_line_gap)
        if lines is None:
            self.logger.warning("No lines detected by Hough Transform.")
            return None
        self.logger.info(f"Raw lines detected: {len(lines)}")

        grouped_lines = self._filter_and_group_lines(lines, max_angle_diff_deg=10.0)
        horizontal_lines = grouped_lines["horizontal"]
        vertical_lines = grouped_lines["vertical"]
        self.logger.info(f"Filtered to {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical lines.")

        if not horizontal_lines: return None

        min_ruler_len = max(self.config.min_line_length, original_width // 8)
        ruler_body_lines = self._get_main_ruler_body_lines(horizontal_lines, vertical_lines, min_ruler_len)

        if ruler_body_lines is None: return None

        line_u, line_l = ruler_body_lines
        y_upper_avg = (line_u[0][1] + line_u[0][3]) / 2
        y_lower_avg = (line_l[0][1] + line_l[0][3]) / 2

        roi_x_start = int(min(line_u[0][0], line_u[0][2], line_l[0][0], line_l[0][2]))
        roi_x_end = int(max(line_u[0][0], line_u[0][2], line_l[0][0], line_l[0][2]))
        roi_y_padding = (y_lower_avg - y_upper_avg)
        roi_y_start = int(y_upper_avg - roi_y_padding)
        roi_y_end = int(y_lower_avg + roi_y_padding)

        roi = (max(0, roi_x_start), max(0, roi_y_start),
               min(original_width, roi_x_end), min(original_height, roi_y_end))

        self.logger.info(f"Ruler ROI defined: x({roi[0]}-{roi[2]}), y({roi[1]}-{roi[3]})")

        ticks = self._filter_ticks_in_roi(vertical_lines, roi, y_upper_avg, y_lower_avg)

        if len(ticks) < 2:
            self.logger.warning(f"Not enough tick candidates found ({len(ticks)}). Cannot calculate scale.")
            return None
        
        tick_positions = [f"{t.x_position:.0f}" for t in ticks]
        self.logger.info(f"Found {len(ticks)} tick candidates in ROI at x-positions: {tick_positions}")

        if self.config.use_deep_gp and self.deep_gp_model:
            calculated_scale = self._calculate_scale_with_deep_gp(ticks, original_width)
        else:
            calculated_scale = self._calculate_scale_from_ticks(ticks)

        if calculated_scale is None:
            self.logger.warning("Scale calculation failed, cannot create Ruler object.")
            return None

        ruler_mask = np.zeros((original_height, original_width), dtype=bool)
        ruler_top = int(y_upper_avg - 5)
        ruler_bottom = int(y_lower_avg + 5)
        ruler_left = int(roi[0] - 5)
        ruler_right = int(roi[2] + 5)
        ruler_mask[max(0, ruler_top):min(original_height, ruler_bottom),
                   max(0, ruler_left):min(original_width, ruler_right)] = True

        return Ruler(
            body_lines=ruler_body_lines,
            ticks=ticks,
            scale_px_per_mm=calculated_scale,
            mask=ruler_mask,
            roi=roi,
        )

    def detect_scale(self, image: np.ndarray) -> Optional[float]:
        """
        A convenience wrapper for `detect()` that returns only the scale.

        Args:
            image: The input image as a NumPy array.

        Returns:
            The calculated scale in pixels per millimeter, or None if detection fails.
        """
        ruler = self.detect(image, exclusion_mask=None)
        if ruler:
            return ruler.scale_px_per_mm
        return None

    def detect_scale_with_mask(self, image: np.ndarray, exclusion_mask: np.ndarray = None) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """
        A convenience wrapper for `detect()` that returns the scale and ruler mask.

        Args:
            image: The input image as a NumPy array.
            exclusion_mask: An optional boolean mask for areas to ignore.

        Returns:
            A tuple containing the calculated scale (or None) and the ruler
            mask (or None).
        """
        ruler = self.detect(image, exclusion_mask=exclusion_mask)
        if ruler:
            return ruler.scale_px_per_mm, ruler.mask
        return None, None

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    config = RulerDetectionConfig(
        enabled=True, min_line_length=30, max_line_gap=5,
        expected_tick_distance_mm=10.0,  # Assume major ticks are 1cm (10mm) apart
        canny_threshold1=50, canny_threshold2=150, hough_threshold=15
    )
    detector = RulerDetector(config=config)

    dummy_image_height = 300
    dummy_image_width = 500
    dummy_image = np.full((dummy_image_height, dummy_image_width, 3), (230, 230, 230), dtype=np.uint8)

    ruler_y_top, ruler_y_bottom = 140, 160
    cv2.line(dummy_image, (50, ruler_y_top), (450, ruler_y_top), (20, 20, 20), 2)
    cv2.line(dummy_image, (50, ruler_y_bottom), (450, ruler_y_bottom), (20, 20, 20), 2)

    tick_y_start, tick_y_end = ruler_y_top - 15, ruler_y_bottom + 15

    num_ticks = 5
    spacing = 75  # pixels, so 75 pixels = 10mm
    start_x = 70
    for i in range(num_ticks):
        x = start_x + i * spacing
        cv2.line(dummy_image, (x, tick_y_start), (x, tick_y_end), (10, 10, 10), 2)

    ruler = detector.detect(dummy_image)

    if ruler is not None and ruler.scale_px_per_mm is not None:
        logger.info(f"FINAL Calculated scale: {ruler.scale_px_per_mm:.2f} pixels/mm")
        expected_scale = spacing / config.expected_tick_distance_mm
        if abs(ruler.scale_px_per_mm - expected_scale) < 0.5:
            logger.info("Scale calculation seems correct for the dummy image.")
        else:
            logger.error(f"Scale calculation might be off. Expected around {expected_scale:.2f}")

        logger.info(f"Detected {len(ruler.ticks)} ticks.")
        logger.info(f"Ruler mask created with shape: {ruler.mask.shape} and {np.sum(ruler.mask)} foreground pixels.")

        # To visualize the result (optional, requires a display environment)
        # output_image = dummy_image.copy()
        # output_image[ruler.mask] = [0, 255, 0] # Highlight ruler area in green
        # for tick in ruler.ticks:
        #     p1 = (tick.line[0][0], tick.line[0][1])
        #     p2 = (tick.line[0][2], tick.line[0][3])
        #     cv2.line(output_image, p1, p2, (255, 0, 0), 2) # Draw detected ticks in blue
        # cv2.imshow("Detection Result", output_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        logger.error("Ruler detection FAILED for the dummy image.")
