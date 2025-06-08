import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict
from esnf_mat_analyzer.core.data_types import RulerDetectionConfig
import math

class RulerDetector:
    """
    Detects a ruler in an image and calculates the scale (pixels per metric unit).
    """

    def __init__(self, config: RulerDetectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"RulerDetector initialized with config: {self.config}")

    def _filter_and_group_lines(self, lines: np.ndarray, max_angle_diff_deg: float = 5.0) -> Dict[str, List[np.ndarray]]:
        """Filters lines and groups them into horizontal and vertical sets."""
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

    def _get_main_ruler_body_lines(self, horizontal_lines: List[np.ndarray],
                                   min_length: int = 50,
                                   image_height: int = 0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Identifies the two primary parallel lines forming the ruler body."""
        if len(horizontal_lines) < 2: return None
        line_lengths = []
        for i, line_seg in enumerate(horizontal_lines):
            x1, y1, x2, y2 = line_seg[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length >= min_length:
                line_lengths.append({'index': i, 'length': length, 'y_avg': (y1 + y2) / 2, 'line': line_seg,
                                     'x_coords': (min(x1,x2), max(x1,x2))}) # Store x extents
        if len(line_lengths) < 2: return None
        sorted_lines = sorted(line_lengths, key=lambda item: item['length'], reverse=True)

        best_pair = None
        max_combined_length = 0

        for i in range(len(sorted_lines)):
            for j in range(i + 1, len(sorted_lines)):
                line1_data = sorted_lines[i]
                line2_data = sorted_lines[j]
                y_diff = abs(line1_data['y_avg'] - line2_data['y_avg'])
                min_ruler_thickness = max(5, image_height * 0.01)
                max_ruler_thickness = max(15,image_height * 0.1) # Ensure max_ruler_thickness is at least 15px

                if min_ruler_thickness < y_diff < max_ruler_thickness:
                    # Check for significant x-overlap
                    overlap_start = max(line1_data['x_coords'][0], line2_data['x_coords'][0])
                    overlap_end = min(line1_data['x_coords'][1], line2_data['x_coords'][1])
                    x_overlap = overlap_end - overlap_start

                    if x_overlap > min_length * 0.75: # Require substantial overlap (e.g. 75% of min_length)
                        current_combined_length = line1_data['length'] + line2_data['length']
                        if current_combined_length > max_combined_length:
                            max_combined_length = current_combined_length
                            if line1_data['y_avg'] < line2_data['y_avg']:
                                best_pair = (line1_data['line'], line2_data['line'])
                            else:
                                best_pair = (line2_data['line'], line1_data['line'])
        if best_pair:
             self.logger.info(f"Selected ruler body lines based on length, y-separation, and overlap.")
             return best_pair
        self.logger.warning("Could not identify a suitable pair of ruler body lines.")
        return None

    def _filter_ticks_in_roi(self, vertical_lines: List[np.ndarray], roi: Tuple[int, int, int, int],
                             line_upper_y: float, line_lower_y: float) -> List[Dict]:
        """Filters vertical lines to keep those likely to be ticks within the ROI."""
        tick_candidates = []
        roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi

        for line_seg in vertical_lines:
            x1, y1, x2, y2 = line_seg[0]
            tick_x_center = (x1 + x2) / 2
            tick_y_center = (y1 + y2) / 2
            tick_length = abs(y2 - y1)

            if not (roi_x_start <= tick_x_center <= roi_x_end and \
                    roi_y_start <= tick_y_center <= roi_y_end) : # Ensure center of tick is in ROI
                continue

            y_check_tolerance = (line_lower_y - line_upper_y) * 0.75 # Allow ticks to be 75% outside main body lines
            if not ( (min(y1,y2) > line_upper_y - y_check_tolerance) and \
                       (max(y1,y2) < line_lower_y + y_check_tolerance) ):
                continue
            min_tick_len = max(5.0, (line_lower_y - line_upper_y) * 0.3) # Must be at least 30% of ruler thickness or 5px
            max_tick_len = (line_lower_y - line_upper_y) * 3.0 # Not more than 3x ruler thickness
            if not (min_tick_len <= tick_length <= max_tick_len):
                continue
            tick_candidates.append({'x': tick_x_center, 'y': tick_y_center, 'len': tick_length, 'line': line_seg})

        sorted_ticks = sorted(tick_candidates, key=lambda t: t['x'])
        # Filter out ticks that are too close to each other (likely noise or same tick detected multiple times)
        if not sorted_ticks: return []

        deduplicated_ticks = [sorted_ticks[0]]
        min_px_separation_between_ticks = 5 # Could be a config
        for i in range(1, len(sorted_ticks)):
            if (sorted_ticks[i]['x'] - deduplicated_ticks[-1]['x']) >= min_px_separation_between_ticks:
                deduplicated_ticks.append(sorted_ticks[i])
        return deduplicated_ticks

    def _calculate_scale_from_ticks(self, tick_candidates: List[Dict]) -> Optional[float]:
        """Calculates scale from a list of tick candidates."""
        if len(tick_candidates) < 2:
            self.logger.warning("Not enough tick candidates (<2) to calculate distances.")
            return None

        distances = []
        for i in range(len(tick_candidates) - 1):
            dist = tick_candidates[i+1]['x'] - tick_candidates[i]['x']
            distances.append(dist)

        if not distances:
            self.logger.warning("No distances calculated between ticks.")
            return None

        self.logger.debug(f"Calculated distances between ticks: {distances}")

        # Use median distance as a robust measure of typical spacing
        median_spacing_px = float(np.median(distances))
        self.logger.info(f"Median tick spacing: {median_spacing_px:.2f} pixels.")

        if median_spacing_px <= 0: # Should not happen if ticks are sorted and deduplicated
            self.logger.error("Median spacing is zero or negative, cannot calculate scale.")
            return None

        # Check variability of distances. If too high, result might be unreliable.
        if len(distances) > 1: # Need at least two distances to calculate std dev
            std_dev_spacing = np.std(distances)
            coeff_of_variation = std_dev_spacing / median_spacing_px
            self.logger.debug(f"Tick spacing StdDev: {std_dev_spacing:.2f}, Coeff of Variation: {coeff_of_variation:.2f}")
            # Tolerance for CoV could be a config, e.g. 0.2 (20% variation)
            max_spacing_variation_coeff = 0.25
            if coeff_of_variation > max_spacing_variation_coeff:
                self.logger.warning(f"High variability in tick spacing (CoV = {coeff_of_variation:.2f}). Scale may be unreliable.")
                # Optionally return None here if too unreliable
                # return None

        # Minimum number of intervals to consider the pattern reliable
        min_reliable_intervals = 2 # e.g., at least 3 ticks making 2 intervals
        if len(distances) < min_reliable_intervals:
            self.logger.warning(f"Too few tick intervals ({len(distances)}) to reliably determine scale.")
            return None

        calculated_scale = median_spacing_px / self.config.expected_tick_distance_mm
        self.logger.info(f"Calculated scale: {calculated_scale:.2f} pixels / {self.config.expected_tick_distance_mm} mm.")
        return calculated_scale


    def detect_scale(self, image: np.ndarray) -> Optional[float]:
        if not self.config.enabled:
            self.logger.info("Ruler detection is disabled. Skipping.")
            return None

        self.logger.debug(f"Starting scale detection on image shape {image.shape}")
        original_height, original_width = image.shape[:2]

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, self.config.canny_threshold1, self.config.canny_threshold2)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, self.config.hough_threshold,
                                minLineLength=self.config.min_line_length,
                                maxLineGap=self.config.max_line_gap)
        if lines is None:
            self.logger.warning("No lines detected.")
            return None
        self.logger.info(f"Raw lines: {len(lines)}")

        grouped_lines = self._filter_and_group_lines(lines, max_angle_diff_deg=10.0)
        horizontal_lines = grouped_lines["horizontal"]
        vertical_lines = grouped_lines["vertical"]
        self.logger.info(f"H-lines: {len(horizontal_lines)}, V-lines: {len(vertical_lines)}")

        if not horizontal_lines: return None

        min_ruler_len = max(self.config.min_line_length, original_width // 8)
        ruler_body_cand = self._get_main_ruler_body_lines(horizontal_lines, min_ruler_len, original_height=original_height)

        if ruler_body_cand is None: return None
        line_u, line_l = ruler_body_cand
        y_upper_avg = (line_u[0][1] + line_u[0][3]) / 2
        y_lower_avg = (line_l[0][1] + line_l[0][3]) / 2

        roi_x_start = int(min(line_u[0][0], line_u[0][2], line_l[0][0], line_l[0][2]))
        roi_x_end = int(max(line_u[0][0], line_u[0][2], line_l[0][0], line_l[0][2]))
        roi_y_padding = (y_lower_avg - y_upper_avg)
        roi_y_start = int(y_upper_avg - roi_y_padding)
        roi_y_end = int(y_lower_avg + roi_y_padding)

        roi_x_start = max(0, roi_x_start); roi_y_start = max(0, roi_y_start)
        roi_x_end = min(original_width, roi_x_end); roi_y_end = min(original_height, roi_y_end)

        if (roi_x_end - roi_x_start) < min_ruler_len * 0.5 or \
           (roi_y_end - roi_y_start) < max(5, (y_lower_avg - y_upper_avg) * 0.5) : # ROI height check
            self.logger.warning("Defined ROI is too small. Aborting.")
            return None
        self.logger.info(f"Ruler ROI: x({roi_x_start}-{roi_x_end}), y({roi_y_start}-{roi_y_end})")

        tick_candidates = self._filter_ticks_in_roi(vertical_lines,
                                                    (roi_x_start, roi_y_start, roi_x_end, roi_y_end),
                                                    y_upper_avg, y_lower_avg)

        if len(tick_candidates) < 2: # Need at least 2 ticks to form an interval
            self.logger.warning(f"Not enough tick candidates found ({len(tick_candidates)}). Cannot calculate scale.")
            return None
        self.logger.info(f"Found {len(tick_candidates)} tick candidates in ROI: {[f'{t["x"]:.0f}' for t in tick_candidates]}")

        # Calculate scale from these ticks
        calculated_scale = self._calculate_scale_from_ticks(tick_candidates)

        if calculated_scale is not None:
            self.logger.info(f"Successfully calculated scale: {calculated_scale:.2f} pixels/mm (based on expected {self.config.expected_tick_distance_mm} mm interval)")
            return calculated_scale
        else:
            self.logger.warning("Failed to calculate a reliable scale from detected ticks.")
            return None

# (Keep the if __name__ == '__main__': block)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    config = RulerDetectionConfig(
        enabled=True, min_line_length=30, max_line_gap=5,
        expected_tick_distance_mm=10.0, # Assume major ticks are 1cm (10mm) apart
        canny_threshold1=50, canny_threshold2=150, hough_threshold=15
    )
    detector = RulerDetector(config=config)

    dummy_image_height = 300 # Increased height for better ruler visibility
    dummy_image_width = 500 # Increased width
    dummy_image = np.full((dummy_image_height, dummy_image_width, 3), (230, 230, 230), dtype=np.uint8)

    ruler_y_top, ruler_y_bottom = 140, 160
    cv2.line(dummy_image, (50, ruler_y_top), (450, ruler_y_top), (20, 20, 20), 2)
    cv2.line(dummy_image, (50, ruler_y_bottom), (450, ruler_y_bottom), (20, 20, 20), 2)

    tick_y_start, tick_y_end = ruler_y_top - 15, ruler_y_bottom + 15

    num_ticks = 5 # 5 ticks means 4 intervals
    spacing = 75 # pixels, so 75 pixels = 10mm (expected_tick_distance_mm)
    start_x = 70
    for i in range(num_ticks):
        x = start_x + i * spacing
        cv2.line(dummy_image, (x, tick_y_start), (x, tick_y_end), (10, 10, 10), 2) # Thicker ticks

    # cv2.imshow("Dummy Ruler Image For Detection", dummy_image)
    # cv2.waitKey(0)
    scale = detector.detect_scale(dummy_image)

    if scale is not None:
        logger.info(f"FINAL Calculated scale: {scale:.2f} pixels/mm")
        # Expected: 75 pixels / 10 mm = 7.5 pixels/mm
        if abs(scale - (spacing / config.expected_tick_distance_mm)) < 0.5:
             logger.info("Scale calculation seems correct for the dummy image.")
        else:
             logger.error(f"Scale calculation might be off. Expected around {spacing/config.expected_tick_distance_mm:.2f}")
    else:
        logger.error("Scale detection FAILED for the dummy image.")
    # cv2.destroyAllWindows()
