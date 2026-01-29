"""
Intensity-to-Thickness Calibration for ESNF Mat Analyzer.

Provides calibration methods for converting pixel intensity to normalized
thickness scale (0-100) and optionally to physical units (micrometers).

Calibration Strategy:
1. Black Reference (0): Background intensity outside deposition area
2. White Reference (100): Either saturation level OR maximum intensity if no saturation
3. Normalized Scale: Linear mapping from black/white references to 0-100

For physical calibration (optional):
- User provides a known thickness measurement at a marked location
- Converts normalized scale to absolute micrometers

Author: ESNF Mat Analyzer Team
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List, Any
import logging


@dataclass
class CalibrationPoint:
    """A calibration reference point in the image."""
    x: int
    y: int
    normalized_value: float  # 0-100 scale
    physical_value_um: Optional[float] = None  # Physical thickness in micrometers
    label: str = ""


@dataclass
class CalibrationResult:
    """Result from intensity calibration."""
    
    # Reference levels (0-255 intensity scale)
    black_reference: float  # Background intensity (maps to 0)
    white_reference: float  # Max/saturation intensity (maps to 100)
    
    # Calibration status
    saturation_detected: bool = False
    saturation_regions: int = 0
    saturation_warning: str = ""
    
    # Physical calibration (if provided)
    physical_calibration_available: bool = False
    um_per_normalized_unit: float = 0.0  # Micrometers per normalized unit
    calibration_point: Optional[CalibrationPoint] = None
    
    # Quality metrics
    dynamic_range: float = 0.0  # white_reference - black_reference
    quality_score: float = 0.0  # 0-1 calibration quality
    
    # Metadata
    method: str = "auto"
    notes: List[str] = field(default_factory=list)
    
    def intensity_to_normalized(self, intensity: np.ndarray) -> np.ndarray:
        """Convert intensity values to normalized 0-100 scale."""
        if self.dynamic_range <= 0:
            return np.zeros_like(intensity, dtype=float)
        
        normalized = 100.0 * (intensity.astype(float) - self.black_reference) / self.dynamic_range
        return np.clip(normalized, 0, 100)
    
    def normalized_to_physical(self, normalized: np.ndarray) -> Optional[np.ndarray]:
        """Convert normalized values to physical micrometers (if calibrated)."""
        if not self.physical_calibration_available:
            return None
        return normalized * self.um_per_normalized_unit


@dataclass 
class DepositionParameters:
    """
    Optional electrospinning process parameters for theoretical mass estimation.
    
    Note: Actual deposited mass may differ due to:
    - Polymer absorbed into gel region (unknown loss)
    - Fiber divergence/collection efficiency
    - Evaporation losses
    
    These parameters provide a ROUGH ESTIMATE only.
    """
    flow_rate_ml_per_hr: Optional[float] = None
    collection_time_min: Optional[float] = None
    polymer_concentration_wt_pct: Optional[float] = None
    solution_density_g_per_ml: float = 1.0  # Default to water-like
    
    def calculate_theoretical_mass_mg(self) -> Optional[float]:
        """
        Calculate theoretical deposited polymer mass (mg).
        
        Returns:
            Theoretical mass in mg, or None if parameters incomplete
        """
        if any(p is None for p in [
            self.flow_rate_ml_per_hr,
            self.collection_time_min,
            self.polymer_concentration_wt_pct
        ]):
            return None
        
        # At this point we know all values are not None
        flow_rate = float(self.flow_rate_ml_per_hr)  # type: ignore[arg-type]
        collection_time = float(self.collection_time_min)  # type: ignore[arg-type]
        concentration = float(self.polymer_concentration_wt_pct)  # type: ignore[arg-type]
        
        # Volume dispensed (mL)
        time_hr = collection_time / 60.0
        volume_ml = flow_rate * time_hr
        
        # Mass of solution (g)
        solution_mass_g = volume_ml * self.solution_density_g_per_ml
        
        # Mass of polymer (mg)
        polymer_mass_mg = solution_mass_g * (concentration / 100.0) * 1000
        
        return polymer_mass_mg


class IntensityCalibrator:
    """
    Calibrator for converting image intensity to normalized thickness scale.
    
    The 0-100 normalized scale:
    - 0: Background level (no nanofibers, just collection surface)
    - 100: Maximum thickness (saturation or highest observed intensity)
    
    Saturation Warning: When saturation is detected, the true thickness
    in those regions is UNKNOWN (could be higher than 100).
    """
    
    def __init__(
        self,
        saturation_threshold: int = 254,
        background_percentile: float = 5.0,
        white_percentile: float = 99.5,
    ):
        """
        Initialize calibrator.
        
        Args:
            saturation_threshold: Intensity above which pixels are considered saturated
            background_percentile: Percentile to use for background estimation
            white_percentile: Percentile to use for white reference (if no saturation)
        """
        self.saturation_threshold = saturation_threshold
        self.background_percentile = background_percentile
        self.white_percentile = white_percentile
        self.logger = logging.getLogger(__name__)
    
    def calibrate(
        self,
        image: np.ndarray,
        mat_mask: np.ndarray,
        background_mask: Optional[np.ndarray] = None,
        user_black_reference: Optional[float] = None,
        user_white_reference: Optional[float] = None,
    ) -> CalibrationResult:
        """
        Calibrate intensity-to-thickness mapping.
        
        Args:
            image: Grayscale image (0-255)
            mat_mask: Mask of mat region (where thickness exists)
            background_mask: Optional mask of background region for black reference
            user_black_reference: Optional user-specified black reference intensity
            user_white_reference: Optional user-specified white reference intensity
            
        Returns:
            CalibrationResult with reference levels and conversion methods
        """
        notes = []
        
        # Determine black reference
        if user_black_reference is not None:
            black_ref = user_black_reference
            notes.append(f"Using user-specified black reference: {black_ref:.1f}")
        elif background_mask is not None and np.sum(background_mask) > 0:
            # Use background region
            bg_values = image[background_mask.astype(bool)]
            black_ref = float(np.percentile(bg_values, self.background_percentile))
            notes.append(f"Black reference from background region: {black_ref:.1f}")
        else:
            # Estimate from image edges or low-intensity regions outside mat
            outside_mat = ~mat_mask.astype(bool)
            if np.sum(outside_mat) > 0:
                outside_values = image[outside_mat]
                black_ref = float(np.percentile(outside_values, self.background_percentile))
                notes.append(f"Black reference from non-mat region: {black_ref:.1f}")
            else:
                black_ref = float(np.min(image))
                notes.append(f"Black reference from image minimum: {black_ref:.1f}")
        
        # Detect saturation in mat region
        mat_values = image[mat_mask.astype(bool)]
        saturation_mask = mat_values >= self.saturation_threshold
        saturation_detected = np.any(saturation_mask)
        saturation_pct = 100.0 * np.sum(saturation_mask) / len(mat_values) if len(mat_values) > 0 else 0
        
        # Determine white reference
        if user_white_reference is not None:
            white_ref = user_white_reference
            notes.append(f"Using user-specified white reference: {white_ref:.1f}")
        elif saturation_detected:
            # Use saturation level as white reference
            white_ref = float(self.saturation_threshold)
            notes.append(f"White reference at saturation threshold: {white_ref:.1f}")
            notes.append(f"WARNING: {saturation_pct:.1f}% of mat is saturated - true thickness unknown in those regions")
        else:
            # Use high percentile of mat values
            white_ref = float(np.percentile(mat_values, self.white_percentile))
            notes.append(f"White reference from {self.white_percentile}th percentile: {white_ref:.1f}")
        
        # Ensure valid range
        dynamic_range = white_ref - black_ref
        if dynamic_range <= 0:
            self.logger.warning("Invalid dynamic range, adjusting references")
            black_ref = float(np.min(mat_values)) if len(mat_values) > 0 else 0
            white_ref = float(np.max(mat_values)) if len(mat_values) > 0 else 255
            dynamic_range = white_ref - black_ref
            notes.append("Adjusted references due to invalid dynamic range")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            dynamic_range, saturation_pct, len(mat_values)
        )
        
        # Saturation warning
        saturation_warning = ""
        if saturation_detected:
            saturation_warning = (
                f"Saturation detected in {saturation_pct:.1f}% of mat region. "
                f"Thickness values in saturated areas are UNDERESTIMATED. "
                f"Consider reducing exposure or using ND filter for accurate measurement."
            )
        
        self.logger.info(
            f"Calibration complete: black={black_ref:.1f}, white={white_ref:.1f}, "
            f"range={dynamic_range:.1f}, saturation={saturation_pct:.1f}%"
        )
        
        return CalibrationResult(
            black_reference=black_ref,
            white_reference=white_ref,
            saturation_detected=bool(saturation_detected),
            saturation_regions=int(np.sum(saturation_mask)),
            saturation_warning=saturation_warning,
            dynamic_range=dynamic_range,
            quality_score=quality_score,
            method="auto" if user_black_reference is None and user_white_reference is None else "manual",
            notes=notes,
        )
    
    def add_physical_calibration(
        self,
        result: CalibrationResult,
        normalized_value: float,
        physical_thickness_um: float,
        location: Tuple[int, int] = (0, 0),
        label: str = "Calibration point"
    ) -> CalibrationResult:
        """
        Add physical thickness calibration to an existing result.
        
        Args:
            result: Existing calibration result
            normalized_value: Normalized thickness value (0-100) at calibration point
            physical_thickness_um: Known physical thickness in micrometers
            location: (x, y) pixel coordinates of calibration point
            label: Description of calibration point
            
        Returns:
            Updated CalibrationResult with physical calibration
        """
        if normalized_value <= 0:
            self.logger.error("Cannot calibrate: normalized value must be positive")
            return result
        
        # Calculate conversion factor
        um_per_unit = physical_thickness_um / normalized_value
        
        # Create calibration point
        cal_point = CalibrationPoint(
            x=location[0],
            y=location[1],
            normalized_value=normalized_value,
            physical_value_um=physical_thickness_um,
            label=label,
        )
        
        # Update result
        result.physical_calibration_available = True
        result.um_per_normalized_unit = um_per_unit
        result.calibration_point = cal_point
        result.notes.append(
            f"Physical calibration: {physical_thickness_um:.1f} µm at "
            f"normalized value {normalized_value:.1f} "
            f"(scale: {um_per_unit:.2f} µm per unit)"
        )
        
        self.logger.info(
            f"Physical calibration added: {um_per_unit:.2f} µm per normalized unit"
        )
        
        return result
    
    def _calculate_quality_score(
        self,
        dynamic_range: float,
        saturation_pct: float,
        mat_pixels: int
    ) -> float:
        """Calculate calibration quality score (0-1)."""
        score = 0.0
        
        # Dynamic range contribution (want at least 100 levels)
        if dynamic_range >= 150:
            score += 0.4
        elif dynamic_range >= 100:
            score += 0.3
        elif dynamic_range >= 50:
            score += 0.2
        else:
            score += 0.1
        
        # Saturation penalty
        if saturation_pct < 1:
            score += 0.3
        elif saturation_pct < 5:
            score += 0.2
        elif saturation_pct < 20:
            score += 0.1
        # Heavy saturation = no additional points
        
        # Sample size contribution
        if mat_pixels >= 10000:
            score += 0.3
        elif mat_pixels >= 1000:
            score += 0.2
        elif mat_pixels >= 100:
            score += 0.1
        
        return min(score, 1.0)


def create_normalized_thickness_map(
    image: np.ndarray,
    calibration: CalibrationResult,
    mat_mask: np.ndarray
) -> np.ndarray:
    """
    Create normalized thickness map (0-100 scale) from image.
    
    Args:
        image: Grayscale image
        calibration: Calibration result
        mat_mask: Mat region mask
        
    Returns:
        Thickness map with values 0-100, background set to 0
    """
    # Convert to normalized scale
    thickness = calibration.intensity_to_normalized(image)
    
    # Set non-mat regions to 0
    thickness[~mat_mask.astype(bool)] = 0
    
    return thickness


def create_physical_thickness_map(
    image: np.ndarray,
    calibration: CalibrationResult,
    mat_mask: np.ndarray
) -> Optional[np.ndarray]:
    """
    Create physical thickness map in micrometers.
    
    Args:
        image: Grayscale image
        calibration: Calibration result (must have physical calibration)
        mat_mask: Mat region mask
        
    Returns:
        Thickness map in µm, or None if physical calibration not available
    """
    if not calibration.physical_calibration_available:
        return None
    
    normalized = create_normalized_thickness_map(image, calibration, mat_mask)
    return calibration.normalized_to_physical(normalized)


if __name__ == "__main__":
    # Demo/test
    print("Intensity Calibration Module")
    print("=" * 40)
    
    # Create synthetic test image
    h, w = 400, 400
    test_image = np.zeros((h, w), dtype=np.uint8)
    
    # Background: 50 intensity
    test_image[:] = 50
    
    # Mat: gradient from 100 to 220
    y, x = np.ogrid[:h, :w]
    center = (h//2, w//2)
    dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    mat_mask = dist < 150
    test_image[mat_mask] = np.clip(220 - dist[mat_mask] * 0.5, 100, 220).astype(np.uint8)
    
    # Add some saturation
    inner_mask = dist < 30
    test_image[inner_mask] = 255
    
    # Background mask
    bg_mask = dist > 180
    
    # Calibrate
    calibrator = IntensityCalibrator()
    result = calibrator.calibrate(test_image, mat_mask, bg_mask)
    
    print(f"Black reference: {result.black_reference:.1f}")
    print(f"White reference: {result.white_reference:.1f}")
    print(f"Dynamic range: {result.dynamic_range:.1f}")
    print(f"Saturation detected: {result.saturation_detected}")
    print(f"Quality score: {result.quality_score:.2f}")
    
    if result.saturation_warning:
        print(f"\n⚠️ {result.saturation_warning}")
    
    print("\nNotes:")
    for note in result.notes:
        print(f"  - {note}")
    
    # Test normalized conversion
    thickness = create_normalized_thickness_map(test_image, result, mat_mask)
    print(f"\nNormalized thickness range: {thickness[mat_mask].min():.1f} - {thickness[mat_mask].max():.1f}")
    
    # Add physical calibration
    result = calibrator.add_physical_calibration(
        result,
        normalized_value=50.0,
        physical_thickness_um=25.0,
        location=(200, 200),
        label="Measured with profilometer"
    )
    
    print(f"\nPhysical calibration: {result.um_per_normalized_unit:.2f} µm per unit")
    
    physical = create_physical_thickness_map(test_image, result, mat_mask)
    if physical is not None:
        print(f"Physical thickness range: {physical[mat_mask].min():.1f} - {physical[mat_mask].max():.1f} µm")
