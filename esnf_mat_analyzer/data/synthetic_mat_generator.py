"""
Synthetic Mat Generator for ESNF Mat Analyzer Validation.

Generates ground-truth thickness maps with known features for validating
the analysis pipeline's ability to preserve real thickness variations.

Features include:
- Radial gradients (simulating center-to-edge thickness variation)
- Gaussian spots (localized thickness features)
- Multi-frequency patterns (testing FFT filtering effects)
- Simulated gel boundaries (dark ring around deposition area)
- Saturation zones (testing white reference handling)
- Background with realistic noise

Author: ESNF Mat Analyzer Team
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum
import json
from pathlib import Path


class PatternType(Enum):
    """Types of synthetic patterns available."""
    UNIFORM = "uniform"
    RADIAL_GRADIENT = "radial_gradient"
    GAUSSIAN_SPOTS = "gaussian_spots"
    MULTI_FREQUENCY = "multi_frequency"
    REALISTIC_MAT = "realistic_mat"
    CALIBRATION_TARGET = "calibration_target"


@dataclass
class SyntheticMatConfig:
    """Configuration for synthetic mat generation."""
    
    # Image dimensions
    width: int = 800
    height: int = 800
    
    # Background properties
    background_level: float = 30.0  # 0-255, simulates black collection surface
    background_noise_std: float = 5.0  # Gaussian noise in background
    
    # Gel boundary properties (dark ring around deposition)
    gel_enabled: bool = True
    gel_inner_radius_ratio: float = 0.35  # Inner edge as fraction of image size
    gel_outer_radius_ratio: float = 0.42  # Outer edge as fraction of image size
    gel_intensity: float = 15.0  # Darker than background (wet polymer)
    gel_irregularity: float = 0.05  # Shape variation (0=perfect circle)
    
    # Mat/deposition area properties
    mat_radius_ratio: float = 0.33  # Mat region as fraction of image size
    mat_base_intensity: float = 180.0  # Base brightness of mat (0-255)
    mat_noise_std: float = 8.0  # Local thickness variation noise
    
    # Pattern-specific parameters
    pattern_type: PatternType = PatternType.REALISTIC_MAT
    
    # Radial gradient parameters
    radial_center_intensity: float = 220.0  # Center brightness
    radial_edge_intensity: float = 140.0  # Edge brightness
    radial_power: float = 1.5  # Gradient curve (1=linear, >1=steeper edge)
    
    # Gaussian spots parameters
    num_spots: int = 5
    spot_intensity_range: Tuple[float, float] = (20.0, 50.0)  # Added intensity
    spot_size_range: Tuple[float, float] = (0.02, 0.08)  # As fraction of mat radius
    
    # Multi-frequency parameters (for FFT validation)
    frequencies: List[float] = field(default_factory=lambda: [0.01, 0.03, 0.08, 0.15])
    frequency_amplitudes: List[float] = field(default_factory=lambda: [30.0, 20.0, 15.0, 10.0])
    
    # Saturation zones
    saturation_enabled: bool = False
    saturation_zones: int = 0  # Number of saturated regions
    saturation_threshold: float = 254.0
    
    # Random seed for reproducibility
    seed: Optional[int] = None


@dataclass
class SyntheticMatResult:
    """Result from synthetic mat generation with ground truth."""
    
    # Generated image (0-255 uint8)
    image: np.ndarray
    
    # Ground truth thickness map (0-100 normalized scale)
    ground_truth_thickness: np.ndarray
    
    # Binary masks for regions
    mat_mask: np.ndarray  # True where mat material exists
    gel_mask: np.ndarray  # True in gel region
    background_mask: np.ndarray  # True in background region
    saturation_mask: np.ndarray  # True where saturated
    
    # Metadata
    config: Optional[SyntheticMatConfig] = None
    reference_black: float = 0.0  # Ground truth black reference
    reference_white: float = 100.0  # Ground truth white reference
    
    # Feature locations for validation
    feature_locations: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export metadata to dictionary (without large arrays)."""
        return {
            "image_shape": list(self.image.shape),
            "reference_black": self.reference_black,
            "reference_white": self.reference_white,
            "feature_locations": self.feature_locations,
            "config": {
                "width": self.config.width,
                "height": self.config.height,
                "pattern_type": self.config.pattern_type.value,
                "background_level": self.config.background_level,
                "gel_enabled": self.config.gel_enabled,
                "mat_radius_ratio": self.config.mat_radius_ratio,
                "seed": self.config.seed,
            } if self.config else None
        }
    
    def save(self, output_dir: Path, name: str = "synthetic_mat"):
        """Save result to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image as PNG
        try:
            from PIL import Image
            img = Image.fromarray(self.image)
            img.save(output_dir / f"{name}_image.png")
        except ImportError:
            np.save(output_dir / f"{name}_image.npy", self.image)
        
        # Save ground truth and masks as NPY
        np.save(output_dir / f"{name}_ground_truth.npy", self.ground_truth_thickness)
        np.save(output_dir / f"{name}_mat_mask.npy", self.mat_mask)
        np.save(output_dir / f"{name}_gel_mask.npy", self.gel_mask)
        np.save(output_dir / f"{name}_saturation_mask.npy", self.saturation_mask)
        
        # Save metadata as JSON
        with open(output_dir / f"{name}_metadata.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class SyntheticMatGenerator:
    """
    Generator for synthetic nanofiber mat images with known ground truth.
    
    Used for validating the analysis pipeline, particularly:
    - FFT filtering effects on real features
    - Calibration accuracy
    - Thickness estimation accuracy
    - Uniformity metric validation
    """
    
    def __init__(self, config: Optional[SyntheticMatConfig] = None):
        """Initialize generator with configuration."""
        self.config = config or SyntheticMatConfig()
        self._rng = np.random.default_rng(self.config.seed)
    
    def generate(self, config: Optional[SyntheticMatConfig] = None) -> SyntheticMatResult:
        """
        Generate a synthetic mat image with ground truth.
        
        Args:
            config: Optional override configuration
            
        Returns:
            SyntheticMatResult with image, ground truth, and masks
        """
        cfg = config or self.config
        if cfg.seed is not None:
            self._rng = np.random.default_rng(cfg.seed)
        
        h, w = cfg.height, cfg.width
        center = (h // 2, w // 2)
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Initialize image with background
        image = np.full((h, w), cfg.background_level, dtype=np.float64)
        image += self._rng.normal(0, cfg.background_noise_std, (h, w))
        
        # Initialize ground truth (0 = background)
        ground_truth = np.zeros((h, w), dtype=np.float64)
        
        # Calculate region radii
        min_dim = min(h, w)
        mat_radius = cfg.mat_radius_ratio * min_dim / 2
        gel_inner_radius = cfg.gel_inner_radius_ratio * min_dim / 2
        gel_outer_radius = cfg.gel_outer_radius_ratio * min_dim / 2
        
        # Create irregular gel boundary if enabled
        gel_mask = np.zeros((h, w), dtype=bool)
        if cfg.gel_enabled:
            gel_mask = self._create_gel_region(
                h, w, center, gel_inner_radius, gel_outer_radius,
                cfg.gel_irregularity
            )
            image[gel_mask] = cfg.gel_intensity
            image[gel_mask] += self._rng.normal(0, cfg.background_noise_std * 0.5, np.sum(gel_mask))
        
        # Create mat region mask
        mat_mask = self._create_irregular_circle(
            h, w, center, mat_radius, cfg.gel_irregularity * 0.5
        )
        
        # Exclude gel from mat
        mat_mask = mat_mask & ~gel_mask
        
        # Generate pattern within mat region
        if cfg.pattern_type == PatternType.UNIFORM:
            pattern, gt_pattern = self._generate_uniform(cfg, mat_mask)
        elif cfg.pattern_type == PatternType.RADIAL_GRADIENT:
            pattern, gt_pattern = self._generate_radial_gradient(
                cfg, mat_mask, dist_from_center, mat_radius
            )
        elif cfg.pattern_type == PatternType.GAUSSIAN_SPOTS:
            pattern, gt_pattern = self._generate_gaussian_spots(
                cfg, mat_mask, center, mat_radius
            )
        elif cfg.pattern_type == PatternType.MULTI_FREQUENCY:
            pattern, gt_pattern = self._generate_multi_frequency(
                cfg, mat_mask, h, w
            )
        elif cfg.pattern_type == PatternType.REALISTIC_MAT:
            pattern, gt_pattern = self._generate_realistic_mat(
                cfg, mat_mask, dist_from_center, mat_radius, center
            )
        elif cfg.pattern_type == PatternType.CALIBRATION_TARGET:
            pattern, gt_pattern = self._generate_calibration_target(
                cfg, mat_mask, center, mat_radius
            )
        else:
            pattern, gt_pattern = self._generate_uniform(cfg, mat_mask)
        
        # Apply pattern to mat region
        image[mat_mask] = pattern[mat_mask]
        ground_truth[mat_mask] = gt_pattern[mat_mask]
        
        # Add saturation zones if enabled
        saturation_mask = np.zeros((h, w), dtype=bool)
        feature_locations = {"saturation_zones": [], "spots": []}
        
        if cfg.saturation_enabled and cfg.saturation_zones > 0:
            saturation_mask, sat_locations = self._add_saturation_zones(
                image, mat_mask, cfg, center, mat_radius
            )
            image[saturation_mask] = cfg.saturation_threshold
            ground_truth[saturation_mask] = 100.0  # Max thickness (saturated)
            feature_locations["saturation_zones"] = sat_locations
        
        # Add local noise to mat region
        mat_noise = self._rng.normal(0, cfg.mat_noise_std, (h, w))
        image[mat_mask] += mat_noise[mat_mask]
        
        # Clip to valid range
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Create background mask
        background_mask = ~mat_mask & ~gel_mask
        
        # Calculate reference values
        reference_black = cfg.background_level  # True background level
        reference_white = float(np.max(image[mat_mask])) if np.any(mat_mask) else 255.0
        
        return SyntheticMatResult(
            image=image,
            ground_truth_thickness=ground_truth,
            mat_mask=mat_mask,
            gel_mask=gel_mask,
            background_mask=background_mask,
            saturation_mask=saturation_mask,
            config=cfg,
            reference_black=reference_black,
            reference_white=reference_white,
            feature_locations=feature_locations,
        )
    
    def _create_irregular_circle(
        self, h: int, w: int, center: Tuple[int, int],
        radius: float, irregularity: float
    ) -> np.ndarray:
        """Create an irregular circular region."""
        y, x = np.ogrid[:h, :w]
        
        if irregularity <= 0:
            # Perfect circle
            dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            return dist <= radius
        
        # Add angular variation
        angle = np.arctan2(y - center[0], x - center[1])
        
        # Generate smooth random boundary variation
        n_harmonics = 8
        variation = np.zeros((h, w))
        for i in range(1, n_harmonics + 1):
            phase = self._rng.uniform(0, 2 * np.pi)
            amp = self._rng.uniform(0.5, 1.0) * irregularity * radius / i
            variation += amp * np.sin(i * angle + phase)
        
        dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        return dist <= (radius + variation)
    
    def _create_gel_region(
        self, h: int, w: int, center: Tuple[int, int],
        inner_radius: float, outer_radius: float, irregularity: float
    ) -> np.ndarray:
        """Create the gel boundary region (annulus with irregular edges)."""
        outer_mask = self._create_irregular_circle(h, w, center, outer_radius, irregularity)
        inner_mask = self._create_irregular_circle(h, w, center, inner_radius, irregularity * 0.8)
        return outer_mask & ~inner_mask
    
    def _generate_uniform(
        self, cfg: SyntheticMatConfig, mat_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate uniform thickness pattern."""
        h, w = mat_mask.shape
        pattern = np.full((h, w), cfg.mat_base_intensity)
        
        # Ground truth: uniform 50% thickness
        gt = np.full((h, w), 50.0)
        return pattern, gt
    
    def _generate_radial_gradient(
        self, cfg: SyntheticMatConfig, mat_mask: np.ndarray,
        dist_from_center: np.ndarray, mat_radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate radial gradient from center to edge."""
        # Normalize distance to [0, 1] within mat
        norm_dist = np.clip(dist_from_center / mat_radius, 0, 1)
        
        # Apply power law for gradient shape
        gradient_factor = norm_dist ** cfg.radial_power
        
        # Intensity gradient (center = bright, edge = dim)
        pattern = (
            cfg.radial_center_intensity * (1 - gradient_factor) +
            cfg.radial_edge_intensity * gradient_factor
        )
        
        # Ground truth: inverse of intensity (thicker = brighter)
        # Normalize to 0-100 scale
        intensity_range = cfg.radial_center_intensity - cfg.radial_edge_intensity
        gt = 100.0 * (1 - gradient_factor)  # Center = 100, edge = 0
        
        return pattern, gt
    
    def _generate_gaussian_spots(
        self, cfg: SyntheticMatConfig, mat_mask: np.ndarray,
        center: Tuple[int, int], mat_radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate pattern with localized Gaussian spots."""
        h, w = mat_mask.shape
        pattern = np.full((h, w), cfg.mat_base_intensity)
        gt = np.full((h, w), 50.0)  # Base thickness at 50%
        
        y, x = np.ogrid[:h, :w]
        spots = []
        
        for _ in range(cfg.num_spots):
            # Random spot location within mat
            angle = self._rng.uniform(0, 2 * np.pi)
            dist = self._rng.uniform(0, mat_radius * 0.8)
            spot_y = center[0] + dist * np.sin(angle)
            spot_x = center[1] + dist * np.cos(angle)
            
            # Random spot size and intensity
            size = self._rng.uniform(*cfg.spot_size_range) * mat_radius
            intensity = self._rng.uniform(*cfg.spot_intensity_range)
            
            # Add Gaussian spot
            spot_dist = np.sqrt((x - spot_x)**2 + (y - spot_y)**2)
            spot = intensity * np.exp(-0.5 * (spot_dist / size)**2)
            
            pattern += spot
            gt += (intensity / cfg.spot_intensity_range[1]) * 25 * np.exp(-0.5 * (spot_dist / size)**2)
            
            spots.append({"x": float(spot_x), "y": float(spot_y), "size": float(size)})
        
        return pattern, gt
    
    def _generate_multi_frequency(
        self, cfg: SyntheticMatConfig, mat_mask: np.ndarray,
        h: int, w: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate multi-frequency pattern for FFT validation.
        
        This creates known spatial frequencies that can be used to verify
        whether FFT filtering preserves or removes them.
        """
        pattern = np.full((h, w), cfg.mat_base_intensity)
        gt = np.full((h, w), 50.0)
        
        y, x = np.mgrid[:h, :w]
        
        for freq, amp in zip(cfg.frequencies, cfg.frequency_amplitudes):
            # Add sinusoidal pattern at this frequency
            # Frequency is in cycles per pixel
            phase_x = self._rng.uniform(0, 2 * np.pi)
            phase_y = self._rng.uniform(0, 2 * np.pi)
            
            wave = amp * (
                np.sin(2 * np.pi * freq * x + phase_x) +
                np.sin(2 * np.pi * freq * y + phase_y)
            ) / 2
            
            pattern += wave
            gt += (amp / cfg.frequency_amplitudes[0]) * 10 * (
                np.sin(2 * np.pi * freq * x + phase_x) +
                np.sin(2 * np.pi * freq * y + phase_y)
            ) / 2
        
        return pattern, gt
    
    def _generate_realistic_mat(
        self, cfg: SyntheticMatConfig, mat_mask: np.ndarray,
        dist_from_center: np.ndarray, mat_radius: float,
        center: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic mat pattern combining multiple features.
        
        Simulates typical electrospun mat characteristics:
        - Center tends to be thicker (radial gradient)
        - Local variations (spots)
        - Some texture/noise
        """
        h, w = mat_mask.shape
        
        # Start with radial gradient
        pattern, gt = self._generate_radial_gradient(
            cfg, mat_mask, dist_from_center, mat_radius
        )
        
        # Add some localized variations
        y, x = np.ogrid[:h, :w]
        n_variations = 3
        
        for _ in range(n_variations):
            angle = self._rng.uniform(0, 2 * np.pi)
            dist = self._rng.uniform(0.2, 0.6) * mat_radius
            var_y = center[0] + dist * np.sin(angle)
            var_x = center[1] + dist * np.cos(angle)
            
            size = self._rng.uniform(0.1, 0.25) * mat_radius
            intensity = self._rng.uniform(-20, 20)
            
            var_dist = np.sqrt((x - var_x)**2 + (y - var_y)**2)
            variation = intensity * np.exp(-0.5 * (var_dist / size)**2)
            
            pattern += variation
            gt += (intensity / 40) * 15 * np.exp(-0.5 * (var_dist / size)**2)
        
        # Add some medium-frequency texture
        freq = 0.02
        y_grid, x_grid = np.mgrid[:h, :w]
        texture = 8 * np.sin(2 * np.pi * freq * x_grid) * np.sin(2 * np.pi * freq * y_grid)
        pattern += texture
        gt += 3 * np.sin(2 * np.pi * freq * x_grid) * np.sin(2 * np.pi * freq * y_grid)
        
        return pattern, gt
    
    def _generate_calibration_target(
        self, cfg: SyntheticMatConfig, mat_mask: np.ndarray,
        center: Tuple[int, int], mat_radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate calibration target with known discrete thickness levels.
        
        Creates concentric rings with exact thickness values for calibration.
        """
        h, w = mat_mask.shape
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Define thickness levels (0, 25, 50, 75, 100)
        levels = [0, 25, 50, 75, 100]
        intensities = [cfg.background_level, 100, 150, 200, 240]
        
        pattern = np.full((h, w), cfg.mat_base_intensity)
        gt = np.full((h, w), 50.0)
        
        # Create concentric rings
        ring_width = mat_radius / (len(levels) + 1)
        
        for i, (level, intensity) in enumerate(zip(levels, intensities)):
            inner = i * ring_width
            outer = (i + 1) * ring_width
            ring_mask = (dist >= inner) & (dist < outer) & mat_mask
            pattern[ring_mask] = intensity
            gt[ring_mask] = level
        
        return pattern, gt
    
    def _add_saturation_zones(
        self, image: np.ndarray, mat_mask: np.ndarray,
        cfg: SyntheticMatConfig, center: Tuple[int, int], mat_radius: float
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Add saturated (white) regions representing very thick areas."""
        h, w = image.shape
        saturation_mask = np.zeros((h, w), dtype=bool)
        locations = []
        
        y, x = np.ogrid[:h, :w]
        
        for _ in range(cfg.saturation_zones):
            # Random location within mat
            angle = self._rng.uniform(0, 2 * np.pi)
            dist = self._rng.uniform(0, mat_radius * 0.5)
            sat_y = center[0] + dist * np.sin(angle)
            sat_x = center[1] + dist * np.cos(angle)
            
            # Small saturated region
            size = self._rng.uniform(0.03, 0.08) * mat_radius
            sat_dist = np.sqrt((x - sat_x)**2 + (y - sat_y)**2)
            zone = (sat_dist < size) & mat_mask
            
            saturation_mask |= zone
            locations.append({"x": float(sat_x), "y": float(sat_y), "radius": float(size)})
        
        return saturation_mask, locations
    
    def generate_validation_suite(
        self, output_dir: Optional[Path] = None
    ) -> Dict[str, SyntheticMatResult]:
        """
        Generate a complete suite of test images for validation.
        
        Returns dictionary of pattern type -> result.
        """
        results = {}
        
        # Generate each pattern type
        for pattern_type in PatternType:
            cfg = SyntheticMatConfig(
                pattern_type=pattern_type,
                seed=42,  # Reproducible
            )
            
            # Special config for multi-frequency
            if pattern_type == PatternType.MULTI_FREQUENCY:
                cfg.frequencies = [0.005, 0.01, 0.02, 0.05, 0.1]
                cfg.frequency_amplitudes = [40.0, 30.0, 20.0, 15.0, 10.0]
            
            result = self.generate(cfg)
            results[pattern_type.value] = result
            
            if output_dir:
                result.save(Path(output_dir), f"synthetic_{pattern_type.value}")
        
        return results


def generate_fft_validation_set(output_dir: Path) -> List[SyntheticMatResult]:
    """
    Generate specific images for validating FFT filtering behavior.
    
    Creates images with known spatial frequencies to test:
    1. What frequencies are preserved vs filtered
    2. Whether real features are being removed
    """
    generator = SyntheticMatGenerator()
    results = []
    
    # Test different frequency ranges
    frequency_sets = [
        ("low_freq", [0.002, 0.005], [40.0, 30.0]),
        ("mid_freq", [0.01, 0.02, 0.03], [30.0, 25.0, 20.0]),
        ("high_freq", [0.05, 0.08, 0.1], [20.0, 15.0, 10.0]),
        ("full_spectrum", [0.002, 0.01, 0.03, 0.08], [35.0, 30.0, 20.0, 15.0]),
    ]
    
    for name, freqs, amps in frequency_sets:
        cfg = SyntheticMatConfig(
            pattern_type=PatternType.MULTI_FREQUENCY,
            frequencies=freqs,
            frequency_amplitudes=amps,
            seed=42,
        )
        result = generator.generate(cfg)
        result.save(output_dir, f"fft_validation_{name}")
        results.append(result)
    
    return results


if __name__ == "__main__":
    # Demo: generate sample images
    from pathlib import Path
    
    output_dir = Path("test_data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = SyntheticMatGenerator()
    
    print("Generating validation suite...")
    results = generator.generate_validation_suite(output_dir)
    
    print(f"\nGenerated {len(results)} synthetic images:")
    for name, result in results.items():
        print(f"  - {name}: shape={result.image.shape}, "
              f"mat coverage={100*result.mat_mask.mean():.1f}%")
    
    print(f"\nGenerating FFT validation set...")
    fft_results = generate_fft_validation_set(output_dir)
    print(f"Generated {len(fft_results)} FFT validation images")
    
    print(f"\nAll files saved to: {output_dir}")
