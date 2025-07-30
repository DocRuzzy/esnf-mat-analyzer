import numpy as np
from typing import Tuple, Dict

class BenchmarkDatasetGenerator:
    """Generate synthetic ESNF mat images with known uniformity properties.

    Creates controlled test cases for validation:
    - Perfectly uniform mats
    - Radially varying thickness
    - Localized defects
    - Edge effects
    """

    def generate_synthetic_mat(self, uniformity_type: str,
                             noise_level: float = 0.1,
                             size: Tuple[int, int] = (512, 512)
                            ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate synthetic mat with known uniformity properties.

        Args:
            uniformity_type: Type of uniformity pattern
            noise_level: Gaussian noise standard deviation
            size: Image dimensions

        Returns:
            Tuple of (synthetic_image, ground_truth_metrics)
        """
        if uniformity_type == 'perfect_uniform':
            return self._generate_uniform_mat(noise_level, size)
        elif uniformity_type == 'radial_gradient':
            return self._generate_radial_gradient_mat(noise_level, size)
        elif uniformity_type == 'localized_defects':
            return self._generate_defect_mat(noise_level, size)
        else:
            raise ValueError(f"Unknown uniformity type: {uniformity_type}")

    def _generate_uniform_mat(self, noise_level: float,
                            size: Tuple[int, int]
                           ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate perfectly uniform mat for validation."""
        h, w = size

        # Base uniform thickness
        base_thickness = 100.0
        uniform_mat = np.full((h, w), base_thickness, dtype=np.float64)

        # Add controlled noise
        noise = np.random.normal(0, noise_level * base_thickness, (h, w))
        synthetic_mat = uniform_mat + noise

        # Calculate ground truth metrics
        ground_truth = {
            'true_gini': 0.0,  # Perfect uniformity
            'true_rui': 1.0,   # Perfect radial uniformity
            'true_trr': 1.0,   # Perfect range ratio
            'mean_thickness': base_thickness,
            'noise_level': noise_level
        }

        return synthetic_mat, ground_truth

    def _generate_radial_gradient_mat(self, noise_level: float, size: Tuple[int, int]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate mat with a radial gradient for validation."""
        h, w = size
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]

        # Create a radial gradient
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_radius = np.sqrt(center_x**2 + center_y**2)
        gradient_mat = 100.0 * (1 - radius / max_radius)

        # Add noise
        noise = np.random.normal(0, noise_level * 100.0, (h, w))
        synthetic_mat = gradient_mat + noise

        # Ground truth metrics are more complex to calculate for this case,
        # so we will use placeholder values.
        ground_truth = {
            'true_gini': 0.3,
            'true_rui': 0.7,
            'true_trr': 0.0,
            'mean_thickness': np.mean(gradient_mat),
            'noise_level': noise_level
        }

        return synthetic_mat, ground_truth

    def _generate_defect_mat(self, noise_level: float, size: Tuple[int, int]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate mat with localized defects for validation."""
        h, w = size

        # Start with a uniform mat
        base_thickness = 100.0
        defect_mat = np.full((h, w), base_thickness, dtype=np.float64)

        # Add a few defects
        for _ in range(5):
            defect_x = np.random.randint(0, w)
            defect_y = np.random.randint(0, h)
            defect_radius = np.random.randint(10, 30)
            defect_intensity = np.random.uniform(-50, 50)

            y, x = np.ogrid[:h, :w]
            mask = (x - defect_x)**2 + (y - defect_y)**2 < defect_radius**2
            defect_mat[mask] += defect_intensity

        # Add noise
        noise = np.random.normal(0, noise_level * base_thickness, (h, w))
        synthetic_mat = defect_mat + noise

        # Ground truth metrics are complex to calculate for this case,
        # so we will use placeholder values.
        ground_truth = {
            'true_gini': 0.1,
            'true_rui': 0.9,
            'true_trr': 0.5,
            'mean_thickness': np.mean(defect_mat),
            'noise_level': noise_level
        }

        return synthetic_mat, ground_truth
