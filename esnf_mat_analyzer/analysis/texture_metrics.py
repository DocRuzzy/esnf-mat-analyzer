import numpy as np
from typing import Dict
from ..core.interfaces import UniformityMetricInterface

from skimage.feature import greycomatrix, greycoprops

class GLCMAnalyzer(UniformityMetricInterface):
    """Gray Level Co-occurrence Matrix features."""

    def calculate_glcm_features(self, thickness_map: np.ndarray, distances: list = [1, 2, 4], angles: list = [0, np.pi/4, np.pi/2, 3*np.pi/4]) -> Dict[str, float]:
        """Extract contrast, correlation, energy, homogeneity."""
        # P(i,j|d,θ) with d=1,2,4 pixels and θ = 0°, 45°, 90°, 135°
        glcm = greycomatrix(thickness_map, distances=distances, angles=angles, symmetric=True, normed=True)

        features = {
            'contrast': greycoprops(glcm, 'contrast').mean(),
            'correlation': greycoprops(glcm, 'correlation').mean(),
            'energy': greycoprops(glcm, 'energy').mean(),
            'homogeneity': greycoprops(glcm, 'homogeneity').mean()
        }

        return features

from skimage.feature import local_binary_pattern

class LBPAnalyzer(UniformityMetricInterface):
    """Local Binary Pattern analysis."""

    def calculate_lbp_uniformity(self, thickness_map: np.ndarray, P: int = 8, R: int = 1) -> float:
        """
        Rotation-invariant texture characterization.
        Reduces feature vectors from 256 to 59 dimensions
        """
        # Calculate the LBP image
        lbp_image = local_binary_pattern(thickness_map, P, R, method='uniform')

        # Calculate the histogram of the LBP image
        hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))

        # Normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)

        # Calculate the uniformity of the LBP histogram
        uniformity = np.sum(hist**2)

        return uniformity

class FractalAnalyzer(UniformityMetricInterface):
    """Fractal dimension analysis using box-counting."""

    def calculate_fractal_dimension(self, thickness_map: np.ndarray) -> float:
        """
        D = lim(log N(ε)/log(1/ε)) as ε→0
        Typical values 1.3-1.9 for nanofiber networks
        """
        # Binarize the image
        threshold = np.mean(thickness_map)
        binary_image = thickness_map > threshold

        # Find the bounding box of the data
        rows = np.any(binary_image, axis=1)
        cols = np.any(binary_image, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # The size of the box is the maximum of the row and column ranges
        size = max(rmax - rmin, cmax - cmin)

        # Number of boxes to use
        n_boxes = 10

        # Box sizes
        box_sizes = np.logspace(np.log10(2), np.log10(size/2), n_boxes)

        # Count the number of boxes that contain data
        counts = []
        for box_size in box_sizes:
            count = 0
            for r in range(rmin, rmax, int(box_size)):
                for c in range(cmin, cmax, int(box_size)):
                    box = binary_image[r:r+int(box_size), c:c+int(box_size)]
                    if np.any(box):
                        count += 1
            counts.append(count)

        # Fit a line to the log-log plot of the counts vs. box sizes
        coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1)

        return -coeffs[0]
