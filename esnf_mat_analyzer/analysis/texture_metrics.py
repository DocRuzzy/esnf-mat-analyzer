import numpy as np
from typing import Dict
from ..core.interfaces import UniformityMetricInterface

"""Texture metrics with skimage API compatibility.

skimage renamed greycomatrix/greycoprops to graycomatrix/graycoprops. We try the
new names first, fall back to old, and gracefully disable if unavailable.
"""

_GLCM_AVAILABLE = True
try:  # pragma: no cover - dynamic import pattern
    try:
        from skimage.feature import graycomatrix as _glcm_fn, graycoprops as _gprops_fn  # type: ignore
    except ImportError:  # older versions
        from skimage.feature import greycomatrix as _glcm_fn, greycoprops as _gprops_fn  # type: ignore
except Exception:  # Any failure disables GLCM features
    _GLCM_AVAILABLE = False
    _glcm_fn = _gprops_fn = None  # type: ignore

class GLCMAnalyzer(UniformityMetricInterface):
    """Gray Level Co-occurrence Matrix feature extraction.

    Provides both a detailed feature dictionary and a single representative
    metric via the UniformityMetricInterface (energy, where higher implies
    more textural uniformity).
    """

    @property
    def name(self) -> str:  # type: ignore[override]
        return "glcm_energy"

    @property
    def description(self) -> str:  # type: ignore[override]
        return "GLCM energy (angular second moment) averaged over distances and angles."

    def calculate(self, thickness_map: np.ndarray, mask: np.ndarray, center) -> float:  # type: ignore[override]
        features = self.calculate_glcm_features(thickness_map)
        return float(features.get("energy", float("nan")))

    def calculate_glcm_features(
        self,
        thickness_map: np.ndarray,
        distances: list = [1, 2, 4],
        angles: list = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    ) -> Dict[str, float]:
        """Extract contrast, correlation, energy, homogeneity.

        Returns NaNs when GLCM features unavailable (e.g., skimage missing).
        """
        if not _GLCM_AVAILABLE:
            return {k: float("nan") for k in ("contrast", "correlation", "energy", "homogeneity")}

        img = thickness_map
        # GLCM expects integer grayscale levels; normalize if needed
        if img.dtype != np.uint8:
            arr = img.astype(float)
            mn, mx = np.nanmin(arr), np.nanmax(arr)
            if mx > mn:
                img = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
            else:  # Flat image
                img = np.zeros_like(arr, dtype=np.uint8)

        glcm = _glcm_fn(  # type: ignore
            img,
            distances=distances,
            angles=angles,
            symmetric=True,
            normed=True,
        )

        features = {
            "contrast": _gprops_fn(glcm, "contrast").mean(),  # type: ignore
            "correlation": _gprops_fn(glcm, "correlation").mean(),  # type: ignore
            "energy": _gprops_fn(glcm, "energy").mean(),  # type: ignore
            "homogeneity": _gprops_fn(glcm, "homogeneity").mean(),  # type: ignore
        }
        return features

try:  # pragma: no cover - optional dependency
    from skimage.feature import local_binary_pattern  # type: ignore
    _LBP_AVAILABLE = True
except Exception:  # skimage missing or incompatible
    _LBP_AVAILABLE = False
    local_binary_pattern = None  # type: ignore

class LBPAnalyzer(UniformityMetricInterface):
    """Local Binary Pattern analysis."""

    @property
    def name(self) -> str:  # type: ignore[override]
        return "lbp_uniformity"

    @property
    def description(self) -> str:  # type: ignore[override]
        return "Uniformity of Local Binary Pattern histogram (sum of squared normalized bin counts)."

    def calculate(self, thickness_map: np.ndarray, mask: np.ndarray, center) -> float:  # type: ignore[override]
        try:
            region = np.where(mask, thickness_map, 0)
        except Exception:
            region = thickness_map
        return float(self.calculate_lbp_uniformity(region))

    def calculate_lbp_uniformity(self, thickness_map: np.ndarray, P: int = 8, R: int = 1) -> float:
        """
        Rotation-invariant texture characterization.
        Reduces feature vectors from 256 to 59 dimensions
        """
        if not _LBP_AVAILABLE:
            # Graceful fallback: use normalized intensity histogram uniformity
            arr = thickness_map.astype(float)
            mn, mx = np.nanmin(arr), np.nanmax(arr)
            if mx > mn:
                norm = (arr - mn) / (mx - mn)
            else:
                norm = np.zeros_like(arr)
            hist, _ = np.histogram(norm.ravel(), bins=16, range=(0, 1))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-6)
            return float(np.sum(hist ** 2))

        # Calculate the LBP image using skimage
        lbp_image = local_binary_pattern(thickness_map, P, R, method="uniform")  # type: ignore
        hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-6)
        return float(np.sum(hist ** 2))

class FractalAnalyzer(UniformityMetricInterface):
    """Fractal dimension analysis using box-counting."""

    @property
    def name(self) -> str:  # type: ignore[override]
        return "fractal_dimension"

    @property
    def description(self) -> str:  # type: ignore[override]
        return "Estimated fractal dimension via box-counting (higher = more structural complexity)."

    def calculate(self, thickness_map: np.ndarray, mask: np.ndarray, center) -> float:  # type: ignore[override]
        try:
            region = np.where(mask, thickness_map, 0)
        except Exception:
            region = thickness_map
        return float(self.calculate_fractal_dimension(region))

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
