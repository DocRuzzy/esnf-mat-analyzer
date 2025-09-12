import numpy as np
from ..core.interfaces import UniformityMetricInterface

class FrequencyAnalyzer(UniformityMetricInterface):
    """FFT-based anisotropy and spatial frequency analysis.

    Implements the UniformityMetricInterface so it can be used seamlessly
    in pipelines expecting a metric calculator. The primary metric exposed
    via the generic calculate() method is the anisotropy index.
    """

    @property
    def name(self) -> str:  # type: ignore[override]
        return "frequency_anisotropy"

    @property
    def description(self) -> str:  # type: ignore[override]
        return "Anisotropy index derived from radial variance of FFT magnitude spectrum (higher = more anisotropy)."

    def calculate(self, thickness_map: np.ndarray, mask: np.ndarray, center) -> float:  # type: ignore[override]
        # Mask the map to avoid background influence
        try:
            region = np.where(mask, thickness_map, 0)
        except Exception:
            region = thickness_map
        return float(self.calculate_anisotropy_index(region))

    def calculate_anisotropy_index(self, thickness_map: np.ndarray) -> float:
        """
        Quantitative anisotropy assessment using 2D FFT.
        A value of 0 indicates perfect isotropy, while a value of 1 indicates perfect anisotropy.
        """
        # Perform 2D FFT
        f_transform = np.fft.fft2(thickness_map)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_transform_shifted)

        # Calculate the radial distribution of the magnitude spectrum
        center_x, center_y = np.array(magnitude_spectrum.shape) / 2
        x, y = np.meshgrid(np.arange(magnitude_spectrum.shape[1]), np.arange(magnitude_spectrum.shape[0]))
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # Calculate the standard deviation of the magnitude spectrum at each radius
        radial_bins = np.arange(0, r.max(), 1)
        radial_profile = np.zeros(len(radial_bins))
        for i in range(len(radial_bins) - 1):
            mask = (r >= radial_bins[i]) & (r < radial_bins[i+1])
            radial_profile[i] = magnitude_spectrum[mask].std()

        # The anisotropy index is the standard deviation of the radial profile
        anisotropy_index = np.std(radial_profile)

        return anisotropy_index

    def power_spectral_density(self, thickness_map: np.ndarray) -> np.ndarray:
        """
        PSD analysis with Tukey windowing.
        PSD(kx,ky) = (1/A)|W(kx,ky)|²/Δkx·Δky
        """
        # Try to import Tukey window; fall back to simple Hann window if scipy missing
        try:  # pragma: no cover - optional dependency branch
            from scipy.signal.windows import tukey  # type: ignore
            window = tukey(thickness_map.shape[0], alpha=0.5)
        except Exception:
            n = thickness_map.shape[0]
            if n > 1:
                x = np.arange(n)
                window = 0.5 - 0.5 * np.cos(2 * np.pi * x / (n - 1))  # Hann window
            else:
                window = np.ones(n, dtype=float)
        windowed_image = thickness_map * window[:, np.newaxis] * window[np.newaxis, :]

        # Perform 2D FFT
        f_transform = np.fft.fft2(windowed_image)
        f_transform_shifted = np.fft.fftshift(f_transform)

        # Calculate the power spectral density
        psd = np.abs(f_transform_shifted)**2

        return psd
