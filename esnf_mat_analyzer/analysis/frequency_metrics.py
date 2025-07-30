import numpy as np
from ..core.interfaces import UniformityMetricInterface

class FrequencyAnalyzer(UniformityMetricInterface):
    """FFT-based anisotropy and spatial frequency analysis."""

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
        from scipy.signal.windows import tukey

        # Apply a Tukey window to the image
        window = tukey(thickness_map.shape[0], alpha=0.5)
        windowed_image = thickness_map * window[:, np.newaxis] * window[np.newaxis, :]

        # Perform 2D FFT
        f_transform = np.fft.fft2(windowed_image)
        f_transform_shifted = np.fft.fftshift(f_transform)

        # Calculate the power spectral density
        psd = np.abs(f_transform_shifted)**2

        return psd
