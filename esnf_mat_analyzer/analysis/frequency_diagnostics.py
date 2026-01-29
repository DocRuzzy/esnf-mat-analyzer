"""
Frequency Analysis Diagnostics for ESNF Mat Analyzer.

Provides tools to analyze spatial frequency content of thickness maps
before and after FFT filtering, helping users understand what features
are being preserved vs. removed by the processing pipeline.

Author: ESNF Mat Analyzer Team
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Any
import warnings


@dataclass
class FrequencyAnalysisResult:
    """Result from frequency analysis of an image or thickness map."""
    
    # Power spectrum (2D)
    power_spectrum: np.ndarray
    
    # Radial power profile (1D average)
    radial_frequencies: np.ndarray  # Frequency in cycles/pixel
    radial_power: np.ndarray  # Average power at each frequency
    
    # Key metrics
    dominant_frequency: float  # Frequency with highest power
    low_freq_power: float  # Power in low frequency band
    mid_freq_power: float  # Power in mid frequency band  
    high_freq_power: float  # Power in high frequency band
    total_power: float  # Total signal power
    
    # Frequency band boundaries (in cycles/pixel)
    low_freq_cutoff: float = 0.02
    high_freq_cutoff: float = 0.1


@dataclass
class FilteringImpactResult:
    """Result comparing signal before and after filtering."""
    
    # Original and filtered analysis
    original_analysis: FrequencyAnalysisResult
    filtered_analysis: FrequencyAnalysisResult
    
    # Power loss by frequency band
    low_freq_loss_pct: float
    mid_freq_loss_pct: float
    high_freq_loss_pct: float
    total_loss_pct: float
    
    # Spatial domain metrics
    rmse: float  # Root mean squared error
    correlation: float  # Pearson correlation
    
    # Per-frequency power ratio (filtered/original)
    frequency_preservation: np.ndarray
    frequencies: np.ndarray


class FrequencyDiagnostics:
    """
    Diagnostic tool for analyzing spatial frequency content.
    
    Helps identify:
    1. What spatial frequencies are present in the original image
    2. What frequencies are removed by filtering
    3. Whether real thickness features are being lost
    """
    
    def __init__(
        self,
        low_freq_cutoff: float = 0.02,
        high_freq_cutoff: float = 0.1
    ):
        """
        Initialize frequency diagnostics.
        
        Args:
            low_freq_cutoff: Boundary between low and mid frequencies (cycles/pixel)
            high_freq_cutoff: Boundary between mid and high frequencies (cycles/pixel)
        """
        self.low_freq_cutoff = low_freq_cutoff
        self.high_freq_cutoff = high_freq_cutoff
    
    def analyze(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> FrequencyAnalysisResult:
        """
        Analyze spatial frequency content of an image.
        
        Args:
            image: Input image or thickness map
            mask: Optional mask to restrict analysis region
            
        Returns:
            FrequencyAnalysisResult with power spectrum and metrics
        """
        # Ensure float
        img = image.astype(np.float64)
        
        # Apply mask if provided
        if mask is not None:
            # Zero out non-mask regions to avoid edge artifacts
            img = img.copy()
            img[~mask] = np.mean(img[mask]) if np.any(mask) else 0
        
        # Compute 2D FFT
        fft = np.fft.fft2(img)
        fft_shifted = np.fft.fftshift(fft)
        
        # Power spectrum
        power_spectrum = np.abs(fft_shifted) ** 2
        
        # Compute radial profile
        radial_freqs, radial_power = self._compute_radial_profile(power_spectrum)
        
        # Compute power in frequency bands
        low_mask = radial_freqs < self.low_freq_cutoff
        mid_mask = (radial_freqs >= self.low_freq_cutoff) & (radial_freqs < self.high_freq_cutoff)
        high_mask = radial_freqs >= self.high_freq_cutoff
        
        low_power = np.sum(radial_power[low_mask]) if np.any(low_mask) else 0
        mid_power = np.sum(radial_power[mid_mask]) if np.any(mid_mask) else 0
        high_power = np.sum(radial_power[high_mask]) if np.any(high_mask) else 0
        total_power = np.sum(radial_power)
        
        # Find dominant frequency (excluding DC)
        non_dc_mask = radial_freqs > 0.001
        if np.any(non_dc_mask):
            dominant_idx = np.argmax(radial_power[non_dc_mask])
            dominant_freq = radial_freqs[non_dc_mask][dominant_idx]
        else:
            dominant_freq = 0.0
        
        return FrequencyAnalysisResult(
            power_spectrum=power_spectrum,
            radial_frequencies=radial_freqs,
            radial_power=radial_power,
            dominant_frequency=dominant_freq,
            low_freq_power=low_power,
            mid_freq_power=mid_power,
            high_freq_power=high_power,
            total_power=total_power,
            low_freq_cutoff=self.low_freq_cutoff,
            high_freq_cutoff=self.high_freq_cutoff,
        )
    
    def compare_filtering_impact(
        self,
        original: np.ndarray,
        filtered: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> FilteringImpactResult:
        """
        Compare frequency content before and after filtering.
        
        Args:
            original: Original image/thickness map
            filtered: Filtered/processed result
            mask: Optional analysis mask
            
        Returns:
            FilteringImpactResult with power loss metrics
        """
        # Analyze both
        orig_analysis = self.analyze(original, mask)
        filt_analysis = self.analyze(filtered, mask)
        
        # Calculate power loss in each band
        def safe_pct_loss(orig, filt):
            if orig > 0:
                return 100 * (orig - filt) / orig
            return 0.0
        
        low_loss = safe_pct_loss(orig_analysis.low_freq_power, filt_analysis.low_freq_power)
        mid_loss = safe_pct_loss(orig_analysis.mid_freq_power, filt_analysis.mid_freq_power)
        high_loss = safe_pct_loss(orig_analysis.high_freq_power, filt_analysis.high_freq_power)
        total_loss = safe_pct_loss(orig_analysis.total_power, filt_analysis.total_power)
        
        # Spatial domain comparison
        if mask is not None:
            orig_masked = original[mask]
            filt_masked = filtered[mask]
        else:
            orig_masked = original.flatten()
            filt_masked = filtered.flatten()
        
        rmse = np.sqrt(np.mean((orig_masked - filt_masked) ** 2))
        
        # Pearson correlation
        if np.std(orig_masked) > 0 and np.std(filt_masked) > 0:
            correlation = np.corrcoef(orig_masked, filt_masked)[0, 1]
        else:
            correlation = 1.0 if np.allclose(orig_masked, filt_masked) else 0.0
        
        # Per-frequency preservation ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            preservation = np.where(
                orig_analysis.radial_power > 0,
                filt_analysis.radial_power / orig_analysis.radial_power,
                1.0
            )
        
        return FilteringImpactResult(
            original_analysis=orig_analysis,
            filtered_analysis=filt_analysis,
            low_freq_loss_pct=low_loss,
            mid_freq_loss_pct=mid_loss,
            high_freq_loss_pct=high_loss,
            total_loss_pct=total_loss,
            rmse=rmse,
            correlation=correlation,
            frequency_preservation=preservation,
            frequencies=orig_analysis.radial_frequencies,
        )
    
    def _compute_radial_profile(
        self,
        power_spectrum: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute radially-averaged power spectrum."""
        h, w = power_spectrum.shape
        center = (h // 2, w // 2)
        
        # Create distance array from center
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Convert to frequency (cycles/pixel)
        max_freq = 0.5  # Nyquist frequency
        freq = dist / max(h, w) * max_freq * 2
        
        # Bin the frequencies
        n_bins = min(h, w) // 2
        freq_bins = np.linspace(0, max_freq, n_bins + 1)
        freq_centers = (freq_bins[:-1] + freq_bins[1:]) / 2
        
        radial_power = np.zeros(n_bins)
        
        for i in range(n_bins):
            bin_mask = (freq >= freq_bins[i]) & (freq < freq_bins[i + 1])
            if np.any(bin_mask):
                radial_power[i] = np.mean(power_spectrum[bin_mask])
        
        return freq_centers, radial_power
    
    def identify_filtered_features(
        self,
        original: np.ndarray,
        filtered: np.ndarray,
        threshold: float = 0.3,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Identify what spatial features were removed by filtering.
        
        Args:
            original: Original image
            filtered: Filtered result  
            threshold: Minimum power loss ratio to flag (0-1)
            mask: Analysis mask
            
        Returns:
            Dictionary with identified lost features
        """
        impact = self.compare_filtering_impact(original, filtered, mask)
        
        lost_features = {
            "significant_loss": False,
            "lost_frequency_bands": [],
            "recommendations": [],
        }
        
        # Check each frequency band
        if impact.low_freq_loss_pct > threshold * 100:
            lost_features["lost_frequency_bands"].append({
                "band": "low",
                "loss_pct": impact.low_freq_loss_pct,
                "interpretation": "Large-scale gradients removed (potential background OR real thickness variation)",
            })
            lost_features["recommendations"].append(
                "Consider reducing high-pass filter strength to preserve gradual thickness changes"
            )
        
        if impact.mid_freq_loss_pct > threshold * 100:
            lost_features["lost_frequency_bands"].append({
                "band": "mid", 
                "loss_pct": impact.mid_freq_loss_pct,
                "interpretation": "Medium-scale features removed (likely real thickness variations)",
            })
            lost_features["recommendations"].append(
                "Mid-frequency loss often indicates real features being filtered - adjust cutoff frequency"
            )
            lost_features["significant_loss"] = True
        
        if impact.high_freq_loss_pct > threshold * 100:
            lost_features["lost_frequency_bands"].append({
                "band": "high",
                "loss_pct": impact.high_freq_loss_pct,
                "interpretation": "Fine details removed (texture, local variations, or noise)",
            })
        
        # Overall assessment
        if impact.correlation < 0.8:
            lost_features["significant_loss"] = True
            lost_features["recommendations"].append(
                f"Low correlation ({impact.correlation:.2f}) suggests significant feature loss"
            )
        
        lost_features["metrics"] = {
            "rmse": impact.rmse,
            "correlation": impact.correlation,
            "total_power_loss_pct": impact.total_loss_pct,
        }
        
        return lost_features
    
    def generate_diagnostic_report(
        self,
        original: np.ndarray,
        filtered: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive diagnostic report.
        
        Args:
            original: Original input image
            filtered: Filtered/processed result
            ground_truth: Optional known ground truth (from synthetic data)
            mask: Analysis region mask
            
        Returns:
            Comprehensive diagnostic dictionary
        """
        report = {
            "original_analysis": {},
            "filtered_analysis": {},
            "filtering_impact": {},
            "feature_loss": {},
            "ground_truth_comparison": None,
        }
        
        # Analyze original
        orig_analysis = self.analyze(original, mask)
        report["original_analysis"] = {
            "dominant_frequency": orig_analysis.dominant_frequency,
            "low_freq_power_pct": 100 * orig_analysis.low_freq_power / max(orig_analysis.total_power, 1),
            "mid_freq_power_pct": 100 * orig_analysis.mid_freq_power / max(orig_analysis.total_power, 1),
            "high_freq_power_pct": 100 * orig_analysis.high_freq_power / max(orig_analysis.total_power, 1),
        }
        
        # Analyze filtered
        filt_analysis = self.analyze(filtered, mask)
        report["filtered_analysis"] = {
            "dominant_frequency": filt_analysis.dominant_frequency,
            "low_freq_power_pct": 100 * filt_analysis.low_freq_power / max(filt_analysis.total_power, 1),
            "mid_freq_power_pct": 100 * filt_analysis.mid_freq_power / max(filt_analysis.total_power, 1),
            "high_freq_power_pct": 100 * filt_analysis.high_freq_power / max(filt_analysis.total_power, 1),
        }
        
        # Filtering impact
        impact = self.compare_filtering_impact(original, filtered, mask)
        report["filtering_impact"] = {
            "low_freq_loss_pct": impact.low_freq_loss_pct,
            "mid_freq_loss_pct": impact.mid_freq_loss_pct,
            "high_freq_loss_pct": impact.high_freq_loss_pct,
            "total_loss_pct": impact.total_loss_pct,
            "rmse": impact.rmse,
            "correlation": impact.correlation,
        }
        
        # Feature loss analysis
        report["feature_loss"] = self.identify_filtered_features(
            original, filtered, mask=mask
        )
        
        # Ground truth comparison if available
        if ground_truth is not None:
            gt_mask = mask if mask is not None else np.ones_like(ground_truth, dtype=bool)
            
            # Compare filtered to ground truth
            if np.std(ground_truth[gt_mask]) > 0 and np.std(filtered[gt_mask]) > 0:
                gt_correlation = np.corrcoef(
                    ground_truth[gt_mask].flatten(),
                    filtered[gt_mask].flatten()
                )[0, 1]
            else:
                gt_correlation = 0.0
            
            gt_rmse = np.sqrt(np.mean((ground_truth[gt_mask] - filtered[gt_mask])**2))
            
            report["ground_truth_comparison"] = {
                "correlation": gt_correlation,
                "rmse": gt_rmse,
                "quality_score": gt_correlation * (1 - min(gt_rmse / 100, 1)),
            }
        
        return report


def visualize_frequency_analysis(
    result: FrequencyAnalysisResult,
    title: str = "Frequency Analysis"
) -> None:
    """
    Visualize frequency analysis results using matplotlib.
    
    Args:
        result: FrequencyAnalysisResult to visualize
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib required for visualization")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Power spectrum (log scale)
    ax1 = axes[0]
    log_power = np.log10(result.power_spectrum + 1)
    ax1.imshow(log_power, cmap='viridis')
    ax1.set_title(f"{title}\n2D Power Spectrum (log scale)")
    ax1.set_xlabel("Frequency X")
    ax1.set_ylabel("Frequency Y")
    
    # Radial profile
    ax2 = axes[1]
    ax2.semilogy(result.radial_frequencies, result.radial_power + 1)
    ax2.axvline(result.low_freq_cutoff, color='g', linestyle='--', label='Low/Mid boundary')
    ax2.axvline(result.high_freq_cutoff, color='r', linestyle='--', label='Mid/High boundary')
    ax2.axvline(result.dominant_frequency, color='orange', linestyle='-', 
                label=f'Dominant: {result.dominant_frequency:.3f}')
    ax2.set_xlabel("Frequency (cycles/pixel)")
    ax2.set_ylabel("Power")
    ax2.set_title("Radial Power Profile")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_filtering_impact(
    impact: FilteringImpactResult,
    original: np.ndarray,
    filtered: np.ndarray,
    title: str = "Filtering Impact"
) -> None:
    """
    Visualize the impact of filtering on spatial frequencies.
    
    Args:
        impact: FilteringImpactResult
        original: Original image
        filtered: Filtered image
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib required for visualization")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original and filtered images
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title("Original")
    
    axes[0, 1].imshow(filtered, cmap='gray')
    axes[0, 1].set_title("Filtered")
    
    # Difference
    diff = original.astype(float) - filtered.astype(float)
    im = axes[0, 2].imshow(diff, cmap='RdBu', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    axes[0, 2].set_title(f"Difference (RMSE: {impact.rmse:.2f})")
    plt.colorbar(im, ax=axes[0, 2])
    
    # Radial power comparison
    axes[1, 0].semilogy(impact.frequencies, impact.original_analysis.radial_power + 1, 
                        label='Original', alpha=0.8)
    axes[1, 0].semilogy(impact.frequencies, impact.filtered_analysis.radial_power + 1,
                        label='Filtered', alpha=0.8)
    axes[1, 0].set_xlabel("Frequency (cycles/pixel)")
    axes[1, 0].set_ylabel("Power")
    axes[1, 0].set_title("Power Spectrum Comparison")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Preservation ratio
    axes[1, 1].plot(impact.frequencies, impact.frequency_preservation)
    axes[1, 1].axhline(1.0, color='g', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(0.5, color='orange', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel("Frequency (cycles/pixel)")
    axes[1, 1].set_ylabel("Preservation Ratio")
    axes[1, 1].set_title("Frequency Preservation (1.0 = preserved)")
    axes[1, 1].set_ylim(0, 1.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Power loss bar chart
    bands = ['Low\nFreq', 'Mid\nFreq', 'High\nFreq', 'Total']
    losses = [impact.low_freq_loss_pct, impact.mid_freq_loss_pct, 
              impact.high_freq_loss_pct, impact.total_loss_pct]
    colors = ['green' if l < 20 else 'orange' if l < 50 else 'red' for l in losses]
    
    axes[1, 2].bar(bands, losses, color=colors)
    axes[1, 2].set_ylabel("Power Loss (%)")
    axes[1, 2].set_title("Power Loss by Frequency Band")
    axes[1, 2].axhline(30, color='orange', linestyle='--', alpha=0.5, label='Warning threshold')
    
    for i, (band, loss) in enumerate(zip(bands, losses)):
        axes[1, 2].text(i, loss + 2, f"{loss:.1f}%", ha='center')
    
    plt.suptitle(f"{title}\nCorrelation: {impact.correlation:.3f}", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Demo with synthetic data
    from pathlib import Path
    import sys
    
    # Try to import synthetic generator
    try:
        from esnf_mat_analyzer.data.synthetic_mat_generator import (
            SyntheticMatGenerator, SyntheticMatConfig, PatternType
        )
        
        print("Generating synthetic test data...")
        generator = SyntheticMatGenerator()
        
        # Generate multi-frequency test pattern
        cfg = SyntheticMatConfig(
            pattern_type=PatternType.MULTI_FREQUENCY,
            frequencies=[0.005, 0.02, 0.05, 0.1],
            frequency_amplitudes=[40, 30, 20, 10],
            seed=42,
        )
        result = generator.generate(cfg)
        
        print("\nAnalyzing frequency content...")
        diagnostics = FrequencyDiagnostics()
        
        # Analyze original
        analysis = diagnostics.analyze(result.image, result.mat_mask)
        print(f"Dominant frequency: {analysis.dominant_frequency:.4f} cycles/pixel")
        print(f"Low freq power: {100*analysis.low_freq_power/analysis.total_power:.1f}%")
        print(f"Mid freq power: {100*analysis.mid_freq_power/analysis.total_power:.1f}%")
        print(f"High freq power: {100*analysis.high_freq_power/analysis.total_power:.1f}%")
        
        # Simulate filtering effect (simple Gaussian blur as example)
        from scipy.ndimage import gaussian_filter
        filtered = gaussian_filter(result.image.astype(float), sigma=5)
        
        print("\nComparing filtering impact...")
        impact = diagnostics.compare_filtering_impact(
            result.image, filtered, result.mat_mask
        )
        print(f"Low freq loss: {impact.low_freq_loss_pct:.1f}%")
        print(f"Mid freq loss: {impact.mid_freq_loss_pct:.1f}%")
        print(f"High freq loss: {impact.high_freq_loss_pct:.1f}%")
        print(f"Correlation: {impact.correlation:.3f}")
        
        # Generate report
        print("\nGenerating diagnostic report...")
        report = diagnostics.generate_diagnostic_report(
            result.image, filtered, 
            ground_truth=result.ground_truth_thickness,
            mask=result.mat_mask
        )
        
        if report["ground_truth_comparison"]:
            print(f"Ground truth correlation: {report['ground_truth_comparison']['correlation']:.3f}")
            print(f"Quality score: {report['ground_truth_comparison']['quality_score']:.3f}")
        
        # Visualize if matplotlib available
        try:
            visualize_filtering_impact(impact, result.image, filtered.astype(np.uint8))
        except Exception as e:
            print(f"Visualization skipped: {e}")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run from project root or install package first")
