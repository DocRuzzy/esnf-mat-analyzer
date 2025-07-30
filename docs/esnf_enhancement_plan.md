# ESNF Mat Analyzer Enhancement Plan - Publication-Ready Uniformity Analysis

## Executive Summary

**Objective**: Transform the ESNF mat analyzer into a publication-ready platform for quantifying thickness uniformity in electrospun nanofiber mats using physics-based methods and comprehensive uniformity metrics.

**Current Status Assessment**: Based on repository analysis, significant foundational work is complete including Beer-Lambert thickness estimation, FFT anisotropy analysis, and GLCM texture features. The enhancement plan focuses on refining these implementations using SOLID principles and adding critical uniformity metrics for peer-reviewed publication.

**Key Innovation**: Integration of physics-based thickness estimation with advanced statistical uniformity metrics specifically designed for mat-scale analysis where individual fiber morphology cannot be resolved.

---

## Phase 1: Architecture Refactoring (SOLID Principles Implementation)

### 1.1 Dependency Injection Framework
**Priority**: Critical for maintainability and testing

```python
# esnf_mat_analyzer/core/interfaces.py
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

@runtime_checkable
class IThicknessEstimator(Protocol):
    """Interface for thickness estimation strategies."""
    
    def estimate_thickness(self, image: np.ndarray, background: np.ndarray) -> np.ndarray:
        """Estimate thickness from image intensity data.
        
        Args:
            image: Preprocessed image array
            background: Background reference image
            
        Returns:
            2D thickness map in calibrated units
            
        Raises:
            ThicknessEstimationError: When estimation fails
        """
        ...

@runtime_checkable  
class IUniformityMetric(Protocol):
    """Interface for uniformity calculation strategies."""
    
    def calculate_metric(self, thickness_map: np.ndarray, roi_mask: np.ndarray) -> float:
        """Calculate uniformity metric from thickness map.
        
        Args:
            thickness_map: 2D thickness distribution
            roi_mask: Boolean mask defining analysis region
            
        Returns:
            Uniformity metric value (normalized 0-1 where appropriate)
        """
        ...

class IImageProcessor(ABC):
    """Abstract base class for image processing operations."""
    
    @abstractmethod
    def process(self, image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
        """Process raw image according to configuration."""
        pass
```

### 1.2 Dependency Container Implementation
```python
# esnf_mat_analyzer/core/container.py
from dataclasses import dataclass
from typing import Dict, Type, Any, Optional
import inspect

@dataclass
class ServiceDescriptor:
    """Describes a service registration in the container."""
    service_type: Type
    implementation: Type
    singleton: bool = False
    factory: Optional[callable] = None

class DependencyContainer:
    """Lightweight dependency injection container following SOLID principles.
    
    Features:
    - Interface-based registration
    - Automatic constructor injection
    - Singleton lifecycle management
    - Factory pattern support
    """
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, Any] = {}
    
    def register(self, interface: Type, implementation: Type, 
                singleton: bool = False, factory: Optional[callable] = None) -> None:
        """Register a service implementation for an interface.
        
        Args:
            interface: The interface type to register
            implementation: The concrete implementation type
            singleton: Whether to use singleton lifecycle
            factory: Optional factory function for complex instantiation
            
        Raises:
            RegistrationError: When interface/implementation mismatch detected
        """
        # Validate implementation conforms to interface
        if not issubclass(implementation, interface):
            if not hasattr(interface, '__protocols__'):  # Not a Protocol
                raise RegistrationError(f"{implementation} does not implement {interface}")
        
        self._services[interface] = ServiceDescriptor(
            interface, implementation, singleton, factory
        )
    
    def resolve(self, service_type: Type) -> Any:
        """Resolve a service instance with automatic dependency injection.
        
        Args:
            service_type: The interface type to resolve
            
        Returns:
            Configured service instance
            
        Raises:
            ResolutionError: When service cannot be resolved
        """
        if service_type in self._instances:
            return self._instances[service_type]
            
        if service_type not in self._services:
            raise ResolutionError(f"Service {service_type} not registered")
        
        descriptor = self._services[service_type]
        
        if descriptor.factory:
            instance = descriptor.factory(self)
        else:
            instance = self._create_instance(descriptor.implementation)
        
        if descriptor.singleton:
            self._instances[service_type] = instance
            
        return instance
    
    def _create_instance(self, implementation: Type) -> Any:
        """Create instance with automatic dependency injection."""
        signature = inspect.signature(implementation.__init__)
        dependencies = {}
        
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
                
            if param.annotation != inspect.Parameter.empty:
                dependencies[param_name] = self.resolve(param.annotation)
        
        return implementation(**dependencies)
```

---

## Phase 2: Enhanced Uniformity Metrics Implementation

### 2.1 Radial Uniformity Index (RUI) - Enhanced Implementation
**Scientific Rationale**: Quantifies how thickness varies as a function of radial distance from center, critical for circular ESNF mats.

```python
# esnf_mat_analyzer/analysis/radial_uniformity.py
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

@dataclass
class RadialAnalysisConfig:
    """Configuration for radial uniformity analysis."""
    num_radial_bins: int = 50
    angle_resolution_degrees: float = 2.0
    smoothing_window: int = 5
    outlier_threshold_std: float = 2.0
    interpolation_method: str = 'cubic'

class RadialUniformityAnalyzer:
    """Enhanced radial uniformity analysis with statistical rigor.
    
    Implements multiple RUI variants for comprehensive characterization:
    - Classical RUI: Coefficient of variation along radial profiles
    - Angular RUI: Variation in circumferential direction
    - Gradient RUI: Spatial derivative-based measure
    """
    
    def __init__(self, config: RadialAnalysisConfig):
        self.config = config
    
    def calculate_radial_uniformity_index(self, thickness_map: np.ndarray, 
                                        center: Tuple[int, int],
                                        roi_radius: float) -> Dict[str, float]:
        """Calculate comprehensive radial uniformity metrics.
        
        Args:
            thickness_map: 2D thickness distribution
            center: (y, x) coordinates of analysis center
            roi_radius: Maximum radius for analysis
            
        Returns:
            Dictionary containing multiple RUI variants and statistics
        """
        cy, cx = center
        y_coords, x_coords = np.ogrid[:thickness_map.shape[0], :thickness_map.shape[1]]
        radius_map = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)
        
        # Create radial bins
        max_radius = min(roi_radius, np.max(radius_map))
        radial_bins = np.linspace(0, max_radius, self.config.num_radial_bins)
        
        # Calculate radial profile with statistical measures
        radial_profile = self._calculate_radial_profile(
            thickness_map, radius_map, radial_bins
        )
        
        # Calculate multiple RUI variants
        classical_rui = self._calculate_classical_rui(radial_profile)
        angular_rui = self._calculate_angular_rui(thickness_map, center, radial_bins)
        gradient_rui = self._calculate_gradient_rui(radial_profile, radial_bins)
        
        # Statistical analysis
        profile_stats = self._calculate_profile_statistics(radial_profile)
        
        return {
            'classical_rui': classical_rui,
            'angular_rui': angular_rui,
            'gradient_rui': gradient_rui,
            'radial_monotonicity': self._calculate_monotonicity(radial_profile),
            'edge_uniformity': self._calculate_edge_uniformity(radial_profile),
            **profile_stats
        }
    
    def _calculate_radial_profile(self, thickness_map: np.ndarray, 
                                radius_map: np.ndarray,
                                radial_bins: np.ndarray) -> np.ndarray:
        """Calculate mean thickness at each radial distance with outlier removal."""
        profile_means = []
        profile_stds = []
        
        for i in range(len(radial_bins) - 1):
            r_inner, r_outer = radial_bins[i], radial_bins[i + 1]
            mask = (radius_map >= r_inner) & (radius_map < r_outer)
            
            if np.any(mask):
                values = thickness_map[mask]
                # Remove outliers
                cleaned_values = self._remove_outliers(values)
                profile_means.append(np.mean(cleaned_values))
                profile_stds.append(np.std(cleaned_values))
            else:
                profile_means.append(np.nan)
                profile_stds.append(np.nan)
        
        return np.array(profile_means)
    
    def _calculate_classical_rui(self, radial_profile: np.ndarray) -> float:
        """Calculate classical RUI as coefficient of variation."""
        valid_profile = radial_profile[~np.isnan(radial_profile)]
        if len(valid_profile) < 2:
            return 0.0
        
        mean_thickness = np.mean(valid_profile)
        std_thickness = np.std(valid_profile)
        
        if mean_thickness == 0:
            return 0.0
        
        # RUI = 1 - CV (higher values indicate better uniformity)
        cv = std_thickness / mean_thickness
        return max(0.0, 1.0 - cv)
    
    def _calculate_angular_rui(self, thickness_map: np.ndarray,
                             center: Tuple[int, int],
                             radial_bins: np.ndarray) -> float:
        """Calculate uniformity in angular direction."""
        cy, cx = center
        angular_variations = []
        
        # Sample at different radii
        for r in radial_bins[1:-1:5]:  # Sample every 5th radius
            angles = np.linspace(0, 2*np.pi, int(360/self.config.angle_resolution_degrees))
            angular_profile = []
            
            for angle in angles:
                y = int(cy + r * np.sin(angle))
                x = int(cx + r * np.cos(angle))
                
                if (0 <= y < thickness_map.shape[0] and 
                    0 <= x < thickness_map.shape[1]):
                    angular_profile.append(thickness_map[y, x])
            
            if len(angular_profile) > 3:
                angular_variations.append(np.std(angular_profile) / np.mean(angular_profile))
        
        if not angular_variations:
            return 0.0
        
        mean_angular_cv = np.mean(angular_variations)
        return max(0.0, 1.0 - mean_angular_cv)
```

### 2.2 Enhanced Gini Coefficient Implementation
**Scientific Rationale**: Measures statistical inequality in thickness distribution, providing insights into material homogeneity.

```python
# esnf_mat_analyzer/analysis/statistical_uniformity.py
class GiniCoefficientAnalyzer:
    """Enhanced Gini coefficient analysis for thickness distribution equality.
    
    Implements multiple Gini variants:
    - Standard Gini: Overall thickness inequality
    - Spatial Gini: Spatial autocorrelation-weighted inequality  
    - Multi-scale Gini: Scale-dependent inequality analysis
    """
    
    def calculate_gini_coefficient(self, thickness_map: np.ndarray,
                                 roi_mask: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive Gini coefficient metrics.
        
        Args:
            thickness_map: 2D thickness distribution
            roi_mask: Boolean mask defining analysis region
            
        Returns:
            Dictionary containing multiple Gini variants
        """
        # Extract thickness values within ROI
        thickness_values = thickness_map[roi_mask]
        thickness_values = thickness_values[~np.isnan(thickness_values)]
        
        if len(thickness_values) < 2:
            return {'standard_gini': 0.0, 'spatial_gini': 0.0, 'normalized_gini': 0.0}
        
        # Standard Gini coefficient
        standard_gini = self._calculate_standard_gini(thickness_values)
        
        # Spatial Gini with location weighting
        spatial_gini = self._calculate_spatial_gini(thickness_map, roi_mask)
        
        # Normalized Gini (0-1 scale)
        normalized_gini = self._normalize_gini(standard_gini, len(thickness_values))
        
        return {
            'standard_gini': standard_gini,
            'spatial_gini': spatial_gini,
            'normalized_gini': normalized_gini,
            'gini_uniformity_index': 1.0 - normalized_gini  # Higher = more uniform
        }
    
    def _calculate_standard_gini(self, values: np.ndarray) -> float:
        """Calculate standard Gini coefficient with numerical stability."""
        sorted_values = np.sort(values)
        n = len(sorted_values)
        
        if n == 0 or np.sum(sorted_values) == 0:
            return 0.0
        
        # Gini coefficient formula with numerical stability
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
    
    def _calculate_spatial_gini(self, thickness_map: np.ndarray,
                              roi_mask: np.ndarray) -> float:
        """Calculate spatial Gini coefficient considering local neighborhoods."""
        from scipy.ndimage import generic_filter
        
        # Calculate local mean in 3x3 neighborhoods
        local_means = generic_filter(thickness_map, np.nanmean, size=3)
        
        # Extract values and local context
        y_coords, x_coords = np.where(roi_mask)
        thickness_values = thickness_map[roi_mask]
        local_context = local_means[roi_mask]
        
        # Weight by spatial coherence
        spatial_weights = 1.0 / (1.0 + np.abs(thickness_values - local_context))
        weighted_values = thickness_values * spatial_weights
        
        return self._calculate_standard_gini(weighted_values)
```

### 2.3 Advanced Thickness Range Ratio (TRR) Implementation
```python
class ThicknessRangeAnalyzer:
    """Advanced thickness range analysis with statistical robustness."""
    
    def calculate_thickness_range_ratio(self, thickness_map: np.ndarray,
                                      roi_mask: np.ndarray,
                                      percentile_range: Tuple[float, float] = (5, 95)
                                     ) -> Dict[str, float]:
        """Calculate robust thickness range metrics.
        
        Args:
            thickness_map: 2D thickness distribution
            roi_mask: Boolean mask defining analysis region
            percentile_range: Percentile range for robust statistics
            
        Returns:
            Dictionary containing multiple TRR variants
        """
        thickness_values = thickness_map[roi_mask]
        thickness_values = thickness_values[~np.isnan(thickness_values)]
        
        if len(thickness_values) < 2:
            return {'standard_trr': 1.0, 'robust_trr': 1.0, 'iqr_trr': 1.0}
        
        # Standard TRR (min/max ratio)
        t_min, t_max = np.min(thickness_values), np.max(thickness_values)
        standard_trr = t_min / t_max if t_max > 0 else 1.0
        
        # Robust TRR using percentiles
        p_low, p_high = np.percentile(thickness_values, percentile_range)
        robust_trr = p_low / p_high if p_high > 0 else 1.0
        
        # Interquartile range TRR
        q1, q3 = np.percentile(thickness_values, [25, 75])
        iqr_trr = q1 / q3 if q3 > 0 else 1.0
        
        return {
            'standard_trr': standard_trr,
            'robust_trr': robust_trr,
            'iqr_trr': iqr_trr,
            'thickness_span': t_max - t_min,
            'robust_thickness_span': p_high - p_low
        }
```

---

## Phase 3: Publication-Ready Validation Framework

### 3.1 Statistical Validation Module
```python
# esnf_mat_analyzer/validation/statistical_validation.py
from dataclasses import dataclass
from typing import List, Dict, Tuple
from scipy import stats
import numpy as np

@dataclass
class ValidationResult:
    """Results of statistical validation testing."""
    metric_name: str
    reproducibility_cv: float
    inter_operator_agreement: float
    test_retest_correlation: float
    sensitivity_analysis: Dict[str, float]
    confidence_interval: Tuple[float, float]
    statistical_power: float

class StatisticalValidator:
    """Comprehensive statistical validation for uniformity metrics.
    
    Ensures publication-ready reliability through:
    - Reproducibility assessment
    - Inter-operator reliability
    - Sensitivity analysis
    - Statistical power calculation
    """
    
    def validate_uniformity_metric(self, metric_calculator: IUniformityMetric,
                                 test_images: List[np.ndarray],
                                 reference_values: Optional[List[float]] = None
                                ) -> ValidationResult:
        """Comprehensively validate a uniformity metric.
        
        Args:
            metric_calculator: The metric implementation to validate
            test_images: List of test images with known properties
            reference_values: Known reference values if available
            
        Returns:
            Complete validation results for publication
        """
        # Reproducibility testing
        reproducibility_cv = self._assess_reproducibility(
            metric_calculator, test_images[0]
        )
        
        # Inter-operator agreement (simulated)
        inter_operator = self._assess_inter_operator_agreement(
            metric_calculator, test_images
        )
        
        # Test-retest correlation
        test_retest_corr = self._assess_test_retest_reliability(
            metric_calculator, test_images
        )
        
        # Sensitivity analysis
        sensitivity = self._perform_sensitivity_analysis(
            metric_calculator, test_images[0]
        )
        
        # Statistical power calculation
        power = self._calculate_statistical_power(
            metric_calculator, test_images, reference_values
        )
        
        return ValidationResult(
            metric_name=metric_calculator.__class__.__name__,
            reproducibility_cv=reproducibility_cv,
            inter_operator_agreement=inter_operator,
            test_retest_correlation=test_retest_corr,
            sensitivity_analysis=sensitivity,
            confidence_interval=(0.0, 1.0),  # Placeholder
            statistical_power=power
        )
```

### 3.2 Benchmark Dataset Generator
```python
# esnf_mat_analyzer/validation/benchmark_generator.py
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
```

---

## Phase 4: Advanced Analysis Integration

### 4.1 Multi-Scale Uniformity Analysis
```python
# esnf_mat_analyzer/analysis/multiscale_uniformity.py
class MultiScaleUniformityAnalyzer:
    """Multi-scale uniformity analysis using wavelet decomposition.
    
    Analyzes uniformity at different length scales to identify:
    - Fine-scale texture variations
    - Medium-scale defects
    - Large-scale thickness gradients
    """
    
    def __init__(self, wavelet: str = 'db4', levels: int = 4):
        self.wavelet = wavelet
        self.levels = levels
    
    def analyze_multiscale_uniformity(self, thickness_map: np.ndarray,
                                    roi_mask: np.ndarray
                                   ) -> Dict[str, np.ndarray]:
        """Perform multi-scale uniformity analysis.
        
        Args:
            thickness_map: 2D thickness distribution
            roi_mask: Boolean mask defining analysis region
            
        Returns:
            Scale-dependent uniformity metrics
        """
        try:
            import pywt
        except ImportError:
            # Fallback to simple multi-resolution analysis
            return self._fallback_multiscale_analysis(thickness_map, roi_mask)
        
        # Apply ROI mask
        masked_thickness = thickness_map.copy()
        masked_thickness[~roi_mask] = np.nan
        
        # Wavelet decomposition
        coeffs = pywt.wavedec2(masked_thickness, self.wavelet, level=self.levels)
        
        scale_uniformity = {}
        for level in range(self.levels):
            # Extract coefficients at this scale
            if level == 0:
                scale_data = coeffs[0]  # Approximation coefficients
            else:
                scale_data = coeffs[level]  # Detail coefficients
            
            # Calculate uniformity at this scale
            valid_data = scale_data[~np.isnan(scale_data)]
            if len(valid_data) > 0:
                scale_uniformity[f'scale_{level}_uniformity'] = 1.0 - (
                    np.std(valid_data) / np.mean(np.abs(valid_data))
                )
        
        return scale_uniformity
```

### 4.2 Uncertainty Quantification
```python
# esnf_mat_analyzer/analysis/uncertainty_quantification.py
class UncertaintyQuantifier:
    """Quantify measurement uncertainties for publication-quality error reporting."""
    
    def quantify_metric_uncertainty(self, metric_values: np.ndarray,
                                  measurement_noise: float,
                                  systematic_error: float = 0.0
                                 ) -> Dict[str, float]:
        """Calculate comprehensive uncertainty estimates.
        
        Args:
            metric_values: Array of metric measurements
            measurement_noise: Estimated measurement noise level
            systematic_error: Known systematic error contribution
            
        Returns:
            Uncertainty quantification results
        """
        n_measurements = len(metric_values)
        
        # Statistical uncertainty
        statistical_uncertainty = np.std(metric_values) / np.sqrt(n_measurements)
        
        # Combined uncertainty (Type A + Type B)
        combined_uncertainty = np.sqrt(
            statistical_uncertainty**2 + 
            measurement_noise**2 + 
            systematic_error**2
        )
        
        # Confidence intervals
        confidence_95 = 1.96 * statistical_uncertainty
        
        return {
            'statistical_uncertainty': statistical_uncertainty,
            'measurement_noise': measurement_noise,
            'systematic_error': systematic_error,
            'combined_uncertainty': combined_uncertainty,
            'confidence_interval_95': confidence_95,
            'relative_uncertainty_percent': (combined_uncertainty / np.mean(metric_values)) * 100
        }
```

---

## Phase 5: Publication Infrastructure

### 5.1 Automated Report Generation
```python
# esnf_mat_analyzer/reporting/scientific_report.py
from jinja2 import Template
import matplotlib.pyplot as plt
from typing import Dict, List

class ScientificReportGenerator:
    """Generate publication-ready analysis reports with figures and statistics."""
    
    def generate_uniformity_report(self, analysis_results: Dict,
                                 sample_metadata: Dict,
                                 output_path: Path) -> None:
        """Generate comprehensive uniformity analysis report.
        
        Args:
            analysis_results: Complete analysis results
            sample_metadata: Sample preparation and imaging metadata
            output_path: Output directory for report files
        """
        # Generate figures
        figure_paths = self._generate_publication_figures(
            analysis_results, output_path / 'figures'
        )
        
        # Statistical summary
        statistical_summary = self._generate_statistical_summary(analysis_results)
        
        # Render report template
        report_content = self._render_report_template(
            analysis_results, sample_metadata, figure_paths, statistical_summary
        )
        
        # Save report
        with open(output_path / 'uniformity_analysis_report.md', 'w') as f:
            f.write(report_content)
        
        # Generate LaTeX version for publication
        self._generate_latex_report(report_content, output_path)
    
    def _generate_publication_figures(self, results: Dict, 
                                    figures_dir: Path) -> Dict[str, Path]:
        """Generate publication-quality figures with proper formatting."""
        figures_dir.mkdir(parents=True, exist_ok=True)
        figure_paths = {}
        
        # Configure matplotlib for publication
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'axes.linewidth': 1.5,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
            'figure.dpi': 300
        })
        
        # Thickness distribution heatmap
        fig, ax = plt.subplots(figsize=(6, 5))
        thickness_map = results['thickness_map']
        im = ax.imshow(thickness_map, cmap='viridis', aspect='equal')
        ax.set_title('Thickness Distribution (μm)')
        plt.colorbar(im, ax=ax, label='Thickness (μm)')
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        plt.tight_layout()
        fig_path = figures_dir / 'thickness_heatmap.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        figure_paths['thickness_heatmap'] = fig_path
        
        # Radial profile analysis
        if 'radial_analysis' in results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Radial profile
            radial_data = results['radial_analysis']
            ax1.plot(radial_data['radii'], radial_data['mean_thickness'], 'b-', linewidth=2)
            ax1.fill_between(radial_data['radii'], 
                           radial_data['mean_thickness'] - radial_data['std_thickness'],
                           radial_data['mean_thickness'] + radial_data['std_thickness'],
                           alpha=0.3)
            ax1.set_xlabel('Radial Distance (pixels)')
            ax1.set_ylabel('Mean Thickness (μm)')
            ax1.set_title('Radial Thickness Profile')
            ax1.grid(True, alpha=0.3)
            
            # Uniformity metrics comparison
            metrics = ['RUI', 'Gini', 'TRR']
            values = [results['rui'], 1-results['gini'], results['trr']]
            bars = ax2.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'])
            ax2.set_ylabel('Uniformity Index')
            ax2.set_title('Uniformity Metrics Comparison')
            ax2.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            fig_path = figures_dir / 'radial_analysis.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            figure_paths['radial_analysis'] = fig_path
        
        return figure_paths
```

### 5.2 Data Export Module
```python
# esnf_mat_analyzer/export/data_export.py
class PublicationDataExporter:
    """Export analysis data in formats suitable for publication and archival."""
    
    def export_analysis_data(self, results: Dict, 
                           metadata: Dict,
                           output_dir: Path,
                           formats: List[str] = ['csv', 'hdf5', 'json']) -> None:
        """Export complete analysis dataset.
        
        Args:
            results: Analysis results dictionary
            metadata: Sample and processing metadata
            output_dir: Output directory
            formats: List of export formats
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if 'csv' in formats:
            self._export_csv(results, metadata, output_dir)
        
        if 'hdf5' in formats:
            self._export_hdf5(results, metadata, output_dir)
        
        if 'json' in formats:
            self._export_json(results, metadata, output_dir)
    
    def _export_csv(self, results: Dict, metadata: Dict, output_dir: Path) -> None:
        """Export tabular data in CSV format."""
        import pandas as pd
        
        # Uniformity metrics summary
        metrics_data = {
            'Metric': [],
            'Value': [],
            'Unit': [],
            'Uncertainty': [],
            'Description': []
        }
        
        # Add all uniformity metrics
        uniformity_metrics = {
            'Radial Uniformity Index': (results.get('rui', 0), 'dimensionless', 0.01, 'Higher = more uniform'),
            'Gini Coefficient': (results.get('gini', 0), 'dimensionless', 0.01, 'Lower = more uniform'),
            'Thickness Range Ratio': (results.get('trr', 0), 'dimensionless', 0.01, 'Higher = more uniform'),
            'Mean Thickness': (results.get('mean_thickness', 0), 'μm', 0.1, 'Average thickness'),
            'Thickness Std Dev': (results.get('thickness_std', 0), 'μm', 0.1, 'Thickness variability')
        }
        
        for metric_name, (value, unit, uncertainty, description) in uniformity_metrics.items():
            metrics_data['Metric'].append(metric_name)
            metrics_data['Value'].append(value)
            metrics_data['Unit'].append(unit)
            metrics_data['Uncertainty'].append(uncertainty)
            metrics_data['Description'].append(description)
        
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_csv(output_dir / 'uniformity_metrics.csv', index=False)
        
        # Export thickness map data
        if 'thickness_map' in results:
            thickness_df = pd.DataFrame(results['thickness_map'])
            thickness_df.to_csv(output_dir / 'thickness_map.csv', index=False)
```

---

## Implementation Timeline and Priorities

### Phase 1 (Weeks 1-2): Architecture Foundation
- **Priority 1**: Implement dependency injection container
- **Priority 2**: Refactor existing analyzers to use interfaces
- **Priority 3**: Create comprehensive unit tests

### Phase 2 (Weeks 3-4): Enhanced Metrics
- **Priority 1**: Implement enhanced RUI with multiple variants
- **Priority 2**: Develop robust Gini coefficient analysis
- **Priority 3**: Create advanced TRR implementation

### Phase 3 (Weeks 5-6): Validation Framework
- **Priority 1**: Build statistical validation module
- **Priority 2**: Create benchmark dataset generator
- **Priority 3**: Implement uncertainty quantification

### Phase 4 (Weeks 7-8): Advanced Analysis
- **Priority 1**: Multi-scale uniformity analysis
- **Priority 2**: Enhanced texture analysis integration
- **Priority 3**: Performance optimization

### Phase 5 (Weeks 9-10): Publication Infrastructure
- **Priority 1**: Automated report generation
- **Priority 2**: Data export modules
- **Priority 3**: Documentation and examples

---

## Expected Scientific Impact

### Publication Readiness Metrics
- **Methodological Rigor**: Physics-based thickness estimation with literature validation
- **Statistical Robustness**: Comprehensive uncertainty quantification and validation
- **Reproducibility**: Standardized protocols and automated analysis
- **Comparative Analysis**: Multiple uniformity metrics for comprehensive characterization

### Key Research Contributions
1. **Novel Multi-Scale Uniformity Quantification**: Integration of spatial, statistical, and frequency-domain metrics
2. **Robust Statistical Framework**: Publication-ready uncertainty quantification and validation
3. **Automated Analysis Pipeline**: Reproducible workflow for high-throughput uniformity assessment
4. **Benchmarking Infrastructure**: Standardized validation datasets for method comparison

### Target Metrics for Publication
- **Measurement Uncertainty**: < 5% for uniformity metrics
- **Reproducibility**: > 95% correlation between repeated measurements  
- **Sensitivity**: Detect 10% changes in uniformity
- **Processing Speed**: < 30 seconds per 1024x1024 image

This enhanced plan transforms your ESNF mat analyzer into a publication-ready platform that addresses the fundamental limitation of mat-scale analysis while providing comprehensive, statistically robust uniformity quantification suitable for peer-reviewed research.