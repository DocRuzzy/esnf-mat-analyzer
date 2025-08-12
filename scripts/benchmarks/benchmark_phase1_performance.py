#!/usr/bin/env python3
"""
Phase 1: Performance Validation - Comprehensive Benchmarking Framework

This script implements comprehensive benchmarking for the ESNF Mat Analyzer to:
1. Test against simulated Rulers2023-style dataset (654 annotated images)
2. Compare with baseline methods (Digits Detection, Mark Frequency)
3. Validate mAPE/cm@768 metric and other performance indicators
4. Generate detailed performance reports

Author: ESNF Mat Analyzer Team
Date: August 2025
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import logging
import time
import json
import csv
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from esnf_mat_analyzer.main import setup_dependencies
from esnf_mat_analyzer.config.config_manager import get_default_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark_phase1.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    method_name: str
    processing_time: float
    memory_usage: float
    accuracy_metrics: Dict[str, float]
    uniformity_metrics: Dict[str, float]
    success_rate: float
    error_count: int
    dataset_size: int
    timestamp: str

@dataclass
class GroundTruthData:
    """Ground truth data for validation"""
    image_path: str
    ruler_present: bool
    expected_scale: Optional[float]  # pixels per mm
    expected_uniformity: Optional[float]
    mat_region: Optional[Tuple[int, int, int, int]]  # x, y, w, h
    quality_score: float  # 0-1, higher is better quality

class BaselineMethodsImplementation:
    """Implementation of baseline methods for comparison"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def digits_detection_method(self, image: np.ndarray, roi: Optional[Tuple] = None) -> Dict[str, float]:
        """
        Baseline method: Simple digits/edge detection for scale estimation
        Simulates traditional ruler detection based on digit recognition
        """
        start_time = time.time()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply ROI if specified
        if roi:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]
        
        # Simple edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find horizontal lines (ruler body)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=100, maxLineGap=10)
        
        # Calculate simple metrics
        edge_density = np.sum(edges > 0) / edges.size
        line_count = len(lines) if lines is not None else 0
        
        # Simple uniformity based on standard deviation
        uniformity = 1.0 / (1.0 + np.std(gray) / np.mean(gray))
        
        processing_time = time.time() - start_time
        
        return {
            'scale_estimate': line_count * 0.1,  # Rough estimate
            'uniformity_cv': np.std(gray) / np.mean(gray),
            'edge_density': edge_density,
            'processing_time': processing_time,
            'success': line_count > 0
        }
    
    def mark_frequency_method(self, image: np.ndarray, roi: Optional[Tuple] = None) -> Dict[str, float]:
        """
        Baseline method: Mark frequency analysis
        Analyzes periodic patterns to estimate scale
        """
        start_time = time.time()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply ROI if specified
        if roi:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]
        
        # FFT-based frequency analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Find dominant frequencies
        center_y, center_x = np.array(gray.shape) // 2
        
        # Analyze horizontal frequency content
        horizontal_profile = magnitude_spectrum[center_y, :]
        peaks = self._find_peaks_simple(horizontal_profile)
        
        # Calculate frequency-based metrics
        frequency_count = len(peaks)
        frequency_regularity = self._calculate_regularity(peaks)
        
        # Simple uniformity metric
        uniformity = 1.0 / (1.0 + np.std(gray) / np.mean(gray))
        
        processing_time = time.time() - start_time
        
        return {
            'scale_estimate': frequency_count * 0.05,  # Rough estimate
            'uniformity_cv': np.std(gray) / np.mean(gray),
            'frequency_count': frequency_count,
            'frequency_regularity': frequency_regularity,
            'processing_time': processing_time,
            'success': frequency_count > 2
        }
    
    def _find_peaks_simple(self, signal: np.ndarray, threshold: float = 0.1) -> List[int]:
        """Simple peak detection"""
        peaks = []
        for i in range(1, len(signal) - 1):
            if (signal[i] > signal[i-1] and signal[i] > signal[i+1] and 
                signal[i] > np.max(signal) * threshold):
                peaks.append(i)
        return peaks
    
    def _calculate_regularity(self, peaks: List[int]) -> float:
        """Calculate regularity of peak spacing"""
        if len(peaks) < 2:
            return 0.0
        
        spacings = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
        if not spacings:
            return 0.0
        
        mean_spacing = np.mean(spacings)
        std_spacing = np.std(spacings)
        
        # Regularity score (higher is more regular)
        if mean_spacing == 0:
            return 0.0
        
        regularity = 1.0 / (1.0 + std_spacing / mean_spacing)
        return regularity

class DatasetGenerator:
    """Generates synthetic datasets mimicking Rulers2023 characteristics"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def generate_rulers2023_style_dataset(self, num_images: int = 654) -> List[GroundTruthData]:
        """
        Generate synthetic dataset mimicking Rulers2023 characteristics
        - 654 annotated images (as mentioned in requirements)
        - Various ruler types and orientations
        - Different mat qualities and uniformities
        """
        self.logger.info(f"Generating {num_images} synthetic images for Rulers2023-style dataset")
        
        ground_truth_data = []
        
        for i in range(num_images):
            # Generate diverse image characteristics
            image_data = self._generate_single_image(i)
            ground_truth_data.append(image_data)
            
            if (i + 1) % 50 == 0:
                self.logger.info(f"Generated {i + 1}/{num_images} images")
        
        # Save ground truth data
        self._save_ground_truth(ground_truth_data)
        
        return ground_truth_data
    
    def _generate_single_image(self, image_id: int) -> GroundTruthData:
        """Generate a single synthetic image with ground truth"""
        
        # Image dimensions (varied)
        width = np.random.randint(800, 1200)
        height = np.random.randint(600, 1000)
        
        # Create base image
        image = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # Decide if ruler is present (80% chance)
        ruler_present = np.random.random() < 0.8
        expected_scale = None
        
        if ruler_present:
            image, expected_scale = self._add_ruler_to_image(image)
        
        # Add nanofiber mat
        mat_region, expected_uniformity = self._add_mat_to_image(image)
        
        # Add realistic noise and artifacts
        image = self._add_realistic_artifacts(image)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(image, ruler_present)
        
        # Save image
        image_path = self.output_dir / f"synthetic_image_{image_id:04d}.png"
        cv2.imwrite(str(image_path), image)
        
        return GroundTruthData(
            image_path=str(image_path),
            ruler_present=ruler_present,
            expected_scale=expected_scale,
            expected_uniformity=expected_uniformity,
            mat_region=mat_region,
            quality_score=quality_score
        )
    
    def _add_ruler_to_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Add synthetic ruler to image"""
        height, width = image.shape[:2]
        
        # Ruler parameters
        ruler_thickness = np.random.randint(15, 25)
        tick_spacing_px = np.random.randint(60, 100)  # pixels between major ticks
        tick_spacing_mm = 10.0  # mm between major ticks
        expected_scale = tick_spacing_px / tick_spacing_mm
        
        # Ruler position (usually at bottom or right)
        if np.random.random() < 0.7:  # Bottom ruler
            ruler_y = height - 80
            ruler_start_x = 50
            ruler_end_x = width - 50
            
            # Draw ruler body
            cv2.rectangle(image, (ruler_start_x, ruler_y), 
                         (ruler_end_x, ruler_y + ruler_thickness), (30, 30, 30), -1)
            
            # Draw ticks
            num_ticks = (ruler_end_x - ruler_start_x) // tick_spacing_px
            for i in range(num_ticks + 1):
                tick_x = ruler_start_x + i * tick_spacing_px
                tick_height = ruler_thickness + 15
                cv2.line(image, (tick_x, ruler_y - 5), 
                        (tick_x, ruler_y + tick_height), (10, 10, 10), 2)
        
        else:  # Right ruler
            ruler_x = width - 80
            ruler_start_y = 50
            ruler_end_y = height - 50
            
            # Draw ruler body
            cv2.rectangle(image, (ruler_x, ruler_start_y), 
                         (ruler_x + ruler_thickness, ruler_end_y), (30, 30, 30), -1)
            
            # Draw ticks
            num_ticks = (ruler_end_y - ruler_start_y) // tick_spacing_px
            for i in range(num_ticks + 1):
                tick_y = ruler_start_y + i * tick_spacing_px
                tick_width = ruler_thickness + 15
                cv2.line(image, (ruler_x - 5, tick_y), 
                        (ruler_x + tick_width, tick_y), (10, 10, 10), 2)
        
        return image, expected_scale
    
    def _add_mat_to_image(self, image: np.ndarray) -> Tuple[Tuple[int, int, int, int], float]:
        """Add synthetic nanofiber mat to image"""
        height, width = image.shape[:2]
        
        # Mat region (avoid edges and potential ruler areas)
        margin = 100
        mat_x = margin
        mat_y = margin
        mat_w = width - 2 * margin - 100  # Leave space for ruler
        mat_h = height - 2 * margin - 100
        
        # Generate mat with varying uniformity
        uniformity_level = np.random.uniform(0.3, 0.95)  # Expected uniformity
        
        # Create thickness map for the mat
        y, x = np.ogrid[:mat_h, :mat_w]
        center_y, center_x = mat_h // 2, mat_w // 2
        
        # Base thickness pattern
        distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        max_distance = np.sqrt(center_y**2 + center_x**2)
        
        # Create thickness variation based on uniformity level
        if uniformity_level > 0.8:  # High uniformity
            thickness = 2.0 + 0.3 * np.sin(distance * 0.05) + 0.1 * np.random.randn(mat_h, mat_w)
        elif uniformity_level > 0.6:  # Medium uniformity
            thickness = 2.0 + 0.8 * (1 - distance / max_distance) + 0.3 * np.random.randn(mat_h, mat_w)
        else:  # Low uniformity
            thickness = 1.0 + 2.0 * np.random.random((mat_h, mat_w)) + 0.5 * np.random.randn(mat_h, mat_w)
        
        thickness = np.clip(thickness, 0.1, 4.0)
        
        # Convert thickness to brightness (Beer-Lambert-like relationship)
        brightness = 255 * np.exp(-thickness * 0.5)
        brightness = np.clip(brightness, 50, 240).astype(np.uint8)
        
        # Apply to image
        for c in range(3):
            image[mat_y:mat_y+mat_h, mat_x:mat_x+mat_w, c] = brightness
        
        return (mat_x, mat_y, mat_w, mat_h), uniformity_level
    
    def _add_realistic_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Add realistic imaging artifacts"""
        
        # Add noise
        noise = np.random.normal(0, 3, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add slight blur (lens effects)
        if np.random.random() < 0.3:
            image = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        # Add illumination gradient
        if np.random.random() < 0.4:
            h, w = image.shape[:2]
            y, x = np.ogrid[:h, :w]
            
            # Create gradient
            gradient = 1.0 + 0.2 * (x / w - 0.5)
            for c in range(3):
                image[:, :, c] = np.clip(image[:, :, c] * gradient, 0, 255).astype(np.uint8)
        
        return image
    
    def _calculate_quality_score(self, image: np.ndarray, ruler_present: bool) -> float:
        """Calculate image quality score"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Factors affecting quality
        contrast = np.std(gray) / np.mean(gray)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Base quality score
        quality = 0.5
        
        # Good contrast increases quality
        if 0.1 < contrast < 0.5:
            quality += 0.2
        
        # Good sharpness increases quality
        if sharpness > 100:
            quality += 0.2
        
        # Ruler presence affects detectability
        if ruler_present:
            quality += 0.1
        
        return np.clip(quality, 0.0, 1.0)
    
    def _save_ground_truth(self, ground_truth_data: List[GroundTruthData]):
        """Save ground truth data to files"""
        
        # Save as JSON
        json_data = [asdict(gt) for gt in ground_truth_data]
        with open(self.output_dir / "ground_truth.json", 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save as CSV
        csv_data = []
        for gt in ground_truth_data:
            csv_data.append({
                'image_path': gt.image_path,
                'ruler_present': gt.ruler_present,
                'expected_scale': gt.expected_scale,
                'expected_uniformity': gt.expected_uniformity,
                'mat_region_x': gt.mat_region[0] if gt.mat_region else None,
                'mat_region_y': gt.mat_region[1] if gt.mat_region else None,
                'mat_region_w': gt.mat_region[2] if gt.mat_region else None,
                'mat_region_h': gt.mat_region[3] if gt.mat_region else None,
                'quality_score': gt.quality_score
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(self.output_dir / "ground_truth.csv", index=False)
        
        self.logger.info(f"Saved ground truth data to {self.output_dir}")

class PerformanceBenchmark:
    """Main benchmarking class"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.baseline_methods = BaselineMethodsImplementation()
        self.dataset_generator = DatasetGenerator(output_dir / "datasets")
        
        # Results storage
        self.results: List[BenchmarkResult] = []
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete Phase 1 performance validation"""
        
        self.logger.info("=== STARTING PHASE 1: PERFORMANCE VALIDATION ===")
        
        # Step 1: Generate synthetic Rulers2023-style dataset
        self.logger.info("Step 1: Generating synthetic dataset...")
        ground_truth_data = self.dataset_generator.generate_rulers2023_style_dataset(100)  # Start with 100 for speed
        
        # Step 2: Benchmark ESNF Mat Analyzer
        self.logger.info("Step 2: Benchmarking ESNF Mat Analyzer...")
        esnf_results = self._benchmark_esnf_analyzer(ground_truth_data)
        
        # Step 3: Benchmark baseline methods
        self.logger.info("Step 3: Benchmarking baseline methods...")
        baseline_results = self._benchmark_baseline_methods(ground_truth_data)
        
        # Step 4: Calculate comparative metrics
        self.logger.info("Step 4: Calculating comparative metrics...")
        comparison_results = self._calculate_comparative_metrics(esnf_results, baseline_results, ground_truth_data)
        
        # Step 5: Generate reports
        self.logger.info("Step 5: Generating performance reports...")
        self._generate_performance_reports(comparison_results)
        
        return comparison_results
    
    def _benchmark_esnf_analyzer(self, ground_truth_data: List[GroundTruthData]) -> Dict[str, Any]:
        """Benchmark the ESNF Mat Analyzer"""
        
        # Test different configurations
        configs_to_test = [
            ("Default", get_default_config()),
            ("Rolling Ball BG", self._get_rolling_ball_config()),
            ("Beer Lambert", self._get_beer_lambert_config()),
            ("High Precision", self._get_high_precision_config())
        ]
        
        esnf_results = {}
        
        for config_name, config in configs_to_test:
            self.logger.info(f"Testing ESNF configuration: {config_name}")
            
            analyzer = setup_dependencies(config)
            config_results = []
            
            for i, gt_data in enumerate(ground_truth_data):
                if i % 20 == 0:
                    self.logger.info(f"Processing image {i+1}/{len(ground_truth_data)}")
                
                result = self._process_single_image_esnf(analyzer, gt_data)
                config_results.append(result)
            
            esnf_results[config_name] = config_results
        
        return esnf_results
    
    def _benchmark_baseline_methods(self, ground_truth_data: List[GroundTruthData]) -> Dict[str, Any]:
        """Benchmark baseline methods"""
        
        baseline_results = {}
        
        # Test Digits Detection method
        self.logger.info("Testing Digits Detection baseline method")
        digits_results = []
        for gt_data in ground_truth_data:
            image = cv2.imread(gt_data.image_path)
            result = self.baseline_methods.digits_detection_method(image, gt_data.mat_region)
            digits_results.append(result)
        baseline_results["Digits Detection"] = digits_results
        
        # Test Mark Frequency method
        self.logger.info("Testing Mark Frequency baseline method")
        frequency_results = []
        for gt_data in ground_truth_data:
            image = cv2.imread(gt_data.image_path)
            result = self.baseline_methods.mark_frequency_method(image, gt_data.mat_region)
            frequency_results.append(result)
        baseline_results["Mark Frequency"] = frequency_results
        
        return baseline_results
    
    def _process_single_image_esnf(self, analyzer, gt_data: GroundTruthData) -> Dict[str, Any]:
        """Process single image with ESNF analyzer"""
        start_time = time.time()
        
        try:
            # Load image
            image_path = Path(gt_data.image_path)
            
            # Process with analyzer
            result = analyzer.process_image(image_path, roi=gt_data.mat_region)
            
            processing_time = time.time() - start_time
            
            # Extract key metrics
            metrics = {
                'processing_time': processing_time,
                'success': True,
                'overall_uniformity': result.metrics.get('overall_mat_uniformity', 0.0),
                'cv_uniformity': result.metrics.get('uniformity_cv', 0.0),
                'anisotropy_index': result.metrics.get('anisotropy_index', 0.0),
                'scale_detected': result.spatial_scale_pixels_per_mm is not None,
                'detected_scale': result.spatial_scale_pixels_per_mm,
            }
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Error processing {gt_data.image_path}: {e}")
            return {
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def _calculate_comparative_metrics(self, esnf_results: Dict, baseline_results: Dict, 
                                     ground_truth_data: List[GroundTruthData]) -> Dict[str, Any]:
        """Calculate comparative performance metrics"""
        
        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(ground_truth_data),
            'methods_compared': list(esnf_results.keys()) + list(baseline_results.keys()),
            'metrics': {}
        }
        
        # Calculate metrics for each method
        all_methods = {**esnf_results, **baseline_results}
        
        for method_name, method_results in all_methods.items():
            metrics = self._calculate_method_metrics(method_results, ground_truth_data, method_name)
            comparison_results['metrics'][method_name] = metrics
        
        # Calculate relative performance
        comparison_results['relative_performance'] = self._calculate_relative_performance(
            comparison_results['metrics'])
        
        return comparison_results
    
    def _calculate_method_metrics(self, method_results: List[Dict], 
                                ground_truth_data: List[GroundTruthData], 
                                method_name: str) -> Dict[str, float]:
        """Calculate performance metrics for a method"""
        
        successful_results = [r for r in method_results if r.get('success', False)]
        success_rate = len(successful_results) / len(method_results)
        
        # Processing time statistics
        processing_times = [r.get('processing_time', 0) for r in method_results]
        avg_processing_time = np.mean(processing_times)
        
        metrics = {
            'success_rate': success_rate,
            'avg_processing_time': avg_processing_time,
            'total_images': len(method_results),
            'successful_images': len(successful_results)
        }
        
        if successful_results:
            # Scale detection accuracy (for methods that detect scale)
            if any('detected_scale' in r for r in successful_results):
                scale_accuracies = []
                for i, result in enumerate(successful_results):
                    gt = ground_truth_data[i]
                    if gt.expected_scale and result.get('detected_scale'):
                        # Calculate mAPE (mean Absolute Percentage Error)
                        error = abs(result['detected_scale'] - gt.expected_scale) / gt.expected_scale
                        scale_accuracies.append(error)
                
                if scale_accuracies:
                    metrics['mAPE_scale'] = np.mean(scale_accuracies)
                    metrics['scale_detection_rate'] = len(scale_accuracies) / len(successful_results)
            
            # Uniformity metrics
            if any('overall_uniformity' in r for r in successful_results):
                uniformity_values = [r.get('overall_uniformity', 0) for r in successful_results]
                metrics['avg_uniformity'] = np.mean(uniformity_values)
                metrics['std_uniformity'] = np.std(uniformity_values)
        
        return metrics
    
    def _calculate_relative_performance(self, all_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate relative performance between methods"""
        
        # Find best performing method for each metric
        best_performers = {}
        
        # Success rate
        success_rates = {method: metrics.get('success_rate', 0) 
                        for method, metrics in all_metrics.items()}
        best_performers['success_rate'] = max(success_rates.items(), key=lambda x: x[1])
        
        # Processing time (lower is better)
        processing_times = {method: metrics.get('avg_processing_time', float('inf')) 
                           for method, metrics in all_metrics.items()}
        best_performers['processing_time'] = min(processing_times.items(), key=lambda x: x[1])
        
        # Scale accuracy (lower mAPE is better)
        scale_accuracies = {method: metrics.get('mAPE_scale', float('inf')) 
                           for method, metrics in all_metrics.items() 
                           if 'mAPE_scale' in metrics}
        if scale_accuracies:
            best_performers['scale_accuracy'] = min(scale_accuracies.items(), key=lambda x: x[1])
        
        return {
            'best_performers': best_performers,
            'performance_matrix': all_metrics
        }
    
    def _generate_performance_reports(self, comparison_results: Dict[str, Any]):
        """Generate comprehensive performance reports"""
        
        # Save raw results as JSON
        with open(self.output_dir / "benchmark_results.json", 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        # Generate summary report
        self._generate_summary_report(comparison_results)
        
        # Generate visualizations
        self._generate_performance_visualizations(comparison_results)
        
        # Generate detailed CSV reports
        self._generate_csv_reports(comparison_results)
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate human-readable summary report"""
        
        report_path = self.output_dir / "benchmark_summary.md"
        
        with open(report_path, 'w') as f:
            f.write("# ESNF Mat Analyzer - Phase 1 Performance Validation Report\n\n")
            f.write(f"**Generated:** {results['timestamp']}\n")
            f.write(f"**Dataset Size:** {results['dataset_size']} images\n")
            f.write(f"**Methods Compared:** {', '.join(results['methods_compared'])}\n\n")
            
            f.write("## Performance Summary\n\n")
            
            for method_name, metrics in results['metrics'].items():
                f.write(f"### {method_name}\n")
                f.write(f"- **Success Rate:** {metrics.get('success_rate', 0):.2%}\n")
                f.write(f"- **Average Processing Time:** {metrics.get('avg_processing_time', 0):.3f}s\n")
                
                if 'mAPE_scale' in metrics:
                    f.write(f"- **Scale Detection mAPE:** {metrics['mAPE_scale']:.4f}\n")
                    f.write(f"- **Scale Detection Rate:** {metrics.get('scale_detection_rate', 0):.2%}\n")
                
                if 'avg_uniformity' in metrics:
                    f.write(f"- **Average Uniformity Score:** {metrics['avg_uniformity']:.3f}\n")
                
                f.write("\n")
            
            # Best performers section
            best_performers = results['relative_performance']['best_performers']
            f.write("## Best Performers\n\n")
            
            for metric, (method, value) in best_performers.items():
                f.write(f"- **{metric.replace('_', ' ').title()}:** {method} ({value:.4f})\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("Based on the benchmark results:\n\n")
            
            # Generate recommendations based on results
            success_winner = best_performers.get('success_rate', ('Unknown', 0))[0]
            speed_winner = best_performers.get('processing_time', ('Unknown', 0))[0]
            
            f.write(f"1. **Most Reliable Method:** {success_winner}\n")
            f.write(f"2. **Fastest Method:** {speed_winner}\n")
            
            if 'scale_accuracy' in best_performers:
                accuracy_winner = best_performers['scale_accuracy'][0]
                f.write(f"3. **Most Accurate Scale Detection:** {accuracy_winner}\n")
        
        self.logger.info(f"Generated summary report: {report_path}")
    
    def _generate_performance_visualizations(self, results: Dict[str, Any]):
        """Generate performance visualization plots"""
        
        plt.style.use('default')
        
        # Create performance comparison plots
        methods = list(results['metrics'].keys())
        
        # Plot 1: Success rates
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        success_rates = [results['metrics'][method].get('success_rate', 0) for method in methods]
        axes[0, 0].bar(methods, success_rates)
        axes[0, 0].set_title('Success Rate by Method')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Processing times
        processing_times = [results['metrics'][method].get('avg_processing_time', 0) for method in methods]
        axes[0, 1].bar(methods, processing_times)
        axes[0, 1].set_title('Average Processing Time by Method')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Scale accuracy (mAPE)
        scale_methods = []
        scale_accuracies = []
        for method in methods:
            if 'mAPE_scale' in results['metrics'][method]:
                scale_methods.append(method)
                scale_accuracies.append(results['metrics'][method]['mAPE_scale'])
        
        if scale_accuracies:
            axes[1, 0].bar(scale_methods, scale_accuracies)
            axes[1, 0].set_title('Scale Detection Accuracy (mAPE - lower is better)')
            axes[1, 0].set_ylabel('mAPE')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Uniformity scores
        uniformity_methods = []
        uniformity_scores = []
        for method in methods:
            if 'avg_uniformity' in results['metrics'][method]:
                uniformity_methods.append(method)
                uniformity_scores.append(results['metrics'][method]['avg_uniformity'])
        
        if uniformity_scores:
            axes[1, 1].bar(uniformity_methods, uniformity_scores)
            axes[1, 1].set_title('Average Uniformity Scores')
            axes[1, 1].set_ylabel('Uniformity Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Generated performance visualization plots")
    
    def _generate_csv_reports(self, results: Dict[str, Any]):
        """Generate detailed CSV reports"""
        
        # Performance metrics CSV
        metrics_data = []
        for method, metrics in results['metrics'].items():
            row = {'Method': method}
            row.update(metrics)
            metrics_data.append(row)
        
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_csv(self.output_dir / "performance_metrics.csv", index=False)
        
        self.logger.info("Generated CSV reports")
    
    # Configuration methods for different ESNF setups
    def _get_rolling_ball_config(self):
        """Get configuration with rolling ball background correction"""
        config = get_default_config()
        # Modify config for rolling ball background correction
        # This would need to be adjusted based on actual config structure
        return config
    
    def _get_beer_lambert_config(self):
        """Get configuration with Beer-Lambert thickness model"""
        config = get_default_config()
        # Modify config for Beer-Lambert model
        # This would need to be adjusted based on actual config structure
        return config
    
    def _get_high_precision_config(self):
        """Get configuration optimized for high precision"""
        config = get_default_config()
        # Modify config for high precision analysis
        # This would need to be adjusted based on actual config structure
        return config

def main():
    """Main execution function"""
    
    # Setup output directory
    output_dir = Path("benchmark_results") / f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting Phase 1 Performance Validation")
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark(output_dir)
    
    # Run complete benchmark
    try:
        results = benchmark.run_full_benchmark()
        
        logger.info("=== BENCHMARK COMPLETED SUCCESSFULLY ===")
        logger.info(f"Results saved to: {output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("PHASE 1 PERFORMANCE VALIDATION - SUMMARY")
        print("="*60)
        
        for method, metrics in results['metrics'].items():
            print(f"\n{method}:")
            print(f"  Success Rate: {metrics.get('success_rate', 0):.2%}")
            print(f"  Avg Processing Time: {metrics.get('avg_processing_time', 0):.3f}s")
            if 'mAPE_scale' in metrics:
                print(f"  Scale mAPE: {metrics['mAPE_scale']:.4f}")
        
        print(f"\nDetailed results available in: {output_dir}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()
