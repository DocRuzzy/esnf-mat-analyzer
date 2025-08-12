#!/usr/bin/env python3
"""
Benchmark Utilities and Helper Functions

This module provides utility functions for the Phase 1 benchmarking framework,
including metric calculations, data analysis helpers, and validation tools.

Author: ESNF Mat Analyzer Team
Date: August 2025
"""

import numpy as np
import cv2
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculate various performance and accuracy metrics"""
    
    @staticmethod
    def calculate_mape(predicted: np.ndarray, actual: np.ndarray, 
                      epsilon: float = 1e-8) -> float:
        """
        Calculate Mean Absolute Percentage Error (mAPE)
        
        Args:
            predicted: Predicted values
            actual: Actual values
            epsilon: Small value to avoid division by zero
            
        Returns:
            mAPE value
        """
        actual_safe = np.where(np.abs(actual) < epsilon, epsilon, actual)
        return np.mean(np.abs((actual - predicted) / actual_safe)) * 100
    
    @staticmethod
    def calculate_mape_at_resolution(predicted: np.ndarray, actual: np.ndarray, 
                                   resolution: int = 768) -> float:
        """
        Calculate mAPE/cm@768 metric (normalized to specific resolution)
        
        Args:
            predicted: Predicted scale values (pixels/mm)
            actual: Actual scale values (pixels/mm)
            resolution: Target resolution for normalization
            
        Returns:
            Normalized mAPE value
        """
        # Normalize to specified resolution
        norm_factor = resolution / 768.0  # Normalize to 768px baseline
        
        predicted_norm = predicted * norm_factor
        actual_norm = actual * norm_factor
        
        return MetricsCalculator.calculate_mape(predicted_norm, actual_norm)
    
    @staticmethod
    def calculate_detection_accuracy(detected: List[bool], 
                                   ground_truth: List[bool]) -> Dict[str, float]:
        """
        Calculate detection accuracy metrics (precision, recall, F1)
        
        Args:
            detected: List of detection results
            ground_truth: List of ground truth values
            
        Returns:
            Dictionary containing accuracy metrics
        """
        detected = np.array(detected, dtype=bool)
        ground_truth = np.array(ground_truth, dtype=bool)
        
        # Calculate confusion matrix elements
        true_positives = np.sum(detected & ground_truth)
        false_positives = np.sum(detected & ~ground_truth)
        false_negatives = np.sum(~detected & ground_truth)
        true_negatives = np.sum(~detected & ~ground_truth)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(detected)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'true_negatives': int(true_negatives)
        }
    
    @staticmethod
    def calculate_uniformity_correlation(predicted: np.ndarray, 
                                       actual: np.ndarray) -> Dict[str, float]:
        """
        Calculate correlation metrics for uniformity measurements
        
        Args:
            predicted: Predicted uniformity values
            actual: Actual uniformity values
            
        Returns:
            Dictionary containing correlation metrics
        """
        # Remove any NaN or infinite values
        mask = np.isfinite(predicted) & np.isfinite(actual)
        pred_clean = predicted[mask]
        actual_clean = actual[mask]
        
        if len(pred_clean) < 2:
            return {'pearson_r': 0, 'spearman_r': 0, 'rmse': float('inf')}
        
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(pred_clean, actual_clean)
        
        # Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(pred_clean, actual_clean)
        
        # RMSE
        rmse = np.sqrt(np.mean((pred_clean - actual_clean) ** 2))
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'rmse': rmse
        }

class BenchmarkValidator:
    """Validate benchmark results and ensure data quality"""
    
    @staticmethod
    def validate_ground_truth_data(ground_truth_data: List[Dict]) -> Dict[str, Any]:
        """
        Validate ground truth data quality and completeness
        
        Args:
            ground_truth_data: List of ground truth data dictionaries
            
        Returns:
            Validation report
        """
        report = {
            'total_samples': len(ground_truth_data),
            'valid_samples': 0,
            'issues': [],
            'quality_stats': {}
        }
        
        valid_count = 0
        quality_scores = []
        
        for i, data in enumerate(ground_truth_data):
            issues = []
            
            # Check required fields
            required_fields = ['image_path', 'ruler_present', 'quality_score']
            for field in required_fields:
                if field not in data or data[field] is None:
                    issues.append(f"Sample {i}: Missing {field}")
            
            # Check file existence
            if 'image_path' in data and data['image_path']:
                if not Path(data['image_path']).exists():
                    issues.append(f"Sample {i}: Image file not found")
            
            # Check quality score range
            if 'quality_score' in data and data['quality_score'] is not None:
                if not (0 <= data['quality_score'] <= 1):
                    issues.append(f"Sample {i}: Quality score out of range [0,1]")
                else:
                    quality_scores.append(data['quality_score'])
            
            # Check scale consistency
            if data.get('ruler_present') and data.get('expected_scale') is None:
                issues.append(f"Sample {i}: Ruler present but no expected scale")
            
            if not issues:
                valid_count += 1
            else:
                report['issues'].extend(issues)
        
        report['valid_samples'] = valid_count
        report['validity_rate'] = valid_count / len(ground_truth_data)
        
        if quality_scores:
            report['quality_stats'] = {
                'mean_quality': np.mean(quality_scores),
                'std_quality': np.std(quality_scores),
                'min_quality': np.min(quality_scores),
                'max_quality': np.max(quality_scores)
            }
        
        return report
    
    @staticmethod
    def validate_benchmark_results(results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate benchmark results for consistency and completeness
        
        Args:
            results: Benchmark results dictionary
            
        Returns:
            Validation report
        """
        report = {
            'validation_passed': True,
            'issues': [],
            'completeness_score': 0.0
        }
        
        # Check required top-level fields
        required_fields = ['timestamp', 'dataset_size', 'methods_compared', 'metrics']
        missing_fields = [field for field in required_fields if field not in results]
        if missing_fields:
            report['issues'].append(f"Missing top-level fields: {missing_fields}")
            report['validation_passed'] = False
        
        # Check metrics completeness
        if 'metrics' in results:
            methods = results.get('methods_compared', [])
            metrics_methods = list(results['metrics'].keys())
            
            missing_methods = set(methods) - set(metrics_methods)
            if missing_methods:
                report['issues'].append(f"Missing metrics for methods: {missing_methods}")
            
            # Check individual method metrics
            expected_metrics = ['success_rate', 'avg_processing_time']
            for method, metrics in results['metrics'].items():
                missing_metrics = [m for m in expected_metrics if m not in metrics]
                if missing_metrics:
                    report['issues'].append(f"Method {method} missing metrics: {missing_metrics}")
        
        # Calculate completeness score
        total_checks = len(required_fields) + len(results.get('methods_compared', []))
        passed_checks = total_checks - len(report['issues'])
        report['completeness_score'] = passed_checks / total_checks if total_checks > 0 else 0
        
        return report

class VisualizationGenerator:
    """Generate advanced visualizations for benchmark results"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_method_comparison_heatmap(self, results: Dict[str, Any], 
                                         save_path: Optional[Path] = None) -> Path:
        """
        Generate heatmap comparing methods across metrics
        
        Args:
            results: Benchmark results
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        # Extract metrics data
        methods = list(results['metrics'].keys())
        
        # Define metrics to include in heatmap
        metric_keys = ['success_rate', 'avg_processing_time', 'mAPE_scale', 'avg_uniformity']
        metric_labels = ['Success Rate', 'Processing Time (s)', 'Scale mAPE', 'Uniformity Score']
        
        # Create data matrix
        data_matrix = []
        available_metrics = []
        
        for metric_key, metric_label in zip(metric_keys, metric_labels):
            row = []
            has_data = False
            
            for method in methods:
                if metric_key in results['metrics'][method]:
                    value = results['metrics'][method][metric_key]
                    row.append(value)
                    has_data = True
                else:
                    row.append(np.nan)
            
            if has_data:
                data_matrix.append(row)
                available_metrics.append(metric_label)
        
        if not data_matrix:
            logger.warning("No comparable metrics found for heatmap")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(data_matrix, index=available_metrics, columns=methods)
        
        # Normalize data (except processing time - lower is better)
        df_norm = df.copy()
        for i, metric in enumerate(available_metrics):
            if 'Processing Time' in metric or 'mAPE' in metric:
                # For these metrics, lower is better - invert scale
                df_norm.iloc[i] = 1 - (df.iloc[i] - df.iloc[i].min()) / (df.iloc[i].max() - df.iloc[i].min())
            else:
                # Higher is better
                df_norm.iloc[i] = (df.iloc[i] - df.iloc[i].min()) / (df.iloc[i].max() - df.iloc[i].min())
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_norm, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Normalized Performance (1.0 = best)'})
        plt.title('Method Performance Comparison Heatmap')
        plt.xlabel('Methods')
        plt.ylabel('Metrics')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "method_comparison_heatmap.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_accuracy_scatter_plots(self, predicted_values: Dict[str, np.ndarray],
                                      actual_values: np.ndarray,
                                      metric_name: str,
                                      save_path: Optional[Path] = None) -> Path:
        """
        Generate scatter plots comparing predicted vs actual values
        
        Args:
            predicted_values: Dictionary of method_name -> predicted values
            actual_values: Array of actual values
            metric_name: Name of the metric being plotted
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        n_methods = len(predicted_values)
        cols = min(3, n_methods)
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if n_methods == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (method_name, predicted) in enumerate(predicted_values.items()):
            ax = axes[i] if n_methods > 1 else axes[0]
            
            # Create scatter plot
            ax.scatter(actual_values, predicted, alpha=0.6, s=50)
            
            # Add perfect prediction line
            min_val = min(np.min(actual_values), np.min(predicted))
            max_val = max(np.max(actual_values), np.max(predicted))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
            
            # Calculate and display R²
            mask = np.isfinite(predicted) & np.isfinite(actual_values)
            if np.sum(mask) > 1:
                r_squared = stats.pearsonr(actual_values[mask], predicted[mask])[0] ** 2
                ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel(f'Actual {metric_name}')
            ax.set_ylabel(f'Predicted {metric_name}')
            ax.set_title(f'{method_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Predicted vs Actual {metric_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"accuracy_scatter_{metric_name.lower().replace(' ', '_')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_performance_radar_chart(self, results: Dict[str, Any],
                                       save_path: Optional[Path] = None) -> Path:
        """
        Generate radar chart comparing methods across multiple metrics
        
        Args:
            results: Benchmark results
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        methods = list(results['metrics'].keys())
        
        # Define metrics for radar chart
        radar_metrics = [
            ('success_rate', 'Success Rate', 1.0),
            ('avg_processing_time', 'Speed', 0.0),  # Inverted - lower is better
            ('mAPE_scale', 'Scale Accuracy', 0.0),  # Inverted - lower is better
            ('avg_uniformity', 'Uniformity Score', 1.0)
        ]
        
        # Filter metrics that are available
        available_metrics = []
        for metric_key, metric_label, _ in radar_metrics:
            if any(metric_key in results['metrics'][method] for method in methods):
                available_metrics.append((metric_key, metric_label))
        
        if len(available_metrics) < 3:
            logger.warning("Not enough metrics for radar chart")
            return None
        
        # Prepare data
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        for method_idx, method in enumerate(methods):
            values = []
            
            for metric_key, metric_label in available_metrics:
                if metric_key in results['metrics'][method]:
                    value = results['metrics'][method][metric_key]
                    
                    # Normalize and invert if necessary
                    if metric_key in ['avg_processing_time', 'mAPE_scale']:
                        # Lower is better - invert scale
                        max_val = max(results['metrics'][m].get(metric_key, 0) for m in methods)
                        min_val = min(results['metrics'][m].get(metric_key, float('inf')) for m in methods if metric_key in results['metrics'][m])
                        if max_val > min_val:
                            value = 1 - (value - min_val) / (max_val - min_val)
                        else:
                            value = 1.0
                    else:
                        # Higher is better - use as is (assuming already normalized 0-1)
                        pass
                    
                    values.append(value)
                else:
                    values.append(0)
            
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[method_idx])
            ax.fill(angles, values, alpha=0.25, color=colors[method_idx])
        
        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([label for _, label in available_metrics])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Method Performance Radar Chart', size=16, pad=20)
        
        if save_path is None:
            save_path = self.output_dir / "performance_radar_chart.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

class ReportGenerator:
    """Generate comprehensive benchmark reports"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_latex_report(self, results: Dict[str, Any], 
                            save_path: Optional[Path] = None) -> Path:
        """
        Generate LaTeX report suitable for academic papers
        
        Args:
            results: Benchmark results
            save_path: Optional path to save the report
            
        Returns:
            Path to saved report
        """
        if save_path is None:
            save_path = self.output_dir / "benchmark_report.tex"
        
        with open(save_path, 'w') as f:
            f.write("\\documentclass{article}\n")
            f.write("\\usepackage{booktabs}\n")
            f.write("\\usepackage{array}\n")
            f.write("\\usepackage{float}\n")
            f.write("\\begin{document}\n\n")
            
            f.write("\\section{ESNF Mat Analyzer Performance Validation}\n\n")
            
            # Summary table
            f.write("\\subsection{Performance Summary}\n\n")
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Method Performance Comparison}\n")
            f.write("\\begin{tabular}{@{}lcccc@{}}\n")
            f.write("\\toprule\n")
            f.write("Method & Success Rate & Avg Time (s) & Scale mAPE & Uniformity \\\\\n")
            f.write("\\midrule\n")
            
            for method, metrics in results['metrics'].items():
                success_rate = metrics.get('success_rate', 0)
                avg_time = metrics.get('avg_processing_time', 0)
                mape = metrics.get('mAPE_scale', '-')
                uniformity = metrics.get('avg_uniformity', '-')
                
                f.write(f"{method} & {success_rate:.3f} & {avg_time:.3f} & ")
                if isinstance(mape, (int, float)):
                    f.write(f"{mape:.4f}")
                else:
                    f.write("-")
                f.write(" & ")
                if isinstance(uniformity, (int, float)):
                    f.write(f"{uniformity:.3f}")
                else:
                    f.write("-")
                f.write(" \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            # Best performers
            if 'relative_performance' in results and 'best_performers' in results['relative_performance']:
                f.write("\\subsection{Best Performing Methods}\n\n")
                f.write("\\begin{itemize}\n")
                
                for metric, (method, value) in results['relative_performance']['best_performers'].items():
                    metric_label = metric.replace('_', ' ').title()
                    f.write(f"\\item \\textbf{{{metric_label}}}: {method} ({value:.4f})\n")
                
                f.write("\\end{itemize}\n\n")
            
            f.write("\\end{document}\n")
        
        return save_path
    
    def generate_json_report(self, results: Dict[str, Any], 
                           save_path: Optional[Path] = None) -> Path:
        """
        Generate machine-readable JSON report
        
        Args:
            results: Benchmark results
            save_path: Optional path to save the report
            
        Returns:
            Path to saved report
        """
        if save_path is None:
            save_path = self.output_dir / "benchmark_results_detailed.json"
        
        # Add metadata
        enhanced_results = results.copy()
        enhanced_results['metadata'] = {
            'generator': 'ESNF Mat Analyzer Benchmark Framework',
            'version': '1.0.0',
            'benchmark_type': 'Phase 1 Performance Validation'
        }
        
        with open(save_path, 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        
        return save_path

def load_benchmark_results(results_path: Path) -> Dict[str, Any]:
    """
    Load benchmark results from JSON file
    
    Args:
        results_path: Path to benchmark results JSON file
        
    Returns:
        Loaded results dictionary
    """
    with open(results_path, 'r') as f:
        return json.load(f)

def compare_benchmark_runs(results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two benchmark runs
    
    Args:
        results1: First benchmark results
        results2: Second benchmark results
        
    Returns:
        Comparison report
    """
    comparison = {
        'run1_timestamp': results1.get('timestamp', 'Unknown'),
        'run2_timestamp': results2.get('timestamp', 'Unknown'),
        'method_comparisons': {},
        'improvements': [],
        'regressions': []
    }
    
    # Compare common methods
    common_methods = set(results1.get('metrics', {}).keys()) & set(results2.get('metrics', {}).keys())
    
    for method in common_methods:
        method_comparison = {}
        metrics1 = results1['metrics'][method]
        metrics2 = results2['metrics'][method]
        
        # Compare common metrics
        common_metrics = set(metrics1.keys()) & set(metrics2.keys())
        
        for metric in common_metrics:
            val1 = metrics1[metric]
            val2 = metrics2[metric]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                change = val2 - val1
                percent_change = (change / val1) * 100 if val1 != 0 else 0
                
                method_comparison[metric] = {
                    'run1': val1,
                    'run2': val2,
                    'change': change,
                    'percent_change': percent_change
                }
                
                # Track significant improvements/regressions
                if abs(percent_change) > 5:  # 5% threshold
                    change_info = {
                        'method': method,
                        'metric': metric,
                        'change': change,
                        'percent_change': percent_change
                    }
                    
                    if (metric in ['success_rate', 'avg_uniformity'] and change > 0) or \
                       (metric in ['avg_processing_time', 'mAPE_scale'] and change < 0):
                        comparison['improvements'].append(change_info)
                    else:
                        comparison['regressions'].append(change_info)
        
        comparison['method_comparisons'][method] = method_comparison
    
    return comparison
