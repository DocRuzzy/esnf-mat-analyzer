#!/usr/bin/env python3
"""
Phase 1: Performance Validation (GPU-Optimized Variant)

This script provides a GPU-aware version of the comprehensive benchmarking
framework defined in `benchmark_phase1_performance.py`. It adds optional
GPU acceleration for baseline methods (edge detection, FFT) and integrates
device reporting while preserving identical output structure so downstream
report consumers remain compatible.

Design Goals:
1. Preserve identical result schema (JSON / CSV / summary markdown files)
2. Add GPU acceleration where low-risk (baseline methods + FFT heavy ops)
3. Provide graceful CPU fallback with zero behavior change if no GPU
4. Minimize code duplication: reuse original classes where possible

Usage:
  python benchmark_phase1_performance_gpu.py --images 654 --full

CLI Options:
  --images N        Number of synthetic images to generate (default 100)
  --full            Shortcut for 654 image full benchmark
  --device DEV      Force device: cpu | cuda (default: auto)
  --no-esnf         Skip ESNF analyzer benchmarking (debug speed focus)
  --no-baselines    Skip baseline methods (ESNF only)

Outputs are written to: benchmark_results/phase1_gpu_TIMESTAMP/

Note: This script does NOT modify core analyzer internals; ESNF pipeline
remains CPU unless internal components adopt torch / cv2.cuda.
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import cv2

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

# Reuse original benchmark components
from benchmark_phase1_performance import (
    DatasetGenerator,
    GroundTruthData,
    PerformanceBenchmark,
    BaselineMethodsImplementation,
    get_default_config,  # imported indirectly in original file
    setup_dependencies
)  # type: ignore

from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark_phase1_gpu.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("benchmark_gpu")


# ----------------------------- Utility ---------------------------------- #
def select_device(preferred: Optional[str] = None) -> str:
    if preferred:
        preferred = preferred.lower()
        if preferred == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        if preferred == "cpu":
            return "cpu"
        logger.warning(f"Preferred device '{preferred}' unavailable. Falling back to auto.")
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def report_device(device: str):
    if device == "cuda" and TORCH_AVAILABLE:
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        logger.info(f"Using GPU: {gpu_name} (compute capability {capability[0]}.{capability[1]})")
        logger.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("Using CPU execution path (no CUDA detected or forced).")


# ------------------ GPU Accelerated Baseline Methods -------------------- #
class BaselineMethodsImplementationGPU(BaselineMethodsImplementation):
    """Extends baseline methods with optional CUDA acceleration.

    Enhancements:
      - Canny edge detection via cv2.cuda if available
      - FFT via torch.fft on GPU (fallback to numpy)
    """

    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.cuda_available = (
            device == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available()
        )
        # Check OpenCV CUDA
        self.cv2_cuda_available = False
        if self.cuda_available and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            try:
                # quick smoke check (allocate GpuMat)
                _ = cv2.cuda_GpuMat()
                self.cv2_cuda_available = True
            except Exception:
                self.cv2_cuda_available = False
        if self.cuda_available:
            self.logger.info(
                f"Baseline GPU acceleration enabled. OpenCV-CUDA={'yes' if self.cv2_cuda_available else 'no'}"
            )

    # ---------------- Digits Detection ---------------- #
    def digits_detection_method(self, image: np.ndarray, roi: Optional[Tuple] = None) -> Dict[str, float]:
        start_time = time.time()
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        if roi:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]

        # Edge detection accelerated
        try:
            if self.cv2_cuda_available:
                gpu_mat = cv2.cuda_GpuMat()
                gpu_mat.upload(gray)
                gpu_blur = cv2.cuda.createGaussianFilter(gpu_mat.type(), gpu_mat.type(), (3, 3), 0)
                gpu_gray_blur = gpu_blur.apply(gpu_mat)
                gpu_edges = cv2.cuda.createCannyEdgeDetector(50, 150).detect(gpu_gray_blur)
                edges = gpu_edges.download()
            else:
                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                edges = cv2.Canny(blur, 50, 150)
        except Exception:
            # Fallback safety
            edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=100, maxLineGap=10)
        edge_density = float(np.sum(edges > 0) / edges.size)
        line_count = 0 if lines is None else len(lines)
        cv_val = float(np.std(gray) / (np.mean(gray) + 1e-8))
        processing_time = time.time() - start_time
        return {
            'scale_estimate': line_count * 0.1,
            'uniformity_cv': cv_val,
            'edge_density': edge_density,
            'processing_time': processing_time,
            'success': line_count > 0
        }

    # ---------------- Mark Frequency ---------------- #
    def mark_frequency_method(self, image: np.ndarray, roi: Optional[Tuple] = None) -> Dict[str, float]:
        start_time = time.time()
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        if roi:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]

        use_torch_fft = self.device == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available()
        try:
            if use_torch_fft:
                t = torch.from_numpy(gray.astype(np.float32)).to(self.device)
                # Centering not strictly necessary but can help numeric stability
                t = (t - t.mean())
                f = torch.fft.fft2(t)
                f_shift = torch.fft.fftshift(f)
                magnitude = torch.log(torch.abs(f_shift) + 1.0).detach()
                magnitude_spectrum = magnitude.cpu().numpy()
            else:
                f_transform = np.fft.fft2(gray)
                f_shift = np.fft.fftshift(f_transform)
                magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        except Exception:
            # Hard fallback to numpy
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)

        cy, cx = np.array(gray.shape) // 2
        horizontal_profile = magnitude_spectrum[cy, :]
        peaks = self._find_peaks_simple(horizontal_profile)
        frequency_count = len(peaks)
        frequency_regularity = self._calculate_regularity(peaks)
        cv_val = float(np.std(gray) / (np.mean(gray) + 1e-8))
        processing_time = time.time() - start_time
        return {
            'scale_estimate': frequency_count * 0.05,
            'uniformity_cv': cv_val,
            'frequency_count': frequency_count,
            'frequency_regularity': frequency_regularity,
            'processing_time': processing_time,
            'success': frequency_count > 2
        }


# ---------------- GPU-Aware Benchmark Wrapper --------------------------- #
class PerformanceBenchmarkGPU(PerformanceBenchmark):
    def __init__(self, output_dir: Path, device: str):
        super().__init__(output_dir)
        # Override baseline methods with GPU version
        self.baseline_methods = BaselineMethodsImplementationGPU(device)
        self.device = device
        self.logger.info(f"Initialized GPU benchmark (device={device}).")

    def run_full_benchmark(self, ground_truth_data: List[GroundTruthData],
                           run_esnf: bool = True, run_baselines: bool = True) -> Dict[str, Any]:
        self.logger.info("=== START GPU PHASE 1 BENCHMARK ===")
        self.logger.info(f"Dataset size: {len(ground_truth_data)} synthetic images")
        if run_esnf:
            self.logger.info("Benchmarking ESNF analyzer configs (CPU path)..")
            esnf_results = self._benchmark_esnf_analyzer(ground_truth_data)
        else:
            esnf_results = {}
        if run_baselines:
            self.logger.info("Benchmarking GPU-accelerated baselines..")
            baseline_results = self._benchmark_baseline_methods(ground_truth_data)
        else:
            baseline_results = {}
        if not esnf_results and not baseline_results:
            self.logger.warning("No methods selected (ESNF and baselines both skipped). Returning empty result structure.")
            return {
                'timestamp': datetime.now().isoformat(),
                'dataset_size': len(ground_truth_data),
                'methods_compared': [],
                'metrics': {},
                'relative_performance': {},
                'device': self.device
            }
        comparison_results = self._calculate_comparative_metrics(esnf_results, baseline_results, ground_truth_data)
        self._generate_performance_reports(comparison_results)
        comparison_results['device'] = self.device
        return comparison_results


# ------------------------------- Main ----------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU-Optimized Phase 1 Benchmark")
    p.add_argument('--images', type=int, default=100, help='Number of synthetic images')
    p.add_argument('--full', action='store_true', help='Use full 654 image benchmark (overrides --images)')
    p.add_argument('--device', type=str, default=None, help='Force device: cpu|cuda')
    p.add_argument('--no-esnf', action='store_true', help='Skip ESNF analyzer benchmarking')
    p.add_argument('--no-baselines', action='store_true', help='Skip baseline methods')
    return p.parse_args()


def main():
    args = parse_args()
    num_images = 654 if args.full else args.images
    device = select_device(args.device)
    report_device(device)

    out_dir = Path("benchmark_results") / f"phase1_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Generating synthetic dataset (n={num_images}) ...")

    dataset_gen = DatasetGenerator(out_dir / "datasets")
    gt_data = dataset_gen.generate_rulers2023_style_dataset(num_images)

    bench = PerformanceBenchmarkGPU(out_dir, device)
    start = time.time()
    results = bench.run_full_benchmark(
        gt_data,
        run_esnf=not args.no_esnf,
        run_baselines=not args.no_baselines
    )
    elapsed = time.time() - start
    logger.info(f"Benchmark completed in {elapsed/60:.2f} min")

    # Console summary
    print("\n" + "="*60)
    print("GPU PHASE 1 PERFORMANCE VALIDATION - SUMMARY")
    print("="*60)
    for method, metrics in results.get('metrics', {}).items():
        print(f"\n{method}:")
        print(f"  Success Rate: {metrics.get('success_rate', 0):.2%}")
        print(f"  Avg Processing Time: {metrics.get('avg_processing_time', 0):.3f}s")
        if 'mAPE_scale' in metrics:
            print(f"  Scale mAPE: {metrics['mAPE_scale']:.4f}")
    print(f"\nDevice: {results.get('device', device)}")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
