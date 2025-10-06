"""Small harness to tune recommendation metric weights using synthetic images.

Produces a short summary of baseline agreement (default recommendation weights)
and best agreement found over a coarse grid search. Designed to be fast but
more realistic than the tiny test images.

Run from project root with the project's Python virtualenv.
"""
import os
from pathlib import Path
import tempfile
import numpy as np

# Ensure matplotlib uses a non-interactive backend before importing project modules
os.environ.setdefault('MPLBACKEND', 'Agg')

from esnf_mat_analyzer.validation.synthetic_ruler_generator import SyntheticRulerGenerator
from esnf_mat_analyzer.core.data_types import BackgroundCorrectionMethod
from esnf_mat_analyzer.config.config_manager import get_default_config
from esnf_mat_analyzer.core.analyzer import NanoFiberAnalyzer
from esnf_mat_analyzer.processing.background_correction import recommendation as rec
from esnf_mat_analyzer.analysis.multiscale_uniformity import MultiScaleUniformityAnalyzer
from esnf_mat_analyzer.analysis.mat_anisotropy import PowerSpectralDensityAnalyzer
from esnf_mat_analyzer.analysis.mat_anisotropy import MatAnisotropyAnalyzer


def run_tuning(input_dir: Path = None, n_images=6, width=512, height=512, downsample=2, output_path: Path = None):
    # If input_dir provided, use its images; otherwise generate synthetic images
    if input_dir is not None:
        input_dir = Path(input_dir)
        print(f"Using benchmark images from {input_dir}")
        img_paths = sorted([p for p in input_dir.glob('*.png')])
        if not img_paths:
            raise ValueError(f"No PNG images found in {input_dir}")
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="rec_tune_"))
        print(f"Writing temporary images to {tmpdir}")

        gen = SyntheticRulerGenerator(width=width, height=height)
        img_paths = []
        for i in range(n_images):
            img = gen.generate_ruler(ruler_thickness=8 + i*2, rotation_angle=0)
            p = tmpdir / f"synthetic_{i}.png"
            try:
                import cv2
                cv2.imwrite(str(p), img)
            except Exception:
                from PIL import Image
                Image.fromarray(img).save(p)
            img_paths.append(p)

    # Compute benchmark best methods by running analyzer per method
    benchmark_best = {}
    methods = list(BackgroundCorrectionMethod)
    print("Computing benchmark best method per image (this runs the analyzer several times)...")
    # Use real analyzers (do not monkeypatch) for a faithful benchmark run.
    for p in img_paths:
        best_score = None
        best_method = None
        for m in methods:
            cfg = get_default_config()
            cfg.processing.background_correction_method = m
            cfg.processing.leveling.enabled = False
            # Disable ruler detection for benchmark tuning to avoid scale-detection
            # side-effects and to speed up per-method runs (we compare uniformity metrics).
            try:
                cfg.ruler_detection.enabled = False
            except Exception:
                # Older configs or dict-like configs may not have attribute access;
                # handle both dataclass and dict-style gracefully.
                if hasattr(cfg, 'ruler_detection') and isinstance(cfg.ruler_detection, dict):
                    cfg.ruler_detection['enabled'] = False
            cfg.cache_intermediates = False

            analyzer = NanoFiberAnalyzer.from_config(cfg)
            res = analyzer.process_image(p)
            metrics = getattr(res, 'metrics', {}) or {}
            score = metrics.get('overall_mat_uniformity')
            if score is None:
                ai = metrics.get('anisotropy_index')
                if ai is not None:
                    score = 1.0 - float(ai)
            if score is None:
                score = 0.0

            if best_score is None or score > best_score:
                best_score = score
                best_method = m.name

        benchmark_best[str(p.name)] = best_method

    # (No monkeypatching in this run; using actual analyzer implementations.)

    # baseline recommendations
    methods_map = rec.get_standard_methods()
    recommendations = {}
    for p in img_paths:
        from PIL import Image
        img = Image.open(p).convert('L')
        img_np = np.array(img)
        best, scores = rec.recommend_method(img_np, methods_map, downsample=downsample)
        recommendations[str(p.name)] = best

    baseline_matches = sum(1 for k in benchmark_best if recommendations[k] == benchmark_best[k])
    print(f"Baseline matches: {baseline_matches} / {len(img_paths)}")

    # coarse grid search
    grid = [0.2, 0.4, 0.6]
    best_weights = None
    best_accuracy = baseline_matches
    print("Starting coarse grid search over weights...")
    for w_dyn in grid:
        for w_sig in grid:
            w_sat = 1.0 - (w_dyn + w_sig)
            if w_sat < 0:
                continue
            tuned_recs = {}
            for p in img_paths:
                from PIL import Image
                img = Image.open(p).convert('L')
                img_np = np.array(img)
                scs = {}
                for mid, func in methods_map.items():
                    try:
                        corr = func(img_np)
                        dyn = rec._dynamic_range_preservation(img_np, corr)
                        sig = rec._signal_preservation(img_np, corr, None)
                        sat = 1.0 - rec._saturation_rate(corr)
                        sc = w_dyn * dyn + w_sig * sig + w_sat * sat
                        scs[mid] = sc
                    except Exception:
                        scs[mid] = -1.0
                tuned_recs[str(p.name)] = max(scs.items(), key=lambda kv: kv[1])[0]

            matches = sum(1 for k in benchmark_best if tuned_recs[k] == benchmark_best[k])
            if matches > best_accuracy:
                best_accuracy = matches
                best_weights = (w_dyn, w_sig, w_sat)
                print(f"New best {best_accuracy}/{len(img_paths)} at weights {best_weights}")

    print("Tuning complete.")
    print(f"Baseline matches: {baseline_matches}/{len(img_paths)}")
    if best_weights:
        print(f"Best tuned matches: {best_accuracy}/{len(img_paths)} with weights {best_weights}")
    else:
        print("No improvement found over baseline in coarse grid.")

    # Optionally write a small summary to CSV
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['image', 'benchmark_best', 'recommendation'])
            for k in benchmark_best:
                w.writerow([k, benchmark_best[k], recommendations.get(k, '')])
            if best_weights:
                w.writerow([])
                w.writerow(['best_weights', best_weights])
        print(f"Wrote summary to {out}")


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Tune recommendation weights on dataset or synthetic images')
    parser.add_argument('input_dir', nargs='?', help='Path to dataset directory containing PNG images (optional)')
    parser.add_argument('--output', '-o', help='Path to write CSV summary (optional)')
    parser.add_argument('--downsample', '-d', type=int, default=2, help='Downsample factor for recommendation (default 2)')
    args = parser.parse_args()

    run_tuning(input_dir=Path(args.input_dir) if args.input_dir else None, downsample=args.downsample, output_path=Path(args.output) if args.output else None)
