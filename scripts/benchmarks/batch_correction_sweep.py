"""Batch acceptance sweep for background correction on real sample images.

Runs every tests/Samples/TCD*.png through the production four-region
pipeline stages (FourRegionDetector -> polynomial illumination fit ->
_step3_correct_image) and measures correction quality per image:

- new mat saturation (pixels newly clipped to white — signal destruction)
- mat signal std retention (detail preservation)
- background std reduction (did leveling actually flatten the background?)

Pass criteria (per the 2026-07 background-correction fix plan):
- 0 images with new mat saturation > 0.5 percentage points
- mat std retention >= 0.9x original on all images
- background std reduced (or equal) on >= 27/30 images

Usage:
    python scripts/benchmarks/batch_correction_sweep.py --operator subtract
    python scripts/benchmarks/batch_correction_sweep.py --operator divide   # legacy baseline

Exits nonzero if pass criteria fail (CI-friendly).
"""
import argparse
import glob
import logging
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from esnf_mat_analyzer.core.data_types import BackgroundCorrectionOperator
from esnf_mat_analyzer.processing.background_correction.advanced import AdvancedBackgroundProcessor
from esnf_mat_analyzer.processing.region_detector import FourRegionDetector

# Acceptance thresholds
MAX_NEW_SATURATION_PP = 0.5      # percentage points of newly saturated mat pixels
MIN_MAT_STD_RETENTION = 0.9      # corrected mat std must be >= 90% of original
MIN_BG_IMPROVED_FRACTION = 0.9   # background std must not worsen on >= 90% of images


def sweep(sample_glob: str, operator: BackgroundCorrectionOperator, verbose: bool = True):
    rows = []
    for path in sorted(glob.glob(sample_glob)):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        name = os.path.basename(path)
        try:
            det = FourRegionDetector().detect(gray)
            bgm = det.background_mask.astype(bool)
            matm = det.mat_mask.astype(bool)
            if matm.sum() == 0 or bgm.sum() == 0:
                rows.append({"name": name, "error": "no regions detected"})
                continue
            proc = AdvancedBackgroundProcessor()
            est = proc._step2a_polynomial_surface_fitting(gray, bgm, 2)
            corr = proc._step3_correct_image(gray, est, background_mask=bgm, operator=operator)

            sat_o = 100.0 * np.sum(gray[matm] >= 254) / matm.sum()
            sat_c = 100.0 * np.sum(corr[matm] >= 254) / matm.sum()
            rows.append({
                "name": name,
                "quality": det.quality_score,
                "illum_min": float(est.min()),
                "new_sat_pp": sat_c - sat_o,
                "mat_std_o": float(gray[matm].std()),
                "mat_std_c": float(corr[matm].std()),
                "bg_std_o": float(gray[bgm].std()),
                "bg_std_c": float(corr[bgm].std()),
            })
        except Exception as exc:  # pragma: no cover - diagnostic path
            rows.append({"name": name, "error": str(exc)})
    return rows


def evaluate(rows, verbose: bool = True):
    ok_rows = [r for r in rows if "error" not in r]
    errors = [r for r in rows if "error" in r]

    sat_fail = [r for r in ok_rows if r["new_sat_pp"] > MAX_NEW_SATURATION_PP]
    detail_fail = [r for r in ok_rows
                   if r["mat_std_c"] < MIN_MAT_STD_RETENTION * r["mat_std_o"]]
    bg_improved = [r for r in ok_rows if r["bg_std_c"] <= r["bg_std_o"]]

    if verbose:
        hdr = (f"{'image':<28}{'qual':>5}{'illumMin':>9}{'newSat':>8}"
               f"{'matStdO':>8}{'matStdC':>8}{'bgStdO':>7}{'bgStdC':>7}")
        print(hdr)
        for r in ok_rows:
            flag = ""
            if r in sat_fail:
                flag += " SAT!"
            if r in detail_fail:
                flag += " DETAIL!"
            if r["bg_std_c"] > r["bg_std_o"]:
                flag += " BG-WORSE!"
            print(f"{r['name']:<28}{r['quality']:>5.2f}{r['illum_min']:>9.0f}"
                  f"{r['new_sat_pp']:>8.2f}{r['mat_std_o']:>8.1f}{r['mat_std_c']:>8.1f}"
                  f"{r['bg_std_o']:>7.1f}{r['bg_std_c']:>7.1f}{flag}")
        for r in errors:
            print(f"{r['name']:<28}ERROR: {r['error']}")

    n = len(ok_rows)
    passed = (
        n > 0
        and not sat_fail
        and not detail_fail
        and len(bg_improved) >= MIN_BG_IMPROVED_FRACTION * n
    )
    print(f"\n{n} images | new-sat >{MAX_NEW_SATURATION_PP}pp: {len(sat_fail)} | "
          f"mat detail lost: {len(detail_fail)} | bg improved: {len(bg_improved)}/{n} | "
          f"errors: {len(errors)}")
    print("RESULT: " + ("PASS" if passed else "FAIL"))
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=["subtract", "divide"], default="subtract")
    parser.add_argument("--samples", default="tests/Samples/TCD*.png",
                        help="Glob for sample images (relative to repo root)")
    args = parser.parse_args()

    logging.disable(logging.WARNING)
    operator = (BackgroundCorrectionOperator.SUBTRACT if args.operator == "subtract"
                else BackgroundCorrectionOperator.DIVIDE)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sample_glob = os.path.join(repo_root, args.samples)

    print(f"Operator: {args.operator}  Samples: {args.samples}\n")
    rows = sweep(sample_glob, operator)
    passed = evaluate(rows)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
