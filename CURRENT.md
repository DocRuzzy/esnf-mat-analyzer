# CURRENT - Active Session Tasks

**Last Updated:** 2026-07-22 (Session 4)

This file tracks active work for the current development focus. Updated at session start/end.

## Active Tasks

| Priority | Task | Notes |
|----------|------|-------|
| 1 | T003: Expose recommendation weights in GUI | Add sliders for (w_dyn, w_sig, w_sat) |
| 2 | T004: Fix OpenCV cvtColor empty-source crash | Awaiting failing image |

## Recently Completed (This Session)

### Session 4 - Background Correction Operator Fix (2026-07-22)

- ✅ **Diagnosed division-based correction blow-out** (30-image batch sweep)
  - Division (`image / model`) blew out 23/30 samples with >5pp new mat saturation
  - 3 catastrophic (TCD6-GPED2.0-1/-2, TCD6-GPED3.0-1): 74–98% of mat clipped to white
  - Root cause: additive stray light on black surface is the wrong physics for division;
    polynomial model went negative under ruler-starved fits and division exploded

- ✅ **Switched to subtraction-based correction (default)**
  - New `BackgroundCorrectionOperator` enum (SUBTRACT/DIVIDE) in `data_types.py`
  - Formula: `corrected = image − (model − model.min())` — removes illumination
    *variation* only; subtracted amount ≥ 0 so new saturation is impossible
  - Division kept as legacy option via `ProcessingConfig.background_correction_operator`
  - Detail-preservation blending removed → warn-only QA via `results['correction_quality']`

- ✅ **Validation harness**
  - `scripts/benchmarks/batch_correction_sweep.py` (--operator subtract|divide, CI exit codes)
  - **Sweep results: SUBTRACT = 30/30 PASS** (0 new-sat, 0 detail-loss, bg improved 30/30);
    DIVIDE = FAIL (25 flagged) — confirms fix
  - Synthetic ground truth: `SyntheticMatConfig` now models additive illumination
    (linear gradient + reflection blobs); `tests/unit/test_correction_synthetic.py` (8 tests)
    verifies flattening to noise floor, <20% residual gradient, mat structure r≥0.98

### Session 3 - GUI Controls

- ✅ **Task 2: Added GUI slider for heatmap gamma (0.1-1.0)**
  - Added `heatmap_gamma` field to `VisualizationConfig` in `data_types.py`
  - Added slider to Image Controls frame with dynamic label
  - Applied to both `show_heatmap` and `export_image` methods
  - Lower gamma (<1) stretches high thickness values (bright mat region)
  
- ✅ **T052: Added deposition parameter inputs**
  - Added Deposition Parameters frame to GUI
  - Inputs: flow rate (mL/hr), collection time (min), concentration (wt%)
  - Calculate button computes theoretical polymer mass
  - Uses `DepositionParameters.calculate_theoretical_mass_mg()` from calibration.py

- ✅ **T053: Upgraded to four-region detection**
  - Added `FourRegionDetector` class in `region_detector.py`
  - Detects 4 regions: background (outside gel), gel (dark wet ring), mat (bright), ruler
  - Legacy `ThreeRegionDetector` preserved for backward compatibility
  - Added `FourRegionDetectionResult` dataclass with all region masks
  
- ✅ **T054: Fixed background reference selection**
  - Four-region detector ensures background reference comes from TRUE background (outside gel ring)
  - No longer confuses gel (darker than background) with actual background
  - Fixes incorrect thickness calibration
  
- ✅ **T055: Background-only gradient correction**
  - Added `complete_uniformity_analysis_with_four_regions()` to `AdvancedBackgroundProcessor`
  - Fits illumination gradient ONLY to true background pixels
  - Preserves mat thickness features that were being removed
  
- ✅ **T050: Integrated into main pipeline**
  - Updated `image_processor.py` with `use_four_region_detection=True` parameter
  - Updated GUI gel boundary analysis to use four-region detection
  - Added intensity histogram and comparison to legacy three-region
  
- ✅ **T051: Added physical calibration UI**
  - Added Physical Calibration frame to GUI
  - Known thickness input (µm)
  - Point picker (click on image to select calibration point)
  - Marker display toggle
  - Helper methods: `_toggle_physical_cal_ui()`, `_start_cal_point_selection()`, `_on_cal_point_click()`, `_draw_calibration_marker()`

### Session 1 - Visualization Improvements

- ✅ T056: Added power-law normalization (PowerNorm) for heatmap contrast enhancement
  - gamma=0.2 allocates ~90% colormap to high thickness values, ~10% to low values
  - Stretches variations in bright mat region, compresses gel/border region
- ✅ Added CV-based auto-contrast for low-variation data (CV<15% triggers full min-max range)
- ✅ Imported PowerNorm from matplotlib.colors in visualization.py

## Known Issues / Blockers

None currently.

## Session Notes

**Key architecture change**: The analyzer now uses `FourRegionDetector` by default which properly identifies:
1. **Background** - Collection surface OUTSIDE gel ring (TRUE black reference)
2. **Gel** - Dark wet ring around mat (DARKER than background due to wet polymer)
3. **Mat** - Bright nanofiber deposition (thickness signal)
4. **Ruler** - Scale bar (excluded from all analysis)

This fixes the critical issue where gel was being used as background reference, causing incorrect thickness calibration.

---
*For full task history, see [task-ledger.md](task-ledger.md)*
