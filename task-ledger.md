# ESNF Mat Analyzer - Task Ledger

Auto-maintained ledger of active, blocked, and completed tasks. Updated by AI assistant after each task-related change.

## Legend

- STATUS: `TODO` | `IN-PROGRESS` | `BLOCKED` | `DONE` | `DEFERRED`
- Prereq: Task IDs that must reach DONE before this can start.
- Block Reason: Short cause if BLOCKED.

## Active Tasks

| ID | Title | Status | Prereq | Block Reason | Notes / Next Action |
|----|-------|--------|--------|--------------|---------------------|
| T003 | Expose recommendation weights in GUI | TODO | T002 | - | Add sliders for (w_dyn, w_sig, w_sat) in `MainWindow`, persist via `config_manager`. Unit tests to verify persistence. |
| T004 | Fix OpenCV cvtColor empty-source crash | IN-PROGRESS | - | Need repro images | Defensive checks added; awaiting failing image to finalize fix and tests. |

## Recently Completed (2026-01-31)

| ID | Title | Status | Completion Date | Notes |
|----|-------|--------|-----------------|-------|
| Task 2 | Add GUI slider for heatmap gamma | DONE | 2026-01-31 | Added gamma slider (0.1-1.0) to Image Controls frame. Updates `VisualizationConfig.heatmap_gamma` which controls power-law normalization in thickness heatmaps. Lower gamma stretches high values (bright mat region). |
| T052 | Add deposition parameter inputs to GUI | DONE | 2026-01-31 | Added Deposition Parameters frame with flow rate (mL/hr), collection time (min), concentration (wt%) inputs. Calculate button computes theoretical mass using `DepositionParameters.calculate_theoretical_mass_mg()`. |
| T053 | Upgrade to four-region detection | DONE | 2026-01-31 | Added `FourRegionDetector` class to `region_detector.py`. Detects 4 regions: background (collection surface OUTSIDE gel), gel (dark wet ring), mat (bright deposition), ruler. Legacy `ThreeRegionDetector` preserved for backward compatibility. |
| T054 | Fix background reference selection | DONE | 2026-01-31 | Four-region detector now ensures background reference comes from true background (outside gel ring), not from gel which is darker. Fixes incorrect thickness calibration when gel was used as background. |
| T055 | Background-only gradient correction | DONE | 2026-01-31 | Added `complete_uniformity_analysis_with_four_regions()` to `AdvancedBackgroundProcessor` that fits illumination gradient ONLY to true background pixels. Preserves mat thickness features. |
| T050 | Integrate four-region detection into main pipeline | DONE | 2026-01-31 | Updated `image_processor.py` to use `FourRegionDetector` via `use_four_region_detection=True` parameter. Updated GUI gel boundary analysis to use four-region detection with comparison view. |
| T051 | Add physical calibration UI with marked point | DONE | 2026-01-31 | Added Physical Calibration frame to GUI with: known thickness input (µm), point picker (click on image), marker display toggle. Methods: `_toggle_physical_cal_ui()`, `_start_cal_point_selection()`, `_on_cal_point_click()`, `_draw_calibration_marker()`. |
| T056 | Add power-law normalization for heatmap contrast | DONE | 2026-01-31 | Added PowerNorm with configurable gamma to `visualization.py`. gamma=0.2 gives ~90% colormap to high thickness values, ~10% to low values. Also added CV-based auto-contrast enhancement for low-variation data (CV<15%). |

## Recently Completed (2026-01-29)

| ID | Title | Status | Completion Date | Notes |
|----|-------|--------|-----------------|-------|
| T045 | Create synthetic data generator module | DONE | 2026-01-29 | Added `esnf_mat_analyzer/data/synthetic_mat_generator.py` with configurable patterns (uniform, radial, Gaussian spots, multi-frequency, realistic mat, calibration target) for validating FFT behavior |
| T046 | Add frequency analysis diagnostic tool | DONE | 2026-01-29 | Added `esnf_mat_analyzer/analysis/frequency_diagnostics.py` for analyzing what spatial frequencies are preserved/removed by filtering |
| T047 | Make FFT enhancement configurable in visualization | DONE | 2026-01-29 | Added `fft_enhancement_enabled`, `fft_blend_ratio`, `fft_sigma_divisor` to VisualizationConfig; updated `visualization.py` to use configurable parameters; **FFT now disabled by default** to preserve real thickness features |
| T048 | Implement three-region detection | DONE | 2026-01-29 | Added `esnf_mat_analyzer/processing/region_detector.py` that identifies background (collection surface), gel (dark perimeter, sub-background), and mat (bright deposition) regions |
| T049 | Add reference region calibration module | DONE | 2026-01-29 | Added `esnf_mat_analyzer/processing/calibration.py` with 0-100 normalized scale calibration, saturation detection, and optional physical thickness calibration in micrometers |
| T040 | Smooth heatmap gradients and color transitions | DONE | 2026-01-26 | Added mild Gaussian display smoothing and bilinear interpolation |
| T041 | Add gel boundary shape analysis | DONE | 2026-01-29 | Added `esnf_mat_analyzer/analysis/gel_boundary_analysis.py` with circularity, irregularity, aspect ratio metrics and uniformity correlation analysis |
| T042 | Update GUI with FFT and calibration controls | DONE | 2026-01-29 | Added FFT Enhancement frame (enable checkbox, blend slider, sigma control), Calibration frame (saturation threshold, gel detection checkbox, gel analysis button) to main_window.py |

## Previously Completed Tasks

- T002 Recommendation tuning harness run (2025-10-05)
- T007 Improve multiscale uniformity computation robustness (2025-10-05)
- T006 Add unit tests for new GUI metrics display logic (2025-10-05)
- T017 Expose per-method tunable parameters via config/UI (2025-10-05)
- T026 Safer mat_anisotropy weighted-average fallback (2025-10-05)
- T009 Add CLI flag to export full mat-scale metrics JSON (2025-10-05)
- T018 CLI benchmarking command for background methods (2025-10-05)
- T021 Improve ruler detection robustness and graceful fallback (2025-10-05)
- T022 Add CI integration test for full pipeline on representative sample image (2025-10-05)
- T019 Add optional `pywt` dependency and graceful fallback for multiscale analysis (2025-09-05)
- T020 Fix multiscale fallback memory usage (2025-09-05)
- T023 GUI: Fit main window for 1080p and add left-panel + canvas scrollbars (2025-09-29)
- T025 Unit tests for GUI image/ROI helpers (2025-09-29)
- T008 Document all uniformity metrics in `docs/guides/uniformity.md` (2025-10-05)
- T032 Add circular ROI selection in GUI (2025-11-02)
- T033 Add detailed interpretation guide to Mat-Scale Analysis GUI (2025-11-02)
- T034 Restore scrollable left-panel helpers in GUI (2025-11-02)
- Task 1: Enhance low-level contrast in heatmap export (2025-12-14)

## Blocked Tasks

None currently.

## Deferred Tasks

None currently.

## Backlog / Ideas

- Batch processing UI for multiple images with aggregated uniformity stats.
- GPU acceleration for anisotropy FFT stage.
- Adaptive ROI suggestion using edge density.
- **Multi-image gel shape correlation study** - correlate gel boundary metrics with uniformity across sample sets
- **Gel absorption modeling** - estimate potential hidden polymer mass in gel region

## Update Policy

After every task completion or status change, this file must be updated by the AI assistant (Requirement R-LEDGER-UPDATE). The assistant also updates `.github/copilot-instructions.md` to include this policy.

## Audit / Recent Activity

- **2026-01-31 (Session 3)**: **Completed Task 2 and T052**.
  - Task 2: Added heatmap gamma slider (0.1-1.0 range) to Image Controls. Added `heatmap_gamma` to `VisualizationConfig`. GUI slider updates label dynamically. Both show_heatmap and export_image now use the gamma setting.
  - T052: Added Deposition Parameters frame with flow rate, collection time, concentration inputs. Calculate button uses `DepositionParameters.calculate_theoretical_mass_mg()` to estimate theoretical polymer mass.

- **2026-01-31 (Session 2)**: **Completed four-region detection system (T050-T055)**. Major architecture changes:
  - `FourRegionDetector` class in `region_detector.py` - detects background (outside gel), gel (dark wet ring), mat (bright), ruler (4 distinct regions)
  - Background reference now comes from TRUE background (collection surface) not gel
  - Added `complete_uniformity_analysis_with_four_regions()` to `AdvancedBackgroundProcessor`
  - Updated `image_processor.py` to use four-region detection by default
  - GUI gel boundary analysis now shows four-region detection with intensity histogram and comparison to legacy three-region
  - Added Physical Calibration UI: thickness input (µm), point picker, marker display

- **2026-01-31 (Session 1)**: **Thickness heatmap visualization improvements**. Added power-law normalization (PowerNorm) to stretch high thickness values and compress low values. gamma=0.2 allocates ~90% of colormap to upper half of values (bright mat region), ~10% to lower half (gel/border). Also added CV-based detection for low-variation data - when CV<15%, uses full min-max range for contrast. This addresses user feedback that thickness maps were showing uniform color (no variation) for uniformly bright mats.

- **2026-01-30 (PM)**: **Critical insight: Four-region model required**. Current three-region detection confuses gel with background. Gel (wet polymer ring) is DARKER than true background but has reflections. True background is the collection surface OUTSIDE the gel ring. Added tasks T053-T055 for proper region detection and background reference fix. Created `scripts/auto_tune_detail_preservation.py` with background-only gradient fitting concept. Fixed `ruler_detector.py` dict access bug.

- **2026-01-30 (AM)**: Created auto-tuning script for detail preservation. Discovered that background gradient correction was fitting to gel region (darkest) instead of true background, causing removal of real mat thickness features in lower-right quadrant.

- **2026-01-29**: Major feature update - Added synthetic data generation, frequency diagnostics, three-region detection, calibration module, gel boundary analysis, and GUI controls for FFT/calibration settings. FFT enhancement now **disabled by default** to preserve real thickness features. This addresses the issue where heatmaps were not registering high-frequency components due to aggressive FFT filtering.

