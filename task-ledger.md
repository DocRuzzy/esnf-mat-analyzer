# ESNF Mat Analyzer - Task Ledger

Auto-maintained ledger of active, blocked, and completed tasks. Updated by AI assistant after each task-related change.

## Legend

- STATUS: `TODO` | `IN-PROGRESS` | `BLOCKED` | `DONE` | `DEFERRED`
- Prereq: Task IDs that must reach DONE before this can start.
- Block Reason: Short cause if BLOCKED.

## Active Tasks

| ID | Title | Status | Prereq | Block Reason | Notes / Next Action |
|----|-------|--------|--------|--------------|---------------------|
| T053 | Upgrade to four-region detection | TODO | - | - | **NEW** Add ruler as 4th region. Regions: background (collection surface), gel (dark wet ring), mat (bright deposition), ruler (scale bar). Current 3-region confuses gel with background. |
| T054 | Fix background reference selection | TODO | T053 | - | **NEW** Background reference must come from collection surface OUTSIDE gel ring. Gel has darkest blacks (wet polymer) but is NOT the true background. |
| T055 | Background-only gradient correction | TODO | T054 | - | **NEW** Fit illumination gradient ONLY to true background pixels, extrapolate into mat region, anchor to darkest true background level. Preserves mat thickness features. |
| T050 | Integrate four-region detection into main pipeline | TODO | T053 | - | Use updated region_detector in analyzer, update GUI overlays |
| T051 | Add physical calibration UI with marked point | TODO | T050 | - | Allow user to mark a point and enter known thickness in µm |
| T052 | Add deposition parameter inputs to GUI | TODO | - | - | Optional inputs for flow rate, time, concentration for theoretical mass estimation |
| T003 | Expose recommendation weights in GUI | TODO | T002 | - | Add sliders for (w_dyn, w_sig, w_sat) in `MainWindow`, persist via `config_manager`. Unit tests to verify persistence. |
| T004 | Fix OpenCV cvtColor empty-source crash | IN-PROGRESS | - | Need repro images | Defensive checks added; awaiting failing image to finalize fix and tests. |
| Task 2 | Add GUI slider for heatmap gamma | TODO | 1 | None | Expose gamma control (0.2–1.5) in Image Controls; persist to user config and use in export. |

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

- **2026-01-30 (PM)**: **Critical insight: Four-region model required**. Current three-region detection confuses gel with background. Gel (wet polymer ring) is DARKER than true background but has reflections. True background is the collection surface OUTSIDE the gel ring. Added tasks T053-T055 for proper region detection and background reference fix. Created `scripts/auto_tune_detail_preservation.py` with background-only gradient fitting concept. Fixed `ruler_detector.py` dict access bug.

- **2026-01-30 (AM)**: Created auto-tuning script for detail preservation. Discovered that background gradient correction was fitting to gel region (darkest) instead of true background, causing removal of real mat thickness features in lower-right quadrant.

- **2026-01-29**: Major feature update - Added synthetic data generation, frequency diagnostics, three-region detection, calibration module, gel boundary analysis, and GUI controls for FFT/calibration settings. FFT enhancement now **disabled by default** to preserve real thickness features. This addresses the issue where heatmaps were not registering high-frequency components due to aggressive FFT filtering.

