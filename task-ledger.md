# ESNF Mat Analyzer - Task Ledger

Auto-maintained ledger of active, blocked, and completed tasks. Updated by AI assistant after each task-related change.

## Legend

- STATUS: `TODO` | `IN-PROGRESS` | `BLOCKED` | `DONE` | `DEFERRED`
- Prereq: Task IDs that must reach DONE before this can start.
- Block Reason: Short cause if BLOCKED.

# ESNF Mat Analyzer - Task Ledger

Auto-maintained ledger of active, blocked, and completed tasks. Updated by the AI assistant after each task-related change.

## Legend

- STATUS: `TODO` | `IN-PROGRESS` | `BLOCKED` | `DONE` | `DEFERRED`
- Prereq: Task IDs that must reach DONE before this can start.
- Block Reason: Short cause if BLOCKED.

## Active Tasks

| ID | Title | Status | Prereq | Block Reason | Notes / Next Action |
|----|-------|--------|--------|--------------|---------------------|
| T002 | Run recommendation tuning on benchmark | DONE | T015 | - | Harness executed and CSV written to `benchmark_results/phase1_demo_20250806_152400/recommendation_tuning_summary.csv`. Note: demo dataset contains synthetic images without ground-truth labels so CSV contains `NONE` entries; recommend re-running on labeled benchmark for meaningful tuning. |
| T003 | Expose recommendation weights in GUI | TODO | T002 | - | Add sliders for (w_dyn, w_sig, w_sat) in `MainWindow`, persist via `config_manager`. Unit tests to verify persistence. |
| T004 | Fix OpenCV cvtColor empty-source crash | IN-PROGRESS | - | Need repro images | Defensive checks added; awaiting failing image to finalize fix and tests. |

| ID | Title | Status | Prereq | Block Reason | Notes / Next Action |
|----|-------|--------|--------|--------------|---------------------|
| T040 | Smooth heatmap gradients and color transitions | DONE | - | - | Added mild Gaussian display smoothing and bilinear interpolation in `esnf_mat_analyzer/visualization/visualization.py` to improve gradient smoothness. Widened percentile range by ±1 for smoother color scaling. Completion: 2026-01-26 |

| ID | Task | Status | Prerequisites | Blocking Reasons | Notes |
|----|------|--------|----------------|------------------|-------|
| 1 | Enhance low-level contrast in heatmap export | DONE | None | None | Applied gamma mapping (0.6) to normalized ROI thickness values and inverted normalization (max→0, min→1) in `esnf_mat_analyzer/gui/main_window.py` to make low thickness brighter and more distinguishable. Completion: 2025-12-14 |
| 2 | Add GUI slider for heatmap gamma | TODO | 1 | None | Expose gamma control (0.2–1.5) in Image Controls; persist to user config and use in export.

## Completed Tasks

- T007 Improve multiscale uniformity computation robustness (2025-10-05)
	- Added percentile-based normalization for detail scales, improved pywt import/error handling, and made fallback more robust to empty/degenerate ROI.
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
 - T033 Add detailed interpretation guide to Mat-Scale Analysis GUI — DONE (2025-11-02): inserted expanded per-metric interpretations into `esnf_mat_analyzer/gui/main_window.py`, fixed syntax/indentation issues discovered during edit, and verified no syntax errors remain with a quick static check.

- T034 Restore scrollable left-panel helpers in GUI — DONE (2025-11-02): Recreated the scrollable left-panel container, canvas, scrollbar, and added the missing `on_frame_configure` and `on_canvas_configure` helpers in `esnf_mat_analyzer/gui/main_window.py` so the left-panel scrolling/bounding behavior no longer raises a NameError at startup. Verified edits applied and file parsed without syntax errors.

- T002 Recommendation tuning harness run (2025-10-05)
	- Ran `scripts/tune_recommendation_on_synthetic.py` over demo dataset; output summary CSV at `benchmark_results/phase1_demo_20250806_152400/recommendation_tuning_summary.csv`. Demo images lacked per-image benchmark labels (entries contain `NONE`) so no tuned weights were produced. Recommend re-running on labeled benchmark for useful tuning.

## Blocked Tasks

None currently.

## Deferred Tasks

None currently.

## Backlog / Ideas

- Batch processing UI for multiple images with aggregated uniformity stats.
- GPU acceleration for anisotropy FFT stage.
- Adaptive ROI suggestion using edge density.

## Update Policy

After every task completion or status change, this file must be updated by the AI assistant (Requirement R-LEDGER-UPDATE). The assistant also updates `.github/copilot-instructions.md` to include this policy.

## Audit / Recent Activity

- Last verification: 2026-01-26 — Updated heatmap gradient smoothing (T040) and visualization interpolation; verified no syntax errors and improved perceived gradient in GUI heatmap. Prior entries preserved. 

