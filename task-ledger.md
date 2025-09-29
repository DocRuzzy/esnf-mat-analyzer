# ESNF Mat Analyzer - Task Ledger

Auto-maintained ledger of active, blocked, and completed tasks. Updated by AI assistant after each task-related change.

## Legend

- STATUS: `TODO` | `IN-PROGRESS` | `BLOCKED` | `DONE` | `DEFERRED`
- Prereq: Task IDs that must reach DONE before this can start.
- Block Reason: Short cause if BLOCKED.

## Active Tasks

| ID | Title | Status | Prereq | Block Reason | Notes / Next Action |
|----|-------|--------|--------|--------------|---------------------|
| T001 | Expose all mat-scale metrics in GUI results tab | DONE | - | - | Implemented display with red tagging for errors/inappropriate (2025-09-05) |
| T002 | Shape-aware appropriateness flag refinement | DONE | T001 | - | Implemented geometric ROI checks (area, solidity, eccentricity) and UI warning/visual flag in `esnf_mat_analyzer/gui/main_window.py`. (2025-09-29) |
| T003 | Add background method selection (full enum) to GUI | TODO | - | - | Add radio buttons + mapping; persist selection in config. |
| T004 | Implement polygon / freeform ROI selection | TODO | T003 | - | Canvas tool for polygon points, mask conversion. |
| T005 | Introduce automated metric validation (range / NaN checks) | TODO | T001 | - | Inject validation step before display; mark failures. |
| T006 | Add unit tests for new GUI metrics display logic | TODO | T001 | - | Snapshot test of text output, error highlighting behavior. |
| T007 | Improve multiscale uniformity computation robustness | TODO | - | - | Verify wavelet path vs fallback; add percentile-based normalization. |
| T008 | Document all uniformity metrics in docs/guides/uniformity.md | TODO | T002,T005 | - | Create detailed formulas + interpretation thresholds. |
| T009 | Add CLI flag to export full mat-scale metrics JSON | TODO | - | - | Extend cli/main.py output options. |
| T010 | Implement task ledger auto-update hook | DONE | - | - | After each change, assistant patches this file. (2025-09-29) |
| T023 | GUI: Fit main window for 1080p and add left-panel + canvas scrollbars | DONE | - | - | Implemented geometry and a scrollable left panel in `esnf_mat_analyzer/gui/main_window.py`. (2025-09-29) |
| T024 | Fix crash: OpenCV error "!_src.empty() in cvtColor" when previewing or analysing | IN-PROGRESS | - | Need failing image(s) and reproduction steps | Added defensive checks and plan to convert PIL images to RGB before OpenCV conversions; awaiting repro for verification. |
| T025 | Unit tests for GUI image/ROI helpers | DONE | T002,T024 | - | Added `tests/unit/test_gui_helpers.py` with tests for `_to_numpy_rgb`, `_to_numpy_gray`, and `_roi_shape_metrics`. (2025-09-29) |
| T011 | Replace manual scale bar (circle) with H-shaped scale widget | TODO | T001 | - | Update `main_window.py` scale drawing to H-shape; ensure draggable handles and physical-length input. |
| T012 | Show GUI popup when analyzing without ROI selected | TODO | T001 | - | Replace terminal warning with `messagebox.showwarning` popup in `analyze()` and add unit test. |
| T013 | Define composite background quality metric | DONE | T001 | - | Implemented in integration test (background_std, signal_mean, dynamic_range, saturation). |
| T014 | Synthetic image generator for background tests | DONE | T013 | - | Added to integration test to produce controlled gradient + features. |
| T015 | Integration test ranking background methods | DONE | T013,T014 | - | New test ensures methods produce varying quality; asserts baseline not best. |
| T016 | Automatic method recommendation feature | TODO | T015 | - | Use composite metric to auto-pick method in GUI/CLI. |
| T017 | Expose per-method tunable parameters via config/UI | TODO | T013 | - | Add sliders/fields (kernel sizes, percentile) and config binding. |
| T018 | CLI benchmarking command for background methods | TODO | T015 | - | Add `esnf-analyzer bg-benchmark <img_dir>` producing CSV of metrics. |
| T019 | Add optional `pywt` dependency and graceful fallback for multiscale analysis | DONE | T007 | - | Make `pywt` optional: use it when available; otherwise use memory-safe fallback. Add install docs and tests. (2025-09-05) |
| T020 | Fix multiscale fallback memory usage (prevent giant allocations) | DONE | T019,T007 | - | Update `_fallback_multiscale_analysis` to downsample, process in chunks, or use percentiles to avoid full-array temporaries. Add unit test for large images. (2025-09-05) |
| T021 | Improve ruler detection robustness and graceful fallback | TODO | T001 | - | Fix ruler detection failures; when scale undetected provide UI prompt and allow manual scale entry. Add test for missing ruler. |
| T022 | Add CI integration test for full pipeline on representative sample image | TODO | T015 | - | Run full pipeline in CI with smaller sample, assert no dependency errors and memory bounds; catch regressions. |

## Completed Tasks

- T001 Expose all mat-scale metrics in GUI results tab (2025-09-05)

## Blocked Tasks

(None currently)

## Deferred Tasks

(None currently)

## Backlog / Ideas

- Batch processing UI for multiple images with aggregated uniformity stats.
- GPU acceleration for anisotropy FFT stage.
- Adaptive ROI suggestion using edge density.

## Update Policy

After every task completion or status change, this file must be updated by the AI assistant (Requirement R-LEDGER-UPDATE). The assistant also updates `.github/copilot-instructions.md` to include this policy.

## Audit / Recent Activity

- Last verification: 2025-09-29 — `task-ledger.md` contents checked and changes confirmed (T010, T023 set to DONE; T024 added as IN-PROGRESS). Assistant internal todo marked accordingly.
