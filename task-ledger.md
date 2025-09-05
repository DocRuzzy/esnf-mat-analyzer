# ESNF Mat Analyzer - Task Ledger

Auto-maintained ledger of active, blocked, and completed tasks. Updated by AI assistant after each task-related change.

## Legend

- STATUS: `TODO` | `IN-PROGRESS` | `BLOCKED` | `DONE` | `DEFERRED`
- Prereq: Task IDs that must reach DONE before this can start.
- Block Reason: Short cause if BLOCKED.

## Active Tasks

| ID | Title | Status | Prereq | Block Reason | Notes / Next Action |
|----|-------|--------|--------|--------------|---------------------|
| T001 | Expose all mat-scale metrics in GUI results tab | DONE | - | - | Implemented display with red tagging for errors/inappropriate |
| T002 | Shape-aware appropriateness flag refinement | TODO | T001 | - | Replace placeholder ROI shape check with geometric analysis (eccentricity, solidity). |
| T003 | Add background method selection (full enum) to GUI | TODO | - | - | Add radio buttons + mapping; persist selection in config. |
| T004 | Implement polygon / freeform ROI selection | TODO | T003 | - | Canvas tool for polygon points, mask conversion. |
| T005 | Introduce automated metric validation (range / NaN checks) | TODO | T001 | - | Inject validation step before display; mark failures. |
| T006 | Add unit tests for new GUI metrics display logic | TODO | T001 | - | Snapshot test of text output, error highlighting behavior. |
| T007 | Improve multiscale uniformity computation robustness | TODO | - | - | Verify wavelet path vs fallback; add percentile-based normalization. |
| T008 | Document all uniformity metrics in docs/guides/uniformity.md | TODO | T002,T005 | - | Create detailed formulas + interpretation thresholds. |
| T009 | Add CLI flag to export full mat-scale metrics JSON | TODO | - | - | Extend cli/main.py output options. |
| T010 | Implement task ledger auto-update hook | IN-PROGRESS | - | - | After each change, assistant patches this file. |
| T011 | Replace manual scale bar (circle) with H-shaped scale widget | TODO | T001 | - | Update `main_window.py` scale drawing to H-shape; ensure draggable handles and physical-length input. |
| T012 | Show GUI popup when analyzing without ROI selected | TODO | T001 | - | Replace terminal warning with `messagebox.showwarning` popup in `analyze()` and add unit test. |
| T013 | Define composite background quality metric | DONE | T001 | - | Implemented in integration test (background_std, signal_mean, dynamic_range, saturation). |
| T014 | Synthetic image generator for background tests | DONE | T013 | - | Added to integration test to produce controlled gradient + features. |
| T015 | Integration test ranking background methods | DONE | T013,T014 | - | New test ensures methods produce varying quality; asserts baseline not best. |
| T016 | Automatic method recommendation feature | TODO | T015 | - | Use composite metric to auto-pick method in GUI/CLI. |
| T017 | Expose per-method tunable parameters via config/UI | TODO | T013 | - | Add sliders/fields (kernel sizes, percentile) and config binding. |
| T018 | CLI benchmarking command for background methods | TODO | T015 | - | Add `esnf-analyzer bg-benchmark <img_dir>` producing CSV of metrics. |

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
