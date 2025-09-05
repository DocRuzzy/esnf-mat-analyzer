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
