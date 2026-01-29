# CURRENT - Active Session Tasks

**Last Updated:** 2026-01-29 (evening)

This file tracks active work for the current development focus. Updated at session start/end.

## Active Tasks

| Priority | Task | Notes |
|----------|------|-------|
| 1 | T053: Upgrade to four-region detection | **NEW** - Add ruler as 4th region (background, gel, mat, ruler). Current 3-region misidentifies gel as background reference |
| 2 | T054: Fix background reference selection | **NEW** - Background reference must come from collection surface OUTSIDE gel ring, not from gel (gel has darkest blacks + wet reflections) |
| 3 | T055: Background-only gradient correction | **NEW** - Fit illumination gradient ONLY to true background, extrapolate into mat, anchor to darkest background level |
| 4 | T050: Integrate region detection into main pipeline | Use updated region_detector in analyzer |
| 5 | T051: Add physical calibration UI with marked point | Allow user to click point and enter known µm thickness |

## Recently Completed (This Session)

- ✅ Created auto-tuning script for detail preservation (`scripts/auto_tune_detail_preservation.py`)
- ✅ Identified issue: background correction was removing real mat thickness features
- ✅ Identified issue: darkest point was being selected from gel region, not true background
- ✅ Fixed ruler_detector.py dict access bug (`self.config.scoring['key']` not `.key`)
- ✅ Hybrid task tracking system (CURRENT.md + task-ledger.md)
- ✅ Synthetic data generator module
- ✅ Frequency diagnostics module
- ✅ FFT enhancement now configurable and disabled by default
- ✅ Three-region detector (needs upgrade to four-region)
- ✅ Intensity calibrator
- ✅ Gel boundary analyzer

## Known Issues / Blockers

1. **Gel vs Background confusion**: Current region detection picks gel pixels as "darkest background" because gel (wet polymer) has darkest blacks. Need to detect gel boundary FIRST, then find true background outside it.
2. **Ruler not excluded**: Ruler region (bottom of image, white ticks) pollutes background statistics. Need explicit ruler detection/exclusion.

## Session Notes

**Key insight from user**: The four regions are:
1. **Background** - Collection surface around the mat (TRUE black reference)
2. **Gel** - Dark wet ring around mat (DARKER than background due to wet polymer, has reflections)
3. **Mat** - Bright nanofiber deposition (thickness signal)
4. **Ruler** - Scale bar at bottom (exclude from all analysis)

The gel is darker than background because wet polymer absorbs light. But we must NOT use gel as black reference - use the collection surface outside the gel ring.

---
*For full task history, see [task-ledger.md](task-ledger.md)*
