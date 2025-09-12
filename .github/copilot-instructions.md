# ESNF Mat Analyzer - AI Coding Agent Instructions

## Project Overview
Scientific Python package for electrospun nanofiber mat analysis with a focus on **background correction methods** and **uniformity metrics**. The project follows JOSS (Journal of Open Source Software) standards with a modular architecture built around dependency injection and interface patterns.

## Architecture & Key Components

### Core Pipeline (Process Order)
1. **Image Loading** → `esnf_mat_analyzer/processing/image_processor.py`
2. **Ruler Detection** → `esnf_mat_analyzer/processing/ruler_detector.py` (spatial calibration)
3. **Background Correction** → `esnf_mat_analyzer/processing/background_correction/` (4 methods: BASIC, rolling ball, RESTORE, homomorphic)
4. **ROI Detection** → `esnf_mat_analyzer/processing/shape_detector.py` (gel boundary detection)
5. **Thickness Estimation** → `esnf_mat_analyzer/processing/thickness_estimator.py`
6. **Uniformity Analysis** → `esnf_mat_analyzer/analysis/uniformity_metrics.py`
7. **Results Export** → `esnf_mat_analyzer/utils/data_exporter.py`

### Background Correction Architecture (Active Development Focus)
```
esnf_mat_analyzer/processing/background_correction/
├── __init__.py           # Unified interface for all approaches
├── advanced.py           # Primary 4-step scientific workflow (used by GUI)
├── mat_level.py          # Continuous mat regions (thick samples)
├── enhanced_scipy.py     # Research-grade with scipy optimization
└── enhanced_opencv.py    # Minimal dependencies (OpenCV only)
```

**Critical Pattern**: Each background correction class provides method selection via enum:
- `BackgroundCorrectionMethod.BASIC` → NMF-based machine learning approach
- `BackgroundCorrectionMethod.ROLLING_BALL` → 3D morphological processing
- `BackgroundCorrectionMethod.RESTORE` → Iterative deconvolution
- `BackgroundCorrectionMethod.HOMOMORPHIC` → Frequency domain filtering

## Entry Points & Interface Patterns

### Primary Entry Points
- **GUI**: `python run_esnf_analyzer.py` → `esnf_mat_analyzer/gui/main_window.py`
- **CLI**: `esnf-analyzer` command → `esnf_mat_analyzer/cli/main.py`
- **API**: `NanoFiberAnalyzer.from_config(config)` → `esnf_mat_analyzer/core/analyzer.py`

### Dependency Injection Pattern
```python
# Core creation pattern - used throughout
from esnf_mat_analyzer.main import setup_dependencies
from esnf_mat_analyzer.config.config_manager import get_default_config

config = get_default_config()
analyzer = setup_dependencies(config)  # Returns fully configured NanoFiberAnalyzer
```

### Configuration System
- **Type-safe dataclasses**: `esnf_mat_analyzer/core/data_types.py`
- **YAML/JSON loading**: `esnf_mat_analyzer/config/config_manager.py`
- **Enum-based method selection**: Background correction, thickness models, etc.

## Development Workflows

### Testing Strategy
```bash
# Run organized test suite
pytest tests/unit/              # Fast unit tests
pytest tests/integration/       # Slower integration tests  
pytest tests/gui/              # GUI-specific tests
pytest -m "not slow"           # Skip computationally expensive tests
```

### Background Correction Development Workflow
```python
# Testing all background correction methods
python test_all_background_methods.py  # CLI comparison tool
python test_background_correction.py   # Validation script

# GUI integration testing
python test_gui_background_integration.py
```

### Adding New Background Correction Methods
1. **Add enum value**: `BackgroundCorrectionMethod` in `data_types.py`
2. **Implement method**: In appropriate `background_correction/*.py` file
3. **Update GUI mapping**: Radio buttons in `gui/main_window.py` lines 120-130
4. **Add configuration**: Update `config_manager.py` enum mappings
5. **Add tests**: Create test in `tests/integration/test_*_background_correction.py`

## Project-Specific Conventions

### File Organization Standards (Recently Reorganized)
- **Tests**: All in `tests/` with subdirectories (unit/, integration/, gui/, data/)
- **Scripts**: Debug/benchmark tools in `scripts/debug/`, `scripts/benchmarks/`
- **Examples**: Demo files in `examples/`
- **Documentation**: Guides in `docs/guides/`, images in `docs/images/`

### Import Path Patterns
```python
# Background correction imports (new structure)
from esnf_mat_analyzer.processing.background_correction.advanced import AdvancedBackgroundProcessor

# Core components
from esnf_mat_analyzer.core.analyzer import NanoFiberAnalyzer
from esnf_mat_analyzer.core.data_types import Config, AnalysisResult, BackgroundCorrectionMethod

# CLI vs GUI patterns
from esnf_mat_analyzer.cli.main import main_cli        # Professional CLI with colors/progress
from esnf_mat_analyzer.gui.main_window import MainWindow  # Tkinter GUI
```

### Scientific Analysis Patterns
- **ROI Handling**: User ROI (rectangular) → Gel detection (optimal boundary) → Analysis mask
- **Scale Calibration**: Ruler detection provides `spatial_scale_pixels_per_mm`
- **Error Handling**: Graceful degradation with fallback methods for each step
- **Validation**: Quality metrics for each processing step (saturation rates, SNR, etc.)

## Testing & Validation Patterns

### Background Correction Testing Pattern
```python
# Standard validation metrics used throughout tests
def analyze_correction_quality(original, corrected, mask=None):
    metrics = {}
    metrics['saturation_pct'] = 100 * np.sum(corrected >= 254) / corrected.size
    metrics['dynamic_range'] = np.percentile(corrected, 98) - np.percentile(corrected, 2)
    metrics['signal_preservation'] = np.mean(corrected[mask]) if mask else 0
    return metrics
```

### GUI Testing Pattern
- **Integration tests**: Load actual images, test UI workflows
- **Method comparison**: Side-by-side visualization of all background methods
- **Configuration validation**: Ensure GUI selections update analysis config properly

## Key External Dependencies
- **OpenCV**: Image processing, background correction algorithms
- **scikit-image**: Advanced image analysis, morphological operations  
- **NumPy/SciPy**: Scientific computing, signal processing
- **Matplotlib**: Visualization, heatmap generation
- **tkinter**: GUI framework (included with Python)
- **Optional**: `colorama` (CLI colors), `tqdm` (progress bars)

## Common Debugging Patterns
- **Image processing issues**: Check `test_all_background_methods.py` for method comparison
- **Configuration problems**: Use `validate_config_file()` from config_manager
- **GUI integration**: Run `test_gui_background_integration.py` to verify UI workflow
- **Import errors**: Check recent reorganization - background correction moved to submodule

When working with this codebase, prioritize understanding the background correction pipeline as it's the primary development focus and affects all downstream analysis quality.

## Task Ledger Update Requirement (R-LEDGER-UPDATE)
An AI-maintained file `task-ledger.md` at the repository root tracks tasks (ID, status, prerequisites, blocking reasons). After every task completion or status change performed via the AI assistant, the assistant MUST:
1. Update the task's status (e.g., TODO → IN-PROGRESS → DONE).
2. Add completion date when moving to DONE.
3. Add or adjust blocking reasons if prerequisites not met.
4. Append any new tasks created during the change.
5. Maintain formatting and table integrity.
6. Ensure this requirement section remains present.

Statuses: `TODO`, `IN-PROGRESS`, `BLOCKED`, `DONE`, `DEFERRED`.

If the ledger is missing, recreate it with current known tasks before proceeding.

## Branch Naming Convention
When creating feature or fix branches, follow a numeric minor-increment policy derived from the current branch name when that branch uses a numeric suffix. Rules:

1. If the current branch matches the pattern `<name>_<major>` or `dev_<major>` (for example `dev_1`), create the new branch by incrementing the minor version: `dev_<major>.<minor>` where `<minor>` starts at `1` for the first child branch. Example: from `dev_1` → `dev_1.1`.
2. If multiple child branches already exist, pick the next unused minor (e.g., `dev_1.1`, `dev_1.2`, ...).
3. Append a short, descriptive suffix after the version separated by a dash, e.g. `dev_1.1-add-ledger`, `feature-2.3-fix-anisotropy`.
4. If the current branch does not use a numeric suffix, fall back to `feature/<short-desc>` or `fix/<short-desc>`.
5. Update `.github/copilot-instructions.md` when modifying this policy.
