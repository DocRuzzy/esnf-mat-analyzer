import tkinter as tk
import numpy as np
from pathlib import Path
import pytest

# Skip tests if tkinter/Tcl isn't available in this environment (headless)
try:
    _tk_root = tk.Tk()
    _tk_root.withdraw()
    _tk_root.destroy()
except Exception:
    pytest.skip("Tkinter/Tcl not available in this environment; skipping GUI tests", allow_module_level=True)

from esnf_mat_analyzer.gui.main_window import MainWindow
from esnf_mat_analyzer.core.data_types import AnalysisResult


def make_fake_result(tmp_path: Path):
    # Create minimal fake arrays
    thickness_map = np.zeros((10, 10), dtype=float)
    mask = np.ones((10, 10), dtype=bool)
    metrics = {
        'overall_mat_uniformity': 0.75,
        'anisotropy_index': 0.12,
        'scale_0_uniformity': 0.9,
        'scale_1_uniformity': 0.8
    }
    res = AnalysisResult(
        image_path=tmp_path / 'fake.png',
        contour=np.array([[0,0],[1,0],[1,1]]),
        thickness_map=thickness_map,
        mask=mask,
        metrics=metrics,
        processing_time=1.23
    )
    return res


def test_display_results_basic(tmp_path):
    win = MainWindow()
    res = make_fake_result(tmp_path)

    # Call display_results (creates a Toplevel)
    win.display_results(res)

    # Find the top-level windows and assert basic content
    tops = [w for w in win.winfo_children() if isinstance(w, tk.Toplevel)]
    assert tops, "Expected a Toplevel results window"

    # Seek a Text widget inside the first Toplevel
    text_widgets = [c for c in tops[0].winfo_children() if isinstance(c, tk.Text)]
    # There are multiple frames; use a recursive search
    def find_texts(widget):
        texts = []
        for c in widget.winfo_children():
            if isinstance(c, tk.Text):
                texts.append(c)
            else:
                texts.extend(find_texts(c))
        return texts

    texts = find_texts(tops[0])
    assert texts, "Expected Text widgets in results window"

    # Read content from the Key Metrics summary tab (first text widget)
    content = texts[0].get('1.0', tk.END)
    assert 'KEY UNIFORMITY METRICS' in content or 'Composite Uniformity Score' in content

    # Also check that detailed metrics are present in some tab
    all_content = " ".join(t.get('1.0', tk.END) for t in texts)
    assert 'Overall Mat Uniformity' in all_content or 'Multi-Scale Uniformity Analysis' in all_content

    # Clean up
    try:
        win.destroy()
    except Exception:
        pass


def test_display_results_multiscale_formatting(tmp_path):
    win = MainWindow()
    res = make_fake_result(tmp_path)
    # Add more scale metrics to test formatting
    res.metrics.update({f'scale_{i}_uniformity': 0.5 + i*0.05 for i in range(5)})

    win.display_results(res)
    tops = [w for w in win.winfo_children() if isinstance(w, tk.Toplevel)]
    assert tops

    def find_texts(widget):
        texts = []
        for c in widget.winfo_children():
            if isinstance(c, tk.Text):
                texts.append(c)
            else:
                texts.extend(find_texts(c))
        return texts

    texts = find_texts(tops[0])
    assert texts
    # Search for the multiscale header in any text widget
    found = any('Multi-Scale Uniformity Analysis' in t.get('1.0', tk.END) for t in texts)
    assert found, 'Multiscale analysis header not found in display'

    try:
        win.destroy()
    except Exception:
        pass
