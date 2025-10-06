import os
from pathlib import Path
import tempfile
import tkinter as tk
import pytest

# Skip if tkinter/Tcl isn't available
try:
    _tk_root = tk.Tk()
    _tk_root.withdraw()
    _tk_root.destroy()
except Exception:
    pytest.skip("Tkinter/Tcl not available in this environment; skipping GUI tests", allow_module_level=True)

from esnf_mat_analyzer.gui.main_window import MainWindow


def test_method_tunables_persist(tmp_path):
    # Create a temp config file path
    cfg_path = tmp_path / "user-config.yaml"

    # Launch MainWindow with the user_config_path
    win = MainWindow(user_config_path=str(cfg_path))

    # Set custom values
    win.blur_kernel_var.set(11)
    win.basic_ncomp_var.set(4)
    win.rolling_radius_var.set(77)
    win.restore_percentile_var.set(12.5)
    win.homomorphic_cutoff_var.set(99.0)
    win.homomorphic_g_low_var.set(0.33)
    win.homomorphic_g_high_var.set(1.99)

    # Force save
    win._save_user_params()

    # Ensure file was created
    assert cfg_path.exists()

    # Create a new window instance that should load persisted values
    win2 = MainWindow(user_config_path=str(cfg_path))

    try:
        assert win2.blur_kernel_var.get() == 11
        assert win2.basic_ncomp_var.get() == 4
        assert win2.rolling_radius_var.get() == 77
        assert abs(win2.restore_percentile_var.get() - 12.5) < 1e-6
        assert abs(win2.homomorphic_cutoff_var.get() - 99.0) < 1e-6
        assert abs(win2.homomorphic_g_low_var.get() - 0.33) < 1e-6
        assert abs(win2.homomorphic_g_high_var.get() - 1.99) < 1e-6
    finally:
        try:
            win.destroy()
        except Exception:
            pass
        try:
            win2.destroy()
        except Exception:
            pass
