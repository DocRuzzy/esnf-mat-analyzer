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
from esnf_mat_analyzer.config.config_manager import get_default_config


def test_gui_propagates_method_tunables(monkeypatch):
    # Create the main window but do not start the mainloop
    root = MainWindow()

    # Set some tunable values in the GUI
    root.blur_kernel_var.set(9)
    root.basic_ncomp_var.set(3)
    root.rolling_radius_var.set(120)
    root.restore_percentile_var.set(7.5)
    root.homomorphic_cutoff_var.set(45)
    root.homomorphic_g_low_var.set(0.7)
    root.homomorphic_g_high_var.set(1.8)

    captured = {}

    # Monkeypatch setup_dependencies to capture the config passed in
    def fake_setup_dependencies(cfg):
        captured['config'] = cfg
        class FakeAnalyzer:
            def process_image(self, *args, **kwargs):
                return None
        return FakeAnalyzer()

    monkeypatch.setattr('esnf_mat_analyzer.gui.main_window.setup_dependencies', fake_setup_dependencies)

    # Prepare a fake image and ROI (bypass file dialogs)
    # Create a tiny blank image
    from PIL import Image
    img = Image.new('RGB', (100, 100), color='white')
    root.current_image = img
    root.selected_files = [__file__]  # dummy path
    # Create an ROI rectangle on the canvas by simulating coords
    root.rect = root.canvas.create_rectangle(0, 0, 10, 10)
    # Ensure a listbox selection exists
    root.file_listbox.insert(0, 'dummy')
    root.file_listbox.selection_set(0)

    # Call analyze which should call our fake_setup_dependencies
    try:
        root.analyze()
    finally:
        # Destroy the Tk root to clean up
        try:
            root.destroy()
        except Exception:
            pass

    assert 'config' in captured
    cfg = captured['config']
    # Check that the values propagated
    assert cfg.processing.blur_kernel_size == 9
    assert cfg.processing.basic_correction_n_components == 3
    assert cfg.processing.rolling_ball_radius == 120
    assert abs(cfg.processing.restore_percentile - 7.5) < 1e-6
    assert abs(cfg.processing.homomorphic_cutoff - 45.0) < 1e-6
    assert abs(cfg.processing.homomorphic_g_low - 0.7) < 1e-6
    assert abs(cfg.processing.homomorphic_g_high - 1.8) < 1e-6
