from unittest.mock import patch
from types import SimpleNamespace

import pytest

from esnf_mat_analyzer.gui.main_window import MainWindow


def test_analyze_shows_warning_when_no_roi_selected():
    mw = MainWindow.__new__(MainWindow)
    # Simulate an image loaded
    mw.current_image = SimpleNamespace(size=(100, 100))
    # Ensure canvas and other attributes exist to avoid attribute errors in analyze()
    mw.canvas = SimpleNamespace()
    mw.canvas.coords = lambda *a, **k: (0, 0, 10, 10)
    mw.scale_factor = 1.0
    mw.file_listbox = SimpleNamespace(curselection=lambda: (0,))
    mw.selected_files = ['dummy.png']
    mw.background_leveling_var = SimpleNamespace(get=lambda: True)

    # Patch get_roi_coordinates to return None (no ROI)
    with patch.object(mw, 'get_roi_coordinates', return_value=None):
        with patch('tkinter.messagebox.showwarning') as mock_warn:
            mw.analyze()
            mock_warn.assert_called_once()
