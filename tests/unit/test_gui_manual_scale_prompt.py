import os
from pathlib import Path
import tempfile

import pytest

import tkinter as tk

from esnf_mat_analyzer.gui.main_window import MainWindow


class FakeResult:
    def __init__(self, metrics=None, spatial_scale=None):
        self.metrics = metrics or {"overall_mat_uniformity": 0.5}
        self.spatial_scale_pixels_per_mm = spatial_scale
        self.processing_time = 0.0
        self.mask = None
        self.thickness_map = None
        self.results_dir = Path('.')


class FakeAnalyzer:
    def __init__(self, first_spatial_scale=None, rerun_spatial_scale=None):
        self.first_spatial_scale = first_spatial_scale
        self.rerun_spatial_scale = rerun_spatial_scale
        self.called = 0

    @classmethod
    def from_config(cls, config):
        return cls()

    def validate_image(self, path):
        return True

    def process_image(self, path, roi=None, spatial_scale_pixels_per_mm=None):
        self.called += 1
        # First call: simulate missing ruler (None). Second call: return with spatial scale
        if self.called == 1:
            return FakeResult(spatial_scale=None)
        else:
            return FakeResult(spatial_scale=spatial_scale_pixels_per_mm or 50.0)


def test_manual_scale_prompt(monkeypatch, tmp_path):
    # Create fake image file
    img = tmp_path / "img.tif"
    img.write_bytes(b"FAKE")

    # Create app instance (do not run mainloop)
    app = MainWindow()

    # Monkeypatch file selection and image
    app.selected_files = [str(img)]
    app.file_listbox.insert(0, str(img))
    app.file_listbox.selection_set(0)
    # Create a simple ROI rectangle
    app.current_image = type('I', (), {'size': (100, 100)})()
    app.scale_factor = 1.0
    app.rect = app.canvas.create_rectangle(0, 0, 50, 50)

    # Monkeypatch analyzer setup to return FakeAnalyzer
    monkeypatch.setattr('esnf_mat_analyzer.gui.main_window.setup_dependencies', lambda cfg: FakeAnalyzer())

    # Monkeypatch simpledialog.askfloat to return manual scale (e.g., 40)
    monkeypatch.setattr('esnf_mat_analyzer.gui.main_window.simpledialog.askfloat', lambda *args, **kwargs: 40.0)

    # Run analyze (should call process_image twice: first fail ruler, then rerun with manual)
    app.analyze()

    # Clean up
    app.destroy()
