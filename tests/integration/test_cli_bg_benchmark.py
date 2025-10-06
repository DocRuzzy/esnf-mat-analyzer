import json
from pathlib import Path
import tempfile
import os

import pytest

from esnf_mat_analyzer.cli import main as cli_main


class StubResult:
    def __init__(self):
        self.metrics = {"overall_mat_uniformity": 0.5}
        self.processing_time = 0.1

class StubAnalyzer:
    @classmethod
    def from_config(cls, cfg):
        return cls()
    def validate_image(self, p):
        return True
    def process_image(self, p):
        return StubResult()


def test_bg_benchmark_writes_csv(monkeypatch, tmp_path):
    # Create a small synthetic image file
    img = tmp_path / "img.tif"
    img.write_bytes(b"FAKE")

    # Create input dir with the image
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    (input_dir / img.name).write_bytes(b"FAKE")

    # Monkeypatch NanoFiberAnalyzer.from_config to return stub
    monkeypatch.setattr('esnf_mat_analyzer.cli.main.NanoFiberAnalyzer', StubAnalyzer)

    # Run benchmark
    out_dir = tmp_path / "out"
    csv_path = cli_main.run_bg_benchmark(StubAnalyzer(), input_dir, out_dir, pattern='*.tif', verbose=False)

    assert csv_path.exists()
    text = csv_path.read_text(encoding='utf-8')
    assert 'overall_mat_uniformity' in text
