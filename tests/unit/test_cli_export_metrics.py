import json
from pathlib import Path
import tempfile
import os

from esnf_mat_analyzer.cli import main as cli_main


def test_export_metrics_writes_file(monkeypatch, tmp_path):
    # Create a fake image file
    img = tmp_path / "img.tif"
    img.write_bytes(b"FAKE")

    # Prepare args: input file and export path
    export_file = tmp_path / "metrics.json"

    monkeypatch.setattr('sys.argv', ['esnf-analyzer', '-i', str(img), '--export-metrics', str(export_file)])

    # Monkeypatch NanoFiberAnalyzer.from_config to a stub that returns an object
    class StubResult:
        def __init__(self):
            self.metrics = {"overall_mat_uniformity": 0.123}
        def validate_completeness(self):
            return []
        def get_metric(self, key):
            return self.metrics.get(key)

    class StubAnalyzer:
        @classmethod
        def from_config(cls, config):
            return cls()
        def validate_image(self, path):
            return True
        def process_image(self, path):
            return StubResult()

    monkeypatch.setattr('esnf_mat_analyzer.cli.main.NanoFiberAnalyzer', StubAnalyzer)

    # Run CLI main
    rc = cli_main.main_cli()
    assert rc == 0

    # Check that file was created
    assert export_file.exists()
    data = json.loads(export_file.read_text(encoding='utf-8'))
    assert data['metrics']['overall_mat_uniformity'] == 0.123
