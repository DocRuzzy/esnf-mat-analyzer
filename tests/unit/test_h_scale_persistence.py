import yaml
import tempfile
from pathlib import Path
from esnf_mat_analyzer.gui.main_window import MainWindow


def test_persist_spatial_scale(tmp_path):
    mw = MainWindow.__new__(MainWindow)
    mw.spatial_scale = 42.5
    out = tmp_path / 'default-config.yaml'
    data = {'processing': {'spatial_scale_pixels_per_mm': float(mw.spatial_scale)}}
    with open(out, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    # Read back and verify
    loaded = yaml.safe_load(out.read_text())
    assert 'processing' in loaded
    assert loaded['processing']['spatial_scale_pixels_per_mm'] == 42.5
