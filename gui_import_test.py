import sys
print('executable:', sys.executable)
try:
    import importlib
    m = importlib.import_module('esnf_mat_analyzer.gui.main_window')
    print('loaded:', getattr(m, '__file__', None))
except Exception:
    import traceback
    traceback.print_exc()
