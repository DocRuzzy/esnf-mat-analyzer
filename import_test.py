import sys
print("executable:", sys.executable)
print("cwd on sys.path[0]:", sys.path[0])
try:
    import esnf_mat_analyzer
    print("esnf_mat_analyzer file:", getattr(esnf_mat_analyzer, "__file__", repr(esnf_mat_analyzer)))
except Exception:
    import traceback
    traceback.print_exc()
