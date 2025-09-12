import sys
import platform
print('python executable:', sys.executable)
print('python version:', sys.version)
print('platform arch:', platform.architecture())
try:
    import PIL
    print('Pillow version:', PIL.__version__)
    print('Pillow file:', getattr(PIL, '__file__', None))
    try:
        from PIL import _imaging
        print('_imaging import OK')
    except Exception as e:
        print('Error importing _imaging:', e)
except Exception as e:
    print('Pillow import failed:', e)
