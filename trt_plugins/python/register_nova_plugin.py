import ctypes
import os
import tensorrt as trt

def load_nova_plugin(plugin_path: str):
    if not os.path.exists(plugin_path):
        raise FileNotFoundError(plugin_path)
    lib = ctypes.CDLL(plugin_path)
    try:
        if hasattr(lib, "pluginInitPlugin"):
            lib.pluginInitPlugin()
    except Exception:
        pass
    print("[*] Nova plugin loaded:", plugin_path)
    return lib
