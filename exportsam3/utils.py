import platform
from pathlib import Path
from importlib.util import find_spec

def get_nvidia_lib_paths() -> tuple[str, str]:
    """
    Returns paths to NVIDIA cuda runtime libs and cudnn libs.
    """
    cuda_path = _get_nvidia_lib_path("nvidia.cuda_runtime")
    cudnn_path = _get_nvidia_lib_path("nvidia.cudnn")
    return cuda_path, cudnn_path

def _get_nvidia_lib_path(package_name):
    spec = find_spec(package_name)
    if spec and spec.submodule_search_locations:
        base_dir = spec.submodule_search_locations[0]
        # Linux uses 'lib', Windows uses 'bin'
        sub_folder = "bin" if platform.system() == "Windows" else "lib"
        package_path = Path(base_dir) / sub_folder
        if package_path.exists():
            return str(package_path)
    return None