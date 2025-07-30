import shutil
import subprocess
import sys
import platform
import os
import re

def get_cuda_version():
    # Try nvcc first
    try:
        output = subprocess.check_output(['nvcc', '--version'], encoding='utf-8')
        match = re.search(r'release (\d+)\.(\d+)', output)
        if match:
            major, minor = match.groups()
            return float(f"{major}.{minor}")
    except Exception:
        pass
    # Fallback to torch
    try:
        import torch
        cuda_version = torch.version.cuda
        if cuda_version is not None:
            return float(cuda_version[:2])
    except Exception:
        pass
    return None

def install_legacy():
    print("Installing legacy SageAttention and triton (CUDA < 12)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sageattention", "triton"])

def install_sageattention2():
    print("Installing SageAttention 2 (CUDA >= 12)...")
    repo_url = "https://github.com/thu-ml/SageAttention"
    clone_dir = "SageAttention"
    start_dir = os.getcwd()
    if not os.path.exists(clone_dir):
        subprocess.check_call(["git", "clone", repo_url, clone_dir])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "triton"])
    os.chdir(clone_dir)
    os.environ["MAX_JOBS"] = "4"
    # Checkout a specific commit to ensure compatibility
    subprocess.check_call(["git", "checkout", "fa1d103"]) # v2.0.1
    subprocess.check_call([sys.executable, "-m", "pip", "install", "."])
    os.chdir(start_dir)
    try:
        shutil.rmtree(clone_dir)
    except Exception as e:
        print(f"Warning: Could not remove {clone_dir}: {e}")

def main():
    if platform.system() != "Linux":
        print("SageAttention is only supported on Linux. Skipping install.")
        return

    cuda_version = get_cuda_version()
    print(f"Detected CUDA version: {cuda_version}")
    if cuda_version is None:
        print("Could not detect CUDA version. Skipping SageAttention install.")
        return

    if cuda_version < 12:
        install_legacy()
    else:
        install_sageattention2()

if __name__ == "__main__":
    main()