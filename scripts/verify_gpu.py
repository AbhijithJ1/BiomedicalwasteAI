"""
CUDA & GPU Verification Script for PyTorch / Ultralytics
"""

import sys
import torch

def verify_environment():
    print("=" * 60)
    print("PyTorch Environment & Hardware Verification")
    print("=" * 60)
    print(f"Python Version       : {sys.version.split()[0]}")
    print(f"PyTorch Version      : {torch.__version__}")
    print(f"CUDA Available       : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Device Count         : {gpu_count}")
        for i in range(gpu_count):
            device_name = torch.cuda.get_device_name(i)
            vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"  [GPU {i}] {device_name} ({vram_gb:.2f} GB VRAM)")
        print("\nSUCCESS: GPU is ready for YOLO training.")
    else:
        print("\nWARNING: No CUDA-capable GPU detected by PyTorch.")
        print("PyTorch will run on CPU. If you have an NVIDIA GPU, make sure:")
        print("  1. NVIDIA GPU drivers are up-to-date.")
        print("  2. PyTorch was installed with CUDA support, e.g.:")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("=" * 60)

if __name__ == "__main__":
    verify_environment()
