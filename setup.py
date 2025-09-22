#!/usr/bin/env python3

# flake8: noqa: E501

import os
import platform
import shutil
import subprocess
import sys

from setuptools import find_packages, setup


def run_command(cmd):
    """Run a command and return True if successful, False otherwise"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


def detect_platform():
    """Detect platform using same logic as train.sh"""
    # Check for Darwin (macOS)
    if platform.system() == "Darwin":
        return "apple"

    # Check for NVIDIA GPU
    if run_command("nvidia-smi > /dev/null 2>&1"):
        return "cuda"

    # Check for ROCm
    if run_command("rocm-smi > /dev/null 2>&1"):
        return "rocm"

    # Check for ROCm environment variables (additional detection)
    if any(env in os.environ for env in ["ROCM_PATH", "HIP_PATH", "ROCM_HOME"]):
        return "rocm"

    # Check if ROCm tools exist
    if shutil.which("rocminfo") or shutil.which("rocm-smi"):
        return "rocm"

    # Default to CUDA on Linux, CPU-only elsewhere
    if platform.system() == "Linux":
        return "cuda"
    else:
        return "cpu"


def get_version():
    """Read version from simpletuner/__init__.py"""
    try:
        with open("simpletuner/__init__.py") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    except:
        pass
    return "1.1.0"


def get_platform_dependencies():
    """Get dependencies based on detected platform"""
    detected_platform = detect_platform()

    # Allow override via environment variable
    platform_override = os.environ.get("SIMPLETUNER_PLATFORM", detected_platform)

    print(f"Detected platform: {detected_platform}")
    if platform_override != detected_platform:
        print(f"Platform overridden to: {platform_override}")

    platform_to_use = platform_override

    # Base PyTorch dependencies
    deps = []

    if platform_to_use == "cuda":
        print("Installing CUDA dependencies...")
        deps.extend(
            [
                # PyTorch with CUDA
                "torch>=2.8.0",
                "torchvision>=0.23.0",
                "torchaudio>=2.8.0",
                # CUDA-specific
                "triton>=3.3.0",
                "bitsandbytes>=0.45.0",
                "deepspeed>=0.17.2",
                "torchao>=0.12.0",
                "nvidia-cudnn-cu12",
                "nvidia-nccl-cu12",
                "lm-eval>=0.4.4",
            ]
        )

    elif platform_to_use == "rocm":
        print("Installing ROCm dependencies...")
        deps.extend(
            [
                # PyTorch with ROCm - using direct URL for now
                "torch @ https://download.pytorch.org/whl/rocm6.3/torch-2.7.0%2Brocm6.3-cp312-cp312-manylinux_2_28_x86_64.whl",
                "torchvision>=0.22.0",  # Will need ROCm version
                "torchaudio>=2.4.1",
                # ROCm-specific
                "torchao>=0.11.0",
                # Note: pytorch_triton_rocm might need special handling
            ]
        )

    elif platform_to_use == "apple":
        print("Installing Apple Silicon (MPS) dependencies...")
        deps.extend(
            [
                # PyTorch with MPS support
                "torch>=2.7.1",
                "torchvision>=0.22.1",
                "torchaudio>=2.7.0",
                "torchao>=0.11.0",
                # No deepspeed on Apple
                # No bitsandbytes on Apple (or use CPU version)
            ]
        )

    else:  # cpu fallback
        print("Installing CPU-only dependencies...")
        deps.extend(
            [
                "torch>=2.7.1",
                "torchvision>=0.22.1",
                "torchaudio>=2.4.1",
            ]
        )

    return deps


# Base dependencies (minimal, works on all platforms)
base_deps = [
    "diffusers>=0.35.1",
    "transformers>=4.55.0",
    "datasets>=3.0.1",
    "wandb>=0.21.0",
    "requests>=2.32.4",
    "pillow>=11.3.0",
    "trainingsample>=0.2.1",
    "accelerate>=1.5.2",
    "safetensors>=0.5.3",
    "compel>=2.1.1",
    "clip-interrogator>=0.6.0",
    "open-clip-torch>=2.26.1",
    "iterutils>=0.1.6",
    "scipy>=1.11.1",
    "boto3>=1.35.83",
    "pandas>=2.2.3",
    "botocore>=1.35.83",
    "urllib3<1.27",
    "torchsde>=0.2.6",
    "torchmetrics>=1.1.1",
    "colorama>=0.4.6",
    "numpy>=2.2.0",
    "peft>=0.17.0",
    "tensorboard>=2.18.0",
    "sentencepiece>=0.2.0",
    "optimum-quanto>=0.2.7",
    "lycoris-lora>=3.2.0.post2",
    "torch-optimi>=0.2.1",
    "toml>=0.10.2",
    "fastapi[standard]>=0.115.0",
    "atomicwrites>=1.4.1",
    "beautifulsoup4>=4.12.3",
    "prodigy-plus-schedule-free>=1.9.2",
    "tokenizers>=0.21.0",
    "huggingface-hub>=0.34.3",
    "imageio-ffmpeg>=0.6.0",
    "imageio[pyav]>=2.37.0",
    "hf-xet>=1.1.5",
    "peft-singlora>=0.2.0",
    "trainingsample>=0.2.1",
    # Minimal PyTorch for base install (CPU-only)
    "torch>=2.7.1",
    "torchvision>=0.22.1",
    "torchaudio>=2.4.1",
    "torchao>=0.11.0",
]

# Optional extras
extras_require = {
    "jxl": ["pillow-jxl-plugin>=1.3.1"],
    "dev": [
        "pytest>=7.0.0",
        "pytest-selenium>=4.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-xdist>=3.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
    ],
    # Platform-specific extras for manual override
    "cuda": [
        "triton>=3.3.0",
        "bitsandbytes>=0.45.0",
        "deepspeed>=0.17.2",
        "torchao>=0.12.0",
        "nvidia-cudnn-cu12",
        "nvidia-nccl-cu12",
        "lm-eval>=0.4.4",
    ],
    "rocm": [
        "torchao>=0.11.0",
    ],
    "apple": [
        "torchao>=0.11.0",
    ],
}

# Read long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except:
    long_description = "Stable Diffusion 2.x and XL tuner."

setup(
    name="simpletuner",
    version=get_version(),
    description="Stable Diffusion 2.x and XL tuner.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="bghira",
    # license handled by pyproject.toml
    packages=find_packages(),
    python_requires=">=3.11,<3.13",
    install_requires=base_deps,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "simpletuner=simpletuner.cli:main",
            "simpletuner-train=simpletuner.train:main",
            "simpletuner-configure=simpletuner.configure:main",
            "simpletuner-inference=simpletuner.inference:main",
            "simpletuner-server=simpletuner.service_worker:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
    ],
    keywords="stable-diffusion machine-learning deep-learning pytorch cuda rocm",
    url="https://github.com/bghira/SimpleTuner",
    project_urls={
        "Bug Reports": "https://github.com/bghira/SimpleTuner/issues",
        "Source": "https://github.com/bghira/SimpleTuner",
        "Documentation": "https://github.com/bghira/SimpleTuner/blob/main/README.md",
    },
)

if __name__ == "__main__":
    print("SimpleTuner Setup")
    print("================")
    print(f"Detected platform: {detect_platform()}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")

    # Show what dependencies would be installed
    platform_deps = get_platform_dependencies()
    print(f"\nPlatform-specific dependencies ({len(platform_deps)}):")
    for dep in platform_deps:
        print(f"  - {dep}")
