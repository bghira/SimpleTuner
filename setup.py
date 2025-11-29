#!/usr/bin/env python3

# flake8: noqa: E501

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

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

    # Default to CPU to avoid pulling incorrect GPU packages
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
    return "3.0.0"


def _rocm_platform_tag():
    """Return the ROCm wheel platform tag, overridable via environment."""
    return os.environ.get("SIMPLETUNER_ROCM_PLATFORM_TAG", "manylinux_2_28_x86_64")


def build_rocm_wheel_url(package: str, version: str, rocm_version: str) -> str:
    """Build a direct wheel URL for ROCm packages."""
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_tag = _rocm_platform_tag()
    filename = f"{package}-{version}%2Brocm{rocm_version}-{py_tag}-{py_tag}-{platform_tag}.whl"
    base_url = os.environ.get("SIMPLETUNER_ROCM_BASE_URL", f"https://download.pytorch.org/whl/rocm{rocm_version}")
    return f"{package} @ {base_url}/{filename}"


def _resolve_ramtorch_dependency() -> str:
    """
    Prefer a local RamTorch checkout (default: ~/src/ramtorch) when present, otherwise fall back to the package name.
    """

    candidate_path = Path(os.environ.get("SIMPLETUNER_RAMTORCH_PATH", "~/src/ramtorch")).expanduser()
    try:
        if candidate_path.exists():
            return f"ramtorch @ {candidate_path.resolve().as_uri()}"
    except Exception:
        # Any failure falls back to the plain package spec.
        pass
    return "ramtorch"


def get_cuda_dependencies():
    ramtorch_dep = _resolve_ramtorch_dependency()
    return [
        "torch>=2.9.0",
        "torchvision>=0.24.0",
        "torchaudio>=2.4.1",
        "triton>=3.3.0",
        "bitsandbytes>=0.45.0",
        "deepspeed>=0.17.2",
        "torchao>=0.14.1",
        "nvidia-cudnn-cu12",
        "nvidia-nccl-cu12",
        "nvidia-ml-py>=12.555",
        "lm-eval>=0.4.4",
        ramtorch_dep,
    ]


def get_rocm_dependencies():
    ramtorch_dep = _resolve_ramtorch_dependency()
    rocm_version = os.environ.get("SIMPLETUNER_ROCM_VERSION", "6.4")
    torch_version = os.environ.get("SIMPLETUNER_ROCM_TORCH_VERSION", "2.9.0")
    torchvision_version = os.environ.get("SIMPLETUNER_ROCM_TORCHVISION_VERSION", "0.24.0")
    torchaudio_version = os.environ.get("SIMPLETUNER_ROCM_TORCHAUDIO_VERSION", "2.4.1")
    triton_version = os.environ.get("SIMPLETUNER_ROCM_TORCHAUDIO_VERSION", "3.5.0")

    try:
        return [
            build_rocm_wheel_url("torch", torch_version, rocm_version),
            build_rocm_wheel_url("torchvision", torchvision_version, rocm_version),
            "pytorch_triton_rocm @ https://download.pytorch.org/whl/pytorch_triton_rocm-3.5.0-cp312-cp312-linux_x86_64.whl#sha256=b3a209621d0433367c489e8dce90ebc4c7c9e3bfe1c2b7adc928344f8290d5f5",
            "torchao>=0.14.1",
            ramtorch_dep,
        ]
    except Exception as exc:
        print(f"Warning: falling back to CPU PyTorch packages because ROCm wheel configuration failed: {exc}")
        return [
            "torch>=2.9.0",
            "torchvision>=0.24.0",
            "torchao>=0.14.1",
            ramtorch_dep,
        ]


def get_apple_dependencies():
    return [
        "torch>=2.9.0",
        "torchvision>=0.24.0",
        "torchao>=0.14.1",
    ]


def get_cpu_dependencies():
    return [
        "torch>=2.9.0",
        "torchvision>=0.24.0",
        "torchao>=0.14.1",
    ]


PLATFORM_DEPENDENCIES = {
    "cuda": get_cuda_dependencies(),
    "rocm": get_rocm_dependencies(),
    "apple": get_apple_dependencies(),
    "cpu": get_cpu_dependencies(),
}


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
    deps = PLATFORM_DEPENDENCIES.get(platform_to_use, PLATFORM_DEPENDENCIES["cpu"])
    print(f"Installing {platform_to_use.upper()} dependencies...")
    return deps


def _collect_package_files(*directories: str):
    """Collect package data files relative to the simpletuner package."""
    collected = []
    package_root = Path("simpletuner")
    for directory in directories:
        root = Path(directory)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    relative = path.relative_to(package_root)
                except ValueError:
                    # Skip files outside package root
                    continue
                collected.append(str(relative))
    return collected


# Base dependencies (minimal, works on all platforms)
base_deps = [
    "diffusers>=0.35.1",
    "transformers>=4.55.0",
    "hf_transfer>=0.1.0",
    "datasets>=3.0.1",
    "wandb>=0.21.0",
    "requests>=2.32.4",
    "pillow>=11.3.0",
    "trainingsample>=0.2.10",
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
    "num2words>=0.5.13",
    "peft>=0.17.0",
    "tensorboard>=2.18.0",
    "py3langid>=0.2.2",
    "pypinyin>=0.50.0",
    "sentencepiece>=0.2.0",
    "spacy>=3.7.4",
    "hangul-romanize>=0.1.0",
    "optimum-quanto>=0.2.7",
    "lycoris-lora>=3.4.0",
    "torch-optimi>=0.2.1",
    "torchaudio>=2.4.1",
    "librosa>=0.10.2",
    "loguru>=0.7.2",
    "toml>=0.10.2",
    "fastapi[standard]>=0.115.0",
    "sse-starlette>=1.6.5",
    "atomicwrites>=1.4.1",
    "beautifulsoup4>=4.12.3",
    "prodigy-plus-schedule-free>=1.9.2",
    "tokenizers>=0.21.0",
    "huggingface-hub>=0.34.3",
    "imageio-ffmpeg>=0.6.0",
    "imageio[pyav]>=2.37.0",
    "hf-xet>=1.1.5",
    "peft-singlora>=0.2.0",
    "cryptography>=41.0.0",
    "torchcodec>=0.8.1",
]

platform_deps_for_install = get_platform_dependencies()

# Optional extras
extras_require = {
    "jxl": ["pillow-jxl-plugin>=1.3.1"],
    "dev": [
        "selenium>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
    ],
    # Platform-specific extras for manual override
    "cuda": list(PLATFORM_DEPENDENCIES["cuda"]),
    "rocm": list(PLATFORM_DEPENDENCIES["rocm"]),
    "apple": list(PLATFORM_DEPENDENCIES["apple"]),
    "cpu": list(PLATFORM_DEPENDENCIES["cpu"]),
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
    include_package_data=True,
    package_data={
        "simpletuner": _collect_package_files(
            "simpletuner/templates",
            "simpletuner/static",
            "simpletuner/config",
            "simpletuner/documentation",
        ),
    },
    python_requires=">=3.11,<3.14",
    install_requires=base_deps + platform_deps_for_install,
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
