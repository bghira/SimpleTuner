#!/usr/bin/env bash
set -u
set -o pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
st_root="$(cd -- "$script_dir/.." >/dev/null 2>&1 && pwd)"

mode="check"
python_bin="${PYTHON:-$st_root/.venv/bin/python}"
umfa_repo_url="${UMFA_REPO_URL:-https://github.com/bghira/universal-metal-flash-attention.git}"
umfa_root="${UMFA_ROOT:-$st_root/../universal-metal-flash-attention}"
skip_python_deps=0

failures=0
warnings=0

usage() {
  cat <<'EOF'
Usage:
  scripts/apple-metal-flash-attention.sh [--check]
  scripts/apple-metal-flash-attention.sh --install [options]

Options:
  --check                 Inspect local dependencies and print the install plan. Default.
  --install               Clone/build/install UMFA and its PyTorch FFI extension.
  --umfa-root PATH        UMFA checkout path. Default: ../universal-metal-flash-attention.
  --python PATH           Python executable. Default: .venv/bin/python.
  --repo-url URL          UMFA git URL used when --install needs to clone it.
  --skip-python-deps      Do not install/upgrade pip, setuptools, wheel, pybind11, numpy.
  -h, --help              Show this help.

Environment overrides:
  UMFA_ROOT, PYTHON, UMFA_REPO_URL
EOF
}

while (($#)); do
  case "$1" in
    --check)
      mode="check"
      shift
      ;;
    --install)
      mode="install"
      shift
      ;;
    --umfa-root)
      if (($# < 2)); then
        echo "Missing value for --umfa-root" >&2
        exit 2
      fi
      umfa_root="$2"
      shift 2
      ;;
    --python)
      if (($# < 2)); then
        echo "Missing value for --python" >&2
        exit 2
      fi
      python_bin="$2"
      shift 2
      ;;
    --repo-url)
      if (($# < 2)); then
        echo "Missing value for --repo-url" >&2
        exit 2
      fi
      umfa_repo_url="$2"
      shift 2
      ;;
    --skip-python-deps)
      skip_python_deps=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ok() {
  printf '[ok] %s\n' "$1"
}

warn() {
  warnings=$((warnings + 1))
  printf '[warn] %s\n' "$1"
}

fail() {
  failures=$((failures + 1))
  printf '[fail] %s\n' "$1"
}

info() {
  printf '[info] %s\n' "$1"
}

quote_arg() {
  printf '%q' "$1"
}

command_path() {
  command -v "$1" 2>/dev/null
}

check_command() {
  local label="$1"
  local cmd="$2"
  local path

  path="$(command_path "$cmd" || true)"
  if [[ -n "$path" ]]; then
    ok "$label: $path"
  else
    fail "$label: missing '$cmd'"
  fi
}

check_macos() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    fail "macOS is required for Metal Flash Attention."
    return
  fi

  local version major arch
  version="$(sw_vers -productVersion 2>/dev/null || echo unknown)"
  major="${version%%.*}"
  arch="$(uname -m)"
  ok "macOS detected: $version"

  if [[ "$arch" == "arm64" ]]; then
    ok "Apple Silicon architecture detected: $arch"
  else
    fail "Apple Silicon arm64 is required; detected '$arch'."
  fi

  if [[ "$major" =~ ^[0-9]+$ ]] && ((major >= 15)); then
    ok "macOS version is compatible with the current UMFA package."
  else
    fail "Current UMFA package declares macOS 15+; detected '$version'."
  fi
}

check_xcode() {
  local selected sdk_path sdk_version metal_path metallib_path swift_version xcode_version xcode_major

  selected="$(xcode-select -p 2>/dev/null || true)"
  if [[ -n "$selected" ]]; then
    ok "Xcode developer directory: $selected"
  else
    fail "Xcode command line tools are not selected. Run 'xcode-select --install' or select Xcode with xcode-select."
  fi

  sdk_path="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"
  if [[ -n "$sdk_path" ]]; then
    ok "macOS SDK: $sdk_path"
  else
    fail "xcrun cannot find the macOS SDK."
  fi

  sdk_version="$(xcrun --sdk macosx --show-sdk-version 2>/dev/null || true)"
  if [[ -n "$sdk_version" ]]; then
    ok "macOS SDK version: $sdk_version"
  fi

  metal_path="$(xcrun --find metal 2>/dev/null || true)"
  if [[ -n "$metal_path" ]]; then
    ok "Metal compiler: $metal_path"
  else
    fail "xcrun cannot find the Metal compiler."
  fi

  metallib_path="$(xcrun --find metallib 2>/dev/null || true)"
  if [[ -n "$metallib_path" ]]; then
    ok "Metal library tool: $metallib_path"
  else
    fail "xcrun cannot find metallib."
  fi

  swift_version="$(swift --version 2>/dev/null | head -n 1 || true)"
  if [[ -n "$swift_version" ]]; then
    ok "Swift: $swift_version"
    if [[ "$swift_version" =~ Swift[[:space:]]version[[:space:]]([0-9]+) ]] && ((${BASH_REMATCH[1]} < 6)); then
      fail "Swift 6+ is required by UMFA."
    fi
  else
    fail "Swift compiler is missing."
  fi

  xcode_version="$(xcodebuild -version 2>/dev/null | awk '/^Xcode / {print $2; exit}' || true)"
  if [[ -n "$xcode_version" ]]; then
    ok "Xcode: $xcode_version"
    xcode_major="${xcode_version%%.*}"
    if [[ "$xcode_major" =~ ^[0-9]+$ ]] && ((xcode_major < 16)); then
      warn "Xcode 16+ is expected by UMFA's current SDK targets; detected Xcode $xcode_version."
    fi
  else
    warn "xcodebuild did not report a full Xcode version; command line tools may still be usable if Swift and Metal are present."
  fi
}

check_python() {
  if [[ ! -x "$python_bin" ]]; then
    fail "Python executable is missing or not executable: $python_bin"
    return
  fi

  ok "Python: $python_bin"

  "$python_bin" - <<'PY'
import importlib.metadata
import importlib.util
import sys

print(f"[info] Python version: {sys.version.split()[0]}")

hard_missing = []
soft_missing = []

for package in ("torch", "numpy", "pybind11", "setuptools", "wheel"):
    spec = importlib.util.find_spec(package)
    if spec is None:
        if package == "torch":
            hard_missing.append(package)
        else:
            soft_missing.append(package)
        print(f"[fail] Python package missing: {package}" if package == "torch" else f"[warn] Python package missing: {package}")
        continue
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    print(f"[ok] Python package: {package} {version}")

if hard_missing:
    raise SystemExit(2)

try:
    import torch
except Exception as exc:
    print(f"[fail] torch import failed: {exc}")
    raise SystemExit(2)

mps_built = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
print(f"[ok] torch version: {torch.__version__}")
print(f"[{'ok' if mps_built else 'fail'}] torch.backends.mps.is_built(): {mps_built}")
print(f"[{'ok' if mps_available else 'fail'}] torch.backends.mps.is_available(): {mps_available}")

if not mps_built or not mps_available:
    raise SystemExit(2)
if soft_missing:
    raise SystemExit(1)
PY
  case "$?" in
    0)
      ok "Python environment is ready for UMFA."
      ;;
    1)
      warn "Python build dependencies are incomplete; --install can install the missing build packages."
      ;;
    *)
      fail "Python environment is not ready for UMFA."
      ;;
  esac
}

check_umfa_source() {
  if [[ ! -d "$umfa_root" ]]; then
    warn "UMFA checkout not found: $umfa_root"
    return
  fi

  ok "UMFA checkout: $umfa_root"

  if [[ -f "$umfa_root/Package.swift" ]]; then
    ok "UMFA Package.swift found."
  else
    fail "UMFA Package.swift is missing; this does not look like a UMFA checkout."
  fi

  if [[ -f "$umfa_root/.gitmodules" ]] && grep -q 'metal-flash-attention' "$umfa_root/.gitmodules"; then
    ok "UMFA declares the MFA+ submodule."
  else
    warn "UMFA .gitmodules does not declare the MFA+ submodule."
  fi

  if [[ -f "$umfa_root/metal-flash-attention/Package.swift" ]]; then
    ok "MFA+ submodule is initialized: metal-flash-attention/Package.swift"
  else
    warn "MFA+ submodule is not initialized. Run: git -C $(quote_arg "$umfa_root") submodule update --init --recursive"
  fi

  if command_path git >/dev/null && [[ -d "$umfa_root/.git" ]]; then
    info "UMFA submodule status:"
    git -C "$umfa_root" submodule status --recursive || warn "git submodule status failed for UMFA."
  fi

  if [[ -f "$umfa_root/.build/arm64-apple-macosx/release/libMFAFFI.dylib" ]]; then
    ok "MFAFFI release library built."
  else
    warn "MFAFFI release library not found. Build with: swift build -c release --product MFAFFI"
  fi

  if [[ -f "$umfa_root/examples/pytorch-custom-op-ffi/setup.py" ]]; then
    ok "PyTorch FFI setup.py found."
  else
    fail "PyTorch FFI setup.py is missing."
  fi
}

check_extension_exports() {
  if [[ ! -x "$python_bin" ]]; then
    warn "Skipping metal_sdpa_extension import check because Python is unavailable."
    return
  fi

  "$python_bin" - <<'PY'
try:
    import metal_sdpa_extension as ext
except Exception as exc:
    print(f"[warn] metal_sdpa_extension import failed: {exc}")
    raise SystemExit(1)

required = [
    "metal_flash_attention_autograd",
    "clear_quantization_mode",
    "get_dispatch_stats",
    "metal_quantized_flash_attention_autograd",
    "set_quantization_mode",
    "QUANT_INT8",
    "QUANT_INT4",
    "QUANT_BLOCK_WISE",
]
optional = ["QUANT_TENSOR_WISE", "rope_scaled_dot_product_attention"]
missing = [name for name in required if not hasattr(ext, name)]
if missing:
    print("[fail] metal_sdpa_extension missing required exports: " + ", ".join(missing))
    raise SystemExit(2)

print("[ok] metal_sdpa_extension imports and exposes required UMFA symbols.")
for name in optional:
    print(f"[{'ok' if hasattr(ext, name) else 'warn'}] optional export {name}: {hasattr(ext, name)}")

try:
    print("[ok] get_dispatch_stats():", ext.get_dispatch_stats())
except Exception as exc:
    print(f"[fail] get_dispatch_stats() failed: {exc}")
    raise SystemExit(2)
PY
  case "$?" in
    0)
      ok "UMFA Python FFI binding is importable."
      ;;
    1)
      warn "UMFA Python FFI binding is not installed in this Python environment."
      ;;
    *)
      fail "UMFA Python FFI binding is installed but missing required behavior."
      ;;
  esac
}

check_simpletuner_backend() {
  if [[ ! -x "$python_bin" ]]; then
    warn "Skipping SimpleTuner backend check because Python is unavailable."
    return
  fi

  PYTHONPATH="$st_root${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" - <<'PY'
try:
    from simpletuner.helpers.training.attention_backend import (
        get_metal_flash_attention_unavailable_reason,
        is_metal_flash_attention_available,
    )
except Exception as exc:
    print(f"[warn] SimpleTuner attention backend import failed: {exc}")
    raise SystemExit(1)

backends = (
    "metal-flash-attention",
    "metal-flash-attention-int8",
    "metal-flash-attention-int4",
)
all_available = True
for backend in backends:
    available = is_metal_flash_attention_available(backend)
    reason = get_metal_flash_attention_unavailable_reason(backend)
    print(f"[{'ok' if available else 'warn'}] {backend}: available={available} reason={reason}")
    all_available = all_available and available

raise SystemExit(0 if all_available else 1)
PY
  if [[ "$?" == "0" ]]; then
    ok "SimpleTuner accepts all Metal Flash Attention backends."
  else
    warn "SimpleTuner does not currently accept every Metal Flash Attention backend; inspect the reason above."
  fi
}

print_manual_steps() {
  local q_python q_umfa q_st q_repo q_script
  q_python="$(quote_arg "$python_bin")"
  q_umfa="$(quote_arg "$umfa_root")"
  q_st="$(quote_arg "$st_root")"
  q_repo="$(quote_arg "$umfa_repo_url")"
  q_script="$(quote_arg "$st_root/scripts/apple-metal-flash-attention.sh")"

  cat <<EOF

Install plan:
  1. Use the same Python environment that runs SimpleTuner:
       export ST_ROOT=$q_st
       export PYTHON=$q_python
       export UMFA_ROOT=$q_umfa

  2. Clone UMFA if the checkout is missing:
       git clone --recursive $q_repo "\$UMFA_ROOT"

  3. Initialize the MFA+ submodule inside UMFA:
       git -C "\$UMFA_ROOT" submodule update --init --recursive

  4. Build the UMFA Swift FFI library:
       cd "\$UMFA_ROOT"
       swift build -c release --product MFAFFI

  5. Build and install the PyTorch custom-op FFI binding into SimpleTuner's Python:
       cd "\$UMFA_ROOT/examples/pytorch-custom-op-ffi"
       "\$PYTHON" -m pip install --upgrade pip setuptools wheel pybind11 numpy
       "\$PYTHON" setup.py build_ext --inplace
       "\$PYTHON" -m pip install --force-reinstall --no-deps --no-build-isolation --no-cache-dir .

  6. Re-run this diagnostic:
       $q_script --check --umfa-root "\$UMFA_ROOT" --python "\$PYTHON"

Run with --install to execute those steps now.
EOF
}

run_cmd() {
  printf '\n[run]'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

run_in_dir() {
  local cwd="$1"
  shift

  printf '\n[run] cd %q &&' "$cwd"
  printf ' %q' "$@"
  printf '\n'
  (cd "$cwd" && "$@")
}

install_umfa() {
  check_command "git" git
  check_command "swift" swift

  if ((failures > 0)); then
    echo
    fail "Required command checks failed; not starting install."
    return 1
  fi

  if [[ ! -d "$umfa_root" ]]; then
    run_cmd mkdir -p "$(dirname "$umfa_root")" || return 1
    run_cmd git clone --recursive "$umfa_repo_url" "$umfa_root" || return 1
  else
    info "Using existing UMFA checkout: $umfa_root"
  fi

  if [[ ! -f "$umfa_root/Package.swift" ]]; then
    fail "UMFA checkout does not contain Package.swift: $umfa_root"
    return 1
  fi

  run_cmd git -C "$umfa_root" submodule update --init --recursive || return 1
  run_in_dir "$umfa_root" swift build -c release --product MFAFFI || return 1

  if ((skip_python_deps == 0)); then
    run_cmd "$python_bin" -m pip install --upgrade pip setuptools wheel pybind11 numpy || return 1
  fi

  run_in_dir "$umfa_root/examples/pytorch-custom-op-ffi" "$python_bin" setup.py build_ext --inplace || return 1
  run_in_dir "$umfa_root/examples/pytorch-custom-op-ffi" "$python_bin" -m pip install --force-reinstall --no-deps --no-build-isolation --no-cache-dir . || return 1
}

echo "SimpleTuner root: $st_root"
echo "Python: $python_bin"
echo "UMFA root: $umfa_root"
echo "Mode: $mode"
echo

check_macos
check_command "git" git
check_command "xcrun" xcrun
check_command "swift" swift
check_command "clang" clang
check_xcode
check_python
check_umfa_source
check_extension_exports
check_simpletuner_backend
print_manual_steps

if [[ "$mode" == "install" ]]; then
  echo
  info "Starting UMFA install."
  install_umfa || exit 1
  echo
  info "Install completed. Running final import checks."
  check_extension_exports
  check_simpletuner_backend
fi

echo
echo "Summary: $failures failure(s), $warnings warning(s)."
if ((failures > 0)); then
  exit 1
fi
