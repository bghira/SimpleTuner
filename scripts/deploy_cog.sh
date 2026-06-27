#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/deploy_cog.sh [options]

Build the advanced-trainer Cog image once, then push the exact same image to
the SimpleTuner Replicate model paths.

Options:
  --no-build              Skip `cog build` and push an existing local image tag.
  --no-push               Build only; do not push to r8.im.
  --tag TAG               Local image tag to build and copy from.
                          Defaults to simpletuner/advanced-trainer-cog:<git-sha>.
  --target IMAGE          Add one push target. Can be repeated. If any target is
                          supplied, the default target list is replaced.
  --include-legacy        Also push r8.im/simpletuner/advanced-trainer.
  --cog-arg ARG           Extra argument passed to `cog build`; can be repeated.
  -h, --help              Show this help.

Environment:
  COG_IMAGE_TAG           Same as --tag.
  INCLUDE_LEGACY_ADVANCED_TRAINER=1
                          Same as --include-legacy.
  COG_BUILD_ARGS          Space-separated extra args for `cog build`.
  DRY_RUN=1               Print commands without running them.

Default targets:
  r8.im/simpletuner/advanced-trainer-h100
  r8.im/simpletuner/advanced-trainer-h100-x2
  r8.im/simpletuner/advanced-trainer-h100-x4
  r8.im/simpletuner/advanced-trainer-h100-x8
  r8.im/simpletuner/advanced-trainer-l40s
  r8.im/simpletuner/advanced-trainer-l40s-x2
  r8.im/simpletuner/advanced-trainer-l40s-x4
  r8.im/simpletuner/advanced-trainer-l40s-x8
EOF
}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "$@"
  fi
}

default_tag() {
  local sha
  sha="$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || true)"
  if [[ -z "${sha}" ]]; then
    sha="$(date -u +%Y%m%d%H%M%S)"
  fi
  printf 'simpletuner/advanced-trainer-cog:%s' "${sha}"
}

build_image=1
push_image=1
include_legacy="${INCLUDE_LEGACY_ADVANCED_TRAINER:-0}"
image_tag="${COG_IMAGE_TAG:-$(default_tag)}"
declare -a cog_build_args=()
declare -a targets=()

if [[ -n "${COG_BUILD_ARGS:-}" ]]; then
  # Intentional shell-style splitting for simple flags such as "--no-cache".
  # Use repeated --cog-arg when values contain spaces.
  read -r -a cog_build_args <<< "${COG_BUILD_ARGS}"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build)
      build_image=0
      shift
      ;;
    --no-push)
      push_image=0
      shift
      ;;
    --tag)
      image_tag="${2:?--tag requires a value}"
      shift 2
      ;;
    --target)
      targets+=("${2:?--target requires a value}")
      shift 2
      ;;
    --include-legacy)
      include_legacy=1
      shift
      ;;
    --cog-arg)
      cog_build_args+=("${2:?--cog-arg requires a value}")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#targets[@]} -eq 0 ]]; then
  targets=(
    "r8.im/simpletuner/advanced-trainer-h100"
    "r8.im/simpletuner/advanced-trainer-h100-x2"
    "r8.im/simpletuner/advanced-trainer-h100-x4"
    "r8.im/simpletuner/advanced-trainer-h100-x8"
    "r8.im/simpletuner/advanced-trainer-l40s"
    "r8.im/simpletuner/advanced-trainer-l40s-x2"
    "r8.im/simpletuner/advanced-trainer-l40s-x4"
    "r8.im/simpletuner/advanced-trainer-l40s-x8"
  )
fi

if [[ "${include_legacy}" == "1" ]]; then
  targets+=("r8.im/simpletuner/advanced-trainer")
fi

cd "${REPO_ROOT}"

if [[ ! -f cog.yaml ]]; then
  echo "cog.yaml not found in ${REPO_ROOT}" >&2
  exit 1
fi

if [[ ${build_image} -eq 1 ]]; then
  run cog build -t "${image_tag}" "${cog_build_args[@]}"
fi

if [[ ${push_image} -eq 0 ]]; then
  exit 0
fi

if ! command -v skopeo >/dev/null 2>&1; then
  echo "skopeo is required to fan out one built image without rebuilding." >&2
  exit 1
fi

for target in "${targets[@]}"; do
  run skopeo copy --all "docker-daemon:${image_tag}" "docker://${target}:latest"
done
