#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/ubuntu/SimpleTuner}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ubuntu/datasets/youtube}
SOURCE=${SOURCE:-youtube_search}
SAM3_CKPT=${SAM3_CKPT:-/home/ubuntu/.cache/huggingface/hub/models--1038lab--sam3/snapshots/f055b060a4de0a040891ba2ebac9c5cb3c1c0132/sam3.pt}
TARGET_COUNT=${TARGET_COUNT:-25000}
GEN_QUERY_COUNT=${GEN_QUERY_COUNT:-50000}
RESULTS_PER_QUERY=${RESULTS_PER_QUERY:-8}
MAX_DURATION=${MAX_DURATION:-360}
SLEEP_SECONDS=${SLEEP_SECONDS:-5}
QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3-VL-4B-Instruct}
LOG_DIR=${LOG_DIR:-/tmp}
BOT_CHALLENGE_FLAG=${BOT_CHALLENGE_FLAG:-${OUTPUT_DIR}/.youtube_bot_challenge}
RATE_LIMIT_FLAG=${RATE_LIMIT_FLAG:-${OUTPUT_DIR}/.youtube_rate_limited}
YOUTUBE_JSON_COOKIE_FILE=${YOUTUBE_JSON_COOKIE_FILE:-}
YOUTUBE_JSON_COOKIES_FROM_BROWSER=${YOUTUBE_JSON_COOKIES_FROM_BROWSER:-}
YOUTUBE_JSON_YT_DLP_JS_RUNTIME=${YOUTUBE_JSON_YT_DLP_JS_RUNTIME:-deno:/home/ubuntu/.deno/bin/deno}
SCENEWALK_DATASET=${SCENEWALK_DATASET:-IVLLab/SceneWalk}
SCENEWALK_SPLIT=${SCENEWALK_SPLIT:-train}
SCENEWALK_SHUFFLE_BUFFER=${SCENEWALK_SHUFFLE_BUFFER:-10000}

workers=(
  "0 720a4 0 1213 540 720"
  "1 720b4 1 1214 540 720"
  "2 512a4 2 1317 480 540"
  "3 512b4 3 1318 480 540"
)

accepted_count() {
  find "$OUTPUT_DIR" -mindepth 2 -type f -name '*.json' 2>/dev/null | wc -l
}

worker_running() {
  local name=$1
  local pid_file="${LOG_DIR}/youtube-json-${name}.pid"
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid=$(<"$pid_file")
  [[ -n "$pid" ]] || return 1
  if kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  rm -f "$pid_file"
  return 1
}

start_worker() {
  local gpu=$1 name=$2 worker_index=$3 seed=$4 min_h=$5 max_h=$6
  local log="${LOG_DIR}/youtube-json-${name}.log"
  mkdir -p "$OUTPUT_DIR"
  (
    cd "$ROOT_DIR"
    cookie_args=()
    if [[ -n "$YOUTUBE_JSON_COOKIE_FILE" ]]; then
      cookie_args+=(--yt-dlp-cookie-file "$YOUTUBE_JSON_COOKIE_FILE")
    fi
    if [[ -n "$YOUTUBE_JSON_COOKIES_FROM_BROWSER" ]]; then
      cookie_args+=(--yt-dlp-cookies-from-browser "$YOUTUBE_JSON_COOKIES_FROM_BROWSER")
    fi
    if [[ -n "$YOUTUBE_JSON_YT_DLP_JS_RUNTIME" ]]; then
      cookie_args+=(--yt-dlp-js-runtime "$YOUTUBE_JSON_YT_DLP_JS_RUNTIME")
    fi
    source_args=(--source "$SOURCE")
    if [[ "$SOURCE" == "scenewalk" ]]; then
      source_args+=(
        --scenewalk-dataset "$SCENEWALK_DATASET"
        --scenewalk-split "$SCENEWALK_SPLIT"
        --scenewalk-worker-index "$worker_index"
        --scenewalk-worker-count "${#workers[@]}"
        --scenewalk-shuffle-buffer "$SCENEWALK_SHUFFLE_BUFFER"
      )
    fi
    exec env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
      .venv/bin/python scripts/datasets/youtube_json_dataset.py \
        "${source_args[@]}" \
        --search-dir "/home/ubuntu/datasets/youtube-search-${name}" \
        --output-dir "$OUTPUT_DIR" \
        --target-count "$TARGET_COUNT" \
        --results-per-query "$RESULTS_PER_QUERY" \
        --generated-query-count "$GEN_QUERY_COUNT" \
        --seed "$seed" \
        --min-height "$min_h" \
        --max-height "$max_h" \
        --max-duration "$MAX_DURATION" \
        --sleep "$SLEEP_SECONDS" \
        --extract-preview-frames \
        --qwen-vl-model "$QWEN_MODEL" \
        --qwen-vl-video-fps 0.5 \
        --qwen-vl-max-new-tokens 900 \
        --sam3-checkpoint "$SAM3_CKPT" \
        --sam3-box-frames 2 \
        --sam3-score-threshold 0.2 \
        --sam3-min-grounded-elements 2 \
        --sam3-min-grounded-ratio 0.4 \
        "${cookie_args[@]}"
  ) >>"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"${LOG_DIR}/youtube-json-${name}.pid"
  echo "$(date -u +%FT%TZ) started ${name} pid=${pid}" >>"${LOG_DIR}/youtube-json-watchdog.log"
}

if [[ -f "$BOT_CHALLENGE_FLAG" ]]; then
  echo "$(date -u +%FT%TZ) paused: bot challenge flag exists at ${BOT_CHALLENGE_FLAG}" >>"${LOG_DIR}/youtube-json-watchdog.log"
  exit 75
fi
if [[ -f "$RATE_LIMIT_FLAG" ]]; then
  echo "$(date -u +%FT%TZ) paused: rate limit flag exists at ${RATE_LIMIT_FLAG}" >>"${LOG_DIR}/youtube-json-watchdog.log"
  exit 75
fi

for spec in "${workers[@]}"; do
  read -r gpu name worker_index seed min_h max_h <<<"$spec"
  if ! worker_running "$name"; then
    start_worker "$gpu" "$name" "$worker_index" "$seed" "$min_h" "$max_h"
  fi
done

while true; do
  if [[ -f "$BOT_CHALLENGE_FLAG" ]]; then
    echo "$(date -u +%FT%TZ) paused: bot challenge flag exists at ${BOT_CHALLENGE_FLAG}" >>"${LOG_DIR}/youtube-json-watchdog.log"
    exit 75
  fi
  if [[ -f "$RATE_LIMIT_FLAG" ]]; then
    echo "$(date -u +%FT%TZ) paused: rate limit flag exists at ${RATE_LIMIT_FLAG}" >>"${LOG_DIR}/youtube-json-watchdog.log"
    exit 75
  fi
  count=$(accepted_count)
  echo "$(date -u +%FT%TZ) accepted=${count}/${TARGET_COUNT}" >>"${LOG_DIR}/youtube-json-watchdog.log"
  if (( count >= TARGET_COUNT )); then
    echo "$(date -u +%FT%TZ) target reached" >>"${LOG_DIR}/youtube-json-watchdog.log"
    exit 0
  fi
  for spec in "${workers[@]}"; do
    read -r gpu name worker_index seed min_h max_h <<<"$spec"
    if ! worker_running "$name"; then
      start_worker "$gpu" "$name" "$worker_index" "$seed" "$min_h" "$max_h"
    fi
  done
  sleep 300
done
