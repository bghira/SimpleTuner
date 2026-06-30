#!/usr/bin/env python3
"""Build a YouTube video dataset with Ideogram-style JSON captions.

The script keeps two separate outputs:

* ``search_dir`` stores generated queries plus candidate/rejection logs.
* ``output_dir`` stores ``<uuid>.mp4`` and ``<uuid>.json`` training pairs.

Per-sample source metadata is written to ``manifest.jsonl`` instead of the
Ideogram sidecar so the caption JSON remains compatible with the strict
Ideogram prompt schema.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_QWEN_VL_MODEL = None
_QWEN_VL_PROCESSOR = None
_SAM3_PROCESSOR = None
STATIC_VIDEO_RE = re.compile(
    r"\b(static images?|series of images|sequence of still images|slideshow|no visible motion|camera remains stationary|text overlays?|title card|"
    r"black screen|presentation|courtroom|judge judy)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VideoFrameSample:
    image: Any
    frame_idx: int
    frame_time: float
    sample_idx: int


SEARCH_SUBJECTS = (
    "archery practice",
    "artisan bakery",
    "backyard chickens",
    "beach cleanup",
    "bee keeping",
    "bird watching",
    "blacksmith workshop",
    "bookbinder",
    "busker performance",
    "ceramic artist",
    "city cycling",
    "community garden",
    "dog agility",
    "drone landscape",
    "fish market",
    "flower garden",
    "food truck",
    "fruit harvest",
    "furniture restoration",
    "gardener",
    "goat farm",
    "woodworker",
    "street food vendor",
    "botanical garden",
    "indie musician",
    "small farm",
    "glassblower",
    "bike mechanic",
    "handmade jewelry",
    "hiking trail",
    "horse riding",
    "kayaking",
    "kitchen prep",
    "leather craft",
    "metal casting",
    "mushroom foraging",
    "night market",
    "parkour training",
    "portrait photographer",
    "repair cafe",
    "sailing harbor",
    "sheep farm",
    "shoe repair",
    "silversmith",
    "snowboarding",
    "street musician",
    "surfboard shaping",
    "urban cycling",
    "vintage car repair",
    "weaving loom",
    "textile studio",
    "local market",
    "skateboarder",
    "dance rehearsal",
    "pottery wheel",
    "printmaking studio",
    "maker workshop",
    "train station",
    "community kitchen",
    "wildlife rescue",
    "urban gardener",
    "small bakery",
    "surf practice",
    "climbing gym",
    "robotics club",
    "street portrait",
    "rainy city walk",
)

SEARCH_CONTEXTS = (
    "artist profile",
    "behind the scenes",
    "daily work",
    "documentary",
    "process",
    "short documentary",
    "day in the life",
    "practice session",
    "field recording",
    "studio tour",
    "tutorial",
    "ambient video",
    "local story",
    "independent creator",
    "creative commons",
    "making of",
    "workshop visit",
)

SEARCH_QUALIFIERS = (
    "480p",
    "720p",
    "4k",
    "1080p",
    "natural light",
    "handheld",
    "cinematic",
    "no commentary",
    "real time",
    "outdoors",
    "close up",
    "wide shot",
    "raw footage",
    "short video",
)

SEARCH_TEMPLATES = (
    "{subject} {context} {qualifier} {term}",
    "{subject} {term}",
    "{subject} {context} {term}",
    "{subject} {qualifier} {term}",
    "{term} {subject} video",
    "{term} {subject} documentary",
    "{term} {subject} process",
)

BLOCKLIST_RE = re.compile(
    r"\b("
    r"trailer|movie|film clip|official trailer|advert|ad break|\bad\b|commercial|sponsored|template|"
    r"reaction|compilation|top 10|news|politics|celebrity gossip|lyrics|music video|"
    r"copyright|all rights reserved|netflix|disney|marvel|pixar|warner|universal pictures|"
    r"free download|soundcloud|spotify|album|single|ep\b|official audio|"
    r"department of education|official government|government agency|state of queensland|municipal meetings|elections|"
    r"free royalty footage|royalty free|stock footage|free stock|free to use video|"
    r"amazon services llc associates|amazon associate|affiliate links?|commission earned|"
    r"buy a life-sized|please consider leaving a like|"
    r"subscribe to our channel|free hd stock|video background|motion background|audio library|"
    r"gameplay|video game|videogame|planet coaster|licensing enquiries|british path[eé]|reuters|"
    r"bbc earth|national geographic|natgeo|nasa|space agency|final fantasy|ffxiv|"
    r"travelogue|tourism|tourist destination|elephant theatre|cinematography basics|filmmaking course"
    r"|expo|trade show|conference booth|see you in 20[0-9]{2}"
    r")\b",
    re.IGNORECASE,
)

CC_LICENSE_RE = re.compile(r"creative commons|cc[- ]?by|reuse allowed", re.IGNORECASE)
HEX_COLOR_RE = re.compile(r"#(?:[0-9a-fA-F]{6})\b")
LOW_VALUE_CATEGORIES = {
    "Entertainment",
    "Film & Animation",
    "Gaming",
    "Music",
    "News & Politics",
}
NON_VISUAL_CONCEPT_CATEGORIES = {
    "Education",
    "Film & Animation",
    "Nonprofits & Activism",
    "People & Blogs",
}
GENERIC_CONCEPT_WORDS = {
    "about",
    "allow",
    "allowed",
    "attribution",
    "blog",
    "blogs",
    "by",
    "cc",
    "channel",
    "commons",
    "creative",
    "download",
    "ed",
    "episode",
    "everything",
    "facts",
    "footage",
    "for",
    "free",
    "full",
    "highlights",
    "how",
    "can",
    "do",
    "does",
    "know",
    "learn",
    "licensed",
    "live",
    "make",
    "need",
    "official",
    "part",
    "people",
    "reuse",
    "shot",
    "short",
    "things",
    "video",
    "videos",
    "vlog",
    "what",
    "where",
    "with",
    "you",
    "your",
}
QUERY_TOPIC_STOP_WORDS = {
    "1080p",
    "480p",
    "4k",
    "720p",
    "ambient",
    "artist",
    "behind",
    "cinematic",
    "close",
    "commons",
    "creative",
    "daily",
    "documentary",
    "field",
    "handheld",
    "independent",
    "life",
    "light",
    "local",
    "making",
    "natural",
    "outdoors",
    "practice",
    "process",
    "profile",
    "raw",
    "real",
    "recording",
    "reuse",
    "session",
    "short",
    "shot",
    "story",
    "studio",
    "the",
    "time",
    "tour",
    "tutorial",
    "video",
    "visit",
    "wide",
    "work",
    "workshop",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-dir", type=Path, default=Path("~/datasets/youtube-search").expanduser())
    parser.add_argument("--output-dir", type=Path, default=Path("~/datasets/youtube").expanduser())
    parser.add_argument("--target-count", type=int, default=25_000)
    parser.add_argument(
        "--source",
        choices=("youtube_search", "scenewalk"),
        default="youtube_search",
        help="Candidate source. scenewalk streams timestamped YouTube clips from IVLLab/SceneWalk.",
    )
    parser.add_argument("--queries", type=Path, help="Optional newline-delimited search queries.")
    parser.add_argument("--generated-query-count", type=int, default=2_000)
    parser.add_argument(
        "--generated-query-license-term",
        default="",
        help="Comma-delimited terms appended to generated queries. Empty by default to avoid Creative Commons-only search bias.",
    )
    parser.add_argument("--results-per-query", type=int, default=12)
    parser.add_argument("--min-duration", type=int, default=15)
    parser.add_argument("--max-duration", type=int, default=360)
    parser.add_argument("--min-height", type=int, default=480)
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument(
        "--resolution-buckets",
        default="512,720",
        help="Comma-delimited base resolution buckets used for output subdirectories.",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between search/download operations.")
    parser.add_argument("--seed", type=int, default=613)
    parser.add_argument(
        "--license-mode",
        choices=("youtube", "creative_commons", "any"),
        default="youtube",
        help="License metadata filter. 'youtube' accepts YouTube's built-in license metadata and records it for review.",
    )
    parser.add_argument("--allow-non-cc", action="store_true", help="Deprecated alias for --license-mode youtube.")
    parser.add_argument("--dry-run", action="store_true", help="Search and log candidates without downloading videos.")
    parser.add_argument("--download-limit", type=int, help="Stop after this many newly downloaded videos.")
    parser.add_argument(
        "--metadata-only-labels",
        action="store_true",
        help="Allow deterministic metadata-only sidecars when no --labeler-model is supplied.",
    )
    parser.add_argument(
        "--qwen-vl-model",
        default=os.environ.get("YOUTUBE_JSON_QWEN_VL_MODEL"),
        help="Local Qwen3-VL model ID/path used to label sampled video frames after download.",
    )
    parser.add_argument("--qwen-vl-frames", type=int, default=4)
    parser.add_argument("--qwen-vl-video-fps", type=float, default=1.0)
    parser.add_argument("--qwen-vl-video-max-pixels", type=int, default=360 * 640)
    parser.add_argument("--qwen-vl-max-new-tokens", type=int, default=1200)
    parser.add_argument(
        "--sam3-checkpoint",
        type=Path,
        help="Local SAM 3 checkpoint path. Required until this machine has access to facebook/sam3.",
    )
    parser.add_argument("--sam3-device", default="cuda")
    parser.add_argument("--sam3-box-frames", type=int, default=3)
    parser.add_argument("--sam3-score-threshold", type=float, default=0.25)
    parser.add_argument("--sam3-min-grounded-elements", type=int, default=2)
    parser.add_argument("--sam3-min-grounded-ratio", type=float, default=0.4)
    parser.add_argument(
        "--no-require-bboxes",
        action="store_true",
        help="Allow samples whose visible elements cannot be grounded by SAM 3. Not recommended for the bbox dataset.",
    )
    parser.add_argument("--labeler-model", help="OpenAI-compatible vLLM model name for metadata-to-caption labeling.")
    parser.add_argument(
        "--labeler-base-url",
        default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible base URL used when --labeler-model is set.",
    )
    parser.add_argument("--labeler-api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--extract-preview-frames", action="store_true")
    parser.add_argument("--preview-frames", type=int, default=4)
    parser.add_argument("--preview-max-width", type=int, default=0, help="Resize preview frames only; 0 keeps native size.")
    parser.add_argument(
        "--yt-dlp-cookie-file",
        type=Path,
        default=Path(os.environ["YOUTUBE_JSON_COOKIE_FILE"]) if os.environ.get("YOUTUBE_JSON_COOKIE_FILE") else None,
        help="Netscape-format cookies.txt passed to yt-dlp for YouTube bot/age checks.",
    )
    parser.add_argument(
        "--yt-dlp-cookies-from-browser",
        default=os.environ.get("YOUTUBE_JSON_COOKIES_FROM_BROWSER"),
        help="Browser name passed to yt-dlp cookiesfrombrowser, e.g. firefox or chrome.",
    )
    parser.add_argument(
        "--yt-dlp-js-runtime",
        default=os.environ.get("YOUTUBE_JSON_YT_DLP_JS_RUNTIME"),
        help="yt-dlp JavaScript runtime spec for YouTube challenge solving, e.g. deno:/path/to/deno.",
    )
    parser.add_argument("--bot-challenge-threshold", type=int, default=20)
    parser.add_argument("--rate-limit-threshold", type=int, default=5)
    parser.add_argument("--scenewalk-dataset", default="IVLLab/SceneWalk")
    parser.add_argument("--scenewalk-split", default="train")
    parser.add_argument("--scenewalk-worker-index", type=int, default=0)
    parser.add_argument("--scenewalk-worker-count", type=int, default=1)
    parser.add_argument("--scenewalk-shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--scenewalk-start-row", type=int, default=0)
    parser.add_argument("--scenewalk-max-rows", type=int)
    return parser.parse_args()


def jsonl_append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_seen_ids(*paths: Path) -> set[str]:
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                video_id = row.get("source_key") or row.get("youtube_id") or row.get("id")
                if video_id:
                    seen.add(str(video_id))
    return seen


def load_reserved_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def normalized_title_key(title: Any) -> str:
    key = clean_title(title).lower()
    key = re.sub(r"[^a-z0-9]+", " ", key)
    return re.sub(r"\s+", " ", key).strip()


def reserve_candidate_identity(output_dir: Path, source_key: str, title: Any) -> bool:
    import fcntl

    lock_path = output_dir / ".youtube_json_dataset.lock"
    seen_path = output_dir / "seen_youtube_ids.txt"
    title_path = output_dir / "seen_titles.txt"
    title_key = normalized_title_key(title)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        seen = load_reserved_ids(seen_path)
        if source_key in seen:
            return False
        seen_titles = load_reserved_ids(title_path)
        if title_key and title_key in seen_titles:
            return False
        with seen_path.open("a", encoding="utf-8") as seen_handle:
            seen_handle.write(source_key + "\n")
        if title_key:
            with title_path.open("a", encoding="utf-8") as title_handle:
                title_handle.write(title_key + "\n")
        return True


def generated_queries(count: int, seed: int, license_terms: str) -> list[str]:
    rng = random.Random(seed)
    terms = [term.strip() for term in license_terms.split(",") if term.strip()]
    if not terms:
        terms = [""]

    queries: set[str] = set()
    for subject in SEARCH_SUBJECTS:
        for context in SEARCH_CONTEXTS:
            for qualifier in SEARCH_QUALIFIERS:
                for term in terms:
                    for template in SEARCH_TEMPLATES:
                        query = template.format(subject=subject, context=context, qualifier=qualifier, term=term).strip()
                        query = re.sub(r"\s+", " ", query)
                        queries.add(query)

    query_list = list(queries)
    rng.shuffle(query_list)
    if count > len(query_list):
        print(
            f"requested {count} generated queries but only {len(query_list)} unique combinations exist; "
            f"using {len(query_list)}",
            flush=True,
        )
    return query_list[: min(count, len(query_list))]


def load_queries(args: argparse.Namespace) -> list[str]:
    if args.queries:
        queries = [line.strip() for line in args.queries.read_text(encoding="utf-8").splitlines()]
        return [query for query in queries if query and not query.startswith("#")]
    return generated_queries(args.generated_query_count, args.seed, args.generated_query_license_term)


def entry_text(entry: dict[str, Any]) -> str:
    fields = (
        entry.get("title"),
        entry.get("description"),
        entry.get("channel"),
        entry.get("uploader"),
        " ".join(str(tag) for tag in entry.get("tags") or []),
    )
    return "\n".join(str(field or "") for field in fields)


def query_topic_terms(query: str) -> set[str]:
    terms = set()
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", query.lower()):
        if word in QUERY_TOPIC_STOP_WORDS or len(word) <= 2:
            continue
        terms.add(word)
    return terms


def entry_matches_query_topic(entry: dict[str, Any], query: str) -> bool:
    terms = query_topic_terms(query)
    if not terms:
        return True
    text = entry_text(entry).lower()
    matches = sum(1 for term in terms if term in text)
    required = 1 if len(terms) <= 1 else 2
    return matches >= required


def rejection_reason(entry: dict[str, Any], args: argparse.Namespace) -> str | None:
    duration = entry.get("duration")
    if not isinstance(duration, (int, float)):
        return "missing_duration"
    if duration < args.min_duration:
        return "too_short"
    if duration > args.max_duration:
        return "too_long"
    if entry.get("is_live") or entry.get("live_status") in {"is_live", "is_upcoming"}:
        return "live_or_upcoming"
    if entry.get("age_limit") not in (None, 0):
        return "age_restricted"
    categories = set(str(category) for category in entry.get("categories") or [])
    if categories & LOW_VALUE_CATEGORIES:
        return "low_value_category"
    if not entry_matches_query_topic(entry, str(entry.get("_search_query") or "")):
        return "off_topic_search_result"
    if BLOCKLIST_RE.search(entry_text(entry)):
        return "blocked_terms"
    license_text = str(entry.get("license") or "")
    license_mode = "youtube" if args.allow_non_cc else args.license_mode
    if license_mode == "creative_commons" and not CC_LICENSE_RE.search(license_text):
        return "license_not_creative_commons"
    return None


def compact_source_metadata(entry: dict[str, Any], query: str, sample_id: str) -> dict[str, Any]:
    license_text = entry.get("license")
    source_key = entry.get("_source_key") or entry.get("id")
    return {
        "sample_id": sample_id,
        "source_key": source_key,
        "source_dataset": entry.get("_source_dataset") or "youtube_search",
        "youtube_id": entry.get("id"),
        "webpage_url": entry.get("webpage_url") or entry.get("url"),
        "title": entry.get("title"),
        "description": entry.get("description"),
        "channel": entry.get("channel") or entry.get("uploader"),
        "channel_id": entry.get("channel_id") or entry.get("uploader_id"),
        "duration": entry.get("duration"),
        "license": license_text,
        "license_metadata_source": "youtube",
        "license_assumption": "youtube_standard_or_unreported" if not license_text else "youtube_reported",
        "categories": entry.get("categories") or [],
        "tags": entry.get("tags") or [],
        "upload_date": entry.get("upload_date"),
        "search_query": query,
        "clip_start": entry.get("_clip_start"),
        "clip_end": entry.get("_clip_end"),
        "clip_duration": entry.get("_clip_duration"),
    }


def parse_hms_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, float(value))
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 1:
            return max(0.0, float(parts[0]))
        seconds = float(parts[-1])
        minutes = int(parts[-2])
        hours = int(parts[-3]) if len(parts) > 2 else 0
    except ValueError:
        return None
    return max(0.0, hours * 3600 + minutes * 60 + seconds)


def conversation_text(row: dict[str, Any]) -> str:
    assistant_values: list[str] = []
    fallback_values: list[str] = []
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for item in conversations:
            if not isinstance(item, dict):
                continue
            value = clean_text(item.get("value"))
            if value:
                speaker = clean_text(item.get("from")).lower()
                if speaker in {"assistant", "gpt"}:
                    assistant_values.append(value)
                elif "<video>" not in value.lower():
                    fallback_values.append(value)
    return " ".join(assistant_values or fallback_values)


def scenewalk_entry_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    url = clean_text(row.get("url"))
    youtube_id = clean_text(row.get("id"))
    if not url or not youtube_id:
        return None
    if url.startswith("youtube.com/"):
        url = f"https://{url}"
    elif url.startswith("www.youtube.com/"):
        url = f"https://{url}"

    time_stamp = row.get("time_stamp")
    if not isinstance(time_stamp, dict):
        return None
    start = parse_hms_seconds(time_stamp.get("start_time"))
    end = parse_hms_seconds(time_stamp.get("end_time"))
    if start is None or end is None or end <= start:
        return None
    clip_duration = end - start
    description = conversation_text(row)
    source_key = f"scenewalk:{youtube_id}:{start:.3f}-{end:.3f}"
    return {
        "id": youtube_id,
        "_source_key": source_key,
        "_source_dataset": "IVLLab/SceneWalk",
        "_search_query": "scenewalk",
        "_clip_start": start,
        "_clip_end": end,
        "_clip_duration": clip_duration,
        "webpage_url": url,
        "url": url,
        "title": sentence_bound_snippet(description, 120) or f"SceneWalk clip {youtube_id}",
        "description": description,
        "duration": clip_duration,
        "license": None,
        "categories": ["SceneWalk"],
        "tags": ["scenewalk", "timestamped video clip"],
        "time_stamp": time_stamp,
        "v2t_score": row.get("v2t_score"),
        "t2t_score": row.get("t2t_score"),
    }


def stream_scenewalk_entries(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.scenewalk_worker_count < 1:
        raise ValueError("--scenewalk-worker-count must be at least 1")
    if not 0 <= args.scenewalk_worker_index < args.scenewalk_worker_count:
        raise ValueError("--scenewalk-worker-index must be in [0, --scenewalk-worker-count)")

    from datasets import load_dataset

    dataset = load_dataset(args.scenewalk_dataset, split=args.scenewalk_split, streaming=True)
    if args.scenewalk_shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=args.scenewalk_shuffle_buffer, seed=args.seed)

    yielded_rows = 0
    for row_index, row in enumerate(dataset):
        if row_index < args.scenewalk_start_row:
            continue
        if row_index % args.scenewalk_worker_count != args.scenewalk_worker_index:
            continue
        entry = scenewalk_entry_from_row(row)
        if entry is None:
            continue
        yielded_rows += 1
        yield entry
        if args.scenewalk_max_rows is not None and yielded_rows >= args.scenewalk_max_rows:
            break


def concept_terms(source: dict[str, Any], limit: int = 12) -> list[str]:
    phrases: list[str] = []
    title = clean_title(source.get("title"))
    if title:
        phrases.append(title)
    for tag in source.get("tags") or []:
        phrases.append(clean_title(tag))
    for category in source.get("categories") or []:
        category = clean_title(category)
        if category and category not in {clean_title(item) for item in NON_VISUAL_CONCEPT_CATEGORIES}:
            phrases.append(category)

    concepts: list[str] = []
    for phrase in phrases:
        concept = normalize_concept_phrase(phrase)
        if not concept or concept in concepts:
            continue
        concepts.append(concept)
        if len(concepts) >= limit:
            break
    return concepts


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def clean_title(value: Any) -> str:
    text = clean_text(value)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^\s*things\s+you\s+need\s+to\s+know\s+about\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*everything\s+you\s+need\s+to\s+know\s+about\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*what\s+you\s+need\s+to\s+know\s+about\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]*(?:creative commons|cc\s*by|reuse allowed)[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:creative commons|cc\s*by|reuse allowed|free hd video footage)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:4k|1080p|720p|hd)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+\b[Ii]\b\s+", " ", text)
    text = re.sub(r"[|_/]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -:!.,")


def sentence_bound_snippet(value: Any, max_chars: int) -> str:
    text = clean_text(value)
    text = re.sub(r"https?://\S+", "", text).strip()
    text = re.split(
        r"\b(?:download link|dimensions|video codec|color profile|duration|fps|music promoted by|audio library)\b\s*:?",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    if len(text) <= max_chars:
        return text
    candidate = text[:max_chars].rsplit(" ", 1)[0].strip()
    sentence_end = max(candidate.rfind("."), candidate.rfind("!"), candidate.rfind("?"))
    if sentence_end >= max_chars // 3:
        return candidate[: sentence_end + 1].strip()
    return candidate.rstrip(" ,;:-") + "."


def normalize_concept_phrase(value: Any) -> str:
    phrase = clean_title(value)
    if not phrase:
        return ""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", phrase)
    filtered = []
    for word in words:
        lower = word.lower().strip("'")
        if lower in GENERIC_CONCEPT_WORDS:
            continue
        if len(lower) <= 2 and not lower.isdigit():
            continue
        filtered.append(word)
    if not filtered:
        return ""
    if filtered and filtered[-1].lower() in {"like", "for"}:
        filtered = filtered[:-1]
    if not filtered:
        return ""
    if len(filtered) > 6:
        filtered = filtered[:6]
    concept = " ".join(filtered).strip()
    if not concept or concept.lower() in GENERIC_CONCEPT_WORDS:
        return ""
    return concept


def canonicalize_ideogram_json_caption(caption: dict[str, Any], *, source_prompt: str = "") -> OrderedDict[str, Any]:
    high_level = clean_text(caption.get("high_level_description") or caption.get("description") or source_prompt)
    style = caption.get("style_description")
    composition = caption.get("compositional_deconstruction")
    if not isinstance(style, dict):
        style = {}
    if not isinstance(composition, dict):
        composition = {}

    ordered_style: OrderedDict[str, Any] = OrderedDict()
    ordered_style["aesthetics"] = clean_text(style.get("aesthetics") or "natural, documentary, creator-shot")
    ordered_style["lighting"] = clean_text(style.get("lighting") or "available light from the recorded scene")
    ordered_style["photo"] = clean_text(style.get("photo") or "real-world video frames with natural perspective")
    ordered_style["medium"] = clean_text(style.get("medium") or "video")
    palette = [str(color).upper() for color in style.get("color_palette", []) if HEX_COLOR_RE.fullmatch(str(color))]
    if palette:
        ordered_style["color_palette"] = palette[:16]

    elements: list[OrderedDict[str, Any]] = []
    raw_elements = composition.get("elements")
    if isinstance(raw_elements, list):
        for raw_element in raw_elements:
            if not isinstance(raw_element, dict):
                continue
            element: OrderedDict[str, Any] = OrderedDict()
            element_type = raw_element.get("type") if raw_element.get("type") in {"obj", "text"} else "obj"
            element["type"] = element_type
            bbox = raw_element.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                element["bbox"] = [int(float(coord)) for coord in bbox]
            frame_idx = raw_element.get("frame_idx")
            if isinstance(frame_idx, int) and frame_idx >= 0:
                element["frame_idx"] = frame_idx
            frame_time = raw_element.get("frame_time")
            if isinstance(frame_time, (int, float)) and frame_time >= 0:
                element["frame_time"] = round(float(frame_time), 3)
            if element_type == "text":
                element["text"] = clean_text(raw_element.get("text"))
            desc = clean_text(raw_element.get("desc") or raw_element.get("description") or raw_element.get("label"))
            if not desc:
                continue
            element["desc"] = desc
            element_palette = [
                str(color).upper() for color in raw_element.get("color_palette", []) if HEX_COLOR_RE.fullmatch(str(color))
            ]
            if element_palette:
                element["color_palette"] = element_palette[:5]
            elements.append(element)
    if not elements:
        elements.append(OrderedDict([("type", "obj"), ("desc", high_level or "licensed YouTube video")]))

    ordered: OrderedDict[str, Any] = OrderedDict()
    ordered["high_level_description"] = high_level
    ordered["style_description"] = ordered_style
    ordered["compositional_deconstruction"] = OrderedDict(
        [
            ("background", clean_text(composition.get("background") or high_level)),
            ("elements", elements),
        ]
    )
    return ordered


def deterministic_caption(source: dict[str, Any]) -> OrderedDict[str, Any]:
    title = clean_title(source.get("title")) or "Untitled YouTube video"
    channel = clean_title(source.get("channel")) or "an independent YouTube creator"
    categories = ", ".join(source.get("categories") or [])
    tag_terms = [normalize_concept_phrase(tag) for tag in source.get("tags") or []]
    tag_terms = [tag for tag in tag_terms if tag][:10]
    tags = ", ".join(tag_terms)
    description = sentence_bound_snippet(source.get("description"), 360)

    high_level = f"{title}. A licensed creator-shot video by {channel}."
    if categories:
        high_level += f" Broad category: {categories}."
    if description:
        high_level += f" Metadata summary: {description}"

    concepts = concept_terms(source)
    elements = [{"type": "obj", "desc": concept.replace("-", " ")} for concept in concepts]
    if not elements:
        elements = [{"type": "obj", "desc": title}]
    if len(elements) < 2 and categories:
        for category in source.get("categories") or []:
            concept = normalize_concept_phrase(category)
            if concept and concept.lower() not in {element["desc"].lower() for element in elements}:
                elements.append({"type": "obj", "desc": concept})
                break
    if all(element["desc"].lower() in GENERIC_CONCEPT_WORDS for element in elements):
        raise ValueError("metadata did not produce useful visual concepts")

    return canonicalize_ideogram_json_caption(
        {
            "high_level_description": high_level,
            "style_description": {
                "aesthetics": "natural, documentary, creator-shot",
                "lighting": "available light from the recorded scene",
                "photo": "real-world video frames with natural perspective",
                "medium": "video",
            },
            "compositional_deconstruction": {
                "background": f"Creator-shot video context inferred from title, description, tags, and channel metadata. Visual tags: {tags or title}",
                "elements": elements,
            },
        },
        source_prompt=high_level,
    )


def call_vllm_labeler(source: dict[str, Any], args: argparse.Namespace) -> OrderedDict[str, Any]:
    prompt = (
        "Convert this YouTube metadata into a strict Ideogram 4 JSON caption. "
        "Use only visible or strongly implied concepts from the metadata. "
        "Do not invent brands, celebrities, copyrighted characters, or private identity. "
        "Return only JSON with high_level_description, style_description, and "
        "compositional_deconstruction.\n\n" + json.dumps(source, ensure_ascii=False, indent=2)
    )
    payload = {
        "model": args.labeler_model,
        "messages": [
            {
                "role": "system",
                "content": "You write concise, schema-valid Ideogram 4 JSON captions for video dataset samples.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    request = Request(
        args.labeler_base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {args.labeler_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"vLLM labeler request failed: {exc}") from exc

    content = data["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.DOTALL).strip()
    parsed = json.loads(content, object_pairs_hook=OrderedDict)
    return canonicalize_ideogram_json_caption(parsed, source_prompt=str(source.get("title") or ""))


def build_caption(source: dict[str, Any], args: argparse.Namespace) -> OrderedDict[str, Any]:
    if not args.labeler_model:
        return deterministic_caption(source)
    return call_vllm_labeler(source, args)


def stream_duration_seconds(container: Any, stream: Any) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration / 1_000_000)
    return 0.0


def stream_fps(stream: Any) -> float | None:
    rate = stream.average_rate or stream.base_rate
    if rate is None:
        return None
    value = float(rate)
    return value if value > 0 else None


def frame_time_seconds(frame: Any, stream: Any) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None and stream.time_base is not None:
        return float(frame.pts * stream.time_base)
    return 0.0


def timestamp_frame_idx(frame_time: float, fps: float | None) -> int:
    if fps is None:
        return 0
    return max(0, int(round(frame_time * fps)))


def extract_label_frames(video_path: Path, frame_count: int) -> list[VideoFrameSample]:
    import av

    frame_count = max(1, frame_count)
    samples: list[VideoFrameSample] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        duration = stream_duration_seconds(container, stream)
        fps = stream_fps(stream)
        if duration <= 0:
            target_times = [0.0]
        else:
            target_times = [duration * (index + 1) / (frame_count + 1) for index in range(frame_count)]

        for sample_idx, target_time in enumerate(target_times):
            if stream.time_base is not None:
                seek_offset = int(target_time / float(stream.time_base))
                container.seek(seek_offset, stream=stream, any_frame=False, backward=True)
            else:
                container.seek(int(target_time * 1_000_000), any_frame=False, backward=True)
            selected_frame = None
            for frame in container.decode(stream):
                selected_frame = frame
                if frame_time_seconds(frame, stream) >= target_time:
                    break
            if selected_frame is None:
                continue
            actual_time = frame_time_seconds(selected_frame, stream)
            samples.append(
                VideoFrameSample(
                    image=selected_frame.to_image().convert("RGB"),
                    frame_idx=timestamp_frame_idx(actual_time, fps),
                    frame_time=actual_time,
                    sample_idx=sample_idx,
                )
            )
    if not samples:
        raise ValueError(f"could not extract label frames from {video_path}")
    return samples


def load_qwen_vl(args: argparse.Namespace):
    global _QWEN_VL_MODEL, _QWEN_VL_PROCESSOR
    if _QWEN_VL_MODEL is not None and _QWEN_VL_PROCESSOR is not None:
        return _QWEN_VL_MODEL, _QWEN_VL_PROCESSOR

    import torch
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    model_id = args.qwen_vl_model
    _QWEN_VL_PROCESSOR = Qwen3VLProcessor.from_pretrained(model_id)
    _QWEN_VL_MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    return _QWEN_VL_MODEL, _QWEN_VL_PROCESSOR


def call_qwen_vl_labeler(source: dict[str, Any], video_path: Path, args: argparse.Namespace) -> OrderedDict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info

    model, processor = load_qwen_vl(args)
    metadata = {
        "title": source.get("title"),
        "description": sentence_bound_snippet(source.get("description"), 700),
        "channel": source.get("channel"),
        "categories": source.get("categories") or [],
        "tags": source.get("tags") or [],
        "duration": source.get("duration"),
        "license": source.get("license"),
    }
    prompt = (
        "You are labeling a licensed YouTube video dataset for generative video training. "
        "Use the provided video as the primary evidence and use metadata only as weak context. "
        "Ignore creator calls-to-action, licensing text, URLs, codec details, stock-footage boilerplate, and download text. "
        "Describe visible motion, camera movement, action changes, and temporal consistency; do not write a single-image caption. "
        "Return only one strict JSON object with this schema: "
        "{"
        '"high_level_description": string, '
        '"style_description": {"aesthetics": string, "lighting": string, "photo": string, "medium": "video"}, '
        '"compositional_deconstruction": {"background": string, "elements": ['
        '{"type": "obj", "desc": string}]}'
        "}. "
        "Elements must be visible subjects, objects, places, actions, or scene concepts from the frames. "
        "Do not include generic words like video, footage, creative commons, channel, stock, download, or license. "
        "Do not identify private people by name. Keep the JSON concise.\n\n"
        f"Metadata context:\n{json.dumps(metadata, ensure_ascii=False, indent=2)}"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "fps": args.qwen_vl_video_fps,
                    "max_pixels": args.qwen_vl_video_max_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    if isinstance(video_kwargs.get("fps"), list) and len(video_kwargs["fps"]) == 1:
        video_kwargs["fps"] = video_kwargs["fps"][0]
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        **video_kwargs,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.qwen_vl_max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[-1]
    decoded = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)[0].strip()
    if decoded.startswith("```"):
        decoded = re.sub(r"^```(?:json)?\s*|\s*```$", "", decoded, flags=re.DOTALL).strip()
    parsed = json.loads(decoded, object_pairs_hook=OrderedDict)
    return canonicalize_ideogram_json_caption(parsed, source_prompt=str(source.get("title") or ""))


def load_sam3_processor(args: argparse.Namespace):
    global _SAM3_PROCESSOR
    if _SAM3_PROCESSOR is not None:
        return _SAM3_PROCESSOR

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    checkpoint_path = str(args.sam3_checkpoint) if args.sam3_checkpoint else None
    model = build_sam3_image_model(
        device=args.sam3_device,
        checkpoint_path=checkpoint_path,
        load_from_HF=checkpoint_path is None,
    )
    processor = Sam3Processor(model)
    processor.set_confidence_threshold(args.sam3_score_threshold)
    _SAM3_PROCESSOR = processor
    return processor


def tensorish_to_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else []


def bbox_xyxy_to_ideogram(box: list[float], width: int, height: int) -> list[int] | None:
    if len(box) != 4 or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x1 >= x2 or y1 >= y2:
        return None
    return [
        round(y1 / height * 1000),
        round(x1 / width * 1000),
        round(y2 / height * 1000),
        round(x2 / width * 1000),
    ]


def sam3_prompt_candidates(desc: str) -> list[str]:
    candidates = [desc]
    shortened = re.split(r",|\bwith\b|\bincluding\b|\bdisplayed\b|\bproviding\b", desc, maxsplit=1, flags=re.IGNORECASE)[0]
    shortened = clean_text(shortened)
    if shortened and shortened.lower() != desc.lower():
        candidates.append(shortened)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", shortened or desc)
    compact = " ".join(words[:6])
    if compact and compact.lower() not in {candidate.lower() for candidate in candidates}:
        candidates.append(compact)
    return candidates


def best_sam3_box_for_prompt(
    processor: Any, samples: list[VideoFrameSample], prompt: str, args: argparse.Namespace
) -> dict[str, Any] | None:
    import contextlib

    import torch

    best: dict[str, Any] | None = None
    for sample in samples:
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if str(args.sam3_device).startswith("cuda")
            else contextlib.nullcontext()
        )
        with autocast_context:
            state = processor.set_image(sample.image)
            output = processor.set_text_prompt(prompt=prompt, state=state)
        boxes = tensorish_to_list(output.get("boxes"))
        scores = tensorish_to_list(output.get("scores"))
        if boxes and boxes and isinstance(boxes[0], (int, float)):
            boxes = [boxes]
        if not scores:
            scores = [1.0] * len(boxes)
        for box, score in zip(boxes, scores):
            if isinstance(score, list):
                score = score[0] if score else 0.0
            score = float(score)
            if score < args.sam3_score_threshold:
                continue
            bbox = bbox_xyxy_to_ideogram(box, sample.image.width, sample.image.height)
            if bbox is None:
                continue
            if best is None or score > best["score"]:
                best = {
                    "bbox": bbox,
                    "score": score,
                    "frame_idx": sample.frame_idx,
                    "frame_time": sample.frame_time,
                    "sample_idx": sample.sample_idx,
                }
    return best


def best_sam3_box_for_desc(
    processor: Any, samples: list[VideoFrameSample], desc: str, args: argparse.Namespace
) -> dict[str, Any] | None:
    for prompt in sam3_prompt_candidates(desc):
        match = best_sam3_box_for_prompt(processor, samples, prompt, args)
        if match is not None:
            match["prompt"] = prompt
            return match
    return None


def add_sam3_bboxes(
    caption: OrderedDict[str, Any], video_path: Path, args: argparse.Namespace
) -> tuple[OrderedDict[str, Any], dict[str, Any]]:
    processor = load_sam3_processor(args)
    samples = extract_label_frames(video_path, args.sam3_box_frames)
    composition = caption["compositional_deconstruction"]
    grounded_elements: list[OrderedDict[str, Any]] = []
    missing: list[str] = []
    grounded = 0

    for element in composition["elements"]:
        desc = clean_text(element.get("desc"))
        if not desc:
            continue
        match = best_sam3_box_for_desc(processor, samples, desc, args)
        if match is None:
            missing.append(desc)
            if args.no_require_bboxes:
                grounded_elements.append(element)
            continue
        element = OrderedDict(element)
        element["bbox"] = match["bbox"]
        element["frame_idx"] = int(match["frame_idx"])
        element["frame_time"] = round(float(match["frame_time"]), 3)
        grounded_elements.append(element)
        grounded += 1

    if not grounded_elements:
        raise ValueError("SAM 3 did not ground any caption elements")
    if grounded < args.sam3_min_grounded_elements and not args.no_require_bboxes:
        raise ValueError(f"SAM 3 grounded only {grounded} elements; minimum is {args.sam3_min_grounded_elements}")
    attempted = grounded + len(missing)
    grounded_ratio = grounded / attempted if attempted else 0.0
    if grounded_ratio < args.sam3_min_grounded_ratio and not args.no_require_bboxes:
        raise ValueError(f"SAM 3 grounded ratio {grounded_ratio:.2f}; minimum is {args.sam3_min_grounded_ratio:.2f}")

    composition["elements"] = grounded_elements
    return canonicalize_ideogram_json_caption(caption), {
        "sam3_grounded_elements": grounded,
        "sam3_missing_elements": missing,
        "sam3_box_frames": len(samples),
        "sam3_grounded_ratio": grounded_ratio,
    }


def verify_caption(caption: OrderedDict[str, Any], *, require_bboxes: bool = False) -> None:
    high_level = caption.get("high_level_description")
    if not isinstance(high_level, str) or not high_level:
        raise ValueError("caption missing high_level_description")
    caption_text = json.dumps(caption, ensure_ascii=False)
    if STATIC_VIDEO_RE.search(caption_text):
        raise ValueError("caption describes static, text-heavy, or courtroom/presentation content")
    style = caption.get("style_description")
    if not isinstance(style, dict):
        raise ValueError("caption missing style_description")
    if "photo" not in style and "art_style" not in style:
        raise ValueError("caption style_description needs photo or art_style")
    composition = caption.get("compositional_deconstruction")
    if not isinstance(composition, dict):
        raise ValueError("caption missing compositional_deconstruction")
    background = composition.get("background")
    if not isinstance(background, str) or not background:
        raise ValueError("caption missing compositional_deconstruction.background")
    elements = composition.get("elements")
    if not isinstance(elements, list) or not elements:
        raise ValueError("caption compositional_deconstruction.elements must be non-empty")
    useful_elements = 0
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            raise ValueError(f"caption element {index} must be an object")
        if element.get("type") not in {"obj", "text"}:
            raise ValueError(f"caption element {index} has invalid type")
        desc = str(element.get("desc") or "").strip()
        if not desc:
            raise ValueError(f"caption element {index} must have a non-empty desc")
        desc_words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", desc.lower())
        meaningful_words = [word for word in desc_words if word not in GENERIC_CONCEPT_WORDS]
        if meaningful_words:
            useful_elements += 1
        bbox = element.get("bbox")
        if bbox is None and require_bboxes:
            raise ValueError(f"caption element {index} missing required bbox")
        if bbox is None:
            continue
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(value, int) for value in bbox):
            raise ValueError(f"caption element {index} bbox must be four integers")
        if not all(0 <= value <= 1000 for value in bbox):
            raise ValueError(f"caption element {index} bbox values must be in [0, 1000]")
        ymin, xmin, ymax, xmax = bbox
        if ymin > ymax or xmin > xmax:
            raise ValueError(f"caption element {index} bbox has inverted coordinates")
        frame_idx = element.get("frame_idx")
        if frame_idx is not None and (not isinstance(frame_idx, int) or frame_idx < 0):
            raise ValueError(f"caption element {index} frame_idx must be a non-negative integer")
        frame_time = element.get("frame_time")
        if frame_time is not None and (
            not isinstance(frame_time, (int, float)) or isinstance(frame_time, bool) or frame_time < 0
        ):
            raise ValueError(f"caption element {index} frame_time must be a non-negative number")
    if useful_elements == 0:
        raise ValueError("caption has no useful visual concept elements")
    if len(high_level) > 900:
        raise ValueError("caption high_level_description is too long")


def ytdlp_auth_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if args.yt_dlp_cookie_file:
        options["cookiefile"] = str(args.yt_dlp_cookie_file)
    if args.yt_dlp_cookies_from_browser:
        options["cookiesfrombrowser"] = (args.yt_dlp_cookies_from_browser,)
    if args.yt_dlp_js_runtime:
        runtime, _, path = str(args.yt_dlp_js_runtime).partition(":")
        if not runtime:
            raise ValueError("--yt-dlp-js-runtime must include a runtime name")
        options["js_runtimes"] = {runtime.lower(): {"path": path or None}}
    return options


def is_youtube_bot_challenge(error: Any) -> bool:
    text = str(error).lower()
    return "confirm you're not a bot" in text or "confirm you’re not a bot" in text or "sign in to confirm" in text


def is_youtube_rate_limited(error: Any) -> bool:
    text = str(error).lower()
    return "rate-limited by youtube" in text or "this content isn't available, try again later" in text


def bot_challenge_flag_path(output_dir: Path) -> Path:
    return output_dir / ".youtube_bot_challenge"


def rate_limit_flag_path(output_dir: Path) -> Path:
    return output_dir / ".youtube_rate_limited"


def write_bot_challenge_flag(output_dir: Path, error: Any) -> None:
    bot_challenge_flag_path(output_dir).write_text(
        json.dumps(
            {
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "error": str(error),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_rate_limit_flag(output_dir: Path, error: Any) -> None:
    rate_limit_flag_path(output_dir).write_text(
        json.dumps(
            {
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "error": str(error),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def raise_for_youtube_pause(
    error: Any, output_dir: Path, *, bot_challenge_count: int, rate_limit_count: int, args: argparse.Namespace
) -> None:
    if is_youtube_bot_challenge(error) and bot_challenge_count >= args.bot_challenge_threshold:
        write_bot_challenge_flag(output_dir, error)
        raise SystemExit(
            f"YouTube bot challenge reached {bot_challenge_count} consecutive failures; "
            f"wrote {bot_challenge_flag_path(output_dir)}"
        )
    if is_youtube_rate_limited(error) and rate_limit_count >= args.rate_limit_threshold:
        write_rate_limit_flag(output_dir, error)
        raise SystemExit(
            f"YouTube rate limit reached {rate_limit_count} consecutive failures; "
            f"wrote {rate_limit_flag_path(output_dir)}"
        )


def search_entries(query: str, count: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    import yt_dlp

    options = {
        "extract_flat": False,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "skip_download": True,
        "ignoreerrors": True,
        **ytdlp_auth_options(args),
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(f"ytsearch{count}:{query}", download=False)
    return [entry for entry in info.get("entries", []) if isinstance(entry, dict)]


def format_selector(args: argparse.Namespace) -> str:
    height_filter = f"[height>={args.min_height}][height<={args.max_height}]"
    return (
        f"b{height_filter}[protocol=m3u8_native][ext=mp4]/"
        f"bv*{height_filter}[protocol=m3u8_native][ext=mp4]+ba[protocol=m3u8_native]/"
        f"bv*{height_filter}[ext=mp4]+ba[ext=m4a]/"
        f"b{height_filter}[ext=mp4]/"
        f"bv*{height_filter}+ba/"
        f"b{height_filter}"
    )


def probe_video_resolution(video_path: Path) -> tuple[int | None, int | None]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        return None, None
    width = streams[0].get("width")
    height = streams[0].get("height")
    return (int(width) if width else None, int(height) if height else None)


def resolution_bucket(height: int | None, buckets: list[int]) -> int:
    if height is None:
        return buckets[0]
    eligible = [bucket for bucket in buckets if height <= bucket]
    if eligible:
        return min(eligible)
    return max(buckets)


def parse_resolution_buckets(raw: str) -> list[int]:
    buckets = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not buckets:
        raise ValueError("--resolution-buckets must contain at least one integer")
    return buckets


def write_dataset_model_card(output_dir: Path, args: argparse.Namespace) -> None:
    license_mode = "youtube" if args.allow_non_cc else args.license_mode
    readme_path = output_dir / "README.md"
    if readme_path.exists():
        return
    readme_path.write_text(
        "\n".join(
            [
                "# SimpleTuner YouTube JSON Dataset",
                "",
                "This dataset is generated from YouTube search results and/or timestamped SceneWalk source rows for video model training experiments.",
                "",
                "## Licensing and Rights",
                "",
                f"- Collector license mode: `{license_mode}`.",
                f"- Collector source mode: `{args.source}`.",
                "- Each sample records the source URL, YouTube ID, channel metadata, and any YouTube-provided license metadata in `manifest.jsonl`.",
                "- SceneWalk-derived samples also record the source dataset name and clip start/end timestamps.",
                "- When YouTube does not expose an explicit license string, the manifest records `license_assumption: youtube_standard_or_unreported`.",
                "- The dataset builder does not assert that every source video is redistributable outside YouTube's terms or the uploader's rights.",
                "- Before publishing, mirroring, or using this dataset commercially, review the recorded source metadata and applicable platform and rights-holder terms.",
                "",
                "## Caption and Box Generation",
                "",
                "- Qwen3-VL is used on the video file itself to produce motion-aware Ideogram-style JSON captions.",
                "- SAM 3 is used to ground each retained visible element with `[ymin, xmin, ymax, xmax]` boxes in 0-1000 integer coordinates.",
                "- Samples whose visible elements cannot be grounded by SAM 3 are rejected unless `--no-require-bboxes` is set.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def preflight_sam3(args: argparse.Namespace) -> None:
    if args.no_require_bboxes:
        return
    if args.sam3_checkpoint:
        if not args.sam3_checkpoint.is_file():
            raise SystemExit(f"--sam3-checkpoint does not exist: {args.sam3_checkpoint}")
        return
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    try:
        hf_hub_download(repo_id="facebook/sam3", filename="config.json")
    except GatedRepoError as exc:
        raise SystemExit(
            "SAM 3 bbox generation is required, but this Hugging Face account cannot access facebook/sam3. "
            "Request access on Hugging Face or pass --sam3-checkpoint with a local checkpoint path."
        ) from exc


def download_video(entry: dict[str, Any], sample_id: str, output_dir: Path, args: argparse.Namespace) -> Path:
    import yt_dlp
    from yt_dlp.utils import download_range_func

    url = entry.get("webpage_url") or entry.get("url")
    if not url:
        raise ValueError("entry has no downloadable URL")

    temp_template = str(output_dir / f"{sample_id}.%(ext)s")
    options = {
        "format": format_selector(args),
        "outtmpl": temp_template,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "merge_output_format": "mp4",
        "postprocessors": [{"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}],
        **ytdlp_auth_options(args),
    }
    clip_start = entry.get("_clip_start")
    clip_end = entry.get("_clip_end")
    if isinstance(clip_start, (int, float)) and isinstance(clip_end, (int, float)) and clip_end > clip_start:
        options["download_ranges"] = download_range_func(None, [(float(clip_start), float(clip_end))])
        options["force_keyframes_at_cuts"] = True
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])

    target = output_dir / f"{sample_id}.mp4"
    if target.exists():
        return target
    candidates = sorted(output_dir.glob(f"{sample_id}.*"))
    if not candidates:
        raise FileNotFoundError(f"yt-dlp did not create an output file for {sample_id}")
    if candidates[0].suffix.lower() != ".mp4":
        shutil.move(str(candidates[0]), target)
    return target


def extract_preview_frames(video_path: Path, preview_dir: Path, frame_count: int, max_width: int) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    pattern = preview_dir / f"{video_path.stem}_%02d.jpg"
    video_filter = "fps=1/5"
    if max_width > 0:
        video_filter += f",scale='min({max_width},iw)':-2"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        video_filter,
        "-frames:v",
        str(frame_count),
        "-q:v",
        "3",
        str(pattern),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def existing_sample_count(output_dir: Path) -> int:
    return len(list(output_dir.rglob("*.mp4")))


def main() -> int:
    args = parse_args()
    args.search_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    buckets = parse_resolution_buckets(args.resolution_buckets)
    if not args.qwen_vl_model and not args.labeler_model and not args.metadata_only_labels:
        raise SystemExit(
            "Refusing to write metadata-only captions by default. Pass --qwen-vl-model for local visual labeling, "
            "start an OpenAI-compatible labeler and pass --labeler-model, or opt in explicitly with "
            "--metadata-only-labels for discovery/smoke tests."
        )
    if not args.dry_run:
        preflight_sam3(args)
    write_dataset_model_card(args.output_dir, args)

    query_path = args.search_dir / "queries.txt"
    candidates_path = args.search_dir / "candidates.jsonl"
    rejected_path = args.search_dir / "rejected.jsonl"
    manifest_path = args.output_dir / "manifest.jsonl"
    preview_dir = args.output_dir / "_preview_frames"

    queries: list[str] = []
    if args.source == "youtube_search":
        queries = load_queries(args)
        if args.queries is None or not query_path.exists():
            query_path.write_text("\n".join(queries) + "\n", encoding="utf-8")
        print(f"loaded {len(queries)} search queries", flush=True)
    else:
        print(
            f"streaming {args.scenewalk_dataset} {args.scenewalk_split} "
            f"worker={args.scenewalk_worker_index}/{args.scenewalk_worker_count}",
            flush=True,
        )

    seen_ids = load_seen_ids(candidates_path, rejected_path, manifest_path)
    seen_ids.update(load_reserved_ids(args.output_dir / "seen_youtube_ids.txt"))
    downloaded = 0
    accepted = 0
    bot_challenge_count = 0
    rate_limit_count = 0

    def handle_entries(entries: Iterable[dict[str, Any]], query: str) -> None:
        nonlocal accepted, bot_challenge_count, downloaded, rate_limit_count
        for entry in entries:
            youtube_id = str(entry.get("id") or "")
            source_key = str(entry.get("_source_key") or youtube_id)
            if not youtube_id or not source_key or source_key in seen_ids:
                continue
            seen_ids.add(source_key)
            entry["_search_query"] = query

            reason = rejection_reason(entry, args)
            if reason:
                jsonl_append(
                    rejected_path,
                    {
                        "id": youtube_id,
                        "title": entry.get("title"),
                        "license": entry.get("license"),
                        "duration": entry.get("duration"),
                        "search_query": query,
                        "reason": reason,
                    },
                )
                continue
            title_for_dedupe = entry.get("title") if args.source == "youtube_search" else None
            if not args.dry_run and not reserve_candidate_identity(args.output_dir, source_key, title_for_dedupe):
                continue

            sample_id = str(uuid.uuid4())
            source = compact_source_metadata(entry, query, sample_id)
            jsonl_append(candidates_path, source)

            if args.dry_run:
                try:
                    if args.qwen_vl_model:
                        caption_status = "video_caption_and_sam3_bbox_pending_download"
                    else:
                        caption = build_caption(source, args)
                        verify_caption(caption)
                        caption_status = "caption_ok"
                except Exception as exc:
                    jsonl_append(rejected_path, source | {"reason": "caption_failed", "error": str(exc)})
                    continue
                print(f"candidate: {source['title']} [{source['license']}; {caption_status}]", flush=True)
                accepted += 1
                if args.download_limit is not None and accepted >= args.download_limit:
                    return
                if accepted >= args.target_count:
                    return
                continue

            try:
                staging_path = download_video(entry, sample_id, args.output_dir, args)
                width, height = probe_video_resolution(staging_path)
                if height is not None and height < args.min_height:
                    staging_path.unlink(missing_ok=True)
                    jsonl_append(
                        rejected_path, source | {"reason": "downloaded_below_min_height", "width": width, "height": height}
                    )
                    continue
                bucket = resolution_bucket(height, buckets)
                bucket_dir = args.output_dir / f"b{bucket}"
                bucket_dir.mkdir(parents=True, exist_ok=True)
                video_path = bucket_dir / staging_path.name
                shutil.move(str(staging_path), video_path)
                try:
                    if args.qwen_vl_model:
                        caption = call_qwen_vl_labeler(source, video_path, args)
                    else:
                        caption = build_caption(source, args)
                    bbox_stats = {}
                    if not args.no_require_bboxes:
                        caption, bbox_stats = add_sam3_bboxes(caption, video_path, args)
                    verify_caption(caption, require_bboxes=not args.no_require_bboxes)
                except Exception as exc:
                    video_path.unlink(missing_ok=True)
                    jsonl_append(rejected_path, source | {"reason": "caption_failed", "error": str(exc)})
                    continue
                sidecar_path = bucket_dir / f"{sample_id}.json"
                sidecar_path.write_text(json.dumps(caption, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                if args.extract_preview_frames:
                    extract_preview_frames(
                        video_path, preview_dir / f"b{bucket}", args.preview_frames, args.preview_max_width
                    )
                jsonl_append(
                    manifest_path,
                    source
                    | {
                        "video_path": str(video_path.relative_to(args.output_dir)),
                        "caption_path": str(sidecar_path.relative_to(args.output_dir)),
                        "width": width,
                        "height": height,
                        "resolution_bucket": bucket,
                        "license_mode": "youtube" if args.allow_non_cc else args.license_mode,
                        "caption_model": args.qwen_vl_model or args.labeler_model or "metadata",
                        "bbox_model": "sam3" if not args.no_require_bboxes else None,
                        **bbox_stats,
                    },
                )
                downloaded += 1
                accepted += 1
                print(
                    f"downloaded: {video_path.relative_to(args.output_dir)} {width}x{height} {source['title']}", flush=True
                )
            except Exception as exc:
                jsonl_append(rejected_path, source | {"reason": "download_failed", "error": str(exc)})
                if is_youtube_bot_challenge(exc):
                    bot_challenge_count += 1
                    rate_limit_count = 0
                    raise_for_youtube_pause(
                        exc,
                        args.output_dir,
                        bot_challenge_count=bot_challenge_count,
                        rate_limit_count=rate_limit_count,
                        args=args,
                    )
                elif is_youtube_rate_limited(exc):
                    rate_limit_count += 1
                    bot_challenge_count = 0
                    raise_for_youtube_pause(
                        exc,
                        args.output_dir,
                        bot_challenge_count=bot_challenge_count,
                        rate_limit_count=rate_limit_count,
                        args=args,
                    )
                else:
                    bot_challenge_count = 0
                    rate_limit_count = 0

            if args.download_limit is not None and accepted >= args.download_limit:
                return
            if existing_sample_count(args.output_dir) >= args.target_count:
                return
            time.sleep(args.sleep)

    if args.source == "youtube_search":
        for query in queries:
            if existing_sample_count(args.output_dir) >= args.target_count:
                break
            if args.download_limit is not None and accepted >= args.download_limit:
                break
            print(f"search: {query}", flush=True)
            try:
                entries = search_entries(query, args.results_per_query, args)
            except Exception as exc:
                jsonl_append(rejected_path, {"search_query": query, "reason": "search_failed", "error": str(exc)})
                if is_youtube_bot_challenge(exc):
                    bot_challenge_count += 1
                    rate_limit_count = 0
                    raise_for_youtube_pause(
                        exc,
                        args.output_dir,
                        bot_challenge_count=bot_challenge_count,
                        rate_limit_count=rate_limit_count,
                        args=args,
                    )
                elif is_youtube_rate_limited(exc):
                    rate_limit_count += 1
                    bot_challenge_count = 0
                    raise_for_youtube_pause(
                        exc,
                        args.output_dir,
                        bot_challenge_count=bot_challenge_count,
                        rate_limit_count=rate_limit_count,
                        args=args,
                    )
                else:
                    bot_challenge_count = 0
                    rate_limit_count = 0
                continue
            if entries:
                bot_challenge_count = 0
                rate_limit_count = 0
            handle_entries(entries, query)
    else:
        handle_entries(stream_scenewalk_entries(args), "scenewalk")

    print(f"complete: downloaded={downloaded} total_mp4={existing_sample_count(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
