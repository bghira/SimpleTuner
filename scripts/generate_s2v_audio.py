#!/usr/bin/env python3
"""
Extract audio from video files for Wan 2.2 S2V training.

This script processes video files to extract their audio tracks into a separate
directory, creating the paired video/audio dataset structure required for S2V training.

Usage:
    python scripts/generate_s2v_audio.py --input-dir datasets/videos --output-dir datasets/audio
    python scripts/generate_s2v_audio.py --input-dir datasets/videos --output-dir datasets/audio --strip-audio

Requirements:
    - ffmpeg must be installed and available in PATH
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
AUDIO_FORMAT = "wav"  # Output format for extracted audio


def check_ffmpeg():
    """Check if ffmpeg is available."""
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found in PATH. Please install ffmpeg first.")
        print("  Ubuntu/Debian: apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        sys.exit(1)


def get_video_files(input_dir: Path) -> list[Path]:
    """Find all video files in the input directory."""
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(input_dir.glob(f"*{ext}"))
        video_files.extend(input_dir.glob(f"*{ext.upper()}"))
    return sorted(video_files)


def extract_audio(video_path: Path, output_path: Path) -> bool:
    """
    Extract audio from a video file.

    Returns True if successful, False otherwise.
    """
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",  # No video
        "-acodec",
        "pcm_s16le",  # PCM 16-bit for WAV
        "-ar",
        "16000",  # 16kHz sample rate (Wav2Vec2 native rate)
        "-ac",
        "1",  # Mono
        "-y",  # Overwrite output
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            # Check if video has no audio stream
            if "does not contain any stream" in result.stderr or "Output file is empty" in result.stderr:
                return False
            print(f"  Warning: ffmpeg returned non-zero exit code for {video_path.name}")
            print(f"  stderr: {result.stderr[:500]}")
            return False
        return True
    except Exception as e:
        print(f"  Error extracting audio from {video_path.name}: {e}")
        return False


def strip_audio_from_video(video_path: Path) -> bool:
    """
    Remove audio track from a video file in-place.

    Returns True if successful, False otherwise.
    """
    temp_path = video_path.with_suffix(f".tmp{video_path.suffix}")

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-an",  # No audio
        "-c:v",
        "copy",  # Copy video stream without re-encoding
        "-y",  # Overwrite output
        str(temp_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f"  Warning: Failed to strip audio from {video_path.name}")
            if temp_path.exists():
                temp_path.unlink()
            return False

        # Replace original with stripped version
        temp_path.replace(video_path)
        return True
    except Exception as e:
        print(f"  Error stripping audio from {video_path.name}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio from video files for Wan 2.2 S2V training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract audio only (keep original videos unchanged)
  python scripts/generate_s2v_audio.py --input-dir datasets/videos --output-dir datasets/audio

  # Extract audio and remove it from source videos
  python scripts/generate_s2v_audio.py --input-dir datasets/videos --output-dir datasets/audio --strip-audio

  # Process specific format
  python scripts/generate_s2v_audio.py --input-dir datasets/videos --output-dir datasets/audio --format mp3
""",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save extracted audio files",
    )
    parser.add_argument(
        "--strip-audio",
        action="store_true",
        help="Remove audio from source video files after extraction",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=AUDIO_FORMAT,
        choices=["wav", "mp3", "flac", "ogg"],
        help="Output audio format (default: wav)",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    if not args.input_dir.is_dir():
        print(f"Error: Input path is not a directory: {args.input_dir}")
        sys.exit(1)

    # Check ffmpeg availability
    check_ffmpeg()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find video files
    video_files = get_video_files(args.input_dir)
    if not video_files:
        print(f"No video files found in {args.input_dir}")
        print(f"Supported formats: {', '.join(sorted(VIDEO_EXTENSIONS))}")
        sys.exit(1)

    print(f"Found {len(video_files)} video files in {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.strip_audio:
        print("Audio will be stripped from source videos after extraction")
    print()

    # Process each video
    extracted_count = 0
    stripped_count = 0
    no_audio_count = 0

    for i, video_path in enumerate(video_files, 1):
        audio_filename = f"{video_path.stem}.{args.format}"
        audio_path = args.output_dir / audio_filename

        print(f"[{i}/{len(video_files)}] Processing {video_path.name}...")

        # Extract audio
        if extract_audio(video_path, audio_path):
            extracted_count += 1
            print(f"  Extracted: {audio_filename}")

            # Optionally strip audio from video
            if args.strip_audio:
                if strip_audio_from_video(video_path):
                    stripped_count += 1
                    print(f"  Stripped audio from video")
        else:
            no_audio_count += 1
            print(f"  Skipped: No audio stream found")

    # Summary
    print()
    print("=" * 50)
    print("Summary:")
    print(f"  Videos processed: {len(video_files)}")
    print(f"  Audio extracted: {extracted_count}")
    print(f"  No audio stream: {no_audio_count}")
    if args.strip_audio:
        print(f"  Audio stripped: {stripped_count}")
    print()

    if extracted_count > 0:
        print("Next steps for S2V training:")
        print(f"  1. Video directory: {args.input_dir}")
        print(f"  2. Audio directory: {args.output_dir}")
        print("  3. Configure your multidatabackend.json with s2v_datasets linking")
        print("  4. See documentation/quickstart/WAN_S2V.md for full setup guide")


if __name__ == "__main__":
    main()
