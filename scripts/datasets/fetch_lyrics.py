#!/usr/bin/env python3
"""
Lyrics Fetcher for Audio Datasets
---------------------------------
This script scans a directory of audio files, reads their metadata (Artist/Title),
and attempts to fetch lyrics for them.

Priorities:
1. Embedded Lyrics (ID3/Vorbis tags)
2. Genius.com (requires GENIUS_ACCESS_TOKEN environment variable)

Outputs:
Writes a .lyrics file next to each audio file.

Usage:
    export GENIUS_ACCESS_TOKEN="your_token_here"
    python scripts/datasets/fetch_lyrics.py --dir datasets/my_music --sleep 5
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Optional dependencies
try:
    import mutagen
    from mutagen.flac import FLAC
    from mutagen.id3 import ID3
    from mutagen.mp4 import MP4
    from mutagen.oggvorbis import OggVorbis
except ImportError:
    print("Error: 'mutagen' is required. Install it via: pip install mutagen")
    sys.exit(1)

try:
    import lyricsgenius
except ImportError:
    print("Warning: 'lyricsgenius' not found. Online fetching will be disabled. Install via: pip install lyricsgenius")
    lyricsgenius = None

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aiff", ".opus"}


def scrape_genius_tokenless(artist, title):
    """
    Scrape Genius without an API token.
    Mimics browser behavior to find and parse lyrics.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    }

    # 1. Search
    search_url = "https://genius.com/api/search/multi"
    params = {"q": f"{artist} {title}", "per_page": 1}

    try:
        logger.info(f"Scraping Genius (Tokenless) for: {artist} - {title}")
        r = requests.get(search_url, params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()
        sections = data.get("response", {}).get("sections", [])
        hits = []
        for sec in sections:
            if sec.get("type") == "song":
                hits = sec.get("hits", [])
                break

        if not hits:
            return None

        song_url = hits[0]["result"]["url"]

        # 2. Fetch Song Page
        r_page = requests.get(song_url, headers=headers, timeout=10)
        if r_page.status_code != 200:
            return None

        # 3. Parse Lyrics
        soup = BeautifulSoup(r_page.text, "html.parser")

        # Genius has changed classes often. Look for data-lyrics-container
        lyrics_divs = soup.find_all("div", attrs={"data-lyrics-container": "true"})

        if not lyrics_divs:
            # Fallback for older layouts
            lyrics_div = soup.find("div", class_="lyrics")
            if lyrics_div:
                return lyrics_div.get_text(separator="\n")
            return None

        # Join multiple containers (e.g. verses separated by ads/images)
        lyrics_text = "\n".join([div.get_text(separator="\n") for div in lyrics_divs])
        return lyrics_text

    except Exception as e:
        logger.warning(f"Tokenless scrape failed: {e}")
        return None


def extract_metadata(filepath):
    """Extract Artist and Title from audio file tags."""
    artist = None
    title = None
    lyrics = None

    try:
        audio = mutagen.File(filepath)
        if audio is None:
            return None, None, None

        # MP3 (ID3)
        if isinstance(audio.tags, ID3):
            if "TPE1" in audio.tags:
                artist = str(audio.tags["TPE1"])
            if "TIT2" in audio.tags:
                title = str(audio.tags["TIT2"])
            # USLT: Unsynchronized Lyric Transcription
            for key in audio.tags.keys():
                if key.startswith("USLT"):
                    lyrics = str(audio.tags[key])
                    break

        # FLAC / Ogg
        elif isinstance(audio, (FLAC, OggVorbis)):
            if "artist" in audio:
                artist = audio["artist"][0]
            if "title" in audio:
                title = audio["title"][0]
            if "lyrics" in audio:
                lyrics = audio["lyrics"][0]
            elif "unsyncedlyrics" in audio:
                lyrics = audio["unsyncedlyrics"][0]

        # M4A (MP4)
        elif isinstance(audio, MP4):
            # ©ART, ©nam, ©lyr
            tags = audio.tags
            if tags:
                if "\xa9ART" in tags:
                    artist = tags["\xa9ART"][0]
                if "\xa9nam" in tags:
                    title = tags["\xa9nam"][0]
                if "\xa9lyr" in tags:
                    lyrics = tags["\xa9lyr"][0]

    except Exception as e:
        logger.warning(f"Could not read tags for {filepath}: {e}")

    return artist, title, lyrics


def fetch_from_genius(genius, artist, title):
    """Fetch lyrics from Genius.com."""
    if not genius or not artist or not title:
        return None

    try:
        logger.info(f"Searching Genius for: {artist} - {title}")
        song = genius.search_song(title, artist)
        if song:
            return song.lyrics
    except Exception as e:
        logger.error(f"Genius fetch error: {e}")

    return None


def clean_lyrics(text):
    """Simple cleanup for lyrics."""
    if not text:
        return None
    # Remove Genius specific headers like "Embed" or "Contributors" at the end if present
    # (lyricsgenius usually handles this, but basic cleanup is good)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Fetch lyrics for audio files.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing audio files.")
    parser.add_argument("--sleep", type=float, default=3.0, help="Seconds to sleep between online requests.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .lyrics files.")
    parser.add_argument("--token", type=str, default=os.environ.get("GENIUS_ACCESS_TOKEN"), help="Genius API Token.")

    args = parser.parse_args()

    root_dir = Path(args.dir)
    if not root_dir.exists():
        print(f"Directory not found: {root_dir}")
        return

    genius = None
    if lyricsgenius and args.token:
        genius = lyricsgenius.Genius(args.token)
        genius.verbose = False  # Quieter output
        genius.remove_section_headers = False  # Keep [Verse], [Chorus] structure - ACE-Step likes this!
    elif lyricsgenius:
        print("Notice: No Genius Token provided. Only local ID3 tags will be checked.")

    files = [p for p in root_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS]
    logger.info(f"Found {len(files)} audio files.")

    for filepath in files:
        lyrics_path = filepath.with_suffix(".lyrics")

        if lyrics_path.exists() and not args.overwrite:
            logger.debug(f"Skipping {filepath.name}, .lyrics exists.")
            continue

        artist, title, local_lyrics = extract_metadata(filepath)

        final_lyrics = None
        source = "None"

        # 1. Try Local Tags
        if local_lyrics:
            final_lyrics = local_lyrics
            source = "ID3/Local"

        # 2. Try Genius
        elif genius and artist and title:
            # Normalize filename junk if artist/title missing?
            # For now, rely on tags.
            final_lyrics = fetch_from_genius(genius, artist, title)
            source = "Genius API"
            time.sleep(args.sleep)  # Be nice

        # 3. Try Tokenless Scraper
        elif artist and title:
            final_lyrics = scrape_genius_tokenless(artist, title)
            source = "Genius Scraper"
            time.sleep(args.sleep)

        if final_lyrics:
            try:
                with open(lyrics_path, "w", encoding="utf-8") as f:
                    f.write(clean_lyrics(final_lyrics))
                logger.info(f"[{source}] Saved lyrics for: {filepath.name}")
            except Exception as e:
                logger.error(f"Failed to write lyrics: {e}")
        else:
            logger.warning(f"No lyrics found for: {filepath.name} (Artist: {artist}, Title: {title})")


if __name__ == "__main__":
    main()
