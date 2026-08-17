"""Token loading and metadata functions.

These functions provide **streaming** access to transcript data without
loading entire channel corpora into memory.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

from .utils import hms_to_seconds, tokenize, extract_video_id


def _input_root(base_dir: Path) -> Path:
    """Return the transcript root directory."""
    conventional = base_dir / "data" / "input"
    if conventional.is_dir():
        return conventional
    return base_dir / "data:input"


def _load_metadata(base_dir: Path, channel: str) -> dict:
    """Load metadata for a channel keyed by video ID."""
    path = _input_root(base_dir) / channel / "metadata.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            entries = json.load(f)
        return {e["id"]: e for e in entries if isinstance(e, dict) and "id" in e}
    except Exception:
        return {}


def _filter_files_by_metadata(
    txt_dir: Path,
    metadata: dict,
    args: argparse.Namespace,
) -> List[Tuple[str, Path]]:
    """Return (upload_date, file_path) for transcripts that pass all filters."""
    min_duration = hms_to_seconds(args.duration_from)
    max_duration = hms_to_seconds(getattr(args, "duration_to", ""))

    files: List[Tuple[str, Path]] = []
    for file_path in txt_dir.glob("*.txt"):
        video_id = extract_video_id(file_path.name)
        meta = metadata.get(video_id, {})
        upload_date = str(meta.get("upload_date", "99999999"))

        if args.date_from and upload_date < args.date_from:
            continue
        if getattr(args, "date_to", "") and upload_date > args.date_to:
            continue

        duration = meta.get("duration") or 0
        if min_duration and duration < min_duration:
            continue
        if max_duration and duration > max_duration:
            continue

        if args.exclude_live and meta.get("was_live") in ("True", True):
            continue

        files.append((upload_date, file_path))
    return files


# ----------------------------------------------------------------------
# Streaming token generator (core of the pipeline)
# ----------------------------------------------------------------------

def stream_tokens(
    base_dir: Path,
    channel: str,
    args: argparse.Namespace,
):
    """Yield tokens one by one from a channel's transcript files.

    Files are processed in reverse chronological order (newest first) so
    that `--token-limit` keeps the most recent material.  The generator
    stops as soon as the limit is reached.

    Yields:
        Lowercase token strings.
    Returns:
        ``None`` immediately if no files pass the filters or the minimum
        token count cannot be met.
    """
    txt_dir = _input_root(base_dir) / channel / "txt_files"
    if not txt_dir.exists():
        return None

    metadata = _load_metadata(base_dir, channel)
    files = _filter_files_by_metadata(txt_dir, metadata, args)
    if not files:
        return None

    files.sort(reverse=True)  # newest first

    token_count = 0
    for _, file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8")
            tokens = tokenize(text)
            if len(tokens) < args.min_tokens_per_file:
                continue

            for token in tokens:
                token_count += 1
                yield token.lower()

                if args.token_limit > 0 and token_count >= args.token_limit:
                    return

        except Exception:
            continue

    # If we never yielded enough tokens, signal failure.
    if token_count < args.min_tokens_total:
        return


# ----------------------------------------------------------------------
# Per‑video loading (used only for per‑video count limiting)
# ----------------------------------------------------------------------

def load_tokens_per_video(
    base_dir: Path,
    channel: str,
    args: argparse.Namespace,
) -> Optional[List[Tuple[str, List[str]]]]:
    """Return a list of ``(video_id, tokens)`` tuples for a channel.

    This is used exclusively for the ``--focus-video-contribute`` feature,
    which must cap counts **per video** and is therefore not cached.
    """
    txt_dir = _input_root(base_dir) / channel / "txt_files"
    if not txt_dir.exists():
        return None

    metadata = _load_metadata(base_dir, channel)
    files = _filter_files_by_metadata(txt_dir, metadata, args)
    if not files:
        return None

    files.sort()  # chronological order
    video_tokens: List[Tuple[str, List[str]]] = []
    for _, file_path in files:
        try:
            video_id = extract_video_id(file_path.name)
            text = file_path.read_text(encoding="utf-8")
            tokens = tokenize(text)
            if len(tokens) < args.min_tokens_per_file:
                continue
            video_tokens.append((video_id, tokens))
        except Exception:
            continue

    if not video_tokens:
        return None
    return video_tokens


# ----------------------------------------------------------------------
# Channel discovery
# ----------------------------------------------------------------------

def _discover_channels(base_dir: Path) -> List[str]:
    """List all channel directories under the transcript root."""
    root = _input_root(base_dir)
    if not root.is_dir():
        return []
    return sorted(folder.name for folder in root.iterdir() if folder.is_dir())