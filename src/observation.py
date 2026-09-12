"""Observation feature for Fightin' Words project.

This module implements the --allow-observe feature that allows users to:
1. View a channel's fightin' ngrams from the last run
2. Check which phrases in specific videos were said in previous videos
3. See phrases ranked by their fightin' words z-score

The observation feature is designed to coexist with other features and
not interfere with normal operation.
"""

import argparse
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .token_loading import _input_root, _load_metadata, _filter_files_by_metadata, stream_tokens, load_tokens_per_video
from .utils import tokenize, extract_video_id
from .scoring import PhraseScore, compute_scores, partition_scores
from .candidate_cache import (
    get_candidate_cache_key,
    get_channel_last_run_phrases,
    get_phrases_across_videos,
    store_video_phrases,
    begin_candidate_cache_batch,
    end_candidate_cache_batch,
)


def parse_observe_args(args: argparse.Namespace) -> Optional[argparse.Namespace]:
    """Parse observation-specific arguments.
    
    Args:
        args: The main argument namespace
        
    Returns:
        Namespace with observation parameters, or None if not enabled
    """
    if not getattr(args, "allow_observe", False):
        return None
    
    # Observation feature is enabled
    observe_args = argparse.Namespace()
    observe_args.observe = True
    observe_args.observe_channel = getattr(args, "observe_channel", None)
    observe_args.observe_video_ids = getattr(args, "observe_video_ids", [])
    observe_args.observe_min_z = getattr(args, "observe_min_z", 0.0)
    
    return observe_args


def _load_channel_videos_info(base_dir: Path, channel: str, args: argparse.Namespace) -> List[Tuple[str, Path, dict]]:
    """Load video information for a channel.
    
    Returns:
        List of (video_id, file_path, metadata) tuples sorted by upload date
    """
    txt_dir = _input_root(base_dir) / channel / "txt_files"
    if not txt_dir.exists():
        return []
    
    metadata = _load_metadata(base_dir, channel)
    files = _filter_files_by_metadata(txt_dir, metadata, args)
    
    # Sort by upload date (newest first)
    files.sort(reverse=True)
    
    result = []
    for upload_date, file_path in files:
        video_id = extract_video_id(file_path.name)
        meta = metadata.get(video_id, {})
        result.append((video_id, file_path, meta))
    
    return result


def _compute_video_ngrams_with_limit(
    base_dir: Path,
    channel: str,
    video_id: str,
    ngram_sizes: List[int],
    args: argparse.Namespace,
    max_contribute: int,
    nlp=None,
) -> Optional[Dict[int, Dict[str, int]]]:
    """Compute n-gram counts for a single video with per-video contribution limit.
    
    This respects the --focus-video-contribute limit when computing n-grams.
    
    Args:
        base_dir: Base directory
        channel: Channel name
        video_id: Video ID
        ngram_sizes: List of n-gram sizes to compute
        args: Arguments for filtering
        max_contribute: Maximum count each video can contribute per phrase
        nlp: Lemmatizer (optional)
        
    Returns:
        Dict of {n: {phrase: count}} or None if video not found
    """
    txt_dir = _input_root(base_dir) / channel / "txt_files"
    if not txt_dir.exists():
        return None
    
    # Find the video file
    video_file = txt_dir / f"{video_id}.txt"
    if not video_file.exists():
        # Try other formats
        for f in txt_dir.glob("*.txt"):
            if extract_video_id(f.name) == video_id:
                video_file = f
                break
        else:
            return None
    
    try:
        text = video_file.read_text(encoding="utf-8")
        tokens = tokenize(text)
        
        if len(tokens) < args.min_tokens_per_file:
            return None
        
        # Count n-grams with per-video contribution limit
        result = {n: Counter() for n in ngram_sizes}
        
        for n in ngram_sizes:
            if n == 1:
                # For unigrams, cap each phrase at max_contribute
                unigram_counts = Counter(tokens)
                for phrase, count in unigram_counts.items():
                    result[1][phrase] = min(count, max_contribute)
            else:
                # For n-grams, cap each phrase at max_contribute
                for i in range(len(tokens) - n + 1):
                    phrase = " ".join(tokens[i:i + n])
                    result[n][phrase] = min(result[n].get(phrase, 0) + 1, max_contribute)
        
        return result
    except Exception:
        return None


def _compute_video_ngrams(
    base_dir: Path,
    channel: str,
    video_id: str,
    ngram_sizes: List[int],
    args: argparse.Namespace,
    nlp=None,
    max_contribute: Optional[int] = None,
) -> Optional[Dict[int, Dict[str, int]]]:
    """Compute n-gram counts for a single video.
    
    Args:
        base_dir: Base directory
        channel: Channel name
        video_id: Video ID
        ngram_sizes: List of n-gram sizes to compute
        args: Arguments for filtering
        nlp: Lemmatizer (optional)
        max_contribute: Optional per-video contribution limit
        
    Returns:
        Dict of {n: {phrase: count}} or None if video not found
    """
    if max_contribute is not None:
        return _compute_video_ngrams_with_limit(
            base_dir, channel, video_id, ngram_sizes, args, max_contribute, nlp
        )
    
    txt_dir = _input_root(base_dir) / channel / "txt_files"
    if not txt_dir.exists():
        return None
    
    # Find the video file
    video_file = txt_dir / f"{video_id}.txt"
    if not video_file.exists():
        # Try other formats
        for f in txt_dir.glob("*.txt"):
            if extract_video_id(f.name) == video_id:
                video_file = f
                break
        else:
            return None
    
    try:
        text = video_file.read_text(encoding="utf-8")
        tokens = tokenize(text)
        
        if len(tokens) < args.min_tokens_per_file:
            return None
        
        # Count n-grams
        result = {n: Counter() for n in ngram_sizes}
        
        for n in ngram_sizes:
            if n == 1:
                for token in tokens:
                    result[1][token] += 1
            else:
                for i in range(len(tokens) - n + 1):
                    phrase = " ".join(tokens[i:i + n])
                    result[n][phrase] += 1
        
        return result
    except Exception:
        return None


def run_observation(
    args: argparse.Namespace,
    base_dir: Path,
    cache_dir: Path,
    ngram_sizes: List[int],
    nlp=None,
) -> int:
    """Run the observation feature.
    
    This function handles:
    1. Displaying a channel's fightin' words from last run (if --observe-channel)
    2. Showing which phrases in specific videos appear in previous videos (if --observe-video-ids)
    
    Args:
        args: Parsed arguments
        base_dir: Base directory
        cache_dir: Cache directory
        ngram_sizes: List of n-gram sizes
        nlp: Lemmatizer (optional)
        
    Returns:
        0 on success, 1 on error
    """
    print("\n" + "=" * 80)
    print("OBSERVATION MODE")
    print("=" * 80)
    
    # Get observation-specific args
    observe_args = parse_observe_args(args)
    if observe_args is None:
        print("Observation feature not enabled (use --allow-observe)")
        return 1
    
    # Check if we have a channel to observe
    if not observe_args.observe_channel:
        print("No channel specified for observation (use --observe-channel)")
        return 1
    
    channel = observe_args.observe_channel
    print(f"\nObserving channel: {channel}")
    
    # Get the cache key from the current args
    filter_hash, cache_ngram_max, cache_min_count = get_candidate_cache_key(args)
    
    # Try to get the last run's phrases for this channel
    print(f"\n[Looking for cached fightin' words from last run...]")
    last_run_phrases = get_channel_last_run_phrases(
        cache_dir, channel, filter_hash, cache_ngram_max, cache_min_count
    )
    
    if last_run_phrases:
        print(f"Found {sum(len(v) for v in last_run_phrases.values())} cached phrases from last run")
        for n in sorted(last_run_phrases.keys()):
            phrases = last_run_phrases[n]
            print(f"  {n}-grams: {len(phrases)} phrases")
            # Show top 10
            top_phrases = sorted(phrases.items(), key=lambda x: -x[1])[:10]
            for phrase, count in top_phrases:
                print(f"    '{phrase}': {count:,}")
    else:
        print("No cached phrases found from last run")
    
    # Handle video-specific observation
    if observe_args.observe_video_ids:
        print(f"\n[Checking phrases across videos...]")
        
        # Get max_contribute from focus_video_contribute
        max_contribute = getattr(args, "focus_video_contribute", None)
        
        # Load all videos for this channel
        videos_info = _load_channel_videos_info(base_dir, channel, args)
        if not videos_info:
            print(f"No videos found for channel {channel}")
            return 1
        
        print(f"Found {len(videos_info)} videos for channel {channel}")
        
        # Get video IDs to check
        video_ids_to_check = observe_args.observe_video_ids
        
        # Build a mapping of video_id to index for sorting
        video_date_map = {}
        for video_id, file_path, meta in videos_info:
            upload_date = meta.get("upload_date", "")
            video_date_map[video_id] = upload_date
        
        # Sort video IDs by date (newest first)
        sorted_videos = sorted(
            video_ids_to_check,
            key=lambda vid: video_date_map.get(vid, ""),
            reverse=True
        )
        
        # Process each video
        for video_id in sorted_videos:
            print(f"\n--- Video: {video_id} ---")
            
            # Compute n-grams for this video with per-video contribution limit
            video_ngrams = _compute_video_ngrams(
                base_dir, channel, video_id, ngram_sizes, args, nlp,
                max_contribute=max_contribute
            )
            
            if video_ngrams is None:
                print(f"  No tokens found for video {video_id}")
                continue
            
            # For each phrase in this video, check if it appears in previous videos
            # We need to compute z-scores for the phrases
            # For observation, we'll use a simplified scoring approach
            
            # Get all previous videos (uploaded before this one)
            target_date = video_date_map.get(video_id, "")
            previous_videos = [
                vid for vid in video_date_map
                if vid != video_id and video_date_map.get(vid, "") < target_date
            ]
            
            print(f"  Previous videos: {len(previous_videos)}")
            
            # Collect all phrases from this video
            all_video_phrases = {}
            for n in ngram_sizes:
                if n in video_ngrams:
                    for phrase, count in video_ngrams[n].items():
                        all_video_phrases[phrase] = count
            
            # For each phrase, check if it appears in previous videos
            # We'll do this by loading the previous videos' content
            phrases_in_prev = defaultdict(list)  # phrase -> list of (video_id, count)
            
            for prev_video_id in previous_videos:
                prev_ngrams = _compute_video_ngrams(
                    base_dir, channel, prev_video_id, ngram_sizes, args, nlp,
                    max_contribute=max_contribute
                )
                if prev_ngrams:
                    for n in ngram_sizes:
                        if n in prev_ngrams:
                            for phrase, count in prev_ngrams[n].items():
                                if phrase in all_video_phrases:
                                    phrases_in_prev[phrase].append((prev_video_id, count))
            
            # Now compute a simple z-score-like metric for each phrase
            # We'll use: (count_in_this_video - avg_count_in_prev) / std_count_in_prev
            # For phrases not in previous videos, we'll assign a high score
            
            scored_phrases = []
            for phrase, current_count in all_video_phrases.items():
                prev_counts = [count for _, count in phrases_in_prev.get(phrase, [])]
                
                if not prev_counts:
                    # New phrase - high score
                    z_score = float('inf')
                    dominance = 1.0
                else:
                    avg_prev = sum(prev_counts) / len(prev_counts)
                    std_prev = (sum((c - avg_prev) ** 2 for c in prev_counts) / len(prev_counts)) ** 0.5 if len(prev_counts) > 1 else avg_prev
                    
                    if std_prev > 0:
                        z_score = (current_count - avg_prev) / std_prev
                    else:
                        z_score = float('inf') if current_count > avg_prev else 0.0
                    
                    total_count = current_count + sum(prev_counts)
                    dominance = current_count / total_count if total_count > 0 else 1.0
                
                # Find which n this phrase belongs to
                n_size = 1
                for n in ngram_sizes:
                    if n in video_ngrams and phrase in video_ngrams[n]:
                        n_size = n
                        break
                
                scored_phrases.append({
                    "phrase": phrase,
                    "n": n_size,
                    "current_count": current_count,
                    "z_score": z_score,
                    "dominance": dominance,
                    "in_prev_videos": len(prev_counts),
                    "prev_video_ids": [vid for vid, _ in phrases_in_prev.get(phrase, [])],
                })
            
            # Sort by z-score descending
            scored_phrases.sort(key=lambda x: -x["z_score"])
            
            # Apply minimum z-score filter
            min_z = observe_args.observe_min_z
            filtered_phrases = [p for p in scored_phrases if p["z_score"] >= min_z or p["z_score"] == float('inf')]
            
            print(f"  Phrases with z-score >= {min_z} or new: {len(filtered_phrases)}")
            
            # Display results
            for i, phrase_data in enumerate(filtered_phrases[:50], 1):  # Show top 50
                n_label = ["", "word", "bigram", "trigram"][phrase_data["n"]] if phrase_data["n"] <= 3 else f"{phrase_data['n']}-gram"
                z_str = "NEW" if phrase_data["z_score"] == float('inf') else f"{phrase_data['z_score']:.2f}"
                prev_str = ", ".join(phrase_data["prev_video_ids"][:3]) if phrase_data["prev_video_ids"] else "none"
                if len(phrase_data["prev_video_ids"]) > 3:
                    prev_str += f" (+{len(phrase_data['prev_video_ids']) - 3} more)"
                
                print(f"  {i:>3}. [{n_label}] '{phrase_data['phrase']}' "
                      f"(z={z_str}, count={phrase_data['current_count']}, "
                      f"prev={prev_str})")
            
            if len(filtered_phrases) > 50:
                print(f"  ... and {len(filtered_phrases) - 50} more phrases")
    
    else:
        print("\nNo specific videos to check (use --observe-video-ids)")
    
    return 0


def add_observation_args(parser: argparse.ArgumentParser) -> None:
    """Add observation feature arguments to the parser.
    
    Args:
        parser: The argument parser to add to
    """
    # Observation feature
    parser.add_argument(
        "--allow-observe",
        action="store_true",
        help="Enable observation mode for viewing fightin' words and cross-video phrase tracking"
    )
    parser.add_argument(
        "--observe-channel",
        type=str,
        default=None,
        help="Channel to observe (required when --allow-observe is used)"
    )
    parser.add_argument(
        "--observe-video-ids",
        nargs="+",
        default=[],
        metavar="VIDEO_ID",
        help="Video IDs to check for phrases that appear in previous videos"
    )
    parser.add_argument(
        "--observe-min-z",
        type=float,
        default=0.0,
        help="Minimum z-score threshold for phrases to display (default: 0.0)"
    )


def store_observation_data(
    base_dir: Path,
    channel: str,
    video_id: str,
    ngram_sizes: List[int],
    video_ngrams: Dict[int, Dict[str, int]],
    scores: List[PhraseScore],
    cache_dir: Path,
    batch_conns: Optional[Dict[str, sqlite3.Connection]] = None,
) -> int:
    """Store observation data for a video.
    
    This stores the n-grams and their z-scores for a video, enabling
    the observation feature to track phrases across videos.
    
    Args:
        base_dir: Base directory
        channel: Channel name
        video_id: Video ID
        ngram_sizes: List of n-gram sizes
        video_ngrams: Dict of {n: {phrase: count}} for this video
        scores: List of PhraseScore objects for this channel
        cache_dir: Cache directory
        batch_conns: Optional batch connection dictionary
        
    Returns:
        Number of entries stored
    """
    # Build z-score mapping for phrases
    z_scores = {}
    for score in scores:
        z_scores[score.phrase] = score.z_score
    
    # Store video phrases
    return store_video_phrases(
        cache_dir, channel, video_id, ngram_sizes,
        video_ngrams, z_scores, batch_conns
    )


def get_observation_summary(
    cache_dir: Path,
    channel: str,
    video_ids: List[str],
    ngram_sizes: List[int],
    min_z_score: float = 0.0,
) -> Dict[str, Dict]:
    """Get a summary of phrases across videos for observation.
    
    Args:
        cache_dir: Cache directory
        channel: Channel name
        video_ids: List of video IDs to check
        ngram_sizes: List of n-gram sizes
        min_z_score: Minimum z-score threshold
        
    Returns:
        Dict with observation summary data
    """
    result = {}
    
    for video_id in video_ids:
        video_data = get_phrases_across_videos(
            cache_dir, channel, video_id, ngram_sizes, min_z_score
        )
        
        if video_data:
            result[video_id] = {
                "phrases_by_n": video_data,
                "total_phrases": sum(len(v) for v in video_data.values()),
            }
    
    return result
