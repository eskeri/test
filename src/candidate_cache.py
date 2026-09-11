"""Candidate phrase cache for incremental accumulation of fightin' words.

This module implements a disk-based cache that accumulates ONLY candidate phrases
(phrases that meet min_count threshold) from channels, allowing incremental
buildup across multiple runs without loading everything into memory.

The cache is structured as:
- One SQLite database per (filter_hash, ngram_max, min_count) combination
- Stores only phrases that meet the min_count threshold in their channel
- Tracks which channel contributed each phrase
- Allows efficient lookup and accumulation

Key properties:
1. Disk-based: Only loads what's needed, doesn't accumulate in memory
2. Incremental: New runs add to existing cache
3. Efficient: Indexed lookups stay fast regardless of cache size
4. Filter-aware: Different filter combinations use different cache files
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter


def _get_candidate_cache_path(cache_dir: Path, filter_hash: str, ngram_max: int, min_count: int) -> Path:
    """Generate cache file path based on filter parameters."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Include ngram_max and min_count in the filename since they affect what's cached
    cache_name = f"candidate_{filter_hash}_n{ngram_max}_min{min_count}.db"
    return cache_dir / cache_name


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS channels (
    name TEXT PRIMARY KEY,
    token_count INTEGER NOT NULL DEFAULT 0,
    phrase_count INTEGER NOT NULL DEFAULT 0,
    last_updated TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS phrases (
    n INTEGER NOT NULL,
    phrase TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    contributing_channels TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (n, phrase)
);
CREATE TABLE IF NOT EXISTS video_phrases (
    channel TEXT NOT NULL,
    video_id TEXT NOT NULL,
    n INTEGER NOT NULL,
    phrase TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    z_score REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (channel, video_id, n, phrase)
);
CREATE INDEX IF NOT EXISTS idx_phrases_n ON phrases(n);
CREATE INDEX IF NOT EXISTS idx_phrases_phrase ON phrases(phrase);
CREATE INDEX IF NOT EXISTS idx_video_phrases_channel ON video_phrases(channel);
CREATE INDEX IF NOT EXISTS idx_video_phrases_video ON video_phrases(video_id);
"""


def _connect(cache_dir: Path, filter_hash: str, ngram_max: int, min_count: int) -> sqlite3.Connection:
    """Open connection to candidate cache database."""
    db_path = _get_candidate_cache_path(cache_dir, filter_hash, ngram_max, min_count)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.executescript(_SCHEMA)
    return conn


def get_candidate_cache_key(args) -> Tuple[str, int, int]:
    """Generate cache key from arguments that affect candidate phrase selection.
    
    Returns:
        Tuple of (filter_hash, ngram_max, min_count)
    """
    key_data = {
        "date_from": getattr(args, "date_from", ""),
        "date_to": getattr(args, "date_to", ""),
        "duration_from": getattr(args, "duration_from", ""),
        "duration_to": getattr(args, "duration_to", ""),
        "exclude_live": getattr(args, "exclude_live", False),
        "token_limit": getattr(args, "token_limit", 0),
        "min_tokens_per_file": getattr(args, "min_tokens_per_file", 0),
        "min_tokens_total": getattr(args, "min_tokens_total", 0),
        "lemmatize": getattr(args, "lemmatize", False),
        "lemmatizer": getattr(args, "lemmatizer", "simplemma"),
    }
    key_str = json.dumps(key_data, sort_keys=True)
    filter_hash = hashlib.sha1(key_str.encode()).hexdigest()[:16]
    ngram_max = getattr(args, "ngram_max", 2)
    min_count = getattr(args, "min_count", 5)
    return filter_hash, ngram_max, min_count


def begin_candidate_cache_batch() -> Dict[str, sqlite3.Connection]:
    """Start a batch write session for candidate cache.
    
    Returns dictionary to store connections for batch writes.
    """
    return {}


def end_candidate_cache_batch(batch_conns: Dict[str, sqlite3.Connection]) -> None:
    """End batch write session, committing all changes."""
    for conn in batch_conns.values():
        try:
            conn.execute("COMMIT")
        except sqlite3.OperationalError:
            pass
        conn.close()


def _get_batch_conn(
    batch_conns: Dict[str, sqlite3.Connection],
    cache_dir: Path,
    filter_hash: str,
    ngram_max: int,
    min_count: int
) -> sqlite3.Connection:
    """Get or create connection for batch writes."""
    cache_key = f"{filter_hash}_n{ngram_max}_min{min_count}"
    if cache_key not in batch_conns:
        conn = _connect(cache_dir, filter_hash, ngram_max, min_count)
        conn.execute("BEGIN")
        batch_conns[cache_key] = conn
    return batch_conns[cache_key]


def add_channel_candidates(
    cache_dir: Path,
    channel: str,
    filter_hash: str,
    ngram_max: int,
    min_count: int,
    ngram_counts: Dict[int, Dict[str, int]],
    token_count: int,
    batch_conns: Optional[Dict[str, sqlite3.Connection]] = None,
) -> Tuple[int, int]:
    """Add candidate phrases from a channel to the cache.
    
    Only phrases that meet the min_count threshold are stored.
    
    Args:
        cache_dir: Directory for cache files
        channel: Channel name
        filter_hash: Filter hash for this run
        ngram_max: Maximum n-gram size
        min_count: Minimum count threshold
        ngram_counts: Dict mapping n -> {phrase: count}
        token_count: Total tokens for this channel
        batch_conns: Optional batch connection dictionary
        
    Returns:
        Tuple of (new_phrases_added, existing_phrases_updated)
    """
    # Filter to candidate phrases only
    candidate_phrases = {}
    for n in range(1, ngram_max + 1):
        if n not in ngram_counts:
            continue
        for phrase, count in ngram_counts[n].items():
            if count >= min_count:
                candidate_phrases.setdefault(n, {})[phrase] = count
    
    if not candidate_phrases:
        return 0, 0
    
    # Get connection
    if batch_conns is not None:
        conn = _get_batch_conn(batch_conns, cache_dir, filter_hash, ngram_max, min_count)
    else:
        conn = _connect(cache_dir, filter_hash, ngram_max, min_count)
        conn.execute("BEGIN")
    
    try:
        new_phrases = 0
        updated_phrases = 0
        
        # Update channel info
        conn.execute(
            "INSERT INTO channels(name, token_count, phrase_count) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET token_count = token_count + ?, "
            "phrase_count = phrase_count + ?",
            (channel, token_count, len(candidate_phrases.get(1, {})) if 1 in candidate_phrases else 0,
             token_count, len(candidate_phrases.get(1, {})) if 1 in candidate_phrases else 0)
        )
        
        # Process each n-gram size
        for n in range(1, ngram_max + 1):
            if n not in candidate_phrases:
                continue
            
            for phrase, count in candidate_phrases[n].items():
                # Check if phrase already exists
                row = conn.execute(
                    "SELECT count, contributing_channels FROM phrases WHERE n=? AND phrase=?",
                    (n, phrase)
                ).fetchone()
                
                if row is None:
                    # New phrase
                    conn.execute(
                        "INSERT INTO phrases(n, phrase, count, contributing_channels) VALUES (?, ?, ?, ?)",
                        (n, phrase, count, channel)
                    )
                    new_phrases += 1
                else:
                    # Existing phrase - update count and add to contributing channels
                    existing_count = row["count"]
                    existing_channels = set(row["contributing_channels"].split(",") if row["contributing_channels"] else [])
                    
                    if channel not in existing_channels:
                        existing_channels.add(channel)
                        new_channels_str = ",".join(sorted(existing_channels))
                        new_count = existing_count + count
                        
                        conn.execute(
                            "UPDATE phrases SET count=?, contributing_channels=? WHERE n=? AND phrase=?",
                            (new_count, new_channels_str, n, phrase)
                        )
                        updated_phrases += 1
        
        if batch_conns is None:
            conn.execute("COMMIT")
        
        return new_phrases, updated_phrases
        
    finally:
        if batch_conns is None:
            conn.execute("ROLLBACK")
            conn.close()


def get_cached_candidates(
    cache_dir: Path,
    filter_hash: str,
    ngram_max: int,
    min_count: int,
    target_phrases: Optional[Dict[int, Set[str]]] = None,
) -> Optional[Tuple[int, Dict[int, Dict[str, int]]]]:
    """Retrieve cached candidate phrases, optionally filtered to target phrases.
    
    This is the main read path - it only loads the phrases we need, not the
    entire cache into memory.
    
    Args:
        cache_dir: Directory for cache files
        filter_hash: Filter hash
        ngram_max: Maximum n-gram size
        min_count: Minimum count threshold
        target_phrases: Optional dict of {n: set of phrases} to filter for
        
    Returns:
        Tuple of (total_phrase_count, {n: {phrase: count}}) or None if cache doesn't exist
    """
    db_path = _get_candidate_cache_path(cache_dir, filter_hash, ngram_max, min_count)
    if not db_path.exists():
        return None
    
    conn = _connect(cache_dir, filter_hash, ngram_max, min_count)
    
    try:
        # Get total phrase count
        row = conn.execute("SELECT COUNT(*) as cnt FROM phrases").fetchone()
        total_count = row["cnt"] if row else 0
        
        if target_phrases is None:
            # Load all phrases (rare - usually we have targets)
            result = {}
            for n in range(1, ngram_max + 1):
                rows = conn.execute(
                    "SELECT phrase, count FROM phrases WHERE n=?",
                    (n,)
                ).fetchall()
                if rows:
                    result[n] = {r["phrase"]: r["count"] for r in rows}
            return total_count, result
        else:
            # Load only target phrases
            result = {}
            for n in range(1, ngram_max + 1):
                if n not in target_phrases or not target_phrases[n]:
                    continue
                
                phrases_list = list(target_phrases[n])
                if not phrases_list:
                    continue
                
                # Batch queries to avoid huge IN clauses
                BATCH_SIZE = 900
                n_result = {}
                for i in range(0, len(phrases_list), BATCH_SIZE):
                    batch = phrases_list[i:i + BATCH_SIZE]
                    placeholders = ",".join("?" for _ in batch)
                    rows = conn.execute(
                        f"SELECT phrase, count FROM phrases WHERE n=? AND phrase IN ({placeholders})",
                        [n, *batch]
                    ).fetchall()
                    for r in rows:
                        n_result[r["phrase"]] = r["count"]
                
                if n_result:
                    result[n] = n_result
            
            return total_count, result
    finally:
        conn.close()


def get_channel_candidates(
    cache_dir: Path,
    filter_hash: str,
    ngram_max: int,
    min_count: int,
    channel: str,
) -> Optional[Dict[int, Dict[str, int]]]:
    """Get all candidate phrases contributed by a specific channel.
    
    Note: This requires scanning the phrases table since we don't store
    per-channel breakdown directly. For efficiency, we use the
    contributing_channels field to filter.
    
    Args:
        cache_dir: Directory for cache files
        filter_hash: Filter hash
        ngram_max: Maximum n-gram size
        min_count: Minimum count threshold
        channel: Channel name to filter for
        
    Returns:
        Dict of {n: {phrase: count}} for phrases contributed by this channel, or None
    """
    db_path = _get_candidate_cache_path(cache_dir, filter_hash, ngram_max, min_count)
    if not db_path.exists():
        return None
    
    conn = _connect(cache_dir, filter_hash, ngram_max, min_count)
    
    try:
        result = {}
        for n in range(1, ngram_max + 1):
            rows = conn.execute(
                "SELECT phrase, count FROM phrases WHERE n=? AND contributing_channels LIKE ?",
                (n, f"%,{channel},%")
            ).fetchall()
            if rows:
                result[n] = {r["phrase"]: r["count"] for r in rows}
        
        return result if result else None
    finally:
        conn.close()


def get_all_channels_in_cache(
    cache_dir: Path,
    filter_hash: str,
    ngram_max: int,
    min_count: int,
) -> List[str]:
    """Get list of all channels that have contributed to this cache."""
    db_path = _get_candidate_cache_path(cache_dir, filter_hash, ngram_max, min_count)
    if not db_path.exists():
        return []
    
    conn = _connect(cache_dir, filter_hash, ngram_max, min_count)
    
    try:
        rows = conn.execute("SELECT name FROM channels").fetchall()
        return [r["name"] for r in rows]
    finally:
        conn.close()


def get_total_token_count(
    cache_dir: Path,
    filter_hash: str,
    ngram_max: int,
    min_count: int,
) -> int:
    """Get total token count across all channels in this cache."""
    db_path = _get_candidate_cache_path(cache_dir, filter_hash, ngram_max, min_count)
    if not db_path.exists():
        return 0
    
    conn = _connect(cache_dir, filter_hash, ngram_max, min_count)
    
    try:
        row = conn.execute("SELECT SUM(token_count) as total FROM channels").fetchone()
        return row["total"] if row and row["total"] else 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Video-level phrase tracking for observation feature
# ---------------------------------------------------------------------------

def store_video_phrases(
    cache_dir: Path,
    channel: str,
    video_id: str,
    ngram_sizes: List[int],
    video_ngram_counts: Dict[int, Dict[str, int]],
    z_scores: Dict[str, float],
    batch_conns: Optional[Dict[str, sqlite3.Connection]] = None,
) -> int:
    """Store n-gram counts for a specific video with their z-scores.
    
    This enables the observation feature to track which phrases appear in
    which videos and their statistical significance.
    
    Args:
        cache_dir: Directory for cache files
        channel: Channel name
        video_id: Video ID
        ngram_sizes: List of n-gram sizes to store
        video_ngram_counts: Dict of {n: {phrase: count}} for this video
        z_scores: Dict of {phrase: z_score} for this video's phrases
        batch_conns: Optional batch connection dictionary
        
    Returns:
        Number of phrase entries stored
    """
    # For video tracking, we use the same database but different tables
    # We'll use a special filter_hash for video tracking
    video_filter_hash = "video_tracking"
    ngram_max = max(ngram_sizes) if ngram_sizes else 2
    min_count = 1  # For video-level, we want all phrases
    
    cache_key = f"{video_filter_hash}_n{ngram_max}_min{min_count}"
    
    if batch_conns is not None and cache_key in batch_conns:
        conn = batch_conns[cache_key]
    else:
        conn = _connect(cache_dir, video_filter_hash, ngram_max, min_count)
        if batch_conns is None:
            conn.execute("BEGIN")
        else:
            conn.execute("BEGIN")
            batch_conns[cache_key] = conn
    
    try:
        stored_count = 0
        
        for n in ngram_sizes:
            if n not in video_ngram_counts:
                continue
            
            for phrase, count in video_ngram_counts[n].items():
                z_score = z_scores.get(phrase, 0.0)
                
                # Check if this video+phrase already exists
                row = conn.execute(
                    "SELECT 1 FROM video_phrases WHERE channel=? AND video_id=? AND n=? AND phrase=?",
                    (channel, video_id, n, phrase)
                ).fetchone()
                
                if row is None:
                    # Insert new
                    conn.execute(
                        "INSERT INTO video_phrases(channel, video_id, n, phrase, count, z_score) VALUES (?, ?, ?, ?, ?, ?)",
                        (channel, video_id, n, phrase, count, z_score)
                    )
                    stored_count += 1
                else:
                    # Update existing
                    conn.execute(
                        "UPDATE video_phrases SET count=?, z_score=? WHERE channel=? AND video_id=? AND n=? AND phrase=?",
                        (count, z_score, channel, video_id, n, phrase)
                    )
        
        if batch_conns is None:
            conn.execute("COMMIT")
        
        return stored_count
        
    finally:
        if batch_conns is None:
            conn.execute("ROLLBACK")
            conn.close()


def get_video_phrases(
    cache_dir: Path,
    channel: str,
    video_ids: List[str],
    ngram_sizes: List[int],
) -> Dict[str, Dict[int, Dict[str, Tuple[int, float]]]]:
    """Get all phrases and their z-scores for specific videos.
    
    Args:
        cache_dir: Directory for cache files
        channel: Channel name
        video_ids: List of video IDs to query
        ngram_sizes: List of n-gram sizes to retrieve
        
    Returns:
        Dict of {video_id: {n: {phrase: (count, z_score)}}}
    """
    video_filter_hash = "video_tracking"
    ngram_max = max(ngram_sizes) if ngram_sizes else 2
    min_count = 1
    
    db_path = _get_candidate_cache_path(cache_dir, video_filter_hash, ngram_max, min_count)
    if not db_path.exists():
        return {}
    
    conn = _connect(cache_dir, video_filter_hash, ngram_max, min_count)
    
    try:
        result = {}
        
        for video_id in video_ids:
            result[video_id] = {}
            for n in ngram_sizes:
                rows = conn.execute(
                    "SELECT phrase, count, z_score FROM video_phrases "
                    "WHERE channel=? AND video_id=? AND n=?",
                    (channel, video_id, n)
                ).fetchall()
                
                if rows:
                    result[video_id][n] = {
                        r["phrase"]: (r["count"], r["z_score"])
                        for r in rows
                    }
        
        return result
    finally:
        conn.close()


def get_phrases_across_videos(
    cache_dir: Path,
    channel: str,
    video_id: str,
    ngram_sizes: List[int],
    min_z_score: float = 0.0,
) -> Dict[int, List[Tuple[str, int, float, List[str]]]]:
    """Get phrases from a video and find which previous videos contain them.
    
    This is the core of the observation feature - it lets users see what
    phrases in a specific video were also said in previous videos, ranked by
    their z-score.
    
    Args:
        cache_dir: Directory for cache files
        channel: Channel name
        video_id: Video ID to check
        ngram_sizes: List of n-gram sizes to check
        min_z_score: Minimum z-score threshold for phrases
        
    Returns:
        Dict of {n: [(phrase, count, z_score, [previous_video_ids]), ...]}
        sorted by z_score descending
    """
    video_filter_hash = "video_tracking"
    ngram_max = max(ngram_sizes) if ngram_sizes else 2
    min_count = 1
    
    db_path = _get_candidate_cache_path(cache_dir, video_filter_hash, ngram_max, min_count)
    if not db_path.exists():
        return {}
    
    conn = _connect(cache_dir, video_filter_hash, ngram_max, min_count)
    
    try:
        result = {}
        
        for n in ngram_sizes:
            # First, get all phrases from the target video
            target_phrases = conn.execute(
                "SELECT phrase, count, z_score FROM video_phrases "
                "WHERE channel=? AND video_id=? AND n=? AND z_score >= ?",
                (channel, video_id, n, min_z_score)
            ).fetchall()
            
            if not target_phrases:
                continue
            
            # For each phrase, find which other videos in this channel have it
            n_result = []
            for phrase_row in target_phrases:
                phrase = phrase_row["phrase"]
                count = phrase_row["count"]
                z_score = phrase_row["z_score"]
                
                # Find all videos with this phrase
                prev_videos = conn.execute(
                    "SELECT DISTINCT video_id FROM video_phrases "
                    "WHERE channel=? AND n=? AND phrase=? AND video_id != ?",
                    (channel, n, phrase, video_id)
                ).fetchall()
                
                prev_video_ids = [r["video_id"] for r in prev_videos]
                n_result.append((phrase, count, z_score, prev_video_ids))
            
            # Sort by z-score descending
            n_result.sort(key=lambda x: -x[2])
            result[n] = n_result
        
        return result
    finally:
        conn.close()


def get_channel_last_run_phrases(
    cache_dir: Path,
    channel: str,
    filter_hash: str,
    ngram_max: int,
    min_count: int,
) -> Optional[Dict[int, Dict[str, int]]]:
    """Get the candidate phrases from the last run for a channel.
    
    This is used by the observation feature to show a channel's fightin' words
    from the last run.
    
    Args:
        cache_dir: Directory for cache files
        channel: Channel name
        filter_hash: Filter hash from last run
        ngram_max: N-gram max from last run
        min_count: Min count from last run
        
    Returns:
        Dict of {n: {phrase: count}} or None if not found
    """
    # Try to get phrases contributed by this channel
    return get_channel_candidates(cache_dir, filter_hash, ngram_max, min_count, channel)
