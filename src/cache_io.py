"""Simple cache I/O functions for instant cache reads.

The cache is one pickle file per filter hash. It holds a plain dictionary:

    {
        "token_count": int,                 # total unigram tokens over all channels
        "total_channels": int,
        "channels": {
            channel: {
                "token_count": int,         # unigram tokens for this channel
                "counts": {n: {phrase: count}},   # n -> {phrase: count}
            },
            ...
        },
    }

Reads load the whole dictionary at once (fast). Focus channels are removed on
the fly by subtracting their counts from the totals when the rest pool is
requested -- nothing about focus channels is stored separately at build time.
"""

import pickle
from collections import Counter
from pathlib import Path


def get_cache_file_path(cache_dir, filter_hash):
    """Get the path to the cache data file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"cache_{filter_hash}.pkl"


def get_index_file_path(cache_dir, filter_hash):
    """Kept for compatibility; the simple cache has no separate index file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"cache_index_{filter_hash}.json"


def compute_data_hash(data):
    """Kept for API compatibility; not used by the simple cache."""
    import hashlib
    return hashlib.sha256(data).hexdigest()


def load_index(cache_dir, filter_hash):
    """Load the cache dictionary (the file is the index). Returns None if absent."""
    cache_file = get_cache_file_path(cache_dir, filter_hash)
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_index(cache_dir, filter_hash, index_data):
    """Save the cache dictionary."""
    cache_file = get_cache_file_path(cache_dir, filter_hash)
    with open(cache_file, "wb") as f:
        pickle.dump(index_data, f)


def cache_exists(cache_dir, channel, filter_hash):
    """Check if a channel has cached data."""
    index = load_index(cache_dir, filter_hash)
    if index is None:
        return False
    return channel in index.get("channels", {})


def get_channel_info(cache_dir, channel, filter_hash):
    """Get channel info from the cache."""
    index = load_index(cache_dir, filter_hash)
    if index is None:
        return None
    return index.get("channels", {}).get(channel)


def write_channel_cache(cache_dir, channel, filter_hash, ngram_pairs):
    """Write a channel's n-gram data to the cache.

    Args:
        cache_dir: Cache directory
        channel: Channel name
        filter_hash: Filter hash for this cache
        ngram_pairs: Iterable of (n, phrase) tuples

    Returns:
        Number of tokens (unigrams) processed
    """
    # Count this channel's n-grams.
    counters = {}
    token_count = 0
    for n, phrase in ngram_pairs:
        if n not in counters:
            counters[n] = Counter()
        counters[n][phrase] += 1
        if n == 1:
            token_count += 1

    if not counters:
        return 0

    # Plain dicts store better than Counter in pickle.
    channel_counts = {n: dict(c) for n, c in counters.items()}

    index = load_index(cache_dir, filter_hash)
    if index is None:
        index = {
            "token_count": 0,
            "total_channels": 0,
            "channels": {},
        }

    index["channels"][channel] = {
        "token_count": token_count,
        "counts": channel_counts,
    }
    index["token_count"] += token_count
    index["total_channels"] = len(index["channels"])

    save_index(cache_dir, filter_hash, index)
    return token_count


def read_channel_cache(cache_dir, channel, filter_hash):
    """Read a channel's full cache data.

    Returns:
        (token_count, {n: {phrase: count}}) or None if not found
    """
    channel_info = get_channel_info(cache_dir, channel, filter_hash)
    if channel_info is None:
        return None
    return channel_info.get("token_count", 0), channel_info.get("counts", {})


def read_channel_cache_filtered(cache_dir, channel, filter_hash, target_phrases):
    """Read a channel's cache, keeping only target phrases.

    Args:
        cache_dir: Cache directory
        channel: Channel name
        filter_hash: Filter hash
        target_phrases: {n: set of phrases} to filter for

    Returns:
        (token_count, {n: {phrase: count}}) or None if not found
    """
    channel_info = get_channel_info(cache_dir, channel, filter_hash)
    if channel_info is None:
        return None

    counts = channel_info.get("counts", {})
    result = {}
    for n, phrases in target_phrases.items():
        bucket = counts.get(n, {})
        kept = {p: c for p, c in bucket.items() if p in phrases}
        if kept:
            result[n] = kept
    return channel_info.get("token_count", 0), result


def get_all_channels(cache_dir, filter_hash):
    """Get list of all channels in the cache."""
    index = load_index(cache_dir, filter_hash)
    if index is None:
        return []
    return list(index.get("channels", {}).keys())


def get_total_token_count(cache_dir, filter_hash, exclude_channels=None):
    """Get total token count for all channels, excluding some if specified."""
    index = load_index(cache_dir, filter_hash)
    if index is None:
        return 0

    exclude_set = set(exclude_channels or [])
    total = 0
    for channel, info in index.get("channels", {}).items():
        if channel not in exclude_set:
            total += info.get("token_count", 0)
    return total


def read_multiple_channels_filtered(cache_dir, channels, filter_hash, target_phrases):
    """Read multiple channels with filtering.

    Args:
        cache_dir: Cache directory
        channels: List of channel names
        filter_hash: Filter hash
        target_phrases: {n: set of phrases} to filter for

    Returns:
        {channel: (token_count, {n: {phrase: count}})}
    """
    index = load_index(cache_dir, filter_hash)
    if index is None:
        return {}

    results = {}
    for channel in channels:
        info = index.get("channels", {}).get(channel)
        if info is None:
            continue
        counts = info.get("counts", {})
        filtered = {}
        for n, phrases in target_phrases.items():
            bucket = counts.get(n, {})
            kept = {p: c for p, c in bucket.items() if p in phrases}
            if kept:
                filtered[n] = kept
        results[channel] = (info.get("token_count", 0), filtered)
    return results


def read_aggregate_filtered(cache_dir, filter_hash, target_phrases, exclude_channels=None):
    """Read aggregate data filtered by target phrases, excluding focus channels.

    This is the key function for instant rest pool computation. It loads the whole
    cache once, then sums every non-excluded channel's target-phrase counts.
    Focus (excluded) channels are simply skipped -- their counts never enter the
    total, so the rest pool is correct with no subtract step.

    Args:
        cache_dir: Cache directory
        filter_hash: Filter hash
        target_phrases: {n: set of phrases} to filter for
        exclude_channels: List of channels to exclude (e.g., focus channels)

    Returns:
        (total_token_count, {n: {phrase: count}}) or None if cache not available
    """
    index = load_index(cache_dir, filter_hash)
    if index is None:
        return None

    exclude_set = set(exclude_channels or [])
    channels = index.get("channels", {})

    total_tokens = 0
    aggregate_counts = {n: Counter() for n in target_phrases}

    for channel, info in channels.items():
        if channel in exclude_set:
            continue
        total_tokens += info.get("token_count", 0)
        counts = info.get("counts", {})
        for n, phrases in target_phrases.items():
            bucket = counts.get(n)
            if not bucket:
                continue
            agg = aggregate_counts[n]
            for phrase in phrases:
                c = bucket.get(phrase)
                if c:
                    agg[phrase] += c

    return total_tokens, aggregate_counts