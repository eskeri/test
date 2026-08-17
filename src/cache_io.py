"""Simple cache I/O functions for instant cache reads.

The cache is dumb and simple: one pickle file per filter hash holding the
combined n-gram counts for ALL channels together. There is no per-channel
breakdown -- the cache is just:

    {
        "token_count": int,            # total unigram tokens over all channels
        "total_channels": int,         # how many channels were folded in
        "channels": {name: True, ...}, # set of channel names already folded (for dedup)
        "counts": {n: {phrase: count}},# combined counts for every channel
    }

Reads load the whole dictionary at once (fast). Focus channels are subtracted
on the fly by the caller (the comparison code already has the focus counts in
memory), never by the cache itself.
"""

import pickle
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# In-process batch accumulator.
#
# When populating a cache over thousands of channels (--cache), calling
# load_index + save_index for every channel re-reads and re-writes the whole
# growing pickle each time -- O(N^2). Instead, a --cache run calls
# begin_cache_batch once, folds every channel into an in-memory aggregate,
# then calls flush_cache_batch once at the end. One read at the start, one
# write at the end.
# ---------------------------------------------------------------------------
_batch = None  # dict keyed by (cache_dir, filter_hash) -> accumulator dict


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


def _new_index():
    return {
        "token_count": 0,
        "total_channels": 0,
        "channels": {},
        "counts": {},
    }


def begin_cache_batch():
    """Start a batch accumulation session (see module docstring)."""
    global _batch
    _batch = {}


def flush_cache_batch():
    """Write any accumulated batch data to disk and end the session."""
    global _batch
    if not _batch:
        return
    for (cache_dir, filter_hash), index in _batch.items():
        save_index(cache_dir, filter_hash, index)
    _batch = None


def close_cache_connection(cache_dir=None):
    """Flush any pending batch writes. `cache_dir` is accepted for call-site
    compatibility but ignored -- a single run only has one batch session."""
    flush_cache_batch()


def cache_exists(cache_dir, channel, filter_hash):
    """Check if a channel has been folded into the aggregate cache.

    During a batch (--cache run) the in-memory accumulator is authoritative, so
    we check it first and avoid re-reading the whole pickle from disk for every
    channel (which made re-runs over hundreds of channels O(N^2) in reads even
    though every channel was already cached). On the first read during a batch
    we memoize the loaded index back into the batch so subsequent calls hit
    memory.
    """
    if _batch is not None:
        key = (cache_dir, filter_hash)
        index = _batch.get(key)
        if index is None:
            index = load_index(cache_dir, filter_hash)
            _batch[key] = index if index is not None else _new_index()
        if index is None:
            return False
        return channel in index.get("channels", {})
    index = load_index(cache_dir, filter_hash)
    if index is None:
        return False
    return channel in index.get("channels", {})


def get_channel_info(cache_dir, channel, filter_hash):
    """Kept for API compatibility. The aggregate cache has no per-channel info,
    so this returns None (a single channel's counts live only in memory in the
    comparison code, not in the cache)."""
    return None


def _fold(index, channel, token_count, counts):
    """Fold one channel's counted n-grams into the aggregate index in place."""
    if channel in index.get("channels", {}):
        # Already folded; don't double-count.
        return
    index.setdefault("channels", {})[channel] = True
    index["token_count"] = index.get("token_count", 0) + token_count
    index["total_channels"] = len(index["channels"])
    agg = index.setdefault("counts", {})
    for n, bucket in counts.items():
        target = agg.setdefault(n, {})
        for phrase, c in bucket.items():
            target[phrase] = target.get(phrase, 0) + c


def write_channel_counts(cache_dir, channel, filter_hash, counts, token_count):
    """Fold a channel's already-counted n-grams into the aggregate cache.

    Args:
        cache_dir: Cache directory
        channel: Channel name (for dedup so re-runs are safe)
        filter_hash: Filter hash for this cache
        counts: {n: {phrase: count}}
        token_count: number of unigram tokens for this channel

    Returns:
        Number of tokens processed
    """
    global _batch
    if _batch is not None:
        key = (cache_dir, filter_hash)
        index = _batch.get(key)
        if index is None:
            index = load_index(cache_dir, filter_hash) or _new_index()
            _batch[key] = index
        _fold(index, channel, token_count, counts)
        return token_count

    index = load_index(cache_dir, filter_hash) or _new_index()
    _fold(index, channel, token_count, counts)
    save_index(cache_dir, filter_hash, index)
    return token_count


def write_channel_cache(cache_dir, channel, filter_hash, ngram_pairs):
    """Count and fold a channel's n-grams into the aggregate cache.

    Args:
        cache_dir: Cache directory
        channel: Channel name
        filter_hash: Filter hash for this cache
        ngram_pairs: Iterable of (n, phrase) tuples

    Returns:
        Number of tokens (unigrams) processed
    """
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

    counts = {n: dict(c) for n, c in counters.items()}
    return write_channel_counts(cache_dir, channel, filter_hash, counts, token_count)


def read_channel_cache(cache_dir, channel, filter_hash):
    """Kept for API compatibility. The aggregate cache cannot return a single
    channel's counts -- the comparison code computes a single channel's counts
    directly from transcripts. Returns None so callers fall through to compute."""
    return None


def read_channel_cache_filtered(cache_dir, channel, filter_hash, target_phrases):
    """Kept for API compatibility. See read_channel_cache -- returns None."""
    return None


def get_all_channels(cache_dir, filter_hash):
    """Get list of all channels folded into the aggregate cache."""
    index = load_index(cache_dir, filter_hash)
    if index is None:
        return []
    return list(index.get("channels", {}).keys())


def get_total_token_count(cache_dir, filter_hash, exclude_channels=None):
    """Get total token count for the whole aggregate.

    exclude_channels is ignored because the aggregate has no per-channel token
    breakdown; the caller subtracts focus tokens itself when computing a rest
    pool.
    """
    index = load_index(cache_dir, filter_hash)
    if index is None:
        return 0
    return index.get("token_count", 0)


def read_multiple_channels_filtered(cache_dir, channels, filter_hash, target_phrases):
    """Kept for API compatibility. The aggregate cache has no per-channel
    breakdown, so this returns {} (callers use read_aggregate_filtered)."""
    return {}


def read_aggregate_filtered(cache_dir, filter_hash, target_phrases, exclude_channels=None):
    """Read the whole-corpus aggregate, filtered to target phrases.

    The cache holds the combined counts for ALL channels (focus included).
    This returns that total filtered to the candidate phrases; the caller
    subtracts the focus channel's counts on the fly to get the rest pool.

    exclude_channels is accepted for call-site compatibility but ignored --
    the aggregate has no per-channel breakdown to skip.

    Args:
        cache_dir: Cache directory
        filter_hash: Filter hash
        target_phrases: {n: set of phrases} to filter for
        exclude_channels: ignored (kept for API compatibility)

    Returns:
        (total_token_count, {n: {phrase: count}}) or None if cache not available
    """
    index = load_index(cache_dir, filter_hash)
    if index is None:
        return None

    counts = index.get("counts", {})
    total_tokens = index.get("token_count", 0)

    aggregate_counts = {}
    for n, phrases in target_phrases.items():
        bucket = counts.get(n, {})
        if not bucket or not phrases:
            continue
        agg = Counter()
        # Iterate over the smaller of the two sets.
        if len(phrases) <= len(bucket):
            for phrase in phrases:
                c = bucket.get(phrase)
                if c:
                    agg[phrase] = c
        else:
            for phrase, c in bucket.items():
                if phrase in phrases:
                    agg[phrase] = c
        if agg:
            aggregate_counts[n] = agg

    return total_tokens, aggregate_counts
