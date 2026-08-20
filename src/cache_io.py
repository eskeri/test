"""Instant cache I/O backed by a single SQLite file per filter hash.

The cache is dumb and simple: ONE file per filter hash holding the combined
n-gram counts for ALL channels together. There is no per-channel breakdown --
the cache is just three tables inside one SQLite database:

    meta(key TEXT PRIMARY KEY, value TEXT)
        - token_count   : total unigram tokens over all channels
        - total_channels: how many channels were folded in
    channels(name TEXT PRIMARY KEY)
        - the set of channel names already folded (for dedup so re-runs are safe)
    counts(n INTEGER, phrase TEXT, count INTEGER, PRIMARY KEY(n, phrase))

Reads are indexed by (n, phrase), so a comparison that only needs a few
thousand candidate phrases returns in milliseconds WITHOUT loading the whole
3-billion-word corpus into memory. Focus channels are subtracted on the fly
by the caller (the comparison code already has the focus counts in memory),
never by the cache itself.

Why SQLite instead of a pickle: a single in-memory pickle dict holding every
n-gram for a multi-billion-word corpus is tens of GB of RAM and takes tens of
seconds to (de)serialize on every read. SQLite keeps the data on disk in an
indexed B-tree, so membership checks (cache_exists) and filtered reads
(read_aggregate_filtered) are O(log N) point lookups that stay instant at any
corpus size, and peak RAM is bounded by the result set, not the corpus.
"""

import sqlite3
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# In-process batch accumulator.
#
# When populating a cache over thousands of channels (--cache), calling
# write_channel_counts with autocommit for every channel would fsync the
# whole growing database each time. Instead a --cache run calls
# begin_cache_batch once; writes are folded into a single open transaction and
# committed once at flush_cache_batch / close_cache_connection time. One
# transaction, one fsync at the end.
# ---------------------------------------------------------------------------
_batch = None  # set of (cache_dir, filter_hash) connections with an open txn


def get_cache_file_path(cache_dir, filter_hash):
    """Get the path to the cache database file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"cache_{filter_hash}.db"


def get_index_file_path(cache_dir, filter_hash):
    """Kept for compatibility; the SQLite cache has no separate index file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"cache_index_{filter_hash}.json"


def compute_data_hash(data):
    """Kept for API compatibility; not used by the SQLite cache."""
    import hashlib
    return hashlib.sha256(data).hexdigest()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS channels (
    name TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS counts (
    n      INTEGER NOT NULL,
    phrase TEXT    NOT NULL,
    count  INTEGER NOT NULL,
    PRIMARY KEY (n, phrase)
);
"""


def _connect(cache_dir, filter_hash):
    """Open (and initialize) the SQLite cache DB in WAL + normal sync mode.

    Returns a sqlite3.Connection with row factory. Caller owns the connection.
    """
    db_path = get_cache_file_path(cache_dir, filter_hash)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    # WAL: readers don't block the writer; normal sync trades some durability
    # for much faster commits (we re-populate from transcripts on failure).
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.executescript(_SCHEMA)
    return conn


def load_index(cache_dir, filter_hash):
    """Kept for API compatibility.

    The SQLite cache has no single in-memory index object. Returns a small
    summary dict {token_count, total_channels, channels:set, counts:{}} so any
    caller that inspects it still works; the heavy counts live in the DB and
    are queried through read_aggregate_filtered. Returns None if no cache file
    exists.
    """
    if not get_cache_file_path(cache_dir, filter_hash).exists():
        return None
    conn = _connect(cache_dir, filter_hash)
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='token_count'"
        ).fetchone()
        if row is None:
            return None
        token_count = int(row["value"])
        total = conn.execute(
            "SELECT value FROM meta WHERE key='total_channels'"
        ).fetchone()
        total_channels = int(total["value"]) if total else 0
        names = [r["name"] for r in conn.execute("SELECT name FROM channels")]
        return {
            "token_count": token_count,
            "total_channels": total_channels,
            "channels": {n: True for n in names},
            "counts": {},  # not loaded; query via read_aggregate_filtered
        }
    finally:
        conn.close()


def save_index(cache_dir, filter_hash, index_data):
    """Kept for API compatibility.

    The SQLite cache is written incrementally via write_channel_counts; there
    is no single dict to dump. This accepts an index_data dict shaped like the
    old pickle (token_count, total_channels, channels, counts) and bulk-loads
    it into the DB, replacing any existing data for this filter hash. Used only
    by callers that still hand-build a full aggregate dict.
    """
    db_path = get_cache_file_path(cache_dir, filter_hash)
    conn = _connect(cache_dir, filter_hash)
    try:
        conn.execute("BEGIN")
        conn.execute("DELETE FROM counts")
        conn.execute("DELETE FROM channels")
        conn.execute("DELETE FROM meta")
        counts = index_data.get("counts", {}) or {}
        for n, bucket in counts.items():
            if not bucket:
                continue
            conn.executemany(
                "INSERT INTO counts(n, phrase, count) VALUES (?, ?, ?)",
                [(n, phrase, c) for phrase, c in bucket.items()],
            )
        for name in (index_data.get("channels", {}) or {}):
            conn.execute("INSERT OR IGNORE INTO channels(name) VALUES (?)", (name,))
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('token_count', ?)",
            (str(index_data.get("token_count", 0)),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('total_channels', ?)",
            (str(index_data.get("total_channels", len(index_data.get("channels", {}) or {}))),),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def _new_index():
    """Kept for API compatibility; the SQLite cache initializes lazily."""
    return {"token_count": 0, "total_channels": 0, "channels": {}, "counts": {}}


def begin_cache_batch():
    """Start a batch accumulation session (see module docstring).

    During a batch, write_channel_counts folds into a single open transaction
    per (cache_dir, filter_hash) and commits once at flush time.
    """
    global _batch
    _batch = {}


def flush_cache_batch():
    """Commit any open batch transactions and end the session."""
    global _batch
    if not _batch:
        _batch = None
        return
    for key, conn in list(_batch.items()):
        try:
            conn.execute("COMMIT")
        except sqlite3.OperationalError:
            # No transaction open (already committed); ignore.
            pass
        conn.close()
    _batch = None


def close_cache_connection(cache_dir=None):
    """Flush any pending batch writes. `cache_dir` is accepted for call-site
    compatibility but ignored -- a single run only has one batch session."""
    flush_cache_batch()


def _batch_conn(cache_dir, filter_hash):
    """Get or open the batch transaction connection for this cache."""
    global _batch
    if _batch is None:
        return None
    key = (cache_dir, filter_hash)
    conn = _batch.get(key)
    if conn is None:
        conn = _connect(cache_dir, filter_hash)
        conn.execute("BEGIN")
        _batch[key] = conn
    return conn


def cache_exists(cache_dir, channel, filter_hash):
    """Check if a channel has been folded into the aggregate cache.

    This is a single indexed point lookup (O(log N)), NOT a full load, so it
    stays instant at any corpus size -- the hot path during --cache re-runs
    where every channel is already cached.
    """
    conn = _batch_conn(cache_dir, filter_hash) if _batch is not None else None
    if conn is not None:
        row = conn.execute(
            "SELECT 1 FROM channels WHERE name=? LIMIT 1", (channel,)
        ).fetchone()
        return row is not None
    if not get_cache_file_path(cache_dir, filter_hash).exists():
        return False
    conn = _connect(cache_dir, filter_hash)
    try:
        row = conn.execute(
            "SELECT 1 FROM channels WHERE name=? LIMIT 1", (channel,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def get_channel_info(cache_dir, channel, filter_hash):
    """Kept for API compatibility. The aggregate cache has no per-channel
    breakdown, so this returns None."""
    return None


def _fold_into_conn(conn, channel, token_count, counts):
    """Fold one channel's counted n-grams into the open-transaction connection.

    Uses INSERT ... ON CONFLICT DO UPDATE so existing counts are incremented,
    and INSERT OR IGNORE for the channel name so re-folding is a no-op.
    """
    cur = conn.execute(
        "SELECT 1 FROM channels WHERE name=? LIMIT 1", (channel,)
    ).fetchone()
    if cur is not None:
        return  # already folded; don't double-count
    conn.execute("INSERT OR IGNORE INTO channels(name) VALUES (?)", (channel,))
    for n, bucket in counts.items():
        if not bucket:
            continue
        conn.executemany(
            "INSERT INTO counts(n, phrase, count) VALUES (?, ?, ?) "
            "ON CONFLICT(n, phrase) DO UPDATE SET count = count + excluded.count",
            [(n, phrase, c) for phrase, c in bucket.items()],
        )
    # Update running totals (atomic increment).
    if token_count:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('token_count', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = CAST("
            "CAST(meta.value AS INTEGER) + ? AS TEXT)",
            (str(token_count), token_count),
        )
    conn.execute(
        "INSERT INTO meta(key, value) VALUES ('total_channels', "
        "(SELECT COUNT(*) FROM channels)) "
        "ON CONFLICT(key) DO UPDATE SET value = "
        "CAST((SELECT COUNT(*) FROM channels) AS TEXT)"
    )


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
    if _batch is not None:
        conn = _batch_conn(cache_dir, filter_hash)
        _fold_into_conn(conn, channel, token_count, counts)
        return token_count
    conn = _connect(cache_dir, filter_hash)
    try:
        conn.execute("BEGIN")
        _fold_into_conn(conn, channel, token_count, counts)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
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
    directly from transcripts. Returns None so callers fall through to compute.
    """
    return None


def read_channel_cache_filtered(cache_dir, channel, filter_hash, target_phrases):
    """Kept for API compatibility. See read_channel_cache -- returns None."""
    return None


def get_all_channels(cache_dir, filter_hash):
    """Get list of all channels folded into the aggregate cache."""
    if not get_cache_file_path(cache_dir, filter_hash).exists():
        return []
    conn = _connect(cache_dir, filter_hash)
    try:
        return [r["name"] for r in conn.execute("SELECT name FROM channels")]
    finally:
        conn.close()


def get_total_token_count(cache_dir, filter_hash, exclude_channels=None):
    """Get total token count for the whole aggregate.

    exclude_channels is ignored because the aggregate has no per-channel token
    breakdown; the caller subtracts focus tokens itself when computing a rest
    pool.
    """
    if not get_cache_file_path(cache_dir, filter_hash).exists():
        return 0
    conn = _connect(cache_dir, filter_hash)
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='token_count'"
        ).fetchone()
        return int(row["value"]) if row else 0
    finally:
        conn.close()


def read_multiple_channels_filtered(cache_dir, channels, filter_hash, target_phrases):
    """Kept for API compatibility. The aggregate cache has no per-channel
    breakdown, so this returns {} (callers use read_aggregate_filtered)."""
    return {}


def read_aggregate_filtered(cache_dir, filter_hash, target_phrases, exclude_channels=None):
    """Read the whole-corpus aggregate, filtered to target phrases.

    The cache holds the combined counts for ALL channels (focus included).
    This returns that total filtered to the candidate phrases; the caller
    subtracts the focus channel's counts on the fly to get the rest pool.

    This is an indexed point lookup (batched queries per n), so it returns in
    milliseconds regardless of corpus size and only loads the requested
    phrases into memory.

    Args:
        cache_dir: Cache directory
        filter_hash: Filter hash
        target_phrases: {n: set of phrases} to filter for
        exclude_channels: ignored (kept for API compatibility)

    Returns:
        (total_token_count, {n: {phrase: count}}) or None if cache not available
    """
    if not get_cache_file_path(cache_dir, filter_hash).exists():
        return None
    conn = _connect(cache_dir, filter_hash)
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='token_count'"
        ).fetchone()
        if row is None:
            return None
        total_tokens = int(row["value"])

        aggregate_counts = {}
        # Batch size for IN clauses - SQLite handles ~1000 parameters efficiently
        BATCH_SIZE = 900
        
        for n, phrases in target_phrases.items():
            if not phrases:
                continue
            phrases_list = list(phrases)
            results_for_n = {}
            
            # Process in batches to avoid huge IN clauses
            for i in range(0, len(phrases_list), BATCH_SIZE):
                batch = phrases_list[i:i + BATCH_SIZE]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT phrase, count FROM counts WHERE n=? AND phrase IN ({placeholders})",
                    [n, *batch],
                ).fetchall()
                for r in rows:
                    results_for_n[r["phrase"]] = r["count"]
            
            if results_for_n:
                aggregate_counts[n] = results_for_n

        return total_tokens, aggregate_counts
    finally:
        conn.close()
