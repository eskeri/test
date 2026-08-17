"""N-gram counting functions with per-channel caching, streaming, and candidate-aware counting."""

import argparse
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator

from .cache_hashes import _base_filters_hash, _filters_hash
from .cache_io_simple import (
    read_channel_cache,
    write_channel_cache,
)


def _ngram_pairs_from_token_stream(token_stream, ngram_sizes):
    """Turn a stream of tokens into a stream of (n, phrase) pairs.

    `token_stream` yields either plain token strings, or (lemma, original)
    tuples when lemmatization is on. Unigrams use the lemma (if present);
    higher n-grams always use the original token, so word order/context
    isn't affected by lemmatization.

    This is a generator - it never holds more than one small sliding
    window (per n-gram size) in memory, no matter how long the stream is.
    """
    windows = {n: deque(maxlen=n) for n in ngram_sizes if n > 1}
    for item in token_stream:
        if item is None:
            continue
        if isinstance(item, tuple) and len(item) == 2:
            lemma, original = item
        else:
            lemma = original = item

        if 1 in ngram_sizes:
            yield (1, lemma)

        for n, window in windows.items():
            window.append(original)
            if len(window) == n:
                yield (n, " ".join(window))


def _compute_ngrams_from_pair_streaming(
    token_stream,
    ngram_sizes,
    candidates=None,
):
    """Compute n-gram counts from a streaming token source, fully in memory.

    Used for analysis-time reads: either a candidate set is supplied
    (bounding memory to that set - typically a few thousand phrases from
    one focus channel), or the caller already knows the corpus is small
    (e.g. a single comparison channel, not the whole 1500-channel dataset).
    For caching every channel of a large dataset, use
    `get_channel_ngrams(..., use_cache_write=True)` instead, which streams
    to disk in chunks and never materializes a whole channel in memory.
    """
    result = {n: Counter() for n in ngram_sizes}
    token_count = 0
    saw_token = False

    for n, phrase in _ngram_pairs_from_token_stream(token_stream, ngram_sizes):
        saw_token = True
        if n == 1:
            token_count += 1
        if candidates is not None and candidates.get(n) is not None and phrase not in candidates[n]:
            continue
        result[n][phrase] += 1

    if not saw_token:
        return None
    return token_count, result


def _compute_ngrams_per_video(
    video_tokens,
    ngram_sizes,
    max_contribute,
    lemmatised_video_tokens=None,
):
    """Count n-grams across videos, capping each video's contribution.

    Used only for --focus-video-contribute, which is deliberately never
    cached (see discover_signature_phrases.py) and only ever applied to a
    handful of focus channels, so an in-memory result here is fine.
    """
    result = {n: Counter() for n in ngram_sizes}

    for idx, (video_id, tokens) in enumerate(video_tokens):
        lemm_tokens = None
        if lemmatised_video_tokens and idx < len(lemmatised_video_tokens):
            if lemmatised_video_tokens[idx][0] == video_id:
                lemm_tokens = lemmatised_video_tokens[idx][1]

        for n in ngram_sizes:
            source_tokens = lemm_tokens if (n == 1 and lemm_tokens) else tokens
            if not source_tokens:
                continue
            counter = Counter(source_tokens) if n == 1 else _count_ngrams_simple(source_tokens, n)
            for gram, count in counter.items():
                result[n][gram] += min(count, max_contribute)
    return result


def _count_ngrams_simple(tokens, n):
    result = Counter()
    limit = len(tokens) - n + 1
    if limit <= 0:
        return result
    for i in range(limit):
        result[" ".join(tokens[i:i + n])] += 1
    return result


def _make_token_stream(base_dir, channel, args, nlp):
    """Build the (possibly lemmatized) token stream for a channel."""
    from .token_loading import stream_tokens

    raw_stream = stream_tokens(base_dir, channel, args)
    if raw_stream is None:
        return None

    need_lemmatise = getattr(args, "lemmatize", False) and nlp is not None
    if not need_lemmatise:
        return raw_stream

    from .lemmatization import _lemmatize_token, _lemmatize_token_simplemma
    lemma_cache = {}
    lemmatiser = (
        (lambda t: _lemmatize_token_simplemma(t)) if nlp == "simplemma"
        else (lambda t: _lemmatize_token(t, nlp))
    )

    def cached_lemmatiser(token):
        cached = lemma_cache.get(token)
        if cached is None:
            cached = lemmatiser(token)
            lemma_cache[token] = cached
        return cached

    def decorated():
        for token in raw_stream:
            if token is None:
                continue
            yield (cached_lemmatiser(token), token)

    return decorated()


def get_channel_ngrams(
    base_dir,
    channel,
    args,
    cache_dir,
    ngram_sizes,
    nlp=None,
    verbose=True,
    use_base_filters=False,
    candidates=None,
):
    """Return n-gram counts for *channel*.

    Read path: try the per-channel cache file first (one sequential
    decompress, no seeking - "instant" in the sense that it's a single
    linear read of one file).

    Write path (when args.cache is set and candidates is None, i.e. we're
    caching the *full* channel, not a candidate-filtered subset): stream
    tokens straight to disk in bounded-size chunks and merge them into one
    compressed cache file. Peak memory for this path does not grow with
    channel size - a channel with 50M tokens takes the same peak memory as
    one with 50K tokens (see cache_io.CHUNK_TOKEN_LIMIT). This is what
    makes `--cache` safe to run over a 1500+ channel / billions-of-tokens
    dataset on a laptop.
    """
    filter_hash = _base_filters_hash(args) if use_base_filters else _filters_hash(args)

    # 1 - try cache first with requested hash
    cached = read_channel_cache(cache_dir, channel, filter_hash)
    
    # 1.5 - if not found and using focus filters without per-video limiting, try base hash for compatibility
    focus_video_contribute = getattr(args, "focus_video_contribute", None)
    if cached is None and not use_base_filters and focus_video_contribute is None:
        base_hash = _base_filters_hash(args)
        cached = read_channel_cache(cache_dir, channel, base_hash)
        if cached is not None:
            print(f"  {channel}: using base-filter cache (focus filters not cached)")
    
    if cached is not None:
        cached_token_count, cached_data = cached
        if all(n in cached_data for n in ngram_sizes):
            if args.token_limit > 0 and cached_token_count > args.token_limit:
                if verbose:
                    print(f"  {channel}: cached ({cached_token_count:,} tokens) exceeds "
                          f"limit ({args.token_limit:,}), recomputing…")
            else:
                result = {n: Counter(cached_data.get(n, {})) for n in ngram_sizes}
                if verbose:
                    unique = sum(len(c) for c in result.values())
                    print(f"  {channel}: instant cache hit ({cached_token_count:,} tokens, {unique:,} unique)")
                return cached_token_count, result

    # 2 - compute
    if verbose:
        print(f"  {channel}: computing n-grams…", end=" ", flush=True)

    token_stream = _make_token_stream(base_dir, channel, args, nlp)
    if token_stream is None:
        if verbose:
            print("no tokens")
        return None

    should_write_cache = getattr(args, "cache", False) and candidates is None

    if should_write_cache:
        # Stream to simple cache format
        pairs = _ngram_pairs_from_token_stream(token_stream, ngram_sizes)
        token_count = write_channel_cache(cache_dir, channel, filter_hash, pairs)
        if token_count == 0:
            if verbose:
                print("no tokens")
            return None
        if verbose:
            print(f"done ({token_count:,} tokens) [cached]")
        # Read the counts back so callers still get a result this run
        cached_data = read_channel_cache(cache_dir, channel, filter_hash)
        if cached_data is None:
            return token_count, {n: Counter() for n in ngram_sizes}
        _, data = cached_data
        result = {n: Counter(data.get(n, {})) for n in ngram_sizes}
        return token_count, result

    # In-memory path: either not caching, or this is a candidate-filtered
    # read, whose memory footprint is bounded by the candidate set size.
    pair_result = _compute_ngrams_from_pair_streaming(token_stream, ngram_sizes, candidates)
    if pair_result is None:
        if verbose:
            print("no tokens")
        return None

    token_count, ngram_counts = pair_result
    if verbose:
        unique = sum(len(c) for c in ngram_counts.values())
        print(f"done ({token_count:,} tokens, {unique:,} unique)")
    return token_count, ngram_counts