"""N-gram counting functions with per-channel caching, streaming, and candidate-aware counting."""

import argparse
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator

from .cache_hashes import _base_filters_hash, _filters_hash
from .cache_io_simple import (
    write_channel_cache,
    write_channel_counts,
    cache_exists,
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

    The cache is a dumb aggregate over ALL channels combined, so it cannot
    return a single channel's counts. Therefore this function always counts
    a single channel from its transcripts in memory.

    Write path (when args.cache is set and candidates is None): the counted
    n-grams are folded into the single global aggregate pickle via
    cache_io.write_channel_counts. During a --cache run the caller wraps the
    loop in begin_cache_batch / flush_cache_batch so the aggregate is loaded
    once and written once, not re-read/re-written per channel.
    """
    filter_hash = _base_filters_hash(args) if use_base_filters else _filters_hash(args)

    # --cache mode skips channels already folded into the aggregate.
    should_write_cache = getattr(args, "cache", False) and candidates is None
    if should_write_cache and cache_exists(cache_dir, channel, filter_hash):
        if verbose:
            print(f"  {channel}: already in aggregate cache, skipping")
        # Caller (cache prepopulation) only needs a token count; return 0
        # tokens as a signal that it was already cached.
        return 0, {n: Counter() for n in ngram_sizes}

    if verbose:
        print(f"  {channel}: computing n-grams…", end=" ", flush=True)

    token_stream = _make_token_stream(base_dir, channel, args, nlp)
    if token_stream is None:
        if verbose:
            print("no tokens")
        return None

    # Count once in memory (bounded by this channel's vocabulary).
    pair_result = _compute_ngrams_from_pair_streaming(token_stream, ngram_sizes, candidates)
    if pair_result is None:
        if verbose:
            print("no tokens")
        return None

    token_count, ngram_counts = pair_result

    if should_write_cache:
        # Fold the counted dict straight into the aggregate (no re-streaming).
        write_channel_counts(
            cache_dir, channel, filter_hash,
            {n: dict(c) for n, c in ngram_counts.items()},
            token_count,
        )
        if verbose:
            print(f"done ({token_count:,} tokens) [cached]")
    else:
        if verbose:
            unique = sum(len(c) for c in ngram_counts.values())
            print(f"done ({token_count:,} tokens, {unique:,} unique)")

    return token_count, ngram_counts
