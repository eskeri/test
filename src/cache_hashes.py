"""Cache hash functions for generating unique cache keys."""

import hashlib
import json
from typing import List

from .constants import CACHE_VERSION


def _base_filters_hash(args) -> str:
    """Hash of filters that don't depend on which channel is the focus."""
    key = json.dumps({
        "cache_version": CACHE_VERSION,
        "date_from": args.date_from,
        "date_to": getattr(args, "date_to", ""),
        "duration_from": args.duration_from,
        "duration_to": getattr(args, "duration_to", ""),
        "exclude_live": args.exclude_live,
        "token_limit": args.token_limit,
        "min_tokens_per_file": args.min_tokens_per_file,
        "min_tokens_total": args.min_tokens_total,
        "lemmatize": bool(getattr(args, "lemmatize", False)),
        "lemmatizer": getattr(args, "lemmatizer", "simplemma"),
    }, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _filters_hash(args) -> str:
    """Hash of all filters, including focus-specific ones."""
    key = json.dumps({
        "cache_version": CACHE_VERSION,
        "date_from": args.date_from,
        "date_to": getattr(args, "date_to", ""),
        "duration_from": args.duration_from,
        "duration_to": getattr(args, "duration_to", ""),
        "exclude_live": args.exclude_live,
        "token_limit": args.token_limit,
        "min_tokens_per_file": args.min_tokens_per_file,
        "min_tokens_total": args.min_tokens_total,
        "lemmatize": bool(getattr(args, "lemmatize", False)),
        "lemmatizer": getattr(args, "lemmatizer", "simplemma"),
        "focus_date_from": getattr(args, "focus_date_from", ""),
        "focus_date_to": getattr(args, "focus_date_to", ""),
        "focus_duration_from": getattr(args, "focus_duration_from", ""),
        "focus_duration_to": getattr(args, "focus_duration_to", ""),
        "focus_exclude_live": getattr(args, "focus_exclude_live", False),
        "focus_video_contribute": getattr(args, "focus_video_contribute", None),
        # focus_token_limit is intentionally excluded - it only affects
        # subset selection, not which tokens exist.
        "second_focus_date_from": getattr(args, "second_focus_date_from", ""),
        "second_focus_date_to": getattr(args, "second_focus_date_to", ""),
        "second_focus_duration_from": getattr(args, "second_focus_duration_from", ""),
        "second_focus_duration_to": getattr(args, "second_focus_duration_to", ""),
        "second_focus_exclude_live": getattr(args, "second_focus_exclude_live", False),
    }, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


# Alias kept only for call-site clarity ("this is the rest-pool cache key").
# Must never diverge from _base_filters_hash, so it has no body of its own.
_rest_pool_filters_hash = _base_filters_hash


def _channels_hash(channels: List[str]) -> str:
    return hashlib.sha1(",".join(sorted(channels)).encode()).hexdigest()[:16]


def _ngram_sizes_hash(ngram_sizes: List[int]) -> str:
    return hashlib.sha1(",".join(map(str, sorted(ngram_sizes))).encode()).hexdigest()[:8]