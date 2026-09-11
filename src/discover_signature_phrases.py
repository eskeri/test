#!/usr/bin/env python
"""Discover signature phrases: n-grams a channel uses more than all others.

Usage:
    # Pre-compute caches for all channels (no analysis, no output):
    python3 -m src.discover_signature_phrases --cache --duration-from 8:00 ...

    # Normal focus comparison (uses caches if present):
    python3 -m src.discover_signature_phrases --focus ctop ...
"""

import argparse
import sys
from pathlib import Path

# Optional imports for lemmatization
try:
    import spacy
except ImportError:
    spacy = None

try:
    import simplemma
except ImportError:
    simplemma = None

if sys.version_info < (3, 8):
    raise RuntimeError("Python 3.8+ required")

from .comparison_modes import (
    run_single_focus_comparison,
    run_two_channel_comparison,
    run_ttr_zscore,
)
from .token_loading import _discover_channels
from .cache_hashes import _base_filters_hash
from .cache_io import close_cache_connection, begin_cache_batch
from .observation import run_observation, add_observation_args
from .candidate_cache import (
    get_candidate_cache_key,
    add_channel_candidates,
    begin_candidate_cache_batch,
    end_candidate_cache_batch,
)


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    p = argparse.ArgumentParser(
        description="Discover which words/phrases a YouTube channel uses more than all others."
    )
    
    # Add observation feature arguments
    add_observation_args(p)

    # Channel selection
    p.add_argument("channels", nargs="*", help="Channel folder names. Defaults to all under data/input.")
    p.add_argument("--focus", nargs="+", metavar="CHANNEL", help="Focus channel(s) to analyze.")
    p.add_argument("--second-focus", metavar="CHANNEL", help="Second focus channel for two-channel comparison.")
    p.add_argument("--prior-channels", nargs="*", metavar="CHANNEL", help="Prior channels for informative Dirichlet prior. Defaults to all others.")
    p.add_argument("--cache-channels", nargs="+", metavar="CHANNEL", help="Specific channels to cache (for testing). If not provided, uses all channels from --cache mode.")

    # Analysis mode
    p.add_argument("--ttr-zscore", action="store_true", help="Run channel-level TTR z-score report.")
    p.add_argument("--ttr-top-n", type=int, default=180)

    # N-gram configuration
    p.add_argument("--ngram-max", type=int, default=2, choices=range(1, 11))
    p.add_argument("--min-count", type=int, default=5, help="Min occurrences in focus channel. Default: 5.")
    p.add_argument("--top-n", type=int, default=1000, help="Top N phrases per n-gram size. Default: 1000.")

    # Scoring
    p.add_argument("--no-logs-under", type=float, default=None, help="Hide phrases with log-odds below threshold.")
    p.add_argument("--show-exclusive-in-distinct", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--alpha", type=float, default=0.01, help="Legacy uniform Dirichlet prior. Default: 0.01.")
    p.add_argument("--alpha-total", type=float, default=2000.0, help="Total prior weight for Monroe Dirichlet. Default: 2000.0.")
    p.add_argument("--size-proportional-prior", action="store_true", help="Scale Dirichlet priors by corpus size.")

    # Additional metrics
    p.add_argument("--extra-collocation-metrics", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--calculate-log-likelihoods", action="store_true", help="Enable log-likelihood calculation in HTML.")

    # Lemmatization
    p.add_argument("--lemmatize", action="store_true", help="Lemmatize unigrams only.")
    p.add_argument("--lemmatizer", choices=["simplemma", "spacy"], default="simplemma",
                   help="Lemmatizer: simplemma (fast) or spacy (accurate).")

    # Funnel chart
    p.add_argument("--funnel-panel-min-font-size", type=float, default=0.0)
    p.add_argument("--force-rest-pool-graph", action="store_true", help="Force funnel chart for focus vs pooled rest.")
    p.add_argument("--no-stops-graph", action="store_true", help="Remove stopwords from funnel chart labels.")

    # Output
    p.add_argument("--graph-only", action="store_true", help="Output HTML with only the funnel chart.")
    p.add_argument("--output-html", default="temps/discover_signature_phrases.html")

    # General filters
    p.add_argument("--date-from", default="")
    p.add_argument("--date-to", default="")
    p.add_argument("--duration-from", default="")
    p.add_argument("--duration-to", default="")
    p.add_argument("--exclude-live", action="store_true")
    p.add_argument("--token-limit", type=int, default=0)
    p.add_argument("--min-tokens-per-file", type=int, default=0)
    p.add_argument("--min-tokens-total", "--min-words", dest="min_tokens_total", type=int, default=0)

    # Focus-specific filters
    p.add_argument("--focus-token-limit", type=int, default=None, help="Token limit for focus channel only.")
    p.add_argument("--focus-date-from", default="", help="Date filter for focus channel only.")
    p.add_argument("--focus-date-to", default="", help="Date filter for focus channel only.")
    p.add_argument("--focus-duration-from", default="", help="Duration filter for focus channel only.")
    p.add_argument("--focus-duration-to", default="", help="Duration filter for focus channel only.")
    p.add_argument("--focus-exclude-live", action="store_true", help="Exclude live videos for focus channel only.")
    p.add_argument("--focus-video-contribute", type=int, default=None,
                   help="Max count each video can contribute per phrase. Focus channel is NOT cached when used.")

    # Second-focus-specific filters
    p.add_argument("--second-focus-date-from", default="")
    p.add_argument("--second-focus-date-to", default="")
    p.add_argument("--second-focus-duration-from", default="")
    p.add_argument("--second-focus-duration-to", default="")
    p.add_argument("--second-focus-exclude-live", action="store_true")

    # Caching
    p.add_argument("--cache", action="store_true",
                   help="Save per-channel n-gram counts for instant reuse. Can be used alone to pre-populate caches. Use --cache-channels to specify specific channels.")

    args, unknown = p.parse_known_args()
    if unknown:
        print(f"Ignoring unknown args: {' '.join(unknown)}")

    base_dir = Path(__file__).resolve().parent.parent

    # --- Validation ---------------------------------------------------------
    if args.second_focus and not args.focus:
        p.error("--focus is required when using --second-focus.")

    if not args.channels:
        discovered = _discover_channels(base_dir)
        if not discovered:
            p.error("No channels given and none found under data/input.")
        args.channels = discovered
        print(f"No channels passed; using all {len(args.channels)} found in data/input.")

    # Defaults
    if args.focus is None:
        args.focus = []
    if args.focus_token_limit is None:
        args.focus_token_limit = args.token_limit
    if not args.focus_date_from:
        args.focus_date_from = args.date_from
    if not args.focus_date_to:
        args.focus_date_to = getattr(args, "date_to", "")
    if not args.focus_duration_from:
        args.focus_duration_from = args.duration_from
    if not args.focus_duration_to:
        args.focus_duration_to = getattr(args, "duration_to", "")
    if not args.focus_exclude_live:
        args.focus_exclude_live = args.exclude_live

    if not args.second_focus_date_from:
        args.second_focus_date_from = args.date_from
    if not args.second_focus_date_to:
        args.second_focus_date_to = getattr(args, "date_to", "")
    if not args.second_focus_duration_from:
        args.second_focus_duration_from = args.duration_from
    if not args.second_focus_duration_to:
        args.second_focus_duration_to = getattr(args, "duration_to", "")
    if not args.second_focus_exclude_live:
        args.second_focus_exclude_live = args.exclude_live

    if args.funnel_panel_min_font_size < 0:
        p.error("--funnel-panel-min-font-size must be >= 0")

    if args.lemmatize:
        if args.lemmatizer == "spacy" and spacy is None:
            p.error("--lemmatizer spacy requires spaCy. Install: pip install spacy && python -m spacy download en_core_web_sm")
        if args.lemmatizer == "simplemma" and simplemma is None:
            p.error("--lemmatizer simplemma requires simplemma. Install: pip install simplemma")

    if args.second_focus:
        if args.prior_channels is None:
            all_channels = _discover_channels(base_dir)
            args.prior_channels = [c for c in all_channels if c not in [args.focus[0], args.second_focus]]
            print(f"No prior-channels specified; using {len(args.prior_channels)} channels (excluding focus and second-focus)")
        else:
            for pc in args.prior_channels:
                if pc not in args.channels:
                    print(f"Adding prior channel '{pc}' to channels list")
                    args.channels.append(pc)

    # Validate observation mode
    if getattr(args, "allow_observe", False):
        if not args.observe_channel:
            p.error("--observe-channel is required when using --allow-observe")

    return args


def _run_cache_prepopulation(args, base_dir, cache_dir, ngram_sizes, nlp):
    """Cache every channel's n-grams, folding each into the dumb aggregate.

    Each channel's tokens are counted in memory and folded into one global
    aggregate cache (cache_{filter_hash}.db). To avoid re-reading and
    re-writing the whole growing cache for every channel (O(N^2)), the whole
    run is wrapped in a single batch: writes are folded into one open
    transaction and committed once at the end. Channels already folded are skipped,
    so re-running after an interruption picks up where it left off.
    
    Additionally, if candidate cache is enabled, we also store candidate phrases
    (those meeting min_count threshold) in a separate disk-based cache for
    incremental accumulation.
    """
    from .ngram_counting import get_channel_ngrams

    # Use specific channels if provided, otherwise use all channels
    channels_to_cache = args.cache_channels if args.cache_channels else args.channels

    print(f"Pre-populating caches for {len(channels_to_cache)} channels\u2026")
    filter_hash = _base_filters_hash(args)

    # Start both batch caches
    begin_cache_batch()
    candidate_batch_conns = begin_candidate_cache_batch()
    
    # Get candidate cache key
    cand_filter_hash, cand_ngram_max, cand_min_count = get_candidate_cache_key(args)

    try:
        for idx, channel in enumerate(channels_to_cache, 1):
            result = get_channel_ngrams(
                base_dir, channel, args, cache_dir, ngram_sizes, nlp,
                verbose=True, use_base_filters=True,
            )
            if result is None:
                print(f"  {idx:>4}/{len(channels_to_cache)}  {channel}: FAILED (no tokens)")
                continue

            tc, ngram_counts = result
            if tc == 0:
                print(f"  {idx:>4}/{len(channels_to_cache)}  {channel}: already cached, skipping")
                continue
            
            # Also add to candidate cache if enabled
            if getattr(args, "cache", False):
                new, updated = add_channel_candidates(
                    cache_dir, channel, cand_filter_hash, cand_ngram_max, cand_min_count,
                    ngram_counts, tc, candidate_batch_conns
                )
                if new > 0 or updated > 0:
                    print(f"  {idx:>4}/{len(channels_to_cache)}  {channel}: {tc:,} tokens \u2192 cached ({new} new candidates, {updated} updated)")
                else:
                    print(f"  {idx:>4}/{len(channels_to_cache)}  {channel}: {tc:,} tokens \u2192 cached")
            else:
                print(f"  {idx:>4}/{len(channels_to_cache)}  {channel}: {tc:,} tokens \u2192 cached")
    finally:
        # Flushes the batched aggregate writes to the cache database.
        close_cache_connection(cache_dir)
        end_candidate_cache_batch(candidate_batch_conns)

    return 0


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    output_path = (base_dir / args.output_html).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = base_dir / "cache" / "leaderboard"
    ngram_sizes = list(range(1, args.ngram_max + 1))

    # Lemmatizer setup
    nlp = None
    if args.lemmatize:
        lemmatizer_type = args.lemmatizer
        try:
            if lemmatizer_type == "simplemma":
                if simplemma is None:
                    raise RuntimeError("simplemma not installed.")
                print("Lemmatization enabled (simplemma)")
                nlp = "simplemma"
            else:  # spacy
                if spacy is None:
                    raise RuntimeError("spaCy not installed.")
                nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat", "senter"])
                print("Lemmatization enabled (spaCy)")
        except Exception as e:
            print(f"Lemmatizer failed: {e}")
            return 1

    # --- Mode dispatch -------------------------------------------------------
    
    # Check if observation mode is requested
    if getattr(args, "allow_observe", False):
        return run_observation(args, base_dir, cache_dir, ngram_sizes, nlp)
    
    if args.ttr_zscore:
        return run_ttr_zscore(args, base_dir, cache_dir, nlp)

    if not args.focus:
        if not args.cache:
            print("Error: --focus is required unless --ttr-zscore, --cache, or --allow-observe is set.")
            return 1
        return _run_cache_prepopulation(args, base_dir, cache_dir, ngram_sizes, nlp)

    # Normal focus comparison
    for fc in args.focus:
        if fc not in args.channels:
            print(f"Adding focus channel '{fc}' to channels list")
            args.channels.append(fc)

    if args.second_focus:
        return run_two_channel_comparison(args, base_dir, cache_dir, ngram_sizes, nlp)
    else:
        return run_single_focus_comparison(args, base_dir, cache_dir, ngram_sizes, nlp)


if __name__ == "__main__":
    raise SystemExit(main())
