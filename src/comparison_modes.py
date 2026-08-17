"""Comparison mode functions for signature phrase analysis.

These functions handle different analysis modes:
- TTR z-score reports
- Two-channel comparison with informative Dirichlet prior
- Single focus vs pooled rest comparison
"""

import argparse
import subprocess
import sys
import webbrowser
from collections import Counter
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .html_output import build_html
from .ngram_counting import get_channel_ngrams, _compute_ngrams_per_video
from .scoring import compute_collocation_metrics, compute_scores, partition_scores
from .token_loading import load_tokens_per_video
from .funnel_chart import build_funnel_chart
from .cache_hashes import _base_filters_hash, _filters_hash
from .cache_io_simple import (
    cache_exists,
    read_channel_cache_filtered,
    read_multiple_channels_filtered,
    read_aggregate_filtered,
)

def run_ttr_zscore(args, base_dir, cache_dir, nlp=None):
    """Run TTR (Type-Token Ratio) z-score report across all channels."""
    print(f"TTR z-score mode | channels: {len(args.channels)} | top/bottom: {args.ttr_top_n}")

    rows = []
    for i, channel in enumerate(args.channels, 1):
        result = get_channel_ngrams(base_dir, channel, args, cache_dir, [1], nlp, verbose=False)
        if result is None:
            continue
        tc, counts = result
        types = len(counts.get(1, {}))
        rows.append((channel, tc, types, types / tc if tc else 0.0))
        if i % 50 == 0 or i == len(args.channels):
            print(f"  {i}/{len(args.channels)} ({len(rows)} usable)")

    if len(rows) < 2:
        print("Need at least 2 usable channels.")
        return 1

    mean = sum(r[3] for r in rows) / len(rows)
    std = sqrt(sum((r[3] - mean) ** 2 for r in rows) / (len(rows) - 1))
    scored = sorted(
        [(ch, tc, ty, ttr, (ttr - mean) / std if std else 0.0) for ch, tc, ty, ttr in rows],
        key=lambda r: -r[4],
    )

    print(f"\nUsable: {len(scored)} | Mean TTR: {mean:.6f} | Std: {std:.6f}")
    top_n = max(1, args.ttr_top_n)
    print("\nTop TTR z-scores:")
    for rank, (ch, tc, ty, ttr, z) in enumerate(scored[:top_n], 1):
        print(f"  {rank:>3}. {ch:<24} z={z:>7.3f}  ttr={ttr:.6f}  types={ty:,}  tokens={tc:,}")
    print("\nBottom TTR z-scores:")
    for rank, (ch, tc, ty, ttr, z) in enumerate(reversed(scored[-top_n:]), 1):
        print(f"  {rank:>3}. {ch:<24} z={z:>7.3f}  ttr={ttr:.6f}  types={ty:,}  tokens={tc:,}")
    return 0


# ----------------------------------------------------------------------
# Focus channel loaders (cached / per‑video)
# ----------------------------------------------------------------------

def _load_focus_channels(
    base_dir,
    focus_channels,
    args,
    cache_dir,
    ngram_sizes,
    nlp=None,
    use_focus_token_limit=True,
    use_focus_filters=True,
):
    """Load focus channel n-grams with proper filter application."""
    load_args = argparse.Namespace(**vars(args))
    if use_focus_token_limit:
        load_args.token_limit = args.focus_token_limit
    if use_focus_filters:
        load_args.date_from = args.focus_date_from
        load_args.date_to = args.focus_date_to
        load_args.duration_from = args.focus_duration_from
        load_args.duration_to = args.focus_duration_to
        load_args.exclude_live = args.focus_exclude_live

    focus_ngram_counts = {}
    focus_token_counts = {}

    for fc in focus_channels:
        print(f"Loading focus channel {fc}...")
        result = get_channel_ngrams(
            base_dir, fc, load_args, cache_dir, ngram_sizes, nlp,
            verbose=True, use_base_filters=False,
        )
        if result is None:
            print(f"  Skipped {fc}: no usable tokens")
            continue
        focus_token_counts[fc], focus_ngram_counts[fc] = result
        print(f"  {fc}: {focus_token_counts[fc]:,} tokens loaded")

    return focus_ngram_counts, focus_token_counts


def _load_focus_channels_per_video(
    base_dir,
    focus_channels,
    args,
    cache_dir,
    ngram_sizes,
    max_contribute,
    nlp=None,
    use_focus_token_limit=True,
    use_focus_filters=True,
):
    """Load focus channel n-grams with per-video count limiting (not cached)."""
    print(f"DEBUG: _load_focus_channels_per_video called")
    load_args = argparse.Namespace(**vars(args))
    if use_focus_token_limit:
        load_args.token_limit = args.focus_token_limit
    if use_focus_filters:
        load_args.date_from = args.focus_date_from
        load_args.date_to = args.focus_date_to
        load_args.duration_from = args.focus_duration_from
        load_args.duration_to = args.focus_duration_to
        load_args.exclude_live = args.focus_exclude_live

    focus_ngram_counts = {}
    focus_token_counts = {}

    for fc in focus_channels:
        print(f"DEBUG: Loading video tokens for {fc}")
        video_tokens = load_tokens_per_video(base_dir, fc, load_args)
        print(f"DEBUG: Video tokens loaded: {video_tokens is not None}")
        if video_tokens is None:
            print(f"  Skipped {fc}: no usable tokens")
            continue

        # Apply token limit – keep most recent videos
        if load_args.token_limit > 0:
            total_tokens = sum(len(tokens) for _, tokens in video_tokens)
            if total_tokens > load_args.token_limit:
                accumulated = 0
                filtered = []
                for video_id, tokens in reversed(video_tokens):
                    if accumulated + len(tokens) <= load_args.token_limit:
                        filtered.append((video_id, tokens))
                        accumulated += len(tokens)
                    else:
                        remaining = load_args.token_limit - accumulated
                        if remaining > 0:
                            filtered.append((video_id, tokens[-remaining:]))
                        break
                video_tokens = list(reversed(filtered))

        total_tokens = sum(len(tokens) for _, tokens in video_tokens)
        if total_tokens < load_args.min_tokens_total:
            print(f"  Skipped {fc}: not enough tokens ({total_tokens:,})")
            continue

        print(f"  {fc}: {total_tokens:,} tokens (per-video max: {max_contribute}) - counting...", end=" ", flush=True)

        # Lemmatise for unigrams if needed
        lemmatised_video_tokens = None
        if nlp is not None and 1 in ngram_sizes:
            if nlp == "simplemma":
                from .lemmatization import _lemmatize_tokens_simplemma
                lemmatised_video_tokens = [
                    (vid, _lemmatize_tokens_simplemma(tokens)) for vid, tokens in video_tokens
                ]
            else:
                from .lemmatization import _lemmatize_tokens
                lemmatised_video_tokens = [
                    (vid, _lemmatize_tokens(tokens, nlp)) for vid, tokens in video_tokens
                ]

        ngram_counts = _compute_ngrams_per_video(
            video_tokens, ngram_sizes, max_contribute,
            lemmatised_video_tokens,
        )

        total_unique = sum(len(c) for c in ngram_counts.values())
        print(f"done ({total_unique:,} unique)")

        focus_token_counts[fc] = total_tokens
        focus_ngram_counts[fc] = ngram_counts

    return focus_ngram_counts, focus_token_counts


# ----------------------------------------------------------------------
# Custom filter loader (used for second-focus comparison)
# ----------------------------------------------------------------------

def _load_channel_with_custom_filters(
    base_dir,
    channel,
    args,
    cache_dir,
    ngram_sizes,
    nlp=None,
    token_limit=None,
    date_from=None,
    date_to=None,
    duration_from=None,
    duration_to=None,
    exclude_live=None,
):
    """Load a channel with optional filter overrides."""
    load_args = argparse.Namespace(**vars(args))
    if token_limit is not None:
        load_args.token_limit = token_limit
    if date_from is not None:
        load_args.date_from = date_from
    if date_to is not None:
        load_args.date_to = date_to
    if duration_from is not None:
        load_args.duration_from = duration_from
    if duration_to is not None:
        load_args.duration_to = duration_to
    if exclude_live is not None:
        load_args.exclude_live = exclude_live

    return get_channel_ngrams(
        base_dir, channel, load_args, cache_dir, ngram_sizes, nlp,
        verbose=True, use_base_filters=False,
    )


# ----------------------------------------------------------------------
# Rest‑pool loading (uses aggregate when available)
# ----------------------------------------------------------------------

def _load_rest_channels_with_candidates(
    base_dir,
    args,
    cache_dir,
    ngram_sizes,
    focus_channels,
    nlp=None,
    candidate_phrases=None,
):
    """Load rest channels, using the global aggregate when available.

    With the global aggregate, the rest pool is computed by subtracting
    focus channels' base-filter counts from the aggregate. This is nearly
    instantaneous and does not require loading per-channel files.
    """
    rest_args = argparse.Namespace(**vars(args))
    # Reset focus‑specific filters
    rest_args.date_from = args.date_from
    rest_args.date_to = getattr(args, "date_to", "")
    rest_args.duration_from = args.duration_from
    rest_args.duration_to = getattr(args, "duration_to", "")
    rest_args.exclude_live = args.exclude_live
    rest_args.token_limit = args.token_limit
    rest_args.focus_date_from = ""
    rest_args.focus_date_to = ""
    rest_args.focus_duration_from = ""
    rest_args.focus_duration_to = ""
    rest_args.focus_exclude_live = False
    rest_args.second_focus_date_from = ""
    rest_args.second_focus_date_to = ""
    rest_args.second_focus_duration_from = ""
    rest_args.second_focus_duration_to = ""
    rest_args.second_focus_exclude_live = False

    focus_set = set(focus_channels or [])
    rest_counts = {n: Counter() for n in ngram_sizes}
    rest_token_count = 0
    rest_channels_used = []
    skipped = 0
    ngram_hits = 0

    # Deduplicate and exclude focus channels
    seen = set()
    rest_channels = []
    for channel in args.channels:
        if channel in focus_set or channel in seen:
            continue
        seen.add(channel)
        rest_channels.append(channel)

    print(f"  Rest pool: {len(rest_channels)} channels to process")

    filter_hash = _base_filters_hash(rest_args)

    # Try instant aggregate read first
    agg_result = read_aggregate_filtered(
        cache_dir, filter_hash, candidate_phrases or {}, focus_channels
    )
    
    # If not found, try with the original args hash (might have focus-specific params)
    if agg_result is None:
        original_hash = _base_filters_hash(args)
        if original_hash != filter_hash:
            agg_result = read_aggregate_filtered(
                cache_dir, original_hash, candidate_phrases or {}, focus_channels
            )
            if agg_result is not None:
                print("  Using original args cache (different filter hash)")

    if agg_result is not None:
        print("  Using instant aggregate cache read…")
        rest_token_count, rest_counts = agg_result
        rest_channels_used = rest_channels  # all rest channels are considered used
        ngram_hits = len(rest_channels_used)
        skipped = 0
        rest_counts_full = rest_counts.get(1, Counter())
        print(
            f"  {len(rest_channels)}/{len(rest_channels)} - "
            f"{len(rest_channels_used)} loaded, {skipped} skipped, {ngram_hits} cache hits"
        )
        print(
            f"\n  Rest pool complete: {len(rest_channels_used)} channels, "
            f"{rest_token_count:,} tokens, {ngram_hits} cached, {skipped} skipped"
        )
        return (
            rest_counts,
            rest_counts_full,
            rest_token_count,
            len(rest_channels_used),
            rest_channels_used,
            ngram_hits,
            skipped,
        )

    # Fallback: simple sequential filtered reads (no threading)
    print("  Streaming candidate-filtered cache reads (sequential)…")
    channel_data = read_multiple_channels_filtered(
        cache_dir, rest_channels, filter_hash, candidate_phrases or {}
    )

    for channel in rest_channels:
        data = channel_data.get(channel)
        if data is None:
            skipped += 1
            continue
        token_count, channel_counts = data
        rest_token_count += token_count
        rest_channels_used.append(channel)
        ngram_hits += 1
        for n in ngram_sizes:
            rest_counts[n].update(channel_counts.get(n, Counter()))

    print(
        f"  {len(rest_channels)}/{len(rest_channels)} - "
        f"{len(rest_channels_used)} loaded, {skipped} skipped, {ngram_hits} cache hits"
    )

    rest_counts_full = rest_counts.get(1, Counter())
    print(
        f"\n  Rest pool complete: {len(rest_channels_used)} channels, "
        f"{rest_token_count:,} tokens, {ngram_hits} cached, {skipped} skipped"
    )
    return (
        rest_counts,
        rest_counts_full,
        rest_token_count,
        len(rest_channels_used),
        rest_channels_used,
        ngram_hits,
        skipped,
    )


# ----------------------------------------------------------------------
# HTML / filter helpers
# ----------------------------------------------------------------------

def _build_filters_dict(args, alpha_desc, extra_filters=None):
    filters = {
        "date_from": args.date_from or "(none)",
        "date_to": getattr(args, "date_to", "") or "(none)",
        "duration_from": args.duration_from or "(none)",
        "exclude_live": args.exclude_live,
        "token_limit": f"{args.token_limit:,}" if args.token_limit else "(all)",
        "focus_token_limit": f"{args.focus_token_limit:,}" if args.focus_token_limit else "(all)",
        "min_words": f"{args.min_tokens_total:,}" if args.min_tokens_total else "(none)",
        "ngram_max": args.ngram_max,
        "min_count": args.min_count,
        "top_n": args.top_n,
        "alpha": alpha_desc,
        "no_logs_under": f"{args.no_logs_under:.3f}" if args.no_logs_under is not None else "(disabled)",
        "lemmatize": args.lemmatize,
        "lemmatizer": args.lemmatizer,
        "funnel_panel_min_font_size": args.funnel_panel_min_font_size,
        "no_stops_graph": args.no_stops_graph,
    }
    if extra_filters:
        filters.update(extra_filters)
    return filters


def _write_html_output(base_dir, output_path, html_blob):
    output = (base_dir / output_path).resolve()
    output.write_text(html_blob, encoding="utf-8")
    print(f"\nWrote: {output}")
    try:
        if not webbrowser.open(output.as_uri()) and sys.platform == "darwin":
            subprocess.run(["open", str(output)], check=False)
    except Exception:
        pass


def _print_top_results(results):
    for channel, by_n in results.items():
        for n in sorted(by_n):
            n_label = ["", "word", "bigram", "trigram"][n] if n <= 3 else f"{n}-gram"
            for cat in ("exclusive", "distinctive", "dominant"):
                items = by_n[n].get(cat, [])
                if items:
                    s = items[0]
                    print(
                        f"  {channel} top {n_label} [{cat}]: "
                        f"'{s.phrase}' (z={s.z_score:.1f}, {s.focus_count:,} vs {s.rest_count:,})"
                    )
                    break


# ----------------------------------------------------------------------
# Two‑channel comparison with informative Dirichlet prior
# ----------------------------------------------------------------------

def run_two_channel_comparison(
    args,
    base_dir,
    cache_dir,
    ngram_sizes,
    nlp=None,
):
    focus_channel = args.focus[0]
    second_focus = args.second_focus
    prior_channels = args.prior_channels

    # Load focus channel
    print(f"\n[Pass 1] Loading focus channel: {focus_channel}")
    focus_result = _load_channel_with_custom_filters(
        base_dir, focus_channel, args, cache_dir, ngram_sizes, nlp,
        token_limit=args.focus_token_limit,
        date_from=args.focus_date_from, date_to=args.focus_date_to,
        duration_from=args.focus_duration_from, duration_to=args.focus_duration_to,
        exclude_live=args.focus_exclude_live,
    )
    if focus_result is None:
        print(f"  Skipped {focus_channel}: no usable tokens")
        return 1
    focus_token_count, focus_ngram_counts = focus_result
    print(f"  {focus_channel}: {focus_token_count:,} tokens")

    # Load second-focus channel
    print(f"\n[Pass 2] Loading second-focus: {second_focus}")
    second_result = _load_channel_with_custom_filters(
        base_dir, second_focus, args, cache_dir, ngram_sizes, nlp,
        token_limit=args.token_limit,
        date_from=args.second_focus_date_from, date_to=args.second_focus_date_to,
        duration_from=args.second_focus_duration_from, duration_to=args.second_focus_duration_to,
        exclude_live=args.second_focus_exclude_live,
    )
    if second_result is None:
        print(f"  Skipped {second_focus}: no usable tokens")
        return 1
    second_token_count, second_ngram_counts = second_result
    print(f"  {second_focus}: {second_token_count:,} tokens")

    # Load prior channels for informative Dirichlet prior
    print(f"\n[Pass 3] Loading {len(prior_channels)} prior channels...")
    prior_counts = {n: Counter() for n in ngram_sizes}
    prior_token_count = 0
    prior_channels_used = []
    skipped = 0
    for done, channel in enumerate(prior_channels, 1):
        result = get_channel_ngrams(
            base_dir, channel, args, cache_dir, ngram_sizes, nlp,
            verbose=False, use_base_filters=True,
        )
        if result is None:
            skipped += 1
            continue
        tc, nc = result
        prior_token_count += tc
        for n in ngram_sizes:
            prior_counts[n].update(nc.get(n, Counter()))
        prior_channels_used.append(channel)
        if done % 50 == 0 or done == len(prior_channels):
            print(f"  {done}/{len(prior_channels)} - {len(prior_channels_used)} loaded, {skipped} skipped")

    print(f"\nPrior pool: {len(prior_channels_used)} channels, {prior_token_count:,} tokens")

    # Score using informative prior
    print("\n[Scoring] Computing log-odds with informative Dirichlet prior...")
    from .scoring import PhraseScore

    def compute_informative_scores(
        counts_a, count_a, counts_b, count_b,
        alpha_total, ngram_sizes, min_count, no_logs_under,
        size_proportional_prior, prior_counts, prior_token_count,
    ):
        results = []
        total_tokens_all = count_a + count_b
        use_informative = prior_counts is not None and prior_token_count is not None and prior_token_count > 0

        if size_proportional_prior and count_a > 0:
            prior_a = 1.0
            prior_b = count_b / count_a
        else:
            prior_a = prior_b = None

        for n in ngram_sizes:
            ca = counts_a.get(n, Counter())
            cb = counts_b.get(n, Counter())
            valid_phrases = [p for p, f in ca.items() if f >= min_count]
            valid_phrases += [p for p, f in cb.items() if f >= min_count and p not in ca]
            if not valid_phrases:
                continue

            f_a = np.fromiter((ca.get(p, 0) for p in valid_phrases), dtype=np.float64)
            f_b = np.fromiter((cb.get(p, 0) for p in valid_phrases), dtype=np.float64)

            if size_proportional_prior:
                smoothed_a = f_a + prior_a
                smoothed_b = f_b + prior_b
                denom_a = np.maximum(count_a + prior_a * 2 - smoothed_a, 1e-10)
                denom_b = np.maximum(count_b + prior_b * 2 - smoothed_b, 1e-10)
                lo = np.log(smoothed_a / denom_a) - np.log(smoothed_b / denom_b)
                var = (1.0 / smoothed_a) + (1.0 / denom_a) + (1.0 / smoothed_b) + (1.0 / denom_b)
                z = np.divide(lo, np.sqrt(var), out=np.zeros_like(lo), where=var > 0)
            else:
                if use_informative:
                    pc = prior_counts.get(n, Counter())
                    m_w = np.fromiter((pc.get(p, 0) / prior_token_count for p in valid_phrases), dtype=np.float64)
                else:
                    total_word_counts = f_a + f_b
                    m_w = total_word_counts / total_tokens_all if total_tokens_all > 0 else np.zeros_like(f_a)

                numer_a = f_a + alpha_total * m_w
                denom_a = np.maximum(count_a + alpha_total - numer_a, 1e-10)
                numer_b = f_b + alpha_total * m_w
                denom_b = np.maximum(count_b + alpha_total - numer_b, 1e-10)
                lo = np.log(numer_a / denom_a) - np.log(numer_b / denom_b)
                var = (1.0 / np.maximum(numer_a, 1e-10) + 1.0 / denom_a +
                       1.0 / np.maximum(numer_b, 1e-10) + 1.0 / denom_b)
                z = np.divide(lo, np.sqrt(var), out=np.zeros_like(lo), where=var > 0)

            f_a_per_10k = (f_a / count_a) * 10000
            f_b_per_10k = (f_b / count_b) * 10000
            total_counts = f_a + f_b
            dominance = np.divide(f_a, total_counts, out=np.zeros_like(f_a), where=total_counts > 0)
            is_exclusive = (f_b == 0)

            for i, phrase in enumerate(valid_phrases):
                lv = lo[i]
                if no_logs_under is not None and lv < no_logs_under:
                    continue
                results.append(PhraseScore(
                    phrase=phrase, n=n,
                    focus_count=int(f_a[i]), focus_per_10k=f_a_per_10k[i],
                    rest_count=int(f_b[i]), rest_per_10k=f_b_per_10k[i],
                    log_odds=lv, z_score=z[i],
                    is_exclusive=bool(is_exclusive[i]), dominance=dominance[i],
                ))
        return results

    focus_scores = compute_informative_scores(
        focus_ngram_counts, focus_token_count,
        second_ngram_counts, second_token_count,
        args.alpha_total, ngram_sizes, args.min_count, args.no_logs_under,
        args.size_proportional_prior, prior_counts, prior_token_count,
    )
    second_scores = compute_informative_scores(
        second_ngram_counts, second_token_count,
        focus_ngram_counts, focus_token_count,
        args.alpha_total, ngram_sizes, args.min_count, args.no_logs_under,
        args.size_proportional_prior, prior_counts, prior_token_count,
    )

    focus_by_n = partition_scores(focus_scores, args.top_n, args.show_exclusive_in_distinct)
    second_by_n = partition_scores(second_scores, args.top_n, args.show_exclusive_in_distinct)

    results = {focus_channel: focus_by_n, second_focus: second_by_n}
    channel_token_counts = {focus_channel: focus_token_count, second_focus: second_token_count}
    _print_top_results(results)

    print("\n[Building funnel chart]")
    funnel = build_funnel_chart(
        focus_channel=focus_channel,
        rest_label=second_focus,
        focus_counts=focus_ngram_counts.get(1, Counter()),
        rest_counts=second_ngram_counts.get(1, Counter()),
        focus_token_count=focus_token_count,
        rest_token_count=second_token_count,
        top_n=100,
        panel_min_font=args.funnel_panel_min_font_size,
        remove_stopwords=args.no_stops_graph,
        rest_channels=[second_focus],
        min_log_odds=args.no_logs_under,
        alpha_total=args.alpha_total,
        size_proportional_prior=args.size_proportional_prior,
        prior_counts=prior_counts.get(1, Counter()),
        prior_token_count=prior_token_count,
    )

    filters = _build_filters_dict(
        args,
        f"complete Monroe (alpha_total={args.alpha_total})",
        {"prior_channels": f"{len(prior_channels_used)} channels, {prior_token_count:,} tokens"},
    )
    html_blob = build_html(
        results_by_focus=results,
        focus_channels=[focus_channel, second_focus],
        focus_label=f"{focus_channel} vs {second_focus}",
        channel_token_counts=channel_token_counts,
        rest_channel_count=len(prior_channels_used),
        rest_token_count=prior_token_count,
        filters=filters,
        additional_metrics=None,
        funnel_html=funnel,
        graph_only=args.graph_only,
        calculate_log_likelihoods=args.calculate_log_likelihoods,
    )
    _write_html_output(base_dir, args.output_html, html_blob)
    return 0


# ----------------------------------------------------------------------
# Single focus vs pooled rest comparison
# ----------------------------------------------------------------------

def run_single_focus_comparison(
    args,
    base_dir,
    cache_dir,
    ngram_sizes,
    nlp=None,
):
    focus_set = set(args.focus)

    print("\n[Pass 1] Loading focus channels...")
    if args.focus_video_contribute is not None:
        print(f"Using per-video count limiting (max {args.focus_video_contribute} per video)")
        focus_ngram_counts, focus_token_counts = _load_focus_channels_per_video(
            base_dir, args.focus, args, cache_dir, ngram_sizes,
            args.focus_video_contribute, nlp,
        )
    else:
        focus_ngram_counts, focus_token_counts = _load_focus_channels(
            base_dir, args.focus, args, cache_dir, ngram_sizes, nlp,
        )

    if not focus_ngram_counts:
        print("No focus channels with usable tokens.")
        return 1

    valid_focus = [fc for fc in args.focus if fc in focus_ngram_counts]
    focus_label = valid_focus[0] if len(valid_focus) == 1 else f"Focuses ({len(valid_focus)} channels)"

    pooled_focus = {n: Counter() for n in ngram_sizes}
    pooled_focus_total = 0
    for fc in valid_focus:
        pooled_focus_total += focus_token_counts[fc]
        for n in ngram_sizes:
            pooled_focus[n].update(focus_ngram_counts[fc].get(n, Counter()))

    channel_token_counts = {focus_label: pooled_focus_total}
    print(
        f"Focus: {focus_label} | Members: {', '.join(valid_focus)}\n"
        f"Comparison: {len(args.channels)} channels | N-grams: {ngram_sizes} | "
        f"min-count: {args.min_count} | top-n: {args.top_n} | alpha: {args.alpha}"
    )

    # Build candidate set from pooled focus (phrases with count >= min_count)
    candidate_phrases = {}
    for n in ngram_sizes:
        cand = set()
        for phrase, cnt in pooled_focus.get(n, Counter()).items():
            if cnt >= args.min_count:
                cand.add(phrase)
        if cand:
            candidate_phrases[n] = cand
    if not candidate_phrases:
        print("No focus phrases meet min_count; nothing to compare.")
        return 1

    total_candidates = sum(len(c) for c in candidate_phrases.values())
    print(f"Candidate phrases: {total_candidates:,} total (focus >= {args.min_count})")

    print("\n[Pass 2] Loading comparison channels...")
    rest_counts, rest_counts_full, rest_token_count, rest_channel_count, \
    rest_channels_used, ngram_hits, skipped = _load_rest_channels_with_candidates(
        base_dir, args, cache_dir, ngram_sizes, valid_focus, nlp,
        candidate_phrases=candidate_phrases,
    )

    for channel in rest_channels_used:
        if channel in focus_token_counts:
            channel_token_counts[channel] = focus_token_counts[channel]

    print(
        f"\nRest pool: {rest_channel_count:,} channels, {rest_token_count:,} tokens "
        f"({ngram_hits} cache hits, {skipped} skipped)"
    )

    print("\n[Scoring]")
    all_scores = compute_scores(
        pooled_focus, pooled_focus_total,
        rest_counts, rest_token_count,
        ngram_sizes, args.min_count, args.alpha, args.no_logs_under,
        args.alpha_total, args.size_proportional_prior,
    )
    by_n = partition_scores(all_scores, args.top_n, args.show_exclusive_in_distinct)
    results = {focus_label: by_n}

    extra_metrics = None
    if args.extra_collocation_metrics:
        extra_metrics = {
            focus_label: compute_collocation_metrics(
                pooled_focus, pooled_focus_total, ngram_sizes, args.min_count,
                args.top_n, rest_counts, rest_token_count, rest_channel_count,
                args.calculate_log_likelihoods,
            )
        }

    _print_top_results(results)

    # Funnel chart
    funnel = _build_funnel_for_single_focus(
        base_dir, args, cache_dir, focus_label, pooled_focus, pooled_focus_total,
        focus_set, rest_channel_count, rest_counts, rest_counts_full,
        rest_token_count, rest_channels_used, nlp,
    )

    filters = _build_filters_dict(
        args, str(args.alpha),
        {"force_rest_pool_graph": str(args.force_rest_pool_graph)}
    )
    html_blob = build_html(
        results_by_focus=results,
        focus_channels=valid_focus,
        focus_label=focus_label,
        channel_token_counts=channel_token_counts,
        rest_channel_count=rest_channel_count,
        rest_token_count=rest_token_count,
        filters=filters,
        additional_metrics=extra_metrics,
        funnel_html=funnel,
        graph_only=args.graph_only,
        calculate_log_likelihoods=args.calculate_log_likelihoods,
    )
    _write_html_output(base_dir, args.output_html, html_blob)
    return 0


def _build_funnel_for_single_focus(
    base_dir, args, cache_dir, focus_label, pooled_focus, pooled_focus_total,
    focus_set, rest_channel_count, rest_counts, rest_counts_full,
    rest_token_count, rest_channels_used, nlp=None,
):
    """Build funnel chart for single focus comparison."""
    funnel = ""
    non_focus = [c for c in args.channels if c not in focus_set]
    one_rest = len(non_focus) == 1 and rest_channel_count == 1

    if one_rest:
        rest_ch = non_focus[0]
        rest_result = get_channel_ngrams(
            base_dir, rest_ch, args, cache_dir, [1], nlp,
            verbose=False, use_base_filters=True,
        )
        if rest_result:
            _, rest_ng = rest_result
            funnel = build_funnel_chart(
                focus_channel=focus_label,
                rest_label=rest_ch,
                focus_counts=pooled_focus.get(1, Counter()),
                rest_counts=rest_ng.get(1, Counter()),
                focus_token_count=pooled_focus_total,
                rest_token_count=rest_result[0],
                top_n=100,
                panel_min_font=args.funnel_panel_min_font_size,
                remove_stopwords=args.no_stops_graph,
                rest_channels=[rest_ch],
                min_log_odds=args.no_logs_under,
                alpha_total=args.alpha_total,
                size_proportional_prior=args.size_proportional_prior,
            )
    elif args.force_rest_pool_graph and rest_channel_count > 1:
        funnel = build_funnel_chart(
            focus_channel=focus_label,
            rest_label=f"rest pool ({rest_channel_count:,} channels)",
            focus_counts=pooled_focus.get(1, Counter()),
            rest_counts=rest_counts_full or rest_counts.get(1, Counter()),
            focus_token_count=pooled_focus_total,
            rest_token_count=rest_token_count,
            top_n=100,
            panel_min_font=args.funnel_panel_min_font_size,
            remove_stopwords=args.no_stops_graph,
            rest_channels=rest_channels_used,
            min_log_odds=args.no_logs_under,
            alpha_total=args.alpha_total,
            size_proportional_prior=args.size_proportional_prior,
        )
    return funnel