"""Scoring functions for statistical analysis of phrases.

These functions compute statistical scores to identify which phrases are
characteristic of a particular channel compared to others.
"""

from collections import Counter
from math import log
from typing import Dict, List, Optional

import numpy as np


class PhraseScore:
    """Stores statistical scores for a single phrase.
    
    This class holds all the metrics computed for a phrase to determine
    how characteristic it is of a focus channel compared to the rest.
    
    Attributes:
        phrase: The phrase (word or n-gram)
        n: The n-gram size (1=unigram, 2=bigram, etc.)
        focus_count: How many times this phrase appears in the focus channel
        focus_per_10k: Frequency per 10,000 tokens in the focus channel
        rest_count: How many times this phrase appears in the rest pool
        rest_per_10k: Frequency per 10,000 tokens in the rest pool
        log_odds: Log-odds ratio (higher means more characteristic of focus)
        z_score: Statistical significance (higher = more significant)
        is_exclusive: True if the phrase only appears in the focus channel
        dominance: Proportion of this phrase's usage that comes from focus (0-1)
    """
    __slots__ = (
        "phrase", "n", "focus_count", "focus_per_10k",
        "rest_count", "rest_per_10k", "log_odds", "z_score",
        "is_exclusive", "dominance",
    )

    def __init__(self, phrase, n, focus_count, focus_per_10k, rest_count,
                 rest_per_10k, log_odds, z_score, is_exclusive, dominance):
        self.phrase = phrase
        self.n = n
        self.focus_count = focus_count
        self.focus_per_10k = focus_per_10k
        self.rest_count = rest_count
        self.rest_per_10k = rest_per_10k
        self.log_odds = log_odds
        self.z_score = z_score
        self.is_exclusive = is_exclusive
        self.dominance = dominance


def compute_scores(
    focus_counts: Dict[int, Counter],
    focus_token_count: int,
    rest_counts: Dict[int, Counter],
    rest_token_count: int,
    ngram_sizes: List[int],
    min_focus_count: int,
    alpha: float,
    min_log_odds: Optional[float] = None,
    alpha_total: float = 300.0,
    size_proportional_prior: bool = False,
) -> List[PhraseScore]:
    """Compute statistical scores for phrases using Monroe's log-odds method.
    
    This function implements the "Fightin' Words" method (Monroe et al. 2008)
    which uses Dirichlet-smoothed log-odds ratio to score phrases by how much
    more a focus channel uses them versus the rest pool.
    
    The method calculates:
    - Log-odds: How much more likely the phrase is in focus vs rest
    - Z-score: Statistical significance of the difference
    - Dominance: Proportion of usage that comes from focus
    - Exclusivity: Whether the phrase only appears in focus
    
    Args:
        focus_counts: N-gram counts for the focus channel
        focus_token_count: Total tokens in the focus channel
        rest_counts: N-gram counts for the rest pool
        rest_token_count: Total tokens in the rest pool
        ngram_sizes: List of n-gram sizes to score (e.g., [1, 2, 3])
        min_focus_count: Minimum count in focus to consider a phrase
        alpha: Legacy parameter (kept for compatibility)
        min_log_odds: Minimum log-odds threshold (optional)
        alpha_total: Total prior weight for Dirichlet smoothing (default: 300.0)
        size_proportional_prior: If True, scale priors by corpus size
        
    Returns:
        List of PhraseScore objects with computed metrics
    """
    results = []
    total_tokens_all = focus_token_count + rest_token_count
    
    # Calculate size-proportional priors if enabled
    # This helps when comparing corpora of vastly different sizes
    if size_proportional_prior and rest_token_count > 0:
        prior_focus = 1.0
        prior_rest = rest_token_count / focus_token_count
    else:
        prior_focus = None
        prior_rest = None
    
    # Process each n-gram size
    for n in ngram_sizes:
        focus_counter = focus_counts.get(n, Counter())
        rest_counter = rest_counts.get(n, Counter())
        
        # Filter to phrases that appear enough times in focus
        candidates = [phrase for phrase, count in focus_counter.items() if count >= min_focus_count]
        if not candidates:
            continue
        
        # Build arrays of counts for vectorized computation (faster than loops)
        focus_count_array = np.fromiter((focus_counter[p] for p in candidates), dtype=np.float64, count=len(candidates))
        rest_count_array = np.fromiter((rest_counter.get(p, 0) for p in candidates), dtype=np.float64, count=len(candidates))
        
        if size_proportional_prior:
            # Use size-proportional Dirichlet priors
            smoothed_focus = focus_count_array + prior_focus
            smoothed_rest = rest_count_array + prior_rest
            
            denom_focus = focus_token_count + prior_focus * 2 - smoothed_focus
            denom_rest = rest_token_count + prior_rest * 2 - smoothed_rest
            
            # Prevent division by zero
            denom_focus = np.where(denom_focus <= 0, 1e-10, denom_focus)
            denom_rest = np.where(denom_rest <= 0, 1e-10, denom_rest)
            
            # Calculate log-odds
            log_odds_array = np.log(smoothed_focus / denom_focus) - np.log(smoothed_rest / denom_rest)
            
            # Calculate variance (four-cell reciprocal formula)
            variance = (1.0 / smoothed_focus) + (1.0 / denom_focus) + (1.0 / smoothed_rest) + (1.0 / denom_rest)
            z_score_array = np.divide(log_odds_array, np.sqrt(variance), out=np.zeros_like(log_odds_array), where=variance > 0)
        else:
            # Monroe log-odds with informative Dirichlet prior.
            # m_w = P(w) estimated from the rest pool (the background corpus),
            # NOT from focus+rest. This is what makes rare words that the focus
            # shares with the rest pool score lower than truly focus-exclusive words.
            if rest_token_count > 0:
                m_w_array = rest_count_array / rest_token_count
            else:
                m_w_array = np.zeros_like(focus_count_array)
            
            numer_focus = focus_count_array + alpha_total * m_w_array
            denom_focus = focus_token_count + alpha_total - numer_focus
            # Floor numerators at a tiny epsilon so a phrase absent from rest
            # (m_w = 0 -> numer = 0) doesn't yield odds = 0 -> log(0) = -inf.
            # The 1e9 "infinite odds" fill stays for the degenerate denom<=0 case.
            eps = 1e-10
            odds_focus = np.divide(
                np.maximum(numer_focus, eps), denom_focus,
                out=np.full_like(numer_focus, 1e9), where=denom_focus > 0,
            )

            numer_rest = rest_count_array + alpha_total * m_w_array
            denom_rest = rest_token_count + alpha_total - numer_rest
            odds_rest = np.divide(
                np.maximum(numer_rest, eps), denom_rest,
                out=np.full_like(numer_rest, 1e9), where=denom_rest > 0,
            )
            
            log_odds_array = np.log(odds_focus) - np.log(odds_rest)
            
            # Calculate variance (all four cells)
            v1 = np.divide(1.0, numer_focus, out=np.zeros_like(numer_focus), where=numer_focus > 0)
            v2 = np.divide(1.0, denom_focus, out=np.zeros_like(denom_focus), where=denom_focus > 0)
            v3 = np.divide(1.0, numer_rest, out=np.zeros_like(numer_rest), where=numer_rest > 0)
            v4 = np.divide(1.0, denom_rest, out=np.zeros_like(denom_rest), where=denom_rest > 0)
            variance = v1 + v2 + v3 + v4
            z_score_array = np.divide(log_odds_array, np.sqrt(variance), out=np.zeros_like(log_odds_array), where=variance > 0)
        
        # Calculate additional metrics
        focus_per_10k_array = (focus_count_array / focus_token_count) * 10_000
        rest_per_10k_array = (rest_count_array / rest_token_count) * 10_000 if rest_token_count > 0 else np.zeros_like(focus_count_array)
        total_counts = focus_count_array + rest_count_array
        dominance_array = np.divide(focus_count_array, total_counts, out=np.zeros_like(focus_count_array), where=total_counts > 0)
        is_exclusive_array = (rest_count_array == 0)

        # Build PhraseScore objects for each candidate phrase
        for i, phrase in enumerate(candidates):
            log_odds = log_odds_array[i]
            
            # Apply minimum log-odds filter if specified
            if min_log_odds is not None and log_odds < min_log_odds:
                continue
            
            results.append(PhraseScore(
                phrase=phrase,
                n=n,
                focus_count=int(focus_count_array[i]),
                focus_per_10k=focus_per_10k_array[i],
                rest_count=int(rest_count_array[i]),
                rest_per_10k=rest_per_10k_array[i],
                log_odds=log_odds,
                z_score=z_score_array[i],
                is_exclusive=bool(is_exclusive_array[i]),
                dominance=dominance_array[i],
            ))
    
    return results


def partition_scores(
    scores: List[PhraseScore],
    top_n: int,
    show_exclusive_in_distinct: bool = False,
) -> Dict[int, Dict[str, List[PhraseScore]]]:
    """Partition scored phrases into categories for each n-gram size.
    
    This function organizes phrases into three categories:
    - Exclusive: Phrases that only appear in the focus channel
    - Distinctive: Phrases with high statistical significance (z-score >= 2)
    - Dominant: Phrases where focus channel has >= 10% of total usage
    
    Args:
        scores: List of PhraseScore objects to partition
        top_n: Maximum number of phrases to keep in each category
        show_exclusive_in_distinct: If True, include exclusive phrases in distinctive
        
    Returns:
        Dictionary mapping n-gram size to dict of category -> list of PhraseScore
    """
    # Group scores by n-gram size
    by_n: Dict[int, List[PhraseScore]] = {}
    for score in scores:
        by_n.setdefault(score.n, []).append(score)

    result = {}
    for n, items in by_n.items():
        # Exclusive: phrases that only appear in focus channel
        exclusive_items = [s for s in items if s.is_exclusive]
        exclusive = sorted(exclusive_items, key=lambda s: -s.focus_count)[:top_n]
        
        # Distinctive: phrases with high statistical significance
        if show_exclusive_in_distinct:
            distinctive_items = [s for s in items if s.is_exclusive and s.z_score >= 2]
        else:
            distinctive_items = [s for s in items if not s.is_exclusive and s.z_score >= 2]
        distinctive = sorted(distinctive_items, key=lambda s: -s.z_score)[:top_n]
        
        # Dominant: phrases where focus has >= 10% of total usage
        dominant_items = [s for s in items if not s.is_exclusive and s.dominance >= 0.1]
        dominant = sorted(dominant_items, key=lambda s: -s.dominance)[:top_n]
        
        result[n] = {
            "exclusive": exclusive,
            "distinctive": distinctive,
            "dominant": dominant,
        }
    
    return result


def compute_collocation_metrics(
    focus_counts: Dict[int, Counter],
    focus_token_count: int,
    ngram_sizes: List[int],
    min_focus_count: int,
    top_n: int = 1000,
    rest_counts: Optional[Dict[int, Counter]] = None,
    rest_token_count: Optional[int] = None,
    rest_channel_count: Optional[int] = None,
    calculate_log_likelihoods: bool = False,
) -> Dict[str, list]:
    """Compute collocation metrics for n-grams.
    
    Collocation metrics measure how strongly words tend to appear together.
    This function computes:
    
    - Mutual Information (MI): Measures how much knowing one word tells you
      about another. Higher MI means the words appear together more often than
      expected by chance.
    
    - Delta P (ΔP): Measures the difference in probability of seeing a target
      word with vs without a prefix. Useful for predictive relationships.
    
    - Log-Likelihood (G²): Statistical test for whether the association is
      significant across corpora (optional, requires rest corpus data).
    
    Args:
        focus_counts: N-gram counts for the focus channel
        focus_token_count: Total tokens in the focus channel
        ngram_sizes: List of n-gram sizes to analyze (must be >= 2)
        min_focus_count: Minimum count in focus to consider a phrase
        top_n: Maximum number of results to return per metric
        rest_counts: Optional n-gram counts for rest corpus (for log-likelihood)
        rest_token_count: Optional total tokens in rest corpus
        rest_channel_count: Optional number of channels in rest corpus
        calculate_log_likelihoods: Whether to compute log-likelihood metrics
        
    Returns:
        Dictionary with keys "mi", "delta_p", and "log_likelihood",
        each containing a list of result dictionaries
    """
    if focus_token_count <= 1:
        return {"mi": [], "delta_p": [], "log_likelihood": []}

    unigrams = focus_counts.get(1, Counter())
    total = float(focus_token_count)
    mi_rows, dp_rows = [], []

    # Process each n-gram size (skip unigrams since they don't have collocations)
    for n in ngram_sizes:
        if n < 2:
            continue
        ngrams = focus_counts.get(n, Counter())
        prefixes = focus_counts.get(n - 1, Counter())
        if not ngrams or not prefixes:
            continue
        n_pos = float(max(focus_token_count - n + 1, 1))

        for phrase, f_seq in ngrams.items():
            if f_seq < min_focus_count:
                continue
            words = phrase.split()
            if len(words) != n:
                continue
            prefix = " ".join(words[:-1])
            target = words[-1]
            f_prefix = prefixes.get(prefix, 0)
            f_target = unigrams.get(target, 0)
            if not f_prefix or not f_target:
                continue

            # Calculate Mutual Information (MI)
            # MI = log(P(phrase) / P(word1) * P(word2) * ...)
            p_joint = f_seq / n_pos
            p_parts = 1.0
            for tok in words:
                ft = unigrams.get(tok, 0)
                if not ft:
                    p_parts = 0.0
                    break
                p_parts *= ft / total
            if not p_parts:
                continue

            mi_rows.append({
                "n": n, "phrase": phrase, "count": f_seq,
                "score": log(p_joint / p_parts),
                "p_joint": p_joint, "p_parts": p_parts,
            })

            # Calculate Delta P (ΔP)
            # ΔP = P(target|prefix) - P(target|not prefix)
            p_tgp = f_seq / f_prefix
            denom = total - f_prefix
            p_tgnp = max(f_target - f_seq, 0) / denom if denom > 0 else 0.0
            dp_rows.append({
                "n": n, "phrase": phrase, "count": f_seq,
                "score": p_tgp - p_tgnp,
                "prefix": prefix, "target": target,
                "p_target_given_prefix": p_tgp,
                "p_target_given_not_prefix": p_tgnp,
            })

    mi_rows.sort(key=lambda x: -x["score"])
    dp_rows.sort(key=lambda x: -x["score"])
    
    # Apply top_n filtering
    mi_rows = mi_rows[:top_n]
    dp_rows = dp_rows[:top_n]
    
    # Compute Log-Likelihood (G²) with Dice Coefficient and TF-IDF modulators
    ll_rows = []
    if calculate_log_likelihoods and rest_counts and rest_token_count and rest_channel_count:
        ll_rows = _compute_log_likelihood_metrics(
            focus_counts, focus_token_count, rest_counts, rest_token_count,
            rest_channel_count, ngram_sizes, min_focus_count, top_n
        )
    
    return {"mi": mi_rows, "delta_p": dp_rows, "log_likelihood": ll_rows}


def _compute_log_likelihood_metrics(
    focus_counts: Dict[int, Counter],
    focus_token_count: int,
    rest_counts: Dict[int, Counter],
    rest_token_count: int,
    rest_channel_count: int,
    ngram_sizes: List[int],
    min_focus_count: int,
    top_n: int,
) -> List[dict]:
    """Compute Log-Likelihood (G²) with Dice Coefficient and TF-IDF modulators.
    
    Log-Likelihood is a statistical test that measures whether the association
    between a phrase and a corpus is statistically significant. This implementation
    also includes:
    
    - Dice Coefficient: Effect size measure (0-1, higher = stronger association)
    - TF-IDF modulation: Downweights phrases that appear in many channels
    
    Args:
        focus_counts: N-gram counts for focus channel
        focus_token_count: Total tokens in focus channel
        rest_counts: N-gram counts for rest corpus
        rest_token_count: Total tokens in rest corpus
        rest_channel_count: Number of channels in rest corpus (for IDF)
        ngram_sizes: List of n-gram sizes to compute
        min_focus_count: Minimum count in focus to consider
        top_n: Maximum number of results to return
        
    Returns:
        List of dictionaries with log-likelihood metrics
    """
    if focus_token_count <= 0 or rest_token_count <= 0 or rest_channel_count <= 0:
        return []
    
    total_tokens_all = focus_token_count + rest_token_count
    rows = []
    
    for n in ngram_sizes:
        if n < 2:
            continue
        focus_ngrams = focus_counts.get(n, Counter())
        rest_ngrams = rest_counts.get(n, Counter())
        
        if not focus_ngrams:
            continue
        
        # Get unigram counts for TF-IDF computation
        focus_unigrams = focus_counts.get(1, Counter())
        rest_unigrams = rest_counts.get(1, Counter())
        
        for phrase, f_focus in focus_ngrams.items():
            if f_focus < min_focus_count:
                continue
            
            f_rest = rest_ngrams.get(phrase, 0)
            total_count = f_focus + f_rest
            
            # Build 2x2 contingency table for Log-Likelihood
            # This table compares the phrase's occurrence in focus vs rest
            # a = count in focus, b = count in rest
            # c = total tokens in focus - a, d = total tokens in rest - b
            a = f_focus
            b = f_rest
            c = focus_token_count - a
            d = rest_token_count - b
            
            # Expected values under independence
            row1_total = a + c
            row2_total = b + d
            col1_total = a + b
            col2_total = c + d
            n_total = row1_total + row2_total
            
            if n_total == 0:
                continue
            
            e1 = row1_total * col1_total / n_total
            e2 = row1_total * col2_total / n_total
            e3 = row2_total * col1_total / n_total
            e4 = row2_total * col2_total / n_total
            
            # Log-Likelihood G² statistic
            g2 = 0.0
            if e1 > 0 and a > 0:
                g2 += a * log(a / e1)
            if e2 > 0 and c > 0:
                g2 += c * log(c / e2)
            if e3 > 0 and b > 0:
                g2 += b * log(b / e3)
            if e4 > 0 and d > 0:
                g2 += d * log(d / e4)
            g2 *= 2  # G² = 2 * sum(O * log(O/E))
            
            # Dice Coefficient (effect size)
            # Dice = 2*a / (2*a + b + c) for collocation strength
            dice = 2.0 * a / (2.0 * a + b + c) if (2.0 * a + b + c) > 0 else 0.0
            
            # TF-IDF modulator using rest corpus
            # TF: term frequency in focus
            tf = f_focus / focus_token_count if focus_token_count > 0 else 0.0
            
            # IDF: inverse document frequency using rest corpus
            # Document frequency = number of rest channels containing this phrase
            # For efficiency, we use rest presence as binary (1 if present, 0 if not)
            # In a more sophisticated version, we'd track per-channel counts
            df_rest = 1 if f_rest > 0 else 0
            idf = log((rest_channel_count + 1) / (df_rest + 1)) + 1  # +1 smoothing
            
            # TF-IDF modulated score
            tfidf_modulated = g2 * tf * idf
            
            # Word-level TF-IDF for individual components (for multi-word phrases)
            words = phrase.split()
            word_tfidf_sum = 0.0
            for word in words:
                f_word_focus = focus_unigrams.get(word, 0)
                f_word_rest = rest_unigrams.get(word, 0)
                tf_word = f_word_focus / focus_token_count if focus_token_count > 0 else 0.0
                df_word_rest = 1 if f_word_rest > 0 else 0
                idf_word = log((rest_channel_count + 1) / (df_word_rest + 1)) + 1
                word_tfidf_sum += tf_word * idf_word
            avg_word_tfidf = word_tfidf_sum / len(words) if words else 0.0
            
            rows.append({
                "n": n,
                "phrase": phrase,
                "count": f_focus,
                "g2": g2,
                "dice": dice,
                "tfidf_modulated": tfidf_modulated,
                "avg_word_tfidf": avg_word_tfidf,
                "focus_count": f_focus,
                "rest_count": f_rest,
                "focus_per_10k": f_focus / focus_token_count * 10_000,
                "rest_per_10k": f_rest / rest_token_count * 10_000 if rest_token_count > 0 else 0.0,
            })
    
    # Sort by TF-IDF modulated G² score
    rows.sort(key=lambda x: -x["tfidf_modulated"])
    return rows[:top_n]
