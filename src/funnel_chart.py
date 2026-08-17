"""Funnel chart visualization functions for Fighting Words analysis.

These functions generate interactive funnel charts using Plotly or matplotlib
to visualize which words are characteristic of a focus channel vs the rest.
"""

import base64
import hashlib
import html
import io
from collections import Counter
from math import asinh
from typing import List, Optional

import numpy as np

from .constants import (
    FOCUS_COLOR,
    OTHER_COLOR,
    FUNNEL_BG_COLOR,
    FUNNEL_FONT_FAMILY,
    FUNNEL_LABEL_OUTLINE_COLOR,
    FUNNEL_TEXT_COLOR,
    GRAPH_STOPWORDS,
    HTML_ENTRY_EXCLUSION_RULES,
)


def _select_fighting_words(
    focus_counts: Counter,
    rest_counts: Counter,
    focus_token_count: int,
    rest_token_count: int,
    min_log_odds: Optional[float] = None,
    alpha_total: float = 2000.0,
    size_proportional_prior: bool = False,
    prior_counts: Optional[Counter] = None,
    prior_token_count: Optional[int] = None,
    min_count: int = 5,
) -> List[dict]:
    """Select fighting words using complete Monroe log-odds (Dirichlet posterior odds).

    When prior_counts is provided, uses Monroe's informative prior where m_w is derived
    from the independent prior corpus instead of the pooled focus/rest corpora.
    """
    if focus_token_count <= 0 or rest_token_count <= 0:
        return []

    total_pooled = sum(focus_counts.values()) + sum(rest_counts.values())
    if not total_pooled:
        return []

    total_tokens_all = focus_token_count + rest_token_count
    use_informative_prior = prior_counts is not None and prior_token_count is not None and prior_token_count > 0

    if size_proportional_prior:
        prior_focus = 1.0
        prior_bg = rest_token_count / focus_token_count
    else:
        prior_focus = None
        prior_bg = None

    valid_terms = [
        term for term, f_foc in focus_counts.items() if f_foc >= min_count
    ] + [
        term for term, f_rst in rest_counts.items() if f_rst >= min_count and term not in focus_counts
    ]

    if not valid_terms:
        return []

    f_focus_arr = np.fromiter((focus_counts.get(t, 0) for t in valid_terms), dtype=np.float64, count=len(valid_terms))
    f_rest_arr = np.fromiter((rest_counts.get(t, 0) for t in valid_terms), dtype=np.float64, count=len(valid_terms))

    if size_proportional_prior:
        smoothed_f = f_focus_arr + prior_focus
        smoothed_r = f_rest_arr + prior_bg

        denom_f = focus_token_count + prior_focus * 2 - smoothed_f
        denom_r = rest_token_count + prior_bg * 2 - smoothed_r

        denom_f = np.where(denom_f <= 0, 1e-10, denom_f)
        denom_r = np.where(denom_r <= 0, 1e-10, denom_r)

        delta_arr = np.log(smoothed_f / denom_f) - np.log(smoothed_r / denom_r)

        var = (1.0 / smoothed_f) + (1.0 / denom_f) + (1.0 / smoothed_r) + (1.0 / denom_r)
        z_arr = np.divide(delta_arr, np.sqrt(var), out=np.zeros_like(delta_arr), where=var > 0)
    else:
        if use_informative_prior:
            # Informative prior: m_w derived from independent prior corpus (Monroe's proper method)
            m_w_arr = np.fromiter(
                (prior_counts.get(t, 0) / prior_token_count for t in valid_terms),
                dtype=np.float64, count=len(valid_terms),
            )
        else:
            # Self-referential pooled prior (practical shortcut)
            total_word_counts = f_focus_arr + f_rest_arr
            m_w_arr = total_word_counts / total_tokens_all if total_tokens_all > 0 else np.zeros_like(f_focus_arr)

        numer_f = f_focus_arr + alpha_total * m_w_arr
        denom_f = focus_token_count + alpha_total - numer_f
        odds_f = np.divide(numer_f, denom_f, out=np.full_like(numer_f, 1e9), where=denom_f > 0)

        numer_r = f_rest_arr + alpha_total * m_w_arr
        denom_r = rest_token_count + alpha_total - numer_r
        odds_r = np.divide(numer_r, denom_r, out=np.full_like(numer_r, 1e9), where=denom_r > 0)

        delta_arr = np.log(odds_f) - np.log(odds_r)

        v1 = np.divide(1.0, numer_f, out=np.zeros_like(numer_f), where=numer_f > 0)
        v2 = np.divide(1.0, denom_f, out=np.zeros_like(denom_f), where=denom_f > 0)
        v3 = np.divide(1.0, numer_r, out=np.zeros_like(numer_r), where=numer_r > 0)
        v4 = np.divide(1.0, denom_r, out=np.zeros_like(denom_r), where=denom_r > 0)
        var = v1 + v2 + v3 + v4
        z_arr = np.divide(delta_arr, np.sqrt(var), out=np.zeros_like(delta_arr), where=var > 0)

    f_focus_per_10k_arr = (f_focus_arr / focus_token_count) * 10_000
    f_rest_per_10k_arr = (f_rest_arr / rest_token_count) * 10_000
    total_counts = f_focus_arr + f_rest_arr
    x_log_total_arr = np.log(total_counts)

    rows = []
    for i, term in enumerate(valid_terms):
        delta = delta_arr[i]
        if min_log_odds is not None and delta < min_log_odds:
            continue

        rows.append({
            "term": term,
            "focus_count": int(f_focus_arr[i]),
            "rest_count": int(f_rest_arr[i]),
            "total_count": int(total_counts[i]),
            "focus_per_10k": f_focus_per_10k_arr[i],
            "rest_per_10k": f_rest_per_10k_arr[i],
            "delta": delta,
            "z": z_arr[i],
            "z_abs": abs(z_arr[i]),
            "x_log_total": x_log_total_arr[i],
        })
    return rows


def _should_exclude(phrase: str, channel: str, category: str) -> bool:
    """Check if a phrase should be excluded based on HTML exclusion rules."""
    phrase_norm = phrase.strip().lower()
    phrase_words = set(html_word for html_word in phrase_norm.split() if html_word)
    channel_norm = channel.strip().lower()
    cat_norm = category.strip().lower()

    for rule in HTML_ENTRY_EXCLUSION_RULES:
        channels = [str(v).strip().lower() for v in rule.get("channels", [])]
        if channels and channel_norm not in channels:
            continue
        cats = [str(v).strip().lower() for v in rule.get("categories", [])]
        if cats and cat_norm not in cats:
            continue
        if any(p in phrase_norm for p in [str(v).strip().lower() for v in rule.get("contains_any", [])]):
            return True
        if phrase_norm in [str(v).strip().lower() for v in rule.get("exact_any", [])]:
            return True
        if any(w in phrase_words for w in [str(v).strip().lower() for v in rule.get("word_any", [])]):
            return True
    return False


def _compute_marker_sizes(rows: List[dict]) -> dict:
    """Compute visual sizing helpers for the funnel chart."""
    abs_z = sorted(abs(r["z"]) for r in rows)
    if not abs_z:
        return {"z_softness": 1.0, "z_denom": 1.0, "marker_min": 2.7, "marker_max": 28.9}
    z_max = max(abs_z)
    z_softness = max(0.35, abs_z[int(round((len(abs_z) - 1) * 0.60))])
    z_denom = max(1e-9, asinh(max(1e-9, z_max) / z_softness))
    return {"z_softness": z_softness, "z_denom": z_denom, "marker_min": 2.7, "marker_max": 28.9}


def _marker_size(row: dict, sizing: dict) -> float:
    """Compute marker size for a row."""
    z_abs = abs(float(row["z"]))
    ratio = asinh(z_abs / sizing["z_softness"]) / sizing["z_denom"]
    ratio = max(0.0, min(1.0, ratio))
    return sizing["marker_min"] + (ratio ** 1.35) * (sizing["marker_max"] - sizing["marker_min"])


def _label_size(row: dict, sizing: dict) -> float:
    """Compute label size for a row."""
    label_min, label_max = 3, 20
    ms = _marker_size(row, sizing)
    ratio = (ms - sizing["marker_min"]) / max(1e-9, sizing["marker_max"] - sizing["marker_min"])
    return max(label_min, min(label_max, label_min + (ratio ** 3.0) * (label_max - label_min)))


def _compute_label_layout(items: List[dict], min_y, max_y, min_x, max_x, x_span) -> dict:
    """Compute label layout to avoid overlaps."""
    if not items:
        return {}
    y_span = max(1e-9, max_y - min_y)
    min_gap = max(0.04, min(0.13, y_span / (len(items) + 40.0)))
    max_offset = max(0.06, min(0.35, y_span * 0.085))
    base_shift = x_span * 0.007

    layout = {}
    for idx, row in enumerate(items):
        h = hashlib.md5(row["term"].encode()).hexdigest()
        y_jitter = ((int(h[:6], 16) / 0xFFFFFF) - 0.5) * min_gap * 0.8
        x_jitter = ((int(h[6:12], 16) / 0xFFFFFF) - 0.5) * x_span * 0.004
        y_sign = 1.0 if row["z"] >= 0 else -1.0
        x_pos = row["x_log_total"] + base_shift + x_jitter
        x_anchor = "left"
        if x_pos > max_x - x_span * 0.012:
            x_pos = row["x_log_total"] - base_shift + x_jitter
            x_anchor = "right"
        x_pos = max(min_x + x_span * 0.01, min(max_x - x_span * 0.01, x_pos))
        layout[idx] = {
            "x": x_pos, "y": max(min_y, min(max_y, row["z"] + y_sign * y_jitter)),
            "z": row["z"], "xanchor": x_anchor, "ha": "left" if x_anchor == "left" else "right",
        }

    ordered = sorted(layout.items(), key=lambda p: p[1]["y"])
    for i in range(1, len(ordered)):
        ordered[i][1]["y"] = max(ordered[i][1]["y"], ordered[i - 1][1]["y"] + min_gap)
    for i in range(len(ordered) - 2, -1, -1):
        ordered[i][1]["y"] = min(ordered[i][1]["y"], ordered[i + 1][1]["y"] - min_gap)
    for idx, pos in ordered:
        low = max(min_y, pos["z"] - max_offset)
        high = min(max_y, pos["z"] + max_offset)
        pos["y"] = max(low, min(high, pos["y"]))
    return layout


def _mirrored_panel_html(rows: List[dict], top_n: int, min_font: float) -> str:
    """Generate HTML for the mirrored side panel."""
    focus_rows = sorted([r for r in rows if r["z"] >= 0], key=lambda r: -r["z"])[:top_n]
    other_rows = sorted([r for r in rows if r["z"] < 0], key=lambda r: r["z"])[:top_n]
    other_rows = sorted(other_rows, key=lambda r: -r["z"])  # display weakest->strongest

    def _word_style(row, max_abs, count):
        if max_abs <= 0:
            ratio = 0.0
        else:
            ratio = max(0.0, min(1.0, abs(row["z"]) / max_abs))
        if count >= 46:
            fp = max(max(0.0, min_font), min(27.6, 9.8 + (ratio ** 2.2) * 17.8))
        elif count >= 36:
            fp = max(max(0.0, min_font), min(29.2, 10.8 + (ratio ** 2.15) * 18.4))
        else:
            fp = max(max(0.0, min_font), min(31.6, 12.0 + (ratio ** 2.05) * 19.6))
        lh = 0.80 + ratio * 0.22
        return f"font-size:{fp:.1f}px;line-height:{lh:.2f};margin:0"

    max_f = max((abs(r["z"]) for r in focus_rows), default=1.0)
    max_o = max((abs(r["z"]) for r in other_rows), default=1.0)
    top = "".join(
        f"<div class='sw sw-f' style='{_word_style(r, max_f, len(focus_rows))}'>{html.escape(r['term'])}</div>"
        for r in focus_rows
    )
    bot = "".join(
        f"<div class='sw sw-o' style='{_word_style(r, max_o, len(other_rows))}'>{html.escape(r['term'])}</div>"
        for r in other_rows
    )
    mode = ""
    if focus_rows and not other_rows:
        mode = " top-only"
    elif other_rows and not focus_rows:
        mode = " bot-only"
    return (
        f"<aside class='side-panel{mode}'>"
        "<div class='stack stack-top'>" + top + "</div>"
        "<div class='stack-line'></div>"
        "<div class='stack stack-bot'>" + bot + "</div>"
        "</aside>"
    )


def build_funnel_chart(
    focus_channel: str,
    rest_label: str,
    focus_counts: Counter,
    rest_counts: Counter,
    focus_token_count: int,
    rest_token_count: int,
    top_n: int = 100,
    panel_min_font: float = 0.0,
    remove_stopwords: bool = False,
    rest_channels: Optional[List[str]] = None,
    min_log_odds: Optional[float] = None,
    alpha_total: float = 2000.0,
    size_proportional_prior: bool = False,
    prior_counts: Optional[Counter] = None,
    prior_token_count: Optional[int] = None,
    min_count: int = 5,
) -> str:
    """Build a funnel chart comparing focus channel vs rest."""
    rows = _select_fighting_words(
        focus_counts, rest_counts, focus_token_count, rest_token_count,
        min_log_odds, alpha_total, size_proportional_prior,
        prior_counts, prior_token_count, min_count,
    )

    # Build list of channels to check for exclusion
    channels_to_check = [focus_channel] + (rest_channels or [rest_label])

    def _excluded(term: str) -> bool:
        for ch in channels_to_check:
            for cat in ("exclusive", "distinctive", "dominant"):
                if _should_exclude(term, ch, cat):
                    return True
        return False

    rows = [r for r in rows if not _excluded(r["term"])]
    if remove_stopwords:
        rows = [r for r in rows if r["term"].strip().lower() not in GRAPH_STOPWORDS]
    if not rows:
        return ""

    rows.sort(key=lambda r: -r["z_abs"])
    labeled, unlabeled = rows[:top_n], rows[top_n:]
    sizing = _compute_marker_sizes(rows)
    side_panel = _mirrored_panel_html(rows, top_n, panel_min_font)
    title = f"Funnel Chart: {html.escape(focus_channel)} vs {html.escape(rest_label)}"
    subtitle = "Based on Monroe, Colaresi, and Quinn (2008) Fightin' Words"

    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        go = pio = None

    if go and pio:
        fig = go.Figure()

        def _trace(row_list, opacity):
            return go.Scatter(
                x=[r["x_log_total"] for r in row_list],
                y=[r["z"] for r in row_list],
                mode="markers",
                marker={
                    "size": [_marker_size(r, sizing) for r in row_list],
                    "color": [FOCUS_COLOR if r["z"] >= 0 else OTHER_COLOR for r in row_list],
                    "opacity": opacity,
                    "line": {"width": 0.6, "color": "#fff"} if opacity > 0.5 else {},
                },
                hovertemplate=(
                    "%{customdata[0]}<br>Total: %{customdata[1]}<br>"
                    f"{html.escape(focus_channel)}: %{{customdata[2]}} / 10k<br>"
                    f"{html.escape(rest_label)}: %{{customdata[3]}} / 10k<br>"
                    "z-score: %{y:.3f}<extra></extra>"
                ),
                customdata=[
                    [r["term"], r["total_count"], f"{r['focus_per_10k']:.3f}", f"{r['rest_per_10k']:.3f}"]
                    for r in row_list
                ],
                showlegend=False,
            )

        if unlabeled:
            fig.add_trace(_trace(unlabeled, 0.35))
        if labeled:
            fig.add_trace(_trace(labeled, 0.92))

        xs = [r["x_log_total"] for r in rows]
        min_x, max_x = min(xs), max(xs)
        x_span = max(1.0, max_x - min_x)
        max_abs_z = max(abs(r["z"]) for r in rows)
        y_pad = max(0.2, max_abs_z * 0.12)
        y_min, y_max = -(max_abs_z + y_pad), max_abs_z + y_pad

        layout = _compute_label_layout(labeled, y_min, y_max, min_x, max_x, x_span)
        for idx, row in enumerate(labeled):
            pos = layout.get(idx, {"x": row["x_log_total"], "y": row["z"], "xanchor": "left"})
            fig.add_annotation(
                x=pos["x"], y=pos["y"],
                text=f"<b>{html.escape(row['term'])}</b>",
                showarrow=False, xanchor=pos["xanchor"], yanchor="middle",
                font={"size": int(round(_label_size(row, sizing))), "color": FUNNEL_TEXT_COLOR, "family": FUNNEL_FONT_FAMILY},
            )

        fig.update_layout(
            template="plotly_dark",
            title={"text": f"{title}<br><sup>{subtitle}</sup>", "x": 0.5},
            font={"color": FUNNEL_TEXT_COLOR, "family": FUNNEL_FONT_FAMILY, "size": 16},
            paper_bgcolor=FUNNEL_BG_COLOR, plot_bgcolor=FUNNEL_BG_COLOR,
            width=900, height=980,
            margin={"l": 60, "r": 60, "t": 80, "b": 80},
            xaxis={
                "title": "Frequency of Word (log total count)",
                "range": [min_x - x_span * 0.08, max_x + x_span * 0.08],
                "showgrid": True, "gridcolor": "rgba(255,255,255,0.12)",
            },
            yaxis={
                "title": "Log-odds z-score",
                "range": [y_min, y_max],
                "zeroline": True, "zerolinewidth": 1.5,
                "zerolinecolor": "rgba(255,255,255,0.45)",
                "showgrid": True, "gridcolor": "rgba(255,255,255,0.12)",
            },
        )

        chart_id = f"tp-{hashlib.md5((focus_channel + '::' + rest_label).encode()).hexdigest()[:10]}"
        plot_html = pio.to_html(fig, include_plotlyjs=True, full_html=False, config={"displayModeBar": False})
        return _wrap_funnel(chart_id, plot_html, side_panel)

    # Matplotlib fallback
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10), dpi=160, facecolor=FUNNEL_BG_COLOR)
    ax.set_facecolor(FUNNEL_BG_COLOR)

    for row_list, alpha in [(unlabeled, 0.35), (labeled, 0.92)]:
        if not row_list:
            continue
        ax.scatter(
            [r["x_log_total"] for r in row_list],
            [r["z"] for r in row_list],
            s=[_marker_size(r, sizing) * 2.0 for r in row_list],
            c=[FOCUS_COLOR if r["z"] >= 0 else OTHER_COLOR for r in row_list],
            alpha=alpha,
            edgecolors="white" if alpha > 0.5 else "none",
            linewidths=0.6 if alpha > 0.5 else 0,
        )

    xs = [r["x_log_total"] for r in rows]
    min_x, max_x = min(xs), max(xs)
    x_span = max(1.0, max_x - min_x)
    max_abs_z = max(abs(r["z"]) for r in rows)
    y_pad = max(0.2, max_abs_z * 0.12)
    y_min, y_max = -(max_abs_z + y_pad), max_abs_z + y_pad

    layout = _compute_label_layout(labeled, y_min, y_max, min_x, max_x, x_span)
    for idx, row in enumerate(labeled):
        pos = layout.get(idx, {"x": row["x_log_total"], "y": row["z"], "ha": "left"})
        ax.text(
            pos["x"], pos["y"], row["term"],
            va="center", ha=pos["ha"],
            fontsize=max(3, min(22, int(round(_label_size(row, sizing))))),
            fontfamily="DejaVu Sans", fontweight="bold", color=FUNNEL_TEXT_COLOR,
        )

    ax.axhline(0, color="#9ca3af", linewidth=1.2)
    ax.set_xlim(min_x - x_span * 0.08, max_x + x_span * 0.08)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Frequency (log total)", color=FUNNEL_TEXT_COLOR, fontsize=14, fontweight="bold")
    ax.set_ylabel("Log-odds z-score", color=FUNNEL_TEXT_COLOR, fontsize=14, fontweight="bold")
    ax.set_title(f"Funnel Chart: {focus_channel} (top)  {rest_label} (bottom)\n{subtitle}", color=FUNNEL_TEXT_COLOR, fontsize=16, fontweight="bold")
    ax.tick_params(colors=FUNNEL_TEXT_COLOR, labelsize=12)
    for spine in ax.spines.values():
        spine.set_color("#3a4659")
    ax.grid(color="#ffffff", alpha=0.12, linewidth=0.8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=FUNNEL_BG_COLOR)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode()
    img_html = (
        f"<img alt='Funnel chart: {html.escape(focus_channel)} vs {html.escape(rest_label)}' "
        f"src='data:image/png;base64,{encoded}' style='display:block;width:100%;height:auto;background:{FUNNEL_BG_COLOR};'/>"
    )
    return _wrap_funnel(None, img_html, side_panel)


def _wrap_funnel(chart_id: Optional[str], plot_html: str, side_panel: str) -> str:
    """Wrap the funnel chart and side panel in HTML with styling."""
    styles = (
        f".tw{{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap;background:{FUNNEL_BG_COLOR};padding:8px;border-radius:8px;}}"
        f".tm{{flex:1 1 700px;min-width:560px;background:{FUNNEL_BG_COLOR};border:1px solid #2b3445;border-radius:8px;overflow:hidden;}}"
        f".side-panel{{flex:0 0 340px;height:980px;display:flex;flex-direction:column;border:1px solid #2b3445;border-radius:8px;background:{FUNNEL_BG_COLOR};overflow:hidden;}}"
        f".stack{{flex:1;overflow:auto;padding:6px 10px;display:flex;flex-direction:column;background:{FUNNEL_BG_COLOR};}}"
        ".stack-top{justify-content:flex-start;align-items:flex-start;}"
        ".stack-bot{justify-content:flex-start;align-items:flex-start;}"
        ".stack-line{height:1px;background:rgba(255,255,255,0.20);margin:0 8px;}"
        f".sw{{font-family:{FUNNEL_FONT_FAMILY};font-weight:700;letter-spacing:0.01em;white-space:nowrap;opacity:0.98;padding:0;text-shadow:0 0 1.6px rgba(0,0,0,0.95),0 0 3px rgba(0,0,0,0.65);}}"
        f".sw-f{{color:{FOCUS_COLOR};}}"
        f".sw-o{{color:{OTHER_COLOR};}}"
        ".side-panel.top-only .stack-bot,.side-panel.bot-only .stack-top{display:none;}"
        ".side-panel.top-only .stack-line,.side-panel.bot-only .stack-line{display:none;}"
        f".tm .plotly,.tm .plot-container,.tm .svg-container{{background:{FUNNEL_BG_COLOR}!important;}}"
        ".tm .annotation-text,.tm .xaxislayer-above text,.tm .yaxislayer-above text,.tm .gtitle,.tm .xtitle,.tm .ytitle{font-weight:700!important;}"
        "@media(max-width:980px){.tm{min-width:100%;}.side-panel{flex:1 1 100%;height:620px;}}"
    )
    chart_div = f"<div class='tm' id='{chart_id}'>{plot_html}</div>" if chart_id else f"<div class='tm'>{plot_html}</div>"
    script = ""
    if chart_id:
        script = (
            "<script>(function(){"
            f"var r=document.getElementById('{chart_id}');if(!r)return;"
            "function fx(){r.querySelectorAll('.scatterlayer .textpoint text,.annotation-text').forEach(function(e){"
            "e.style.fontWeight='800';"
            f"e.style.stroke='{FUNNEL_LABEL_OUTLINE_COLOR}';"
            "e.style.strokeWidth='3.8px';e.style.strokeOpacity='1';"
            "e.style.paintOrder='stroke fill';e.style.strokeLinejoin='round';"
            f"e.style.filter='drop-shadow(0 0 0.6px {FUNNEL_LABEL_OUTLINE_COLOR})';}})}}"
            "setTimeout(fx,80);new MutationObserver(fx).observe(r,{childList:true,subtree:true});"
            "})();</script>"
        )
    return f"<div class='tw'><style>{styles}</style>{chart_div}{side_panel}{script}</div>"
