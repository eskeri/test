"""HTML output generation functions.

These functions generate HTML reports for the signature phrase analysis results.
"""

import hashlib
import html
import io
from datetime import datetime
from typing import Dict, List, Optional

from .constants import HTML_ENTRY_EXCLUSION_RULES
from .scoring import PhraseScore
from .funnel_chart import _should_exclude


_Z_HEAT = [
    (50, "#0d5c0d", "#c8f5c8"), (20, "#1a7a1a", "#d4f5d4"),
    (10, "#2e9e2e", "#e2f5e2"), (5, "#5cb85c", "#f0faf0"),
    (2, "#999", "#f8f8f8"),     (0, "#bbb", "#fff"),
]


def _z_colors(z: float) -> tuple:
    """Get color pair for a z-score value."""
    for threshold, fg, bg in _Z_HEAT:
        if z >= threshold:
            return fg, bg
    return "#bbb", "#fff"


def _generate_filter_items_html(filters: dict) -> str:
    """Generate HTML for filter items list."""
    return "".join(
        f"<li><strong>{html.escape(str(k))}:</strong> {html.escape(str(v))}</li>"
        for k, v in filters.items()
    )


def _generate_phrase_row_html(
    s: PhraseScore,
    rank: int,
    cat_key: str,
    ch_label: str,
    focus_channels: List[str],
) -> str:
    """Generate HTML for a single phrase row in the results table."""
    if cat_key == "exclusive":
        fg, bg = "#cc6600", "#fff8f0"
        badge = "<span style='font-size:10px;color:#cc6600;font-weight:700'>[EXCLUSIVE]</span> "
        score_txt = "∞"
    elif cat_key == "dominant":
        fg, bg = "#0066cc", "#f0f4ff"
        pct = s.dominance * 100
        badge = f"<span style='font-size:10px;color:#0066cc;font-weight:700'>[{pct:.0f}%]</span> "
        score_txt = f"{pct:.1f}%"
    else:
        fg, bg = _z_colors(s.z_score)
        badge = ""
        score_txt = f"{s.z_score:.1f}"
    
    excluded = any(_should_exclude(s.phrase, fc, cat_key) for fc in focus_channels)
    if excluded:
        return ""
    
    return (
        f"<tr style='background:{bg}' data-fs='phrase' "
        f"data-phrase='{html.escape(s.phrase)}' "
        f"data-lo='{s.log_odds:.6f}' "
        f"data-fc='{s.focus_count}' data-rc='{s.rest_count}'>"
        f"<td class='rk'>{rank}</td>"
        f"<td class='ph'>{badge}{html.escape(s.phrase)}</td>"
        f"<td class='nm'>{s.focus_count:,}</td>"
        f"<td class='nm'>{s.rest_count:,}</td>"
        f"<td class='nm'>{s.log_odds:.3f}</td>"
        f"<td class='nm' style='color:{fg};font-weight:700'>{score_txt}</td>"
        "</tr>"
    )


def _generate_category_tab_panel_html(
    by_n: Dict,
    cat_key: str,
    cat_label: str,
    cat_desc: str,
    ch_label: str,
    focus_channels: List[str],
    top_n: int,
) -> str:
    """Generate HTML for a category tab panel (exclusive/distinctive/dominant)."""
    tab_btns, tab_panels = [], []
    first = True
    for n in sorted(by_n):
        scores = [
            s for s in by_n[n].get(cat_key, [])
            if not any(_should_exclude(s.phrase, fc, cat_key) for fc in focus_channels)
        ]
        if not scores:
            continue
        n_label = ["", "words", "bigrams", "trigrams"][n] if n <= 3 else f"{n}-grams"
        pid = f"p-{hashlib.md5(ch_label.encode()).hexdigest()[:10]}-{cat_key}-{n}"
        active_class = ' active' if first else ''
        tab_btns.append(
            f"<button class='tb{active_class}' onclick='showP(this,\"{pid}\")'>"
            f"{n_label} ({len(scores)})</button>"
        )
        rows_html = io.StringIO()
        for rank, s in enumerate(scores, 1):
            rows_html.write(_generate_phrase_row_html(s, rank, cat_key, ch_label, focus_channels))
        rows_html = rows_html.getvalue()
        display = "block" if first else "none"
        tab_panels.append(
            f"<div class='tp' data-fs='phrase' id='{pid}' style='display:{display}'>"
            "<table><thead><tr>"
            "<th>#</th><th>Phrase</th><th title='Focus count'>Focus #</th>"
            "<th title='Rest count'>Rest #</th><th>Log-odds</th>"
            f"<th title='{cat_desc}'>Score</th>"
            "</tr></thead><tbody>" + rows_html + "</tbody></table></div>"
        )
        first = False

    inner = (
        f"<div class='tab-bar'>{''.join(tab_btns)}</div>" + "".join(tab_panels)
        if tab_btns else "<div class='empty'>No results</div>"
    )
    return (
        f"<div class='cg'><h3>{cat_label}</h3>"
        f"<p class='desc'>{cat_desc}</p>{inner}</div>"
    )


def _generate_extra_metrics_html(
    extra_metrics: dict,
    ch_label: str,
    top_n: int,
    calculate_log_likelihoods: bool = False,
) -> str:
    """Generate HTML for extra collocation metrics (MI, delta_p, log_likelihood) separated by n-gram levels."""
    metric_specs = [
        ("mi", "Mutual Information", "MI"),
        ("delta_p", "ΔP", "ΔP"),
    ]
    if calculate_log_likelihoods:
        metric_specs.append(("log_likelihood", "Log-Likelihood", "G²"))
    
    metric_sections = []
    for metric_key, label, header in metric_specs:
        rows = extra_metrics.get(metric_key, [])
        if not rows:
            continue
        
        # Group by n-gram level
        by_n: Dict[int, List[dict]] = {}
        for row in rows:
            n = int(row.get("n", 0))
            by_n.setdefault(n, []).append(row)
        
        # Generate tabs for each n-gram level
        tab_btns, tab_panels = [], []
        first = True
        for n in sorted(by_n.keys()):
            # Filter by exclusion rules and sort by score
            filtered_rows = [
                r for r in by_n[n]
                if not _should_exclude(r.get("phrase", ""), ch_label, metric_key)
            ]
            if not filtered_rows:
                continue
            
            # Sort by score (descending) and apply top_n
            filtered_rows.sort(key=lambda x: -float(x.get("score", 0.0)))
            filtered_rows = filtered_rows[:top_n]
            
            n_label = ["", "words", "bigrams", "trigrams"][n] if n <= 3 else f"{n}-grams"
            pid = f"p-{hashlib.md5(ch_label.encode()).hexdigest()[:10]}-x-{metric_key}-{n}"
            active_class = ' active' if first else ''
            tab_btns.append(
                f"<button class='tb{active_class}' onclick='showP(this,\"{pid}\")'>"
                f"{n_label} ({len(filtered_rows)})</button>"
            )
            
            body = io.StringIO()
            for rank, row in enumerate(filtered_rows, 1):
                phrase = html.escape(str(row.get("phrase", "")))
                count = int(row.get("count", 0))
                score = float(row.get("score", 0.0))
                if metric_key == "mi":
                    detail = f"P(joint)={row.get('p_joint',0.):.6f}, prod P(wi)={row.get('p_parts',0.):.6f}"
                elif metric_key == "delta_p":
                    detail = f"P(B|A)={row.get('p_target_given_prefix',0.):.4f}, P(B|¬A)={row.get('p_target_given_not_prefix',0.):.4f}"
                else:  # log_likelihood
                    dice = float(row.get("dice", 0.0))
                    tfidf = float(row.get("tfidf_modulated", 0.0))
                    detail = f"Dice={dice:.3f}, TF-IDF modulated={tfidf:.2f}"
                body.write(
                    f"<tr data-fs='metric' data-phrase='{html.escape(str(row.get('phrase','')))}'>"
                    f"<td class='rk'>{rank}</td><td class='ph'>{phrase}</td>"
                    f"<td class='nm'>{count:,}</td>"
                    f"<td class='nm' style='font-weight:700'>{score:.4f}</td>"
                    f"<td>{detail}</td></tr>"
                )
            body = body.getvalue()
            display = "block" if first else "none"
            tab_panels.append(
                f"<div class='tp' data-fs='metric' id='{pid}' style='display:{display}'>"
                "<table><thead><tr><th>#</th><th>Phrase</th><th>Count</th>"
                f"<th>{html.escape(header)}</th><th>Details</th></tr></thead>"
                f"<tbody>{body}</tbody></table></div>"
            )
            first = False
        
        if not tab_btns:
            continue
        
        inner = (
            f"<div class='tab-bar'>{''.join(tab_btns)}</div>" + "".join(tab_panels)
        )
        metric_sections.append(
            f"<div class='cg'><h3>{label}</h3>"
            f"<p class='desc'>{label} collocations.</p>{inner}</div>"
        )
    
    return "".join(metric_sections)


def _generate_channel_section_html(
    ch_label: str,
    by_n: Dict,
    token_count: int,
    extra_metrics: Optional[dict],
    focus_channels: List[str],
    top_n: int,
    calculate_log_likelihoods: bool = False,
) -> str:
    """Generate HTML for a single channel section."""
    categories = [
        ("exclusive", "Exclusive", "Words only this channel uses"),
        ("distinctive", "Distinctive", "High statistical significance"),
        ("dominant", "Dominant", "≥10% of all usage in corpus"),
    ]

    cat_sections = []
    for cat_key, cat_label, cat_desc in categories:
        cat_sections.append(
            _generate_category_tab_panel_html(
                by_n, cat_key, cat_label, cat_desc, ch_label, focus_channels, top_n
            )
        )

    extra_section = ""
    if extra_metrics:
        extra_section = _generate_extra_metrics_html(extra_metrics, ch_label, top_n, calculate_log_likelihoods)

    section_id = hashlib.md5(ch_label.encode()).hexdigest()[:10]
    return (
        f"<section class='cs'>"
        f"<h2>{html.escape(ch_label)}<span class='badge'>{token_count:,} words</span></h2>"
        "<div class='sw-wrap'>"
        "<input class='sb' placeholder='Filter phrases (text, re:pattern, or /pattern/i)…' oninput='sf(this)'/>"
        "<div class='ft'>"
        "<label class='fc'><span class='fs'><input type='checkbox' class='fe lo-e' onchange='sf(this)'/> Log-odds</span>"
        "<div class='fi'><select class='fm lo-m' onchange='sf(this)'><option value='min'>keep ≥</option><option value='max'>keep ≤</option></select>"
        "<input class='fv lo-v' type='number' step='0.001' placeholder='0.000' oninput='sf(this)'/></div></label>"
        "<label class='fc'><span class='fs'><input type='checkbox' class='fe fc-e' onchange='sf(this)'/> Focus #</span>"
        "<div class='fi'><select class='fm fc-m' onchange='sf(this)'><option value='min'>keep ≥</option><option value='max'>keep ≤</option></select>"
        "<input class='fv fc-v' type='number' step='1' min='0' placeholder='0' oninput='sf(this)'/></div></label>"
        "<label class='fc'><span class='fs'><input type='checkbox' class='fe rc-e' onchange='sf(this)'/> Rest #</span>"
        "<div class='fi'><select class='fm rc-m' onchange='sf(this)'><option value='min'>keep ≥</option><option value='max' selected>keep ≤</option></select>"
        "<input class='fv rc-v' type='number' step='1' min='0' placeholder='0' oninput='sf(this)'/></div></label>"
        "</div><div class='fn'>Numeric filters apply to phrase tables only.</div></div>"
        + "".join(cat_sections) + extra_section + "</section>"
    )


def build_html(
    results_by_focus: Dict[str, Dict],
    focus_channels: List[str],
    focus_label: str,
    channel_token_counts: Dict[str, int],
    rest_channel_count: int,
    rest_token_count: int,
    filters: dict,
    additional_metrics: Optional[dict] = None,
    funnel_html: str = "",
    graph_only: bool = False,
    calculate_log_likelihoods: bool = False,
) -> str:
    """Build the complete HTML report."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filter_items = _generate_filter_items_html(filters)

    channel_sections = []
    for ch_label, by_n in results_by_focus.items():
        token_count = channel_token_counts.get(ch_label, 0)
        extra_metrics = (additional_metrics or {}).get(ch_label, {})
        channel_sections.append(
            _generate_channel_section_html(
                ch_label, by_n, token_count, extra_metrics, focus_channels, filters.get("top_n", 1000),
                calculate_log_likelihoods
            )
        )

    if graph_only:
        sections_html = ""
        if not funnel_html:
            funnel_html = "<div class='empty'>No funnel chart could be generated.</div>"
        graph_only_note = "<div class='meta'>Graph-only mode: phrase tables omitted.</div>"
    else:
        sections_html = "".join(channel_sections)
        graph_only_note = ""

    funnel_section = f"<section class='cs'><h2>Funnel Chart</h2>{funnel_html}</section>"

    return (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<title>Fighting Word (aka farting words in this world)</title><style>"
        "body{font-family:Arial,'Segoe UI',sans-serif;max-width:1200px;margin:20px auto;padding:0 16px}"
        "h1{margin-bottom:4px}"
        ".meta{color:#666;font-size:13px;margin-bottom:16px}"
        ".ft-toggle{margin:8px 0 10px}"
        ".ft-btn{border:1px solid #ccc;background:#fff;border-radius:6px;padding:6px 10px;cursor:pointer;font-size:13px}"
        ".filters{background:#f5f5f5;border:1px solid #ddd;border-radius:8px;padding:8px 14px;margin-bottom:22px;font-size:13px}"
        ".filters ul{margin:4px 0;padding-left:18px;column-count:2;column-gap:2em}"
        ".cs{border:1px solid #ddd;border-radius:10px;padding:14px 16px;margin-bottom:28px;background:#fafafa}"
        "h2{margin:0 0 10px;display:flex;align-items:center;gap:10px;font-size:18px}"
        ".badge{font-size:12px;font-weight:400;color:#666;border:1px solid #ddd;border-radius:999px;padding:2px 8px;background:#fff}"
        ".cg{margin-top:16px}"
        "h3{margin:8px 0;color:#444;font-size:14px}"
        "p.desc{margin:0 0 8px;font-size:12px;color:#666}"
        ".tab-bar{display:flex;gap:5px;margin-bottom:8px;flex-wrap:wrap}"
        ".tb{border:1px solid #ccc;background:#fff;border-radius:6px;padding:5px 11px;cursor:pointer;font-size:13px}"
        ".tb.active{background:#222;color:#fff;border-color:#222}"
        ".sw-wrap{margin-bottom:8px}"
        ".sb{width:100%;box-sizing:border-box;padding:6px 10px;border:1px solid #ccc;border-radius:6px;font-size:13px}"
        ".ft{display:grid;grid-template-columns:repeat(3,minmax(150px,1fr));gap:8px;margin-top:8px;align-items:end}"
        ".fc{display:flex;flex-direction:column;gap:4px;min-width:0}"
        ".fs{display:flex;align-items:center;gap:6px;font-size:12px;font-weight:700;color:#444;white-space:nowrap}"
        ".fi{display:flex;gap:6px;align-items:center}"
        ".fm{flex:0 0 auto;box-sizing:border-box;padding:6px 8px;border:1px solid #ccc;border-radius:6px;background:#fff;font-size:12px;min-width:86px}"
        ".fv{flex:1 1 auto;box-sizing:border-box;padding:6px 8px;border:1px solid #ccc;border-radius:6px;font-size:12px;min-width:0}"
        ".fn{margin-top:6px;font-size:11px;color:#666}"
        "table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{border:1px solid #e0e0e0;padding:5px 8px;text-align:left}"
        "thead{background:#f0f0f0;position:sticky;top:0;z-index:1}"
        "th{white-space:nowrap}"
        ".ph{font-family:monospace;font-size:13px}"
        ".rk{color:#999;width:32px;text-align:right}"
        ".nm{text-align:right;white-space:nowrap}"
        ".empty{padding:12px;color:#999;border:1px dashed #ddd;border-radius:6px}"
        ".hr{display:none}"
        "</style></head><body>"
        "<h1>serious political tools used on trivial subject matter</h1>"
        f"<div class='meta'>machine pooped this out on {html.escape(generated_at)}"
        f" · Focus: <strong>{html.escape(focus_label)}</strong>"
        f" · Channels: {', '.join(html.escape(c) for c in focus_channels)}"
        f" · Rest pool: {rest_channel_count:,} channels, {rest_token_count:,} tokens</div>"
        f"{graph_only_note}"
        "<div class='ft-toggle'>"
        "<button id='ftb' class='ft-btn' onclick='toggleFilters()'>Hide CLI Settings</button></div>"
        f"<div id='ftp' class='filters'><strong>CLI Settings:</strong><ul>{filter_items}</ul></div>"
        f"{funnel_section}"
        + sections_html
        + "<script>"
        "function toggleFilters(){"
        "var p=document.getElementById('ftp'),b=document.getElementById('ftb');"
        "var h=p.style.display==='none';"
        "p.style.display=h?'block':'none';"
        "b.textContent=h?'Hide CLI Settings':'Show CLI Settings';}"
        "var FJ=new WeakMap();"
        "function showP(btn,id){"
        "var sec=btn.closest('.cs'),panel=document.getElementById(id);"
        "if(!sec||!panel)return;"
        "var was=btn.classList.contains('active');"
        "sec.querySelectorAll('.tb').forEach(function(b){b.classList.remove('active');});"
        "sec.querySelectorAll('.tp').forEach(function(p){p.style.display='none';});"
        "if(was)return;"
        "btn.classList.add('active');panel.style.display='block';"
        "var s=sec.querySelector('.sb');if(s)sf(s);}"
        "function parseQ(raw){"
        "var t=(raw||'').trim();"
        "if(!t)return{mode:'empty'};"
        "if(t.startsWith('re:')){var b=t.slice(3).trim();if(!b)return{mode:'empty'};try{return{mode:'rx',rx:new RegExp(b,'i')}}catch(e){return{mode:'text',n:t.toLowerCase()}}}"
        "if(t.length>=2&&t[0]=='/'){var ls=t.lastIndexOf('/');if(ls>0){var b=t.slice(1,ls),fl=t.slice(ls+1)||'i';if(b){if(fl.indexOf('i')<0)fl+='i';try{return{mode:'rx',rx:new RegExp(b,fl)}}catch(e){}}}}"
        "return{mode:'text',n:t.toLowerCase()};}"
        "function match(phrase,q){"
        "if(q.mode==='empty')return true;"
        "if(q.mode==='rx')return q.rx.test(phrase);"
        "return phrase.toLowerCase().includes(q.n);}"
        "function readNf(sec,p,def){"
        "var e=sec.querySelector('.'+p+'-e'),m=sec.querySelector('.'+p+'-m'),v=sec.querySelector('.'+p+'-v');"
        "var raw=v?parseFloat(v.value):NaN;"
        "return{on:!!(e&&e.checked),mode:m?m.value:def,val:isFinite(raw)?raw:null};}"
        "function passNf(val,f){"
        "if(!f.on||f.val===null||!isFinite(val))return true;"
        "return f.mode==='max'?val<=f.val:val>=f.val;}"
        "function _runFilter(inp){"
        "var sec=inp.closest('.cs');if(!sec)return;"
        "var panel=null;sec.querySelectorAll('.tp').forEach(function(p){if(!panel&&p.style.display==='block')panel=p;});"
        "if(!panel)return;"
        "var tbody=panel.querySelector('tbody');if(!tbody)return;"
        "var rows=tbody.querySelectorAll('tr');"
        "var q=parseQ((sec.querySelector('.sb')||{value:''}).value);"
        "var lo=readNf(sec,'lo','min'),fc=readNf(sec,'fc','min'),rc=readNf(sec,'rc','max');"
        "var scope=panel.dataset.fs||'phrase';"
        "var jid=(FJ.get(sec)||0)+1;FJ.set(sec,jid);"
        "var i=0;"
        "function step(){"
        "if(FJ.get(sec)!==jid)return;"
        "var end=Math.min(i+350,rows.length);"
        "for(;i<end;i++){"
        "var tr=rows[i];"
        "var phrase=tr.dataset.phrase||((tr.querySelector('.ph')||{textContent:''}).textContent||'');"
        "var show=match(phrase,q);"
        "if(show&&scope==='phrase'){"
        "show=passNf(parseFloat(tr.dataset.lo),lo)&&passNf(parseInt(tr.dataset.fc,10),fc)&&passNf(parseInt(tr.dataset.rc,10),rc);}"
        "tr.classList.toggle('hr',!show);}"
        "if(i<rows.length)requestAnimationFrame(step);}"
        "requestAnimationFrame(step);}"
        "function sf(inp){"
        "var sec=inp.closest('.cs');if(!sec)return;"
        "if(sec._ft)clearTimeout(sec._ft);"
        "sec._ft=setTimeout(function(){_runFilter(inp);},110);}"
        "</script></body></html>"
    )
