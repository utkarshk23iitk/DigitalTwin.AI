"""
viz.py — visual views of the assembly-line simulation.

Three views, each answering a different question about the same run:

    1. state_timeline()      "what was every station doing, moment to moment?"
    2. buffer_heatmap()      "where did work pile up?"
    3. state_composition()   "how did each station spend the shift, in total?"

The first two are the ones that make the bottleneck visible without being told
where it is: a wall of BLOCKED upstream of the slow station, STARVED downstream,
and buffers saturated on one side of it and empty on the other.

Figures are Plotly, so they drop straight into Streamlit with
`st.plotly_chart(fig, use_container_width=True)` and also stand alone as HTML.

    python src/viz.py            # writes docs/line_sim_views.html and opens it
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from line_sim import WORKING, BLOCKED, STARVED, DOWN, AssemblyLine, default_line


# ------------------------------- palette ---------------------------------- #
# Station state is a *status* encoding, not a series encoding, so it uses the
# fixed status palette (never themed) rather than categorical hues. Colour never
# carries the meaning alone — every view keeps a legend, and hover names the state.

STATE_COLOR = {
    WORKING: "#0ca30c",   # good
    STARVED: "#fab219",   # warning     — idle, waiting on upstream
    DOWN:    "#ec835a",   # serious     — broken
    BLOCKED: "#d03b3b",   # critical    — downstream cannot take the unit
}
STATE_ORDER = [WORKING, BLOCKED, STARVED, DOWN]

# Sequential blue ramp (100 -> 700) for buffer occupancy — a continuous
# magnitude, so the palest step is allowed to recede toward the surface.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]

# Ordinal sub-ramp (250 -> 700) for discrete marks ordered by position along the
# line. Discrete marks must stay legible, so it starts darker than BLUE_RAMP.
ORDINAL_RAMP = ["#86b6ef", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

THEMES = {
    "light": dict(surface="#fcfcfb", plane="#f9f9f7", ink="#0b0b0b",
                  ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7"),
    "dark":  dict(surface="#1a1a19", plane="#0d0d0d", ink="#ffffff",
                  ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                  axis="#383835"),
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _chrome(fig: go.Figure, theme: str, height: int) -> go.Figure:
    """Apply surface, ink and recessive grid/axis styling."""
    t = THEMES[theme]
    fig.update_layout(
        height=height,
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        font=dict(family=FONT, size=12, color=t["ink2"]),
        title_font=dict(size=15, color=t["ink"]),
        margin=dict(l=110, r=24, t=64, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, title_text=""),
        hoverlabel=dict(font_family=FONT),
    )
    fig.update_xaxes(gridcolor=t["grid"], zeroline=False,
                     linecolor=t["axis"], tickcolor=t["axis"],
                     tickfont=dict(color=t["muted"]))
    fig.update_yaxes(gridcolor=t["grid"], zeroline=False,
                     linecolor=t["axis"], tickcolor=t["axis"],
                     tickfont=dict(color=t["muted"]))
    return fig


# --------------------------- log -> tidy frames --------------------------- #

def state_spans(line: AssemblyLine) -> pd.DataFrame:
    """
    Turn the change-point log into [start, stop) spans, one row per span.

    The log records a station's state only when it *changes*, so a span runs
    from its own timestamp to that station's next entry. Consecutive entries in
    the same state (e.g. two units processed back to back with no wait between)
    are merged into one span.

    Note the consequence: **a span is not a unit.** A station that is never
    blocked or starved merges its entire run into one WORKING span, so counting
    spans undercounts production catastrophically. Use `station_stats()['produced']`
    for unit counts, and this function only for drawing durations.
    """
    end_t = line.env.now
    by_station: dict[int, list[dict]] = defaultdict(list)
    for e in line.log:
        by_station[e["station"]].append(e)

    rows = []
    for i, evs in by_station.items():
        evs.sort(key=lambda e: e["t"])
        j = 0
        while j < len(evs):
            k = j
            while k + 1 < len(evs) and evs[k + 1]["state"] == evs[j]["state"]:
                k += 1
            stop = evs[k + 1]["t"] if k + 1 < len(evs) else end_t
            rows.append({
                "station": i,
                "name": line.cfg.stations[i].name,
                "state": evs[j]["state"],
                "start": evs[j]["t"],
                "stop": stop,
                "duration": stop - evs[j]["t"],
            })
            j = k + 1
    return pd.DataFrame(rows).sort_values(["station", "start"])


def buffer_levels(line: AssemblyLine, grid: np.ndarray) -> np.ndarray:
    """
    Resample buffer occupancy onto a regular time grid (zero-order hold).

    Returns an array of shape (n_buffers, len(grid)). Buffer i is the input
    buffer of station i, so buffer 0 is the raw-material feed.

    Reads `line.buffer_log` (fixed-interval polling), *not* the event log: the
    event log's buffer readings are taken at state changes, which for a station
    means the instant just after it pulled a unit off its own input queue. That
    is a length-biased sample and reads near-zero for busy stations.
    """
    if not line.buffer_log:
        raise ValueError("no buffer samples — run with LineConfig.sample_interval > 0")
    ts = np.array([s["t"] for s in line.buffer_log])
    levels = np.array([s["levels"] for s in line.buffer_log])   # (samples, buffers)
    idx = np.clip(np.searchsorted(ts, grid, side="right") - 1, 0, len(ts) - 1)
    return levels[idx].T


# ------------------------------- the views -------------------------------- #

def state_timeline(line: AssemblyLine, t0: float | None = None,
                   t1: float | None = None, theme: str = "light") -> go.Figure:
    """
    View 1 — the Gantt. Station on y, sim time on x, spans coloured by state.

    Read it top to bottom: material flows down the y axis, so the bottleneck is
    the row where red (BLOCKED) above meets yellow (STARVED) below.

    Defaults to a 600-second window in the middle of the run — the whole run
    compresses spans below a pixel and the chart turns to mush.
    """
    t = THEMES[theme]
    df = state_spans(line)
    mid = (line.cfg.warmup + line.env.now) / 2
    t0 = mid - 300 if t0 is None else t0
    t1 = mid + 300 if t1 is None else t1
    df = df[(df["stop"] > t0) & (df["start"] < t1)].copy()
    df["start"] = df["start"].clip(lower=t0)
    df["stop"] = df["stop"].clip(upper=t1)
    df["duration"] = df["stop"] - df["start"]

    names = [s.name for s in line.cfg.stations]
    fig = go.Figure()
    for state in STATE_ORDER:
        d = df[df["state"] == state]
        fig.add_bar(
            y=d["name"], x=d["duration"], base=d["start"],
            orientation="h", name=state,
            marker=dict(color=STATE_COLOR[state],
                        line=dict(color=t["surface"], width=0.5)),
            customdata=np.stack([d["start"], d["stop"]], axis=-1),
            hovertemplate=("<b>%{y}</b><br>" + state +
                           "<br>%{customdata[0]:.0f}s → %{customdata[1]:.0f}s"
                           " (%{x:.0f}s)<extra></extra>"),
        )

    fig.update_layout(
        barmode="overlay", bargap=0.35,
        title=("Station state over time — a wall of BLOCKED above the "
               "constraint, STARVED below it"),
    )
    fig.update_yaxes(categoryorder="array", categoryarray=names[::-1],
                     showgrid=False,
                     tickfont=dict(color=t["ink2"], size=11))
    fig.update_xaxes(title_text="simulation time (s)", range=[t0, t1])
    return _chrome(fig, theme, height=520)


def buffer_heatmap(line: AssemblyLine, t0: float | None = None,
                   t1: float | None = None, theme: str = "light",
                   cols: int = 480) -> go.Figure:
    """
    View 2 — where work piles up. Buffer on y, time on x, colour = occupancy.

    Occupancy is a magnitude, so it gets one sequential hue light->dark. Dark
    band = a full buffer (work waiting); pale band = starved. The constraint is
    the boundary between the two.
    """
    t = THEMES[theme]
    # Same default window as state_timeline, so the two stack up readable
    # against each other — the same 600 seconds, states above, queues below.
    # Over the full run the queues are pinned at 4 and 0 and the chart flattens
    # into two solid blocks; view 3 is the one that makes the whole-shift case.
    mid = (line.cfg.warmup + line.env.now) / 2
    t0 = mid - 300 if t0 is None else t0
    t1 = mid + 300 if t1 is None else t1
    grid = np.linspace(t0, t1, cols)
    z = buffer_levels(line, grid)
    labels = [f"→ {s.name}" for s in line.cfg.stations]
    cap = line.cfg.buffer_capacity

    fig = go.Figure(go.Heatmap(
        z=z, x=grid, y=labels,
        colorscale=[[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)],
        zmin=0, zmax=cap,
        colorbar=dict(title=dict(text="units<br>queued", font=dict(size=11)),
                      thickness=12, outlinewidth=0,
                      tickfont=dict(color=t["muted"], size=10)),
        hovertemplate="<b>%{y}</b><br>t=%{x:.0f}s<br>%{z:.0f} of "
                      + str(cap) + " queued<extra></extra>",
    ))
    fig.update_layout(title="Input-buffer occupancy — work piles up in front of "
                            "the constraint and drains behind it")
    fig.update_yaxes(autorange="reversed", showgrid=False,
                     tickfont=dict(color=t["ink2"], size=11))
    fig.update_xaxes(title_text="simulation time (s)", showgrid=False)
    return _chrome(fig, theme, height=460)


def state_composition(line: AssemblyLine, theme: str = "light") -> go.Figure:
    """
    View 3 — the shift summary. One stacked bar per station, fractions of time.

    This is the supervisor's view: the constraint is the only station whose
    WORKING band runs nearly the full width.
    """
    t = THEMES[theme]
    stats = line.station_stats()
    names = [s["name"] for s in stats]
    frac = {
        WORKING: [s["utilisation"] for s in stats],
        BLOCKED: [s["blocked_frac"] for s in stats],
        STARVED: [s["starved_frac"] for s in stats],
        DOWN:    [s["down_frac"] for s in stats],
    }

    fig = go.Figure()
    for state in STATE_ORDER:
        fig.add_bar(
            y=names, x=frac[state], orientation="h", name=state,
            marker=dict(color=STATE_COLOR[state],
                        line=dict(color=t["surface"], width=1)),
            hovertemplate="<b>%{y}</b><br>" + state +
                          " %{x:.1%} of the shift<extra></extra>",
        )

    # Direct-label utilisation only — a number on every segment is noise. The
    # labels sit past the 100% end rather than at the WORKING/next boundary, so
    # they never land on a coloured segment and can stay in muted ink.
    fig.add_scatter(
        y=names, x=[1.02] * len(names), mode="text",
        text=[f"{u:.0%}" for u in frac[WORKING]],
        textposition="middle right", showlegend=False, hoverinfo="skip",
        textfont=dict(color=t["ink2"], size=11),
    )

    fig.update_layout(barmode="stack", bargap=0.3,
                      # plotly reverses the legend for stacked bars by default,
                      # which would list the states backwards from the stack
                      legend_traceorder="normal",
                      title="Where the shift went — utilisation is labelled; "
                            "the constraint is the one station never starved")
    fig.update_yaxes(autorange="reversed", showgrid=False,
                     tickfont=dict(color=t["ink2"], size=11))
    fig.update_xaxes(title_text="fraction of post-warmup time",
                     tickformat=".0%", range=[0, 1.12])
    return _chrome(fig, theme, height=480)


# A cumulative-output-per-station chart was tried here and dropped. At steady
# state every station runs at the constraint's rate by definition, so all twelve
# curves are parallel and there is no "slowest line" to see — the information is
# in the *state* mix, which views 1 and 3 already carry.


# --------------------------------- main ----------------------------------- #

def build_report(line: AssemblyLine, path: str, theme: str = "light") -> str:
    """Write all three views to a single standalone HTML file."""
    t = THEMES[theme]
    figs = [state_timeline(line, theme=theme),
            buffer_heatmap(line, theme=theme),
            state_composition(line, theme=theme)]
    parts = [f.to_html(full_html=False, include_plotlyjs=(i == 0))
             for i, f in enumerate(figs)]
    html = f"""<!doctype html><meta charset="utf-8">
<title>Assembly-line simulation views</title>
<style>
 body {{ background:{t['plane']}; color:{t['ink']}; font-family:{FONT};
        margin:0; padding:32px; }}
 h1 {{ font-size:20px; font-weight:600; margin:0 0 4px; }}
 p.sub {{ color:{t['ink2']}; margin:0 0 28px; font-size:13px; }}
 .card {{ background:{t['surface']}; border:1px solid rgba(11,11,11,.10);
          border-radius:10px; padding:8px 12px; margin-bottom:20px; }}
</style>
<h1>Assembly-line simulation — {len(line.cfg.stations)} stations,
 {line.env.now:.0f}s of sim time</h1>
<p class="sub">Generated by <code>src/viz.py</code>. No view is told where the
 bottleneck is; all three make it visible.</p>
""" + "".join(f'<div class="card">{p}</div>' for p in parts)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


if __name__ == "__main__":
    import os
    import webbrowser

    # Steady state, not the 3000s/200s-warmup default: at that horizon the line
    # is still filling its buffers, so the upstream stations never block and the
    # bottleneck signature these views exist to show has not formed yet. The
    # line needs ~12 stations x 50s to fill once, then ~24 buffer slots x 72s to
    # saturate against Topcoat — roughly t=2300 before the pattern settles.
    line = default_line()
    line.cfg.warmup = 4000.0
    line.run(until=20000)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "line_sim_views.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    build_report(line, out)
    print(f"wrote {out}")
    webbrowser.open(f"file://{out}")
