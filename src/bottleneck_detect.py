"""
bottleneck.py — Bottleneck detection for the assembly-line twin.

Two complementary signals:

1. Active-period method (Roser & Nakano):
   The bottleneck is the station with the largest share of time spent in an
   uninterrupted *active* period (WORKING or BLOCKED — i.e. not starved and not
   idle). Averaged over the horizon this identifies the sustained constraint;
   the station whose "sole active" share is highest is the primary bottleneck.
   We use the practical proxy: the station with the highest (utilisation +
   blocked) time, since a true bottleneck is rarely starved and is often
   blocked-downstream only when *another* station constrains it. The station
   that is busiest and least starved is the constraint.

2. Queue-growth (live early-warning):
   A bottleneck *forming* shows up as the input buffer of a station growing
   faster than it drains. We compute the slope of each station's input-buffer
   level over a recent window; a sustained positive slope is a leading
   indicator, often before utilisation saturates.

Neither signal can be read directly at a sensor-poor station — those rely on
inferred state from virtual_sensor.py, and their bottleneck verdict therefore
carries lower input-confidence (surfaced later via effective_trust.py).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def log_to_frame(line) -> pd.DataFrame:
    """Convert an AssemblyLine's event log into a tidy DataFrame."""
    return pd.DataFrame(line.log)


def active_period_bottleneck(line) -> pd.DataFrame:
    """
    Rank stations by a sustained-constraint score.

    score = utilisation + blocked_frac - starved_frac
    The highest score is the primary bottleneck: it is working (or blocked by
    nothing downstream) most of the time and rarely waiting for work.
    Returns a DataFrame sorted by score, with an `is_bottleneck` flag on the top.
    """
    stats = pd.DataFrame(line.station_stats())

    # A true bottleneck is highly utilised AND starves the stations downstream
    # of it. We measure "downstream starvation pressure": the mean starved
    # fraction of all stations after this one. The constraint is the station
    # that is itself busy while making everything after it wait.
    n = len(stats)
    downstream_starve = []
    for i in range(n):
        after = stats[stats["index"] > i]["starved_frac"]
        downstream_starve.append(after.mean() if len(after) else 0.0)
    stats["downstream_starve"] = downstream_starve

    # Score: own busyness (util+blocked) rewarded, own starvation penalised,
    # and a strong bonus for starving the downstream — which is what a real
    # constraint does. This discounts the front-of-line saturation artifact,
    # where station 0/1 look busy but starve no one meaningfully more.
    stats["score"] = (stats["utilisation"]
                      + stats["blocked_frac"]
                      - stats["starved_frac"]
                      + 1.5 * stats["downstream_starve"]).round(3)
    stats = stats.sort_values("score", ascending=False).reset_index(drop=True)
    stats["is_bottleneck"] = False
    stats.loc[0, "is_bottleneck"] = True
    return stats[["index", "name", "has_sensor", "utilisation",
                  "blocked_frac", "starved_frac", "score", "is_bottleneck"]]


def queue_growth(line, window: float = 400.0) -> pd.DataFrame:
    """
    Estimate the recent slope of each station's input-buffer level.

    A positive slope means work is piling up faster than the station clears it —
    a bottleneck forming. Returns per-station slope (units per 100s) over the
    last `window` seconds of the log.
    """
    df = log_to_frame(line)
    if df.empty:
        return pd.DataFrame(columns=["station", "queue_slope"])
    t_max = df["t"].max()
    recent = df[df["t"] >= t_max - window]
    rows = []
    for st, g in recent.groupby("station"):
        g = g.sort_values("t")
        if len(g) >= 2 and g["t"].nunique() >= 2:
            # linear fit of buffer_in vs time; scale slope to per-100s
            slope = np.polyfit(g["t"], g["buffer_in"], 1)[0] * 100.0
        else:
            slope = 0.0
        rows.append({"station": int(st), "queue_slope": round(float(slope), 3)})
    return pd.DataFrame(rows).sort_values("station").reset_index(drop=True)


def bottleneck_report(line, window: float = 400.0) -> dict:
    """
    Combine both signals into a single report.

    Returns:
        primary        — dict of the primary bottleneck station
        ranking        — active-period ranking (list of dicts)
        forming        — stations with a positive queue slope (list of dicts),
                         i.e. bottlenecks that may be forming
    """
    ranking = active_period_bottleneck(line)
    growth = queue_growth(line, window=window)
    merged = ranking.merge(growth.rename(columns={"station": "index"}),
                           on="index", how="left")
    merged["queue_slope"] = merged["queue_slope"].fillna(0.0)

    primary = merged.iloc[0].to_dict()
    forming = (merged[merged["queue_slope"] > 0.05]
               .sort_values("queue_slope", ascending=False)
               [["index", "name", "queue_slope", "has_sensor"]]
               .to_dict("records"))

    return {
        "primary": {
            "index": int(primary["index"]),
            "name": primary["name"],
            "score": float(primary["score"]),
            "utilisation": float(primary["utilisation"]),
            "has_sensor": bool(primary["has_sensor"]),
        },
        "ranking": merged[["index", "name", "score", "utilisation",
                           "queue_slope", "is_bottleneck", "has_sensor"]]
                   .to_dict("records"),
        "forming": forming,
    }


if __name__ == "__main__":
    from line_sim import default_line

    line = default_line().run(until=3000)
    rep = bottleneck_report(line)
    p = rep["primary"]
    print(f"PRIMARY BOTTLENECK: #{p['index']} {p['name']} "
          f"(score={p['score']}, util={p['utilisation']})\n")
    print("Active-period ranking:")
    for r in rep["ranking"]:
        tag = "  <== bottleneck" if r["is_bottleneck"] else ""
        sensor = "" if r["has_sensor"] else "  [sensor-poor]"
        print(f"  #{r['index']:>2} {r['name']:<14} score={r['score']:>6} "
              f"slope={r['queue_slope']:>6}{sensor}{tag}")
    if rep["forming"]:
        print("\nBottlenecks forming (queue growing):")
        for f in rep["forming"]:
            print(f"  #{f['index']:>2} {f['name']:<14} slope={f['queue_slope']}")
