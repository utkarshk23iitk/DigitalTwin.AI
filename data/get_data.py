"""
get_data.py — Defect-dataset provider for the DigitalTwin.ai prototype.

Strategy (documented in the README):
  1. If a real Bosch Production Line Performance CSV is present locally
     (data/bosch_numeric_sample.csv), use it.
  2. Else, generate a *Bosch-faithful* synthetic sample that reproduces the
     real dataset's structure and difficulty so the modelling code is valid
     against the genuine data with no changes.

To get the real data for tier 1, run `python data/fetch_bosch.py` — it handles
the Kaggle download and cuts a sampled CSV into place. Fetching is deliberately
a separate, explicit script rather than an implicit side effect of loading:
nothing here should silently pull 2 GB over the network mid-training.

Both tiers return an identically-shaped `meta` (including `station_of_col`), so
downstream station-grouped code works against either without changes.

Bosch-faithful properties reproduced:
  - Station-grouped anonymised numeric features named  L{line}_S{station}_F{feat}
  - Missing-NOT-at-random: each part visits only a subset of stations, so a
    part's features are present only for the stations it passed through.
  - Extreme class imbalance: ~0.58% positive (defective) parts.
  - Weak, distributed signal: defects depend on subtle multi-feature drift at a
    few "risky" stations rather than one obvious column.

Usage:
    from get_data import load_defect_data
    X, y, meta = load_defect_data()
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
REAL_SAMPLE = DATA_DIR / "bosch_numeric_sample.csv"

# Bosch names every numeric column L{line}_S{station}_F{feature}, with globally
# unique station numbers across lines. Parsing it is how we recover the station
# grouping from a real CSV, which carries no separate schema file.
_COL_RE = re.compile(r"^L(\d+)_S(\d+)_F(\d+)$")


def _station_map(cols) -> dict[str, int]:
    """Map each feature column -> its station id, by parsing Bosch column names."""
    out = {}
    for c in cols:
        m = _COL_RE.match(str(c))
        if m:
            out[c] = int(m.group(2))
    return out

# ---------------------------------------------------------------------------
# Structure knobs, calibrated against a measured 100k-row sample of the real
# train_numeric.csv (see the comparison table in the README). Targets:
#     968 features · 50 stations · 0.596% defects · 81.1% missing
#     12.1 stations visited per part · 2,737 distinct routes
#     max cross-station feature correlation ~0.74
# ---------------------------------------------------------------------------
N_LINES = 4
STATIONS_PER_LINE = [12, 12, 12, 14]  # 50 stations, uneven — like the real set
TOTAL_FEATURES = 968                  # real numeric-file feature count
POS_RATE = 0.0058                     # real Bosch failure rate ~0.58%

# Routing. Bosch's four lines are *alternative paths*, not sequential stages: a
# part runs one line, so every other line's columns are null for it. That is
# where the bulk of the 81% missingness comes from — it is structural, not
# sensor dropout, and which stations a part skipped is itself informative.
# Measured on the real file: presence is *exactly* all-or-nothing within a
# station (mean recorded fraction = 1.000 when visited, every station). So a
# station is either fully read or fully absent — there is no per-feature
# dropout, and all missingness is structural. Stations are also visited at very
# different rates (real P(visit) ranges ~0.19-0.57), and consecutive stations
# form cells a part enters or skips together, which is what keeps the number of
# distinct routes in the thousands rather than tens of thousands.
# Some real stations are visited by 57% of parts — more than any one line's
# share — so line 0 is a *shared* entry line most parts pass through, with lines
# 1-3 as the alternative main paths. Popular stations also tend to carry fewer
# features (real: the P=0.57 stations have 12/2/3 features), which is how the
# file gets 24% of stations visited but only 19% of features present.
BLOCK_SIZE = 3          # stations per cell entered/skipped together
BLOCK_POP_LO = 0.06     # cell usage rates, spread to mirror the real range
BLOCK_POP_HI = 0.86
SHARED_LINE = 0         # entry line most parts touch
SHARED_P = 0.75         # P(a part passes through the shared entry line)
MAIN_LINE_P = (0.40, 0.25, 0.35)      # relative use of lines 1, 2, 3
CROSS_LINE_P = 0.25     # P(part also passes through part of the final line)
CROSS_LINE_FRAC = 0.30  # fraction of the final line such a part touches

# Latent process factors. Real Bosch stations are strongly predictive of one
# another (measured 0.5-0.74 between co-observed stations) because they share
# physical conditions: material batch, ambient drift, tool wear. We reproduce
# that with a few latent variables per part, with each station's loading drifting
# smoothly along the line (AR(1)), so nearby stations correlate and distant ones
# do not. Without this the virtual-sensor component has nothing to infer from.
N_LATENT = 6
STATION_AR = 0.90       # station-to-station latent similarity
LOADING_LO, LOADING_HI = 0.50, 0.88   # per-feature loading on its station factor

# One risky station per line, so a part meets at least one whichever line it
# runs. Defects still require a multi-feature pattern, never a single column.
RISKY_STATIONS = (4, 16, 28, 40)

STRUCT_SEED = 7         # fixes the *schema* independently of the data seed


def _build_schema():
    """
    Fixed station/column layout, independent of the data seed.

    Feature counts per station are uneven (real Bosch stations carry anywhere
    from a handful to dozens of measurements) and the F-indices run globally
    with gaps, matching real column names like `L3_S50_F4259`.
    """
    srng = np.random.default_rng(STRUCT_SEED)
    n_stations = sum(STATIONS_PER_LINE)

    raw = srng.integers(6, 40, size=n_stations)
    per_station = np.maximum(1, np.round(raw * TOTAL_FEATURES / raw.sum())).astype(int)
    per_station[0] += TOTAL_FEATURES - per_station.sum()   # absorb rounding drift

    line_of_station = np.repeat(np.arange(N_LINES), STATIONS_PER_LINE)

    cols, station_of_col, fidx = [], [], 0
    for sid in range(n_stations):
        for _ in range(per_station[sid]):
            cols.append(f"L{line_of_station[sid]}_S{sid}_F{fidx}")
            station_of_col.append(sid)
            fidx += int(srng.integers(1, 4))     # gaps, as in the real file
    return cols, np.array(station_of_col), line_of_station, n_stations, per_station


def _route_matrix(rng, n_rows, n_stations, line_of_station, per_station):
    """
    Which stations each part visited -> (n_rows, n_stations) boolean.

    A part runs ONE line (so the other three lines are structurally null), skips
    some stations on it, and sometimes crosses into the final line. That yields
    thousands of distinct routes rather than a handful of hard-coded ones.
    """
    srng = np.random.default_rng(STRUCT_SEED + 2)

    # group consecutive stations on each line into cells
    block_of_station = np.zeros(n_stations, dtype=int)
    nxt = 0
    for line in range(N_LINES):
        idx = np.where(line_of_station == line)[0]
        block_of_station[idx] = nxt + np.arange(len(idx)) // BLOCK_SIZE
        nxt = block_of_station[idx].max() + 1
    n_blocks = int(block_of_station.max()) + 1

    # Cells differ a lot in usage, and feature-light cells are the busy ones —
    # that inverse relationship is what lets 24% of stations carry only 19% of
    # the present feature values, as measured on the real file.
    block_feats = np.zeros(n_blocks)
    np.add.at(block_feats, block_of_station, per_station)
    rank = block_feats.argsort().argsort() / max(1, n_blocks - 1)
    block_pop = BLOCK_POP_HI - (BLOCK_POP_HI - BLOCK_POP_LO) * rank
    block_pop = np.clip(block_pop * srng.uniform(0.85, 1.15, n_blocks), 0.05, 1.0)

    # main path: one of lines 1..N-1; line 0 is a shared entry most parts touch
    main = np.array([l for l in range(N_LINES) if l != SHARED_LINE])
    p_main = np.array(MAIN_LINE_P, dtype=float)[:len(main)]
    p_main = p_main / p_main.sum()
    part_line = rng.choice(main, size=n_rows, p=p_main)

    on_line = line_of_station[None, :] == part_line[:, None]
    takes_entry = rng.random(n_rows) < SHARED_P
    on_line |= takes_entry[:, None] & (line_of_station[None, :] == SHARED_LINE)

    draw = rng.random((n_rows, n_blocks)) < block_pop[None, :]
    visited = on_line & draw[:, block_of_station]

    # a minority of parts also touch the final line (cross-line flow)
    final_line = N_LINES - 1
    crossers = (rng.random(n_rows) < CROSS_LINE_P) & (part_line != final_line)
    final_mask = line_of_station[None, :] == final_line
    extra = rng.random((n_rows, n_blocks)) < CROSS_LINE_FRAC
    visited |= crossers[:, None] & final_mask & extra[:, block_of_station]
    return visited


def _synthetic_bosch(n_rows: int = 40000, seed: int = 42):
    """Generate a Bosch-faithful synthetic numeric sample."""
    rng = np.random.default_rng(seed)
    srng = np.random.default_rng(STRUCT_SEED + 1)

    cols, station_of_col, line_of_station, n_stations, per_station = _build_schema()
    visited = _route_matrix(rng, n_rows, n_stations, line_of_station, per_station)

    # --- latent process factors -------------------------------------------
    # Each station's loading on the latent factors drifts smoothly along the
    # line (AR(1)), so adjacent stations share physical conditions and distant
    # ones do not. This is what a virtual sensor exploits.
    W = np.zeros((n_stations, N_LATENT))
    W[0] = srng.normal(size=N_LATENT)
    for s in range(1, n_stations):
        W[s] = (STATION_AR * W[s - 1]
                + np.sqrt(1 - STATION_AR ** 2) * srng.normal(size=N_LATENT))
    W /= np.linalg.norm(W, axis=1, keepdims=True)

    Z = rng.normal(size=(n_rows, N_LATENT))          # per-part conditions
    U = Z @ W.T                                      # (n_rows, n_stations)

    # per-feature loading on its own station's factor; the rest is noise
    load = srng.uniform(LOADING_LO, LOADING_HI, size=len(cols))

    X = np.full((n_rows, len(cols)), np.nan, dtype=np.float32)
    for sid in range(n_stations):
        idx = np.where(station_of_col == sid)[0]
        a = load[idx][None, :]
        vals = (U[:, sid][:, None] * a
                + rng.normal(size=(n_rows, len(idx))) * np.sqrt(1 - a ** 2))
        # all-or-nothing per station, exactly as measured on the real file
        keep = visited[:, sid][:, None]
        X[:, idx] = np.where(keep, vals, np.nan).astype(np.float32)

    # --- defect mechanism (unchanged in spirit) ---------------------------
    # Multi-feature drift at the risky stations: no single giveaway column.
    risk_cols = np.where(np.isin(station_of_col, RISKY_STATIONS))[0]
    rv = X[:, risk_cols]
    present = ~np.isnan(rv)
    n_present = present.sum(axis=1)

    # Mean drift and top-3 co-elevation, computed over PRESENT features only.
    # (Treating missing as 0.0 would let absent sensors masquerade as readings —
    # at 81% missingness that would dominate the signal.)
    mean_drift = np.where(n_present > 0,
                          np.nansum(np.where(present, rv, 0.0), axis=1)
                          / np.maximum(n_present, 1), 0.0)
    filled = np.where(present, rv, -np.inf)
    top3 = np.sort(filled, axis=1)[:, -3:]
    top3 = np.where(np.isfinite(top3), top3, np.nan)
    # A part that visited none of the risky stations has an all-NaN slice here;
    # that is expected under 81% missingness, not an error, so keep it quiet.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        co_elev = np.nanmean(top3, axis=1)
    co_elev = np.nan_to_num(co_elev, nan=0.0)

    latent = (1.6 * mean_drift + 1.2 * co_elev
              + 0.9 * rng.normal(0, 1, size=n_rows))

    # choose threshold to hit the target positive rate
    thresh = np.quantile(latent, 1 - POS_RATE)
    y = (latent >= thresh).astype(np.int8)

    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "Id", np.arange(1, n_rows + 1))
    df["Response"] = y  # Bosch's label column name
    meta = {
        "source": "synthetic-bosch-faithful",
        "n_rows": n_rows,
        "n_features": len(cols),
        "n_stations": n_stations,
        "pos_rate": float(y.mean()),
        "station_of_col": {c: int(s) for c, s in zip(cols, station_of_col)},
        "risky_stations": sorted(RISKY_STATIONS),
    }
    return df, meta


def load_defect_data(n_rows: int = 40000, seed: int = 42):
    """
    Return (X, y, meta).

    X    : DataFrame of numeric station features (with NaNs where not visited)
    y    : Series of 0/1 defect labels (Response)
    meta : dict describing the source and structure
    """
    if REAL_SAMPLE.exists():
        df = pd.read_csv(REAL_SAMPLE, low_memory=False)
        feat_cols = [c for c in df.columns if c not in ("Id", "Response")]
        station_of_col = _station_map(feat_cols)
        meta = {
            "source": "real-bosch-sample",
            "n_rows": len(df),
            "n_features": len(feat_cols),
            "n_stations": len(set(station_of_col.values())),
            "pos_rate": float(df["Response"].mean()),
            "station_of_col": station_of_col,
            # Ground truth is unknown for real data — it is a property of the
            # physical line, not the file. Downstream code must treat this as
            # optional and fall back to model-derived importance.
            "risky_stations": None,
        }
    else:
        df, meta = _synthetic_bosch(n_rows=n_rows, seed=seed)

    y = df["Response"].astype(int)
    X = df.drop(columns=[c for c in ("Id", "Response") if c in df.columns])
    return X, y, meta


if __name__ == "__main__":
    X, y, meta = load_defect_data()
    print("source     :", meta["source"])
    print("rows       :", len(X))
    print("features   :", X.shape[1])
    print("stations   :", meta.get("n_stations", "n/a"))
    print("pos rate   :", round(y.mean(), 5), f"({int(y.sum())} defects)")
    print("missing %  :", round(X.isna().mean().mean() * 100, 1))
    print("risky sta. :", meta.get("risky_stations") or "n/a (unknown for real data)")
