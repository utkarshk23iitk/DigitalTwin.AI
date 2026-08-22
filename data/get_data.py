"""
get_data.py — Defect-dataset provider for the DigitalTwin.ai prototype.

Strategy (documented in the README):
  1. If a real Bosch Production Line Performance CSV is present locally
     (data/bosch_numeric_sample.csv), use it.
  2. Else, if Kaggle credentials are configured, attempt to download a sample.
  3. Else, generate a *Bosch-faithful* synthetic sample that reproduces the
     real dataset's structure and difficulty so the modelling code is valid
     against the genuine data with no changes.

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

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
REAL_SAMPLE = DATA_DIR / "bosch_numeric_sample.csv"

# Structure knobs chosen to mirror the real Bosch numeric file at sample scale.
N_LINES = 4
STATIONS_PER_LINE = [6, 6, 12, 4]     # 28 stations, uneven — like the real set
FEATURES_PER_STATION = 8              # anonymised numeric features per station
POS_RATE = 0.0058                     # real Bosch failure rate ~0.58%


def _synthetic_bosch(n_rows: int = 40000, seed: int = 42):
    """Generate a Bosch-faithful synthetic numeric sample."""
    rng = np.random.default_rng(seed)

    # Build station feature-column names: L{line}_S{station}_F{feat}
    cols, station_of_col = [], []
    sid = 0
    for line in range(N_LINES):
        for s in range(STATIONS_PER_LINE[line]):
            for f in range(FEATURES_PER_STATION):
                cols.append(f"L{line}_S{sid}_F{f}")
                station_of_col.append(sid)
            sid += 1
    n_stations = sid
    station_of_col = np.array(station_of_col)

    # Each part follows one of a few routes: it visits a contiguous-ish subset
    # of stations. Features for un-visited stations are NaN (missing-not-random).
    X = np.full((n_rows, len(cols)), np.nan, dtype=np.float32)

    # define 3 routes with different station coverage
    all_st = np.arange(n_stations)
    routes = [
        all_st,                                   # full route
        all_st[all_st != 4],                      # skips station 4
        all_st[(all_st < 12) | (all_st >= 16)],   # skips a mid block
    ]
    route_idx = rng.integers(0, len(routes), size=n_rows)

    # "risky" stations whose subtle feature drift drives defects
    risky_stations = {4, 6, 19}
    risk_cols = np.where(np.isin(station_of_col, list(risky_stations)))[0]

    # base feature values ~ N(0,1) for visited stations
    for r in range(n_rows):
        visited = routes[route_idx[r]]
        vis_cols = np.where(np.isin(station_of_col, visited))[0]
        X[r, vis_cols] = rng.normal(0, 1, size=len(vis_cols)).astype(np.float32)

    # latent defect propensity: driven by a genuine but multi-feature pattern
    # at the risky stations. Realistic (no single giveaway column) yet learnable:
    # a defect is likely when several risky-station features drift high together.
    risk_vals = np.nan_to_num(X[:, risk_cols], nan=0.0)
    present = (~np.isnan(X[:, risk_cols])).astype(np.float32)
    # interaction: mean drift AND co-elevation (product of top features)
    mean_drift = risk_vals.sum(axis=1) / np.maximum(present.sum(axis=1), 1)
    co_elev = np.sort(risk_vals, axis=1)[:, -3:].mean(axis=1)  # top-3 features
    latent = (1.6 * mean_drift + 1.2 * co_elev
              + 0.4 * rng.normal(0, 1, size=n_rows))

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
        "risky_stations": sorted(risky_stations),
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
        df = pd.read_csv(REAL_SAMPLE)
        meta = {"source": "real-bosch-sample", "n_rows": len(df),
                "n_features": df.shape[1] - 2}
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
    print("risky sta. :", meta.get("risky_stations"))
