"""
fetch_bosch.py — Download the real Bosch numeric data and cut a usable sample.

The full Bosch Production Line Performance dataset is ~14 GB and lives behind a
Kaggle *competition* login. We only need `train_numeric.csv` (~2 GB zipped,
1.18M parts x 970 numeric features), and for a prototype we only need a slice of
it. This script does both steps:

    download train_numeric.csv  ->  stratified row sample  ->  bosch_numeric_sample.csv

`get_data.py` picks that sample up automatically (tier 1) with no code changes.

One-time setup (only you can do this — it needs a human to accept the rules):
  1. Create a free Kaggle account.
  2. Visit https://www.kaggle.com/c/bosch-production-line-performance/rules
     and click "I Understand and Accept". The API returns 403 until you do.
     This step cannot be automated and is the usual reason a download fails.
  3. Authenticate the client (kaggle >= 2.2 uses OAuth, not the old
     kaggle.json):
         kaggle auth login
     Alternatively, generate a token at https://www.kaggle.com/settings/api
     and either export KAGGLE_API_TOKEN=... or save it to
     ~/.kaggle/access_token

Then:
    python data/fetch_bosch.py                 # 100k-row sample, natural defect rate
    python data/fetch_bosch.py --rows 200000   # bigger sample
    python data/fetch_bosch.py --all-positives # keep every defect (raises the rate)

Sampling note: by default we Bernoulli-sample rows *without* touching the class
balance, so the sample keeps Bosch's genuine ~0.58% defect rate — that extreme
imbalance is one of the four properties we claim to model, so preserving it
matters more than having a comfortable number of positives.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
COMPETITION = "bosch-production-line-performance"
# Kaggle serves this competition's files pre-zipped, so the *remote* name carries
# a .zip suffix while the member inside the archive does not. Requesting the
# unsuffixed name returns 404, not 403 — easy to misread as an auth problem.
REMOTE_FILE = "train_numeric.csv.zip"
TARGET_FILE = "train_numeric.csv"
RAW_ZIP = DATA_DIR / REMOTE_FILE
RAW_CSV = DATA_DIR / TARGET_FILE
SAMPLE_CSV = DATA_DIR / "bosch_numeric_sample.csv"

# Documented row count of Bosch train_numeric.csv. Only used to pick the
# sampling probability; override with --total if it ever changes.
BOSCH_TRAIN_ROWS = 1_183_747

CHUNK_ROWS = 50_000


def _credential_help() -> str:
    return (
        "\nKaggle credentials not found or not authorised.\n"
        "  1. Authenticate the client:   kaggle auth login\n"
        f"  2. Accept the competition rules at\n"
        f"     https://www.kaggle.com/c/{COMPETITION}/rules\n"
        "     (the API returns 403 Forbidden until you click Accept)\n"
        "\nAlternatively, download train_numeric.csv by hand from\n"
        f"  https://www.kaggle.com/c/{COMPETITION}/data\n"
        f"and drop it at {RAW_CSV}, then re-run this script — it will skip the\n"
        "download and go straight to sampling.\n"
    )


def download() -> Path:
    """Fetch train_numeric.csv from Kaggle unless it is already on disk."""
    if RAW_CSV.exists():
        print(f"[skip]  {RAW_CSV.name} already present "
              f"({RAW_CSV.stat().st_size / 1e9:.2f} GB)")
        return RAW_CSV

    if not RAW_ZIP.exists():
        # The kaggle client calls sys.exit() itself on missing credentials, so we
        # must catch SystemExit as well or our own guidance — including the
        # rules-acceptance step, which is the usual culprit — never prints.
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except SystemExit:
            sys.exit(_credential_help())
        except Exception as exc:                       # noqa: BLE001
            sys.exit(f"kaggle package unavailable: {exc}\n"
                     "Install it with:  pip install kaggle")

        api = KaggleApi()
        try:
            api.authenticate()
        except SystemExit:
            sys.exit(_credential_help())
        except Exception as exc:                       # noqa: BLE001
            sys.exit(f"Could not authenticate with Kaggle: {exc}\n"
                     + _credential_help())

        print(f"[dl]    downloading {REMOTE_FILE} (283 MB zipped, ~2 GB "
              f"unzipped)...")
        try:
            api.competition_download_file(
                COMPETITION, REMOTE_FILE, path=str(DATA_DIR), quiet=False)
        except SystemExit:
            sys.exit(_credential_help())
        except Exception as exc:                       # noqa: BLE001
            # A 404 means the remote filename is wrong, NOT that you lack access
            # — don't send the reader chasing credentials for a naming problem.
            if "404" in str(exc):
                sys.exit(
                    f"Download failed: {exc}\n\n"
                    f"'{REMOTE_FILE}' was not found in the competition. The file\n"
                    "names may have changed. List what is actually available:\n"
                    f"    kaggle competitions files {COMPETITION}\n"
                    "then update REMOTE_FILE at the top of this script.\n")
            sys.exit(f"Download failed: {exc}\n" + _credential_help())

    if RAW_ZIP.exists():
        with zipfile.ZipFile(RAW_ZIP) as zf:
            members = [n for n in zf.namelist() if n.endswith(".csv")]
            if not members:
                sys.exit(f"No .csv inside {RAW_ZIP.name}: {zf.namelist()}")
            member = members[0]
            print(f"[unzip] {RAW_ZIP.name} -> {member}")
            zf.extract(member, path=DATA_DIR)
            if member != TARGET_FILE:
                (DATA_DIR / member).rename(RAW_CSV)
        # Keep the 283 MB archive: it is gitignored, and retaining it makes
        # re-sampling at a different --rows free instead of a fresh download.
        # Only the 2.14 GB extract is worth reclaiming (see --keep-raw).
        print(f"[keep]  {RAW_ZIP.name} retained for re-sampling")

    if not RAW_CSV.exists():
        sys.exit(f"Expected {RAW_CSV} after download but it is missing.")
    return RAW_CSV


def sample(n_rows: int, total: int, all_positives: bool, seed: int) -> pd.DataFrame:
    """
    Single-pass, memory-safe stratified sample of the 2 GB CSV.

    We never load the whole file: 1.18M x 970 float64 would be ~9 GB in RAM.
    Instead we stream 50k-row chunks as float32 and keep a Bernoulli subset of
    each, which preserves the natural class ratio in expectation.
    """
    if not RAW_CSV.exists():
        sys.exit(f"{RAW_CSV} not found — run the download step first.")

    p_keep = min(1.0, n_rows / max(1, total))
    rng = np.random.default_rng(seed)
    kept: list[pd.DataFrame] = []
    seen = kept_rows = 0

    print(f"[scan]  streaming {RAW_CSV.name} in {CHUNK_ROWS:,}-row chunks "
          f"(keep p={p_keep:.4f})")
    reader = pd.read_csv(RAW_CSV, chunksize=CHUNK_ROWS, dtype=np.float32,
                         low_memory=False)
    for i, chunk in enumerate(reader):
        seen += len(chunk)
        mask = rng.random(len(chunk)) < p_keep
        if all_positives and "Response" in chunk.columns:
            mask |= (chunk["Response"].to_numpy() == 1)
        sub = chunk.loc[mask]
        if len(sub):
            kept.append(sub)
            kept_rows += len(sub)
        if i % 5 == 0:
            print(f"        {seen:>9,} rows scanned | {kept_rows:>7,} kept",
                  end="\r", flush=True)

    print(f"\n[scan]  done: {seen:,} rows scanned, {kept_rows:,} kept")
    df = pd.concat(kept, ignore_index=True)

    # restore integer dtypes that the float32 read flattened
    for col, dtype in (("Id", "int64"), ("Response", "int8")):
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=100_000,
                    help="approximate sample size (default: 100000)")
    ap.add_argument("--total", type=int, default=BOSCH_TRAIN_ROWS,
                    help="row count of the source file, for the keep probability")
    ap.add_argument("--all-positives", action="store_true",
                    help="additionally keep every defect (inflates the defect "
                         "rate above Bosch's real 0.58%% — say so if you use it)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep the 2 GB train_numeric.csv after sampling")
    args = ap.parse_args()

    download()
    df = sample(args.rows, args.total, args.all_positives, args.seed)

    df.to_csv(SAMPLE_CSV, index=False)
    size_mb = SAMPLE_CSV.stat().st_size / 1e6

    n_feat = df.shape[1] - sum(c in df.columns for c in ("Id", "Response"))
    pos = int(df["Response"].sum()) if "Response" in df.columns else 0
    feat_cols = [c for c in df.columns if c not in ("Id", "Response")]
    missing = df[feat_cols].isna().to_numpy().mean() * 100

    print(f"\n[write] {SAMPLE_CSV.name}  ({size_mb:.1f} MB)")
    print(f"        rows      : {len(df):,}")
    print(f"        features  : {n_feat:,}")
    print(f"        defects   : {pos:,}  ({pos / max(1, len(df)) * 100:.3f}%)")
    print(f"        missing   : {missing:.1f}%")
    print("\nDone. get_data.py will now load this automatically (tier 1).")

    if not args.keep_raw and RAW_CSV.exists():
        RAW_CSV.unlink()
        print(f"[clean] removed {RAW_CSV.name} (use --keep-raw to retain it)")


if __name__ == "__main__":
    main()
