# DigitalTwin.ai — Assembly-Line Digital Twin (Round 2 Prototype)

A live digital twin of a vehicle assembly line that (1) detects and predicts
**bottlenecks**, (2) predicts **defects** before they propagate, and — its
differentiator — stays **honest under partial observability**: where a station
has no sensor, it infers the station's state *and attaches a confidence*, then
fuses data-trust with model-confidence into an **Effective Trust** score that
gates every action.

> Accenture Innovation Challenge 2026 — Round 2, Problem Track 4 (DigitalTwin.ai)

---

## Architecture (at a glance)

```
Simulated line (SimPy)  ──►  bottleneck detection (active-period + queue-growth)
        │                                   │
        ▼                                   ▼
  station signals ─► defect model (XGBoost + conformal confidence)
        │                                   │
        ▼                                   ▼
  virtual sensor (sensor-poor stations)   model confidence
   value + input-trust  ─────────────┐     │
                                     ▼     ▼
                         Effective Trust = input_trust × model_confidence
                                     │
                                     ▼
                    Risk × Trust  ──►  action (auto / monitor / human-verify)
                                     │
                                     ▼
                 persona views: supervisor · manager · leadership
```

> **In production, SimPy is replaced by the live plant feed** (PLC / SCADA /
> IoT sensors, historian, MES) — it is only a stand-in for a real line here.
> Everything downstream is **source-agnostic**: the virtual sensor, defect
> model, bottleneck detection and trust layer consume the same station-data
> schema regardless of origin, so swapping in real telemetry needs no model
> changes. (A discrete-event simulator can *optionally* remain in a real
> deployment as the twin's forward-looking "what-if" / bottleneck-forecasting
> engine — a distinct role from generating the data.)

## Repo layout

```
digitaltwin-ai/
├── README.md
├── requirements.txt
├── data/
│   ├── fetch_bosch.py             # [DONE] download real Bosch + cut a sampled CSV
│   ├── get_data.py                # real-Bosch-if-available, else Bosch-faithful synthetic
│   └── generate_training_data.py  # [DONE] persist historical simulated sessions (offline train data)
├── src/
│   ├── line_sim.py            # [DONE] SimPy line model + health-driven 3-tier stations
│   ├── bottleneck_detect.py   # [DONE] active-period + queue-growth detection
│   ├── viz.py                  # [DONE] Plotly views of a sim run
│   ├── defect_model.py        # [DONE] regularised XGBoost + conformal-style confidence
│   ├── virtual_sensor.py      # [TODO] infer sensor-poor station state + confidence
│   ├── effective_trust.py     # [TODO] fusion + Risk×Trust action matrix
│   └── personas.py            # [TODO] supervisor / manager / leadership views
├── notebooks/
│   └── defect_model_training.ipynb   # [TODO] model-training narrative
└── app.py                     # [DONE] Streamlit dashboard (the demo)
```

## Status

| Component | File | State |
|---|---|---|
| 1. Line simulation | `src/line_sim.py` | ✅ working |
| 2. Bottleneck detection | `src/bottleneck_detect.py` | ✅ working (finds the true constraint) |
| 3. Defect model | `src/defect_model.py` + `src/train_defect_model.py` | ✅ retrained on `data/simulated/` — held-out **AUC 0.72** (train 0.88, no overfit), **top-20%-risk recall 54%** / top-10% 37% on 76 held-out defects; regularised, real signal |
| 3b. Feature engineering | `src/feature_engineering.py` | ✅ trend-aware, tier-complete, includes decoy channels as distractors (`model_features.csv`) |
| 4. Virtual sensor | `src/virtual_sensor.py` | ✅ working — method auto-selected from measured correlation, validated against baselines |
| 5. Effective Trust | `src/effective_trust.py` | ✅ input_trust × model_confidence → Risk×Trust action gate; auto-act flags **2.99× more precise** than human-verify on held-out data |
| 6. Persona views | `src/personas.py` | ✅ supervisor / manager / leadership — three lenses over one shared model state (bottleneck + risk + trust) |
| 7. Streamlit dashboard | `app.py` | ✅ `streamlit run app.py` — supervisor/manager/leadership tabs, live trust-policy slider, over the production model |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# run individual components
python src/line_sim.py                    # simulate the line, print station stats
python src/bottleneck_detect.py           # detect the bottleneck
python src/viz.py                         # render docs/line_sim_views.html
python data/get_data.py                   # show the (Bosch-style) defect dataset summary
python data/fetch_bosch.py                # (optional) fetch real Bosch data — see Data note

# launch the dashboard (the demo) — needs a trained model, so run the
# "Full pipeline" steps below once first
streamlit run app.py
```

### Full pipeline: regenerate data and train the defect model

`data/simulated/` and the trained model artifact are gitignored (regenerable,
not committed) — run these three steps in order to reproduce them from a
fresh clone:

```bash
# 1. generate the offline training data (this exact command reproduces the
#    "final calibrated dataset" the reported numbers are based on:
#    60 sessions x 100,000s, 50 train / 10 test, ~82,000 units, ~497 defects.
#    The 10 held-out sessions give ~76 test defects -- enough for a stable
#    held-out AUC, which a smaller 4-test-session run was NOT.)
python data/generate_training_data.py --sessions 60 --duration 100000 --seed 100 --test-sessions 10

# 2. build the trend-aware feature table (data/simulated/model_features.csv)
python src/feature_engineering.py

# 3. train + evaluate the defect model on a chronological holdout
python src/train_defect_model.py
```

Each session's seed is derived deterministically (`seed = --seed + session_index`,
see `generate_training_data.py`), so the same flags always regenerate the same
data — nothing here depends on committing the CSVs. Omitting the flags falls
back to the script's smaller defaults (5 sessions x 50,000s), which is fine for
a quick smoke test but not the dataset the reported model numbers are based on.

### Honesty notes on the model numbers

- **Held-out AUC ~0.72 is a real, stable signal**, measured on ~76 held-out
  defects (an earlier 4-test-session setup gave only ~32, where AUC swings
  ±0.05–0.07 and an optimistic ~0.58 draw turned out to be noise). The number
  moved *down* to ~0.55 under honest evaluation before the fixes below moved
  it genuinely up — it was never inflated on a slide.
- **Defect signal is a stated assumption.** Each defect is a *spontaneous*
  (precursor-less) or *health-driven* (has a sensor precursor) event; we
  calibrate the mix to ~70% health-driven (verified ~74% originate at degraded
  health) so prediction is a learnable task, not mostly-noise. Real lines have
  both; the ~30% spontaneous fraction is deliberately unpredictable and caps
  achievable recall. Overall rate stays ~0.6% (Bosch is 0.596%).
- **Decoy channels + feature selection.** The sim also emits irrelevant
  "decoy" sensor channels (noise + a useless random walk) on an isolated RNG
  stream, so a real historian's mix of useful/useless tags is modelled.
  `train_defect_model.py` reports permutation importance (which scores the
  decoys ~0) and an ablation: dropping the decoys and retraining *improves*
  held-out AUC (0.69 → 0.72) — the model is measurably better without noise.

### Simulated training data (`data/generate_training_data.py`)

`line_sim.py` now models a hidden per-station **health state** that drifts
down in rare, gradual episodes (never instantaneously) and drives three
things at once: sensor drift, cycle-time/failure-rate creep (→ a *forming*
bottleneck), and defect risk for units processed during the dip (→ a defect,
often only caught several stations later at an inspection point). Stations
carry one of three instrumentation tiers instead of a binary sensor flag:

| Tier | Meaning | How it's filled |
|---|---|---|
| **A** | Fully instrumented, dense readings | Ground truth |
| **B** | No sensor, but correlated with other stations sharing a real cause (tooling, calibration rig, material batch) — **not** assumed to be its line-neighbours | Regression / transfer learning across whichever stations share that cause |
| **C** | No sensor, ~zero loading on any shared cause, sparse manual checks | Kalman filter |

Correlation is deliberately *not* modelled as physical adjacency — two stations next to each other on the line may share nothing, while two stations far apart can share the same calibration rig. `line_sim.py` assigns each station a loading vector onto a small set of shared process factors; two stations correlate to the extent they load on the same factor(s), regardless of position. Verified: Door-Fit (body construction, early) and Torque-2 (final assembly, late) correlate at 0.46 through a shared torque-calibration factor, while adjacent stations with no shared cause stay near zero.

`generate_training_data.py` is the *offline* half of the pipeline: it runs
several independent simulated sessions back-to-back (default 5 × 50,000s),
and writes the result to `data/simulated/` (gitignored — regenerate with the
command above, same as the Bosch sample):

- `station_registry.csv` — static per-station config
- `health_log.csv` — **hidden** ground-truth health over time (never a model
  input; kept only to score detection lead time / false-alarm rate)
- `sensor_log.csv` — observable channel readings (dense for tier A, sparse
  for tier C, absent for tier B)
- `unit_features.csv` — per-unit modelling table: only the *observed*
  `S{station}_{channel}` columns (tier B columns don't exist at all; tier C
  columns are ~99.5% empty) — what `defect_model.py` should train on
- `unit_features_true.csv` — the *true* values behind every reading, for
  scoring virtual-sensor imputation accuracy only — never join this into
  model training
- `manifest.json` — generation parameters + a **chronological** train/test
  split (train on all sessions but the last, evaluate on the held-out last
  one) — never split randomly, since that would leak within a single health
  episode

Models train **offline** on this persisted dataset, then get applied to a
separate, freshly-started simulation for the live demo — never the same run
they were trained on.

## Data note (important, and honest)

The real **Bosch Production Line Performance** dataset (~14 GB, Kaggle) requires
a login and is not fetched automatically. `data/get_data.py` will:
1. use `data/bosch_numeric_sample.csv` if a real sample is present, else
2. generate a **Bosch-faithful synthetic sample** that reproduces the real
   dataset's structure and difficulty — station-grouped features
   (`L{line}_S{station}_F{feat}`), ~0.58% defect rate, and missing-not-at-random
   values (each part visits only some stations).

**Out of the box this repo runs on tier 2 (synthetic).** No Bosch data ships
here — it cannot be redistributed, and it is `.gitignore`d.

### Getting the real data

```bash
pip install kaggle
kaggle auth login                    # kaggle >= 2.2 uses OAuth, not kaggle.json
# then ACCEPT THE RULES in a browser — the API 403s until you do:
#   https://www.kaggle.com/c/bosch-production-line-performance/rules
python data/fetch_bosch.py           # ~100k-row sample -> bosch_numeric_sample.csv
```

`fetch_bosch.py` downloads only `train_numeric.csv` (~2 GB of the 14 GB), streams
it in chunks so it never needs the ~9 GB of RAM a full load would take, cuts a
row sample, and deletes the raw file. Rules acceptance is a human step and
cannot be scripted. If the API is a problem, download the file by hand to
`data/train_numeric.csv` and re-run — the script skips straight to sampling.

The sampler **preserves the natural ~0.58% defect rate** rather than balancing
the classes, since that imbalance is one of the properties we claim to model.
Use `--all-positives` to keep every defect instead, and say so if you report
numbers from it.

### Measured on the real 100k sample (verified, not quoted)

Every synthetic-tier constant in `get_data.py` is calibrated against these
measurements rather than guessed:

| Property | Real Bosch (`train_numeric.csv`) | Synthetic generator |
|---|---|---|
| Rows | 99,741 sampled of 1,183,747 | 40,000 |
| Numeric features | 968 | 968 ✅ |
| Stations | 50 | 50 ✅ |
| Defect rate | 0.596% | 0.580% ✅ |
| Missing values | 81.1% | 80.8% ✅ |
| Distinct station-visit routes | 2,737 | 2,914 ✅ |
| Stations visited per part | 12.1 of 50 | 10.9 of 50 ✅ |
| Max cross-station feature corr. | 0.74 | 0.72 ✅ |
| Station P(visit), max | 0.57 | 0.55 ✅ |

Three structural facts we measured on the real file and reproduced, rather than
assuming:

1. **Presence is all-or-nothing within a station** (mean recorded fraction =
   1.000 when visited, for every station). A station is fully read or fully
   absent; there is no per-feature dropout. All 81% missingness is *structural*,
   which is exactly why "which stations did this part skip" is informative.
2. **Lines are alternative paths, not sequential stages** — but line 0 is a
   shared entry that ~75% of parts pass through, which is how some stations
   reach P(visit) = 0.57, higher than any single line's share.
3. **Busy stations are feature-light.** That inverse relationship is why parts
   visit 24% of stations yet only 19% of feature values are present.

The cross-station correlation row is the one the differentiator depends on: real
Bosch stations genuinely predict each other (0.5–0.74 between co-observed
stations), which is what makes `virtual_sensor.py` possible. The generator
reproduces it with latent per-part process factors (material batch, ambient
drift, tool wear) whose station loadings drift smoothly along the line, so
neighbours correlate and distant stations do not.

Two notes on figures quoted elsewhere in our docs: the **~81% missingness is
correct for the numeric file specifically** (measured 81.1%), but the **~4,264
feature count spans all three files** — numeric alone has 968. Say "across the
numeric, date and categorical files" when citing the larger number.

Both tiers return an identically-shaped `meta` (including `station_of_col`,
parsed from the column names for real data), so swapping in the genuine CSV
needs no code changes. On real Bosch, expect MCC in the ~0.2–0.4 range (a
famously hard metric at this imbalance); AUC is the clearer signal that the
model separates defects from good parts.

## Design decisions (defensible in the pitch)

- **Bottleneck** uses the active-period method plus a downstream-starvation
  weighting, so it finds the *true* constraint rather than the front-of-line
  station that always looks busy.
- **Confidence is not raw `predict_proba`** (gradient-boosted probabilities are
  miscalibrated at extreme imbalance). We use a split-conformal-style score:
  distance from the decision boundary, ranked against a held-out calibration set.
- **Effective Trust multiplies** input-trust and model-confidence rather than
  averaging — if either is weak, trust drops. Averaging would hide bad data.
- We **model** station flow, defect propagation, and confidence-tagged state; we
  deliberately **skip** part geometry, tool-wear physics, and PLC/OT integration
  (out of scope for a proof-of-concept).
