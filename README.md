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

## Repo layout

```
digitaltwin-ai/
├── README.md
├── requirements.txt
├── data/
│   ├── fetch_bosch.py         # [DONE] download real Bosch + cut a sampled CSV
│   └── get_data.py            # real-Bosch-if-available, else Bosch-faithful synthetic
├── src/
│   ├── line_sim.py            # [DONE] SimPy assembly-line model + state tracking
│   ├── bottleneck_detect.py   # [DONE] active-period + queue-growth detection
│   ├── defect_model.py        # [DONE] XGBoost + conformal-style confidence
│   ├── virtual_sensor.py      # [TODO] infer sensor-poor station state + confidence
│   ├── effective_trust.py     # [TODO] fusion + Risk×Trust action matrix
│   └── personas.py            # [TODO] supervisor / manager / leadership views
├── notebooks/
│   └── defect_model_training.ipynb   # [TODO] model-training narrative
└── app.py                     # [TODO] Streamlit dashboard (the demo)
```

## Status

| Component | File | State |
|---|---|---|
| 1. Line simulation | `src/line_sim.py` | ✅ working |
| 2. Bottleneck detection | `src/bottleneck_detect.py` | ✅ working (finds the true constraint) |
| 3. Defect model | `src/defect_model.py` | ✅ working — metrics pending re-run on the real sample |
| 4. Virtual sensor | `src/virtual_sensor.py` | ⏳ next |
| 5. Effective Trust | `src/effective_trust.py` | ⏳ next |
| 6. Persona views | `src/personas.py` | ⏳ |
| 7. Streamlit dashboard | `app.py` | ⏳ |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# run individual components
python src/line_sim.py            # simulate the line, print station stats
python src/bottleneck_detect.py   # detect the bottleneck
python data/get_data.py           # show the defect dataset summary
python data/fetch_bosch.py        # (optional) fetch real Bosch data — see Data note
python src/defect_model.py        # train + evaluate the defect model

# (later) launch the dashboard
# streamlit run app.py
```

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
