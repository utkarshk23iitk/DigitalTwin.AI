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
| 3. Defect model | `src/defect_model.py` | ✅ working (AUC ≈ 0.89 on sample) |
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
python src/defect_model.py        # train + evaluate the defect model

# (later) launch the dashboard
# streamlit run app.py
```

## Data note (important, and honest)

The real **Bosch Production Line Performance** dataset (~14 GB, Kaggle) requires
a login and is not fetched automatically. `data/get_data.py` will:
1. use `data/bosch_numeric_sample.csv` if you place a real sample there, else
2. generate a **Bosch-faithful synthetic sample** that reproduces the real
   dataset's structure and difficulty — station-grouped features
   (`L{line}_S{station}_F{feat}`), ~0.58% defect rate, and missing-not-at-random
   values (each part visits only some stations).

The modelling code is written against the real data's schema, so swapping in the
genuine Bosch CSV requires no code changes. On real Bosch, expect MCC in the
~0.2–0.4 range (a famously hard metric at this imbalance); AUC is the clearer
signal that the model separates defects from good parts.

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
