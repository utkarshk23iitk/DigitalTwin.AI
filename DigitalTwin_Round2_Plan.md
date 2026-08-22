# DigitalTwin.ai — Round 2 Master Plan

## 1. What must be delivered (from the brief)
All submitted via a **public GitHub repo**:
1. **Detailed Business Proposal** — problem framing, solution design, target users, business case & impact, phased roadmap, key risks + mitigations.
2. **Working Prototype** — functional demo of the core predictive mechanism on realistic/simulated data. Explicitly *not* production-grade; proof-of-concept encouraged.
3. **Pitch Presentation** — presents proposal + prototype.
Plus submission mechanics: **repo + prototype demo video + README** (implementation approach, architecture, dependencies, execution instructions).

## 2. How our existing design maps to the Round 2 asks
| Round 2 asks for… | We already have… | Status |
|---|---|---|
| Works with uneven sensor coverage | Virtual sensors + Effective Trust | Designed ✓ |
| Stays useful at sensor-poor stations | Kalman / transfer-learning inference | Designed ✓ |
| Validate predictions before trusting them | Confidence scoring + Effective Trust gating | Designed ✓ |
| Bottleneck + defect prediction | SimPy sim + active-period; XGBoost on Bosch | Designed ✓ |
| **Different stakeholder views** (supervisor / manager / leadership) | — | **NEW — must add** |
| Handling data gaps / low-cost sensing | Virtual sensors (+ propose cheap sensing) | Partial |
| Scalability & ROI across lines/plants | Roadmap + business case | To write |

**Verdict:** ~70% of Round 2 is building what we already specified. The one genuinely new build item is the **multi-persona view**.

## 3. Prototype scope — what the demo will actually show
A single coherent story on **simulated + Bosch data**, end to end:

**Core loop (must-have):**
1. **Simulated assembly line** (SimPy) — ~15–30 stations, cycle times, buffers, one or more stations flagged "no sensor."
2. **Bottleneck detection** — active-period method + queue-growth on the live sim → identifies/forecasts the constraint station.
3. **Defect prediction** — gradient-boosted model trained on the **Bosch Production Line Performance** dataset → per-part failure risk.
4. **Virtual sensor** — for a deliberately sensor-less station, infer its state from surrounding data; output value **+ a confidence**.
5. **Effective Trust** — fuse input-trust × model-confidence; **gate the action** (auto-act / monitor / human-verify) via the Risk × Trust matrix.
6. **Dashboard** — shows the line, live bottleneck flags, defect risks, and — crucially — the **confidence/trust on every prediction**, with the 3 persona views.

**Nice-to-have (if time):** feedback loop (override logging), what-if slider (slow a station → see throughput impact), drift note.

**Deliberately skipped (state in README):** real PLC/OT integration, part geometry, tool-wear physics, production-grade scale.

## 4. Recommended tech stack (my advice)
- **Language:** Python (matches the whole ecosystem + team strength).
- **Simulation:** **SimPy** (free, lightweight, Python-native DES).
- **Defect ML:** **XGBoost** (or LightGBM) + scikit-learn, on Bosch data; **MAPIE** for conformal prediction / calibrated confidence.
- **Virtual sensor:** start with a regressor (scikit-learn) for spatial inference; **FilterPy** for a Kalman variant if we do the temporal branch.
- **Dashboard:** **Streamlit** — the right call. It's pure Python (no JS), fast to build, looks clean, runs locally with one command, and screen-records beautifully for the demo video. A supervisor/manager/leadership tab set is trivial in Streamlit.
- **Repo hygiene:** `requirements.txt`, clear `README.md`, `data/` (with a script to fetch Bosch or a small sample), `src/` modules, `app.py` for the dashboard, a `notebooks/` folder for the model-training story.

**Why Streamlit over a notebook:** the brief wants a *working prototype* people can run and *see*, and a demo video. A dashboard demos far better than a notebook, and Streamlit is the lowest-effort way to get one. Notebooks are great for the *model-training* narrative (we'll include one), but the headline demo should be the app.

## 5. Proposed repo structure
```
digitaltwin-ai/
├── README.md                  # architecture, deps, run instructions
├── requirements.txt
├── data/
│   └── get_data.py            # fetch/sample Bosch; generate sim config
├── src/
│   ├── line_sim.py            # SimPy assembly-line model
│   ├── bottleneck.py          # active-period + queue-growth detection
│   ├── defect_model.py        # XGBoost train/predict on Bosch
│   ├── virtual_sensor.py      # inference for sensor-poor stations + confidence
│   ├── effective_trust.py     # fusion + Risk×Trust action matrix
│   └── personas.py            # supervisor / manager / leadership views
├── notebooks/
│   └── defect_model_training.ipynb
├── app.py                     # Streamlit dashboard (the demo)
└── docs/
    ├── business_proposal.md   # or PDF
    └── architecture.md
```

## 6. Build sequence (phased)
- **Phase A — Prototype (highest risk, do first)**
  1. SimPy line sim + basic dashboard skeleton
  2. Bottleneck detection wired to the sim
  3. Bosch defect model (train offline, load in app)
  4. Virtual sensor for a sensorless station (+ confidence)
  5. Effective Trust fusion + action gating
  6. Persona views + polish
- **Phase B — README + repo** (architecture, deps, run steps)
- **Phase C — Business Proposal** (problem → solution → users → business case → roadmap → risks)
- **Phase D — Pitch deck + demo video** (record the running app)

## 7. Open decisions
- Real Bosch data (14GB — we'll use a **sample**) vs. fully synthetic defect data? (Recommend: small Bosch sample for credibility + synthetic for the live sim.)
- How many stations in the sim (15 keeps it readable; 30–50 matches the brief's scale — we can label 30 but visualize a readable subset).
- Assumed jurisdiction / ROI figures for the business case (state assumptions clearly).
```