# Twinly — a Digital Twin That Knows Its Limits

**Catching bottlenecks and defects on a mixed-sensor assembly line before they propagate —
with every prediction tagged by how much to trust it.**

> Accenture Innovation Challenge 2026 — Round 2, Problem Track 4 (DigitalTwin.ai) — Team Twinly

> **In plain terms:**
> - A stalled line costs ~$600/second and a missed defect can trigger a $10M+ recall — both
>   are invisible until it's too late, because real lines are unevenly instrumented.
> - Twinly runs two engines: a simulation engine that predicts where the next bottleneck
>   will form, and an ML engine that flags parts likely to fail downstream.
> - Where a station has no sensor, we don't guess blindly — we infer its behavior from
>   correlated stations or its own history, and tag every inferred value with a confidence score.
> - We fuse input trust and model confidence — two things most systems conflate — into one
>   Effective Trust score that decides whether a flag auto-acts or gets routed to a human.
> - Every claim below is validated against a real held-out test set the model never trained
>   on — not just a static slide's word for it.

---

## 1. Problem Statement

A vehicle assembly line is a chain of stations, each performing one operation. Two failures
recur, and both spread before anyone reacts.

**Bottlenecks.** When one station slows or stalls, every station behind it waits. A stopped
automotive line costs roughly **$2.3M/hour — about $600 every second** (Siemens, 2024).

**Silent defects.** When a station introduces a fault that goes uncaught, the same mistake
repeats across dozens of vehicles before quality control finds it. A defect caught in the
field costs far more than one caught at the station — a single recall can exceed **$10M**.

Today, plant teams discover both problems too late — after the queue has formed or the
defective cars have shipped. The deeper issue is visibility: a line is unevenly instrumented,
so teams never have a complete, trustworthy picture. They don't just find out late — they
don't even know where they are blind.

We prototype this on a **12-station simulated line** (Frame-Weld → Final-QC) as a readable
stand-in for the brief's 30–50-station scale, with a deliberately uneven instrumentation mix
(see [§5](#5-data--assumptions)).

**The challenge**: see bottlenecks forming and predict defects before they propagate,
honestly, under partial observability — without a system pretending to know what it cannot.

---

## 2. Solution Approach

Our edge is simple: **where we don't know something, we say so — and act accordingly.**

### 2.1 Handling stations with no sensors

Not every station on a real line has sensors. Instead of guessing blindly, every station is
assigned one of **three instrumentation tiers** — this is the single most distinctive design
choice in the build:

| Tier | Meaning | How the twin fills the gap |
|---|---|---|
| **A — instrumented** | Dense torque/vibration/temperature readings every cycle | Used directly, as ground truth |
| **B — sensor-poor, correlated** | No sensor, but shares a real physical cause (tooling, calibration rig, material batch) with another station — not assumed to be its line-*neighbour* | **Spatial regression** across whichever station(s) actually share that cause |
| **C — sensor-poor, isolated** | No sensor, ~zero shared-cause loading, only sparse manual-check timestamps | **Kalman filter** over the station's own sparse history |

The key idea: correlation is based on **shared physical cause, not line position**. Two
stations next to each other on the line can share nothing, while two stations far apart can
share a calibration rig. Verified on simulated data: **Door-Fit** (early, body construction)
and **Torque-2** (late, final assembly) correlate at **0.46** through a shared
torque-calibration factor, while line-adjacent stations with no shared cause stay near zero.

We didn't hand-assign these tiers and hope for the best, either — `src/virtual_sensor.py`
**auto-selects** spatial regression vs. Kalman per station from the *measured* correlation on
training data. That auto-detection independently reproduced our hand-designed tiers — itself
a check that the design matches what the data actually supports.

Every inferred reading carries its own confidence score (`*_est` value + `*_conf`), which
feeds directly into Effective Trust below — the twin never treats a guess as equal to a
measurement.

### 2.2 Two engines, and why each was chosen

- **Bottleneck detection** uses the **active-period method** (Roser & Nakano) plus a
  downstream-starvation weighting, so it finds the station that's the *true* constraint —
  not whichever front-of-line station always looks busy. We chose this over a pure ML
  forecaster because it's a well-established, interpretable technique that needs no training
  data to be trustworthy on day one — exactly what a floor supervisor needs to believe on
  sight, not a black box.
- **Defect prediction** uses a regularised **XGBoost** classifier over trend-aware engineered
  features, with hyperparameters and decision threshold Optuna-tuned. We chose gradient
  boosting because it handles the Bosch-style structural missingness (whole blocks of columns
  absent depending on a station's tier) natively through its own split-finding — no separate
  imputation step needed.

Both engines are trained **offline** on historical simulated sessions and applied **live** to
a fresh run for the demo — never trained on the run being predicted. This is the standard
digital-twin train/deploy separation, and the only honest way to report a detection metric
without leaking the answer into training.

### 2.3 Catching defects before they're caught

Defects often aren't caught where they start — they're caught several stations later, once
they've already repeated across other vehicles. The simulator tracks two separate timestamps
per defective unit: **`defect_occurred_at`** (where the fault actually started) vs.
**`defect_caught_at`** (where quality control actually found it). On a verified test run,
**73% of defects were caught at a later station than where they occurred** — the defect
genuinely surfaces late, exactly the brief's complexity.

We measure this directly rather than assert it: the model is re-scored at each station
cutoff, finding the earliest point predicted risk crosses threshold, then compared to where
the defect was actually caught. This is reported as a **directional estimate**, not a
certified lead-time-in-minutes figure — stated honestly rather than inflated.

The same approach applies to bottlenecks, checked against the hidden ground-truth health
process that's never fed to any model: replaying all held-out sessions gave a **median lead
time of 680s** ahead of the bottleneck fully forming, at a **27.9% false-alarm rate** — a
real number that also honestly surfaces where we're still tuning, not a hidden weakness.

### 2.4 One score that decides who acts

Most systems mix up two different questions: *is the data any good?* and *is the model
sure?* We keep them separate and multiply them together into one **Effective Trust** score
(`input_trust × model_confidence`) — so a confident prediction built on shaky, inferred data
still gets flagged for a human, instead of auto-acting on a guess. This score drives the
Risk × Trust action matrix: **AUTO-ACT, HUMAN-VERIFY, MONITOR, or PASS**.

### 2.5 Proof it actually works

Every number below comes from a held-out set the model never trained on — nothing here is
asserted without a check behind it.

- **Defect model**: chronological held-out **ROC AUC 0.7286** (recall 0.3333, precision 0.0294,
  MCC 0.0815) on 13,678 held-out rows containing 84 positive examples — a rare-event ranking
  metric, not raw accuracy, since a model predicting every unit as non-defective would look
  accurate at this class imbalance while catching nothing.
- **Bottleneck detection**: median **680s lead time** ahead of the bottleneck forming, at a
  **27.9% false-alarm rate** we're still tuning down.
- **Effective Trust actually separates good flags from bad ones**: AUTO-ACT flags are
  **1.86× as precise** as HUMAN-VERIFY flags among held-out high-risk units.

<details>
<summary>Full validation detail (thresholds, baselines, feature checks)</summary>

- Recall at other risk thresholds, from the larger 60-session / 76-held-out-defect evaluation
  configuration described in [§5](#5-data--assumptions): top-10% 36.8%, top-5% 22.4%, top-1%
  11.8%.
- Feature selection is validated, not assumed: irrelevant "decoy" sensor channels score ~0 on
  permutation importance, and dropping them improves held-out AUC 0.69 → 0.72 pre-tuning (same
  larger evaluation configuration).
- Virtual sensor accuracy is checked against fair baselines, not itself: Kalman beats naive
  forward-fill by 9–20%; spatial regression beats a naive mean-guess by 29–44% overall, and
  by 54–66% specifically during drift episodes — the case that matters most for prediction.
- AUTO-ACT vs. HUMAN-VERIFY defect rates: 3.08% vs. 1.66%.
- Small-sample AUC variance and the ~30% of defects that are spontaneous and unlearnable by
  design are known caveats, documented in full in [docs/SUBMISSION_REPORT.md](docs/SUBMISSION_REPORT.md).

</details>

We model station flow, defect propagation, and confidence-tagged state — and deliberately
skip part geometry and tool-wear physics, since that's a different problem. The result: a
twin that knows its limits.

---

## 3. Architecture

```
OFFLINE TRAINING

SimPy historical sessions
        |
        +--> observed sensor history ------+
        |                                  |
        +--> hidden truth for validation   +--> virtual-sensor selection
        |                                       | spatial regression
        +--> unit outcomes                       | Kalman filtering
                                                v
                                      trend + imputation features
                                                |
                                                v
                                      XGBoost defect model
                                                |
                              Optuna parameters + model metadata
                                                |
                                                v
                                      local artifacts/ directory

LIVE DEMO / PRODUCTION SHAPE

Fresh demo shift (PLC/SCADA/MES in production)
        |
        +--> station states + buffers --> bottleneck intelligence
        +--> sensors + virtual fills --> feature pipeline
        +--> unit evidence ------------> defect risk + confidence
                                                |
                     input trust x model confidence = Effective Trust
                                                |
                   AUTO-ACT / HUMAN-VERIFY / MONITOR / PASS
                                                |
                                                v
                                      app2.py dashboard
                                                |
                                optional grounded LLM explanation
```

> **Simulated-PLC/OT note:** SimPy stands in for a real plant feed here; every downstream
> module (virtual sensor, defect model, bottleneck detection, trust layer) consumes the same
> station-data schema regardless of origin, so swapping in real telemetry needs no model
> changes — the simulator is a data-generation choice, not an architectural dependency. Real
> PLC/OT wiring itself is explicitly out of scope for this proof-of-concept (see [§5](#5-data--assumptions)).

---

## 4. Key Features

- **Three-tier sensor-poor handling** (not a binary flag) — spatial regression for correlated
  blind stations, Kalman filtering for isolated ones, each auto-selected from measured
  correlation and validated against fair baselines (§2.5).
- **Causally-linked bottleneck + defect signals** — a hidden per-station health state drives
  sensor drift, cycle-time/failure-rate creep, *and* defect risk together, so predictions are
  trend-based on a real shared cause rather than independent threshold trips on decorative
  columns.
- **Late-surfacing defect tracing** — `defect_occurred_at` vs. `defect_caught_at` gives a
  genuine (station-based) detection lead-time metric instead of an assumed one.
- **Effective Trust gates every action** — `input_trust × model_confidence` (multiplied, not
  averaged, so one weak factor cannot be masked by the other), driving a Risk × Trust action
  matrix that routes high-risk-but-low-trust flags to a human instead of auto-acting on
  shaky data.
- **One model, three stakeholder lenses** — Supervisor / Manager / Leadership views are three
  projections of the *same* per-unit assessment, so numbers never diverge across the
  personas — a deliberate design requirement, not three separate tools glued together.
- **Simulated PLC/OT integration, without disrupting a real line** — the entire pipeline
  consumes a source-agnostic station-data schema; SimPy is a swappable stand-in, not a hard
  dependency.
- **False-alarm / trust calibration made visible, not hidden** — bottleneck false-alarm rate
  (27.9%) and the trust-gating precision lift (1.86×) are both reported as measured numbers,
  including where the current calibration is still loose.
- **Optional AI Copilot, narration only** — a grounded explanation layer over the already-computed
  state (never a source of new numbers), with a deterministic local fallback if no API key is
  configured — see [§10](#10-optional-ai-copilot--narration-not-computation).

---

## 5. Data & Assumptions

**Everything here is synthetic and explicitly stated as such — no real plant data was used or
claimed.**

- **Simulated line**: a SimPy discrete-event model of **12 named stations** (Frame-Weld,
  Body-Weld, Door-Fit, Seal, Paint-Inspect, Primer, Topcoat, Cure, Torque-1, Torque-2,
  Electrical, Final-QC) as a **readable representative subset** of the brief's 30–50-station
  scale — cycle times, buffer capacities, failure/repair rates, and factor loadings are
  **assumed, stated parameters**, not measured from a real line.
- **Instrumentation mix in the demo line is deliberately uneven, but on the light side of
  realistic**: 10 of 12 stations are Tier A, 1 is Tier B (Torque-2), 1 is Tier C
  (Paint-Inspect). A real brownfield line typically has a *larger* fraction of poorly
  instrumented legacy stations than this demo's ~17% — scaling that ratio up is a stated
  roadmap item (§9), not a hidden gap.
- **The Tier B and Tier C stations were deliberately chosen as a stress test, not picked for
  convenience.** Torque-2 (Tier B) sits far from any other station on the line but was
  intentionally given a shared calibration-factor loading with an early station (Door-Fit)
  — a case designed specifically to prove correlation isn't line-adjacency. Paint-Inspect
  (Tier C) was intentionally given ~zero shared-cause loading with anything else on the
  line, so it's a genuine test of the isolated-station case (Kalman-only, no correlated
  neighbour to lean on). Both are worst-case placements by design, not the easiest cases we
  could have picked.
 - **Defect realism is a stated assumption, not a hidden knob.** Each defect is either
  *health-driven* — following a slow drift in the station's condition (rising vibration,
  temperature creep, etc.) that shows up in the data before the defect occurs, which is
  exactly what our trend-aware features are built to catch ahead of time — or *spontaneous*,
  with no such lead-up: sudden and uncorrelated with any measurable precursor, so no model
  could reasonably be expected to predict it. We calibrate this split to **~74% health-driven
  / ~26% spontaneous** (overall defect rate ~0.61%), which makes the prediction task
  learnable while stating an honest ceiling on how much recall is even possible — not a
  hidden shortcoming.
- **Bosch dataset used as a structural reference, not as training data.** We used the public
  Bosch Production Line Performance dataset to understand what real station-level production
  data actually looks like — column structure, per-station groupings, and the kind of sparse,
  block-missing patterns a real historian produces. We took a subset of columns from Bosch as
  the schema basis for our simulator's station-data tables, so our synthetic data has a
  realistic shape. The simulator itself, not Bosch, generates every value used in training
  and the live pipeline — Bosch rows are never trained on directly, only used to inform the
  structure our simulated data follows.
- **Offline training dataset**: the final calibrated run is **60 sessions × 100,000 simulated
  seconds (50 train / 10 held-out test)** → 82,047 units, 497 defects (0.61%), ~76 held-out
  defects. Train/test split is **chronological**, never random, so no health episode leaks
  across the split.
- **Deliberately out of scope, stated up front**: real PLC/OT wiring, part geometry,
  tool-wear physics, upstream supply chain, and production-grade scale — this is a
  proof-of-concept of the causal drift → bottleneck/defect → trust-gated action mechanism,
  not a production system.

---
## 6. Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.13 |
| Line simulation | **SimPy** (discrete-event) |
| Defect ML | **XGBoost**, tuned with **Optuna**; scikit-learn for splits/metrics |
| Virtual sensor | scikit-learn (spatial regression) + **FilterPy** (Kalman filter) |
| Dashboard | **Streamlit** + **Plotly** |
| Data | pandas / numpy; optional real-data fetch via `kaggle` |

Full pinned list in [requirements.txt](requirements.txt).

---

## 7. Setup / How to Run

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Full pipeline (regenerate data, train the model, launch the dashboard)

```bash
# 1. generate the offline training data
python data/generate_training_data.py --sessions 60 --duration 100000 --seed 100 --test-sessions 10

# 2. build the trend-aware feature table
python src/feature_engineering.py

# 3. train + evaluate the defect model on a chronological holdout
python src/train_defect_model.py

# 4. launch the dashboard (the demo)
streamlit run app2.py
```

Then open the URL Streamlit prints (default `http://localhost:8501`) and step through the
dashboard's sections: **Live Digital Twin, Defect Intelligence, Bottleneck Intelligence,
Station Health, AI Copilot, Model Analytics, Operational Perspectives (Supervisor / Manager /
Leadership), and Business & Scale.**

### Individual components (no training required)

```bash
python src/line_sim.py                    # simulate the line, print station stats
python src/bottleneck_detect.py           # detect the bottleneck
python src/effective_trust.py             # trust-gating validation
python src/personas.py                    # print all three persona views
python src/lead_time_eval.py              # detection lead-time numbers
```

---

## 8. Demo

- Live dashboard: `streamlit run app2.py` (see [§7](#7-setup--how-to-run)) — walk through
  **Live Digital Twin** first (current state, right now), then **Defect Intelligence** and
  **Bottleneck Intelligence** (the two prediction engines), **Station Health** (per-station
  detail), **AI Copilot** (grounded narration over the same computed state), **Model
  Analytics** (the held-out validation numbers in §2.5), and **Operational Perspectives** /
  **Business & Scale** for the Supervisor/Manager/Leadership and ROI framing.
- Demo video: *link to be added before final submission.*
- Screenshots: *to be added before final submission.*

---

## 9. Known Limitations & Future Improvements

- **Real telemetry integration** — swap the SimPy source for an actual PLC/SCADA/historian
  feed; the schema is already source-agnostic (§3), so this is an ingestion change, not a
  model change.
- **Heavier, more realistic sensor-poor mix** — scale from the demo's 12-station, ~17%-poor
  line toward the brief's full 30–50-station scale with a larger Tier B/C fraction.
- **Tighten the bottleneck false-alarm rate** — the current threshold is measurably
  trigger-happy (27.9%) — worth tuning before this becomes a headline claim.
- **Multi-site generalization** — validate the health-drift mechanism's calibration
  transfers across a second, differently-configured line.
- **Real data would raise recall further** — our synthetic generator adds randomness on top
  of the causal signal, which also caps how sharp our station correlations can get; real
  sensor data should lift recall past what synthetic data alone allows.
- **Action recommendations, not just risk flags** — the twin currently flags *that* something
  is at risk, not *what to do about it*; that needs real domain knowledge of each station's
  failure history, which synthetic data can't provide. With real maintenance logs, a
  retrieval-based (RAG) layer could surface a recommended action alongside each flag.
- **More rigorous production validation** — once real telemetry is available, replace the
  synthetic held-out evaluation with genuine production-outcome validation (real caught-vs-missed
  defects, not simulated ground truth).
---

## 10. Optional AI Copilot — narration, not computation

A narration layer sits **on top of** the fully-computed dashboard state — it never predicts,
scores, or decides anything. Every number (risk, Effective Trust, bottleneck, station health,
action) is computed exactly as described in [§2](#2-solution-approach) and [§3](#3-architecture),
entirely locally, before the Copilot is ever involved.

```
Dashboard state (already computed locally — nothing here changes)
      │
      ▼
Compact structured snapshot (risk scores, actions, bottleneck, trust,
station health — numbers only, nothing the Copilot invents)
      │
      ▼
Google Gemini API
      │
      ▼
Grounded narration + interactive Q&A, shown in the dashboard's own
AI Copilot section
```

**What it does:**
- Generates a plain-language shift summary from the current computed state on demand.
- Answers free-form questions about what's on screen ("why is this station flagged?", "what
  should I do about unit 42?"), grounded strictly in the same structured snapshot — never raw
  CSVs, never a source of new facts.
- A toggle in the dashboard shows the exact structured context sent to the model, so nothing
  about what it saw is hidden.

**What it never does:** invent a number, override a risk score or action, or calculate any
plant metric itself. If no API key is configured, or the API call fails for any reason, the
Copilot automatically falls back to a **grounded local mode** — deterministic, non-LLM answers
built from the same structured snapshot — and every other part of the dashboard (simulations,
models, charts, alerts) keeps working exactly as before. Narration is an add-on, never a
dependency the rest of the system relies on.

**Setup** (optional — the dashboard runs fully without it):

```bash
export GEMINI_API_KEY="your_rotated_server_side_key"
export GEMINI_MODEL="gemini-2.5-flash"
streamlit run app2.py
```

Never commit an API key — `.env`, Streamlit secrets, and key files are gitignored; see
`.env.example` for the placeholder format.

---
## 11. Team — Twinly

| Name | College | Stream | Year of Graduation |
|---|---|---|---|
| Shreyansh Dewangan  | IIT Kanpur | Earth Sciences | 2027 |
| Karan Pratap Lohiya | IIT Kanpur | Statistics and Data Science | 2027 |
| Utkarsh Kesharwani | IIT Kanpur | Statistics and Data Science | 2027 |
