# Twinly Submission Report

**Project:** DigitalTwin.AI / Twinly

**Primary application:** `app2.py`

**Prototype type:** Trust-aware production digital shadow with a path to a live digital twin

**Report purpose:** Technical handoff, judging guide, reproducibility guide, and dashboard manual

## 1. Executive summary

Twinly demonstrates how a production-line digital twin can remain useful when
instrumentation is incomplete. It does not merely show a simulated line. It
combines four operational decisions in one time-aware interface:

1. What is happening at every station now?
2. Which station is the current constraint, and which constraint is forming?
3. Which units have elevated defect risk before final inspection?
4. Is the evidence trustworthy enough to automate an action?

The core differentiator is the last question. Twinly tracks the reliability of
the input data separately from the confidence of the predictive model:

```text
Effective Trust = Input Trust x Model Confidence
```

Multiplication is intentionally conservative. A confident model using weak
virtual-sensor evidence, or a hesitant model using perfect sensor data, should
not be treated as highly trustworthy. The resulting Risk x Trust matrix maps
each unit to one of four actions:

| Defect risk | Effective Trust | Action | Meaning |
|---|---:|---|---|
| High | High | `AUTO-ACT` | Automatically hold or reject, subject to plant policy |
| High | Low | `HUMAN-VERIFY` | Escalate the warning but require human verification |
| Low | Low | `MONITOR` | Do not reject; continue observing weak evidence |
| Low | High | `PASS` | Evidence supports normal progression |

This is a prototype. The data source is a discrete-event simulation and the
dashboard replays a generated shift. In production, PLC/SCADA/historian/MES
events would replace the generated files while retaining the same downstream
schemas, feature pipeline, trust logic, and dashboard concepts.

## 2. Submission boundary

The Git submission contains source code, reproducibility instructions, a compact
demo replay, and the final model required to launch the dashboard immediately.
It does not publish the large historical corpus, tuning databases, or secrets.

### Included in Git

- `app2.py`, the final Streamlit dashboard.
- Simulation, feature engineering, training, trust, and visualization source.
- Data-generation and demo-inference scripts.
- `requirements.txt` and `.env.example`.
- The compact `data/demo_live/` runtime replay and final model/metadata.
- This report and the landing `README.md`.
- Empty `.gitkeep` placeholders for generated directories.
- `scripts/verify_submission.py` for pre-push hygiene checks.

### Kept locally and ignored

- `data/simulated/`: historical sessions and chronological holdout data.
- `data/demo_live/model_features.csv`: rebuild-only intermediate, not needed by the app.
- Historical or experimental model artifacts other than the final runtime model.
- `artifacts/tuning/`: Optuna SQLite studies, best parameters, and checkpoints.
- Generated HTML visualizations, caches, environments, logs, and credentials.

The final verification script enforces this boundary and checks that every file
needed for a clean-clone dashboard launch is tracked.

## 3. End-to-end architecture

```text
                          OFFLINE / HISTORICAL

       +-------------------------------------------------------+
       | src/line_sim.py                                       |
       | stations + finite buffers + unit flow + failures      |
       | hidden gradual health + sensors + delayed defects     |
       +----------------------------+--------------------------+
                                    |
                     data/generate_training_data.py
                                    |
          +-------------------------+--------------------------+
          |                         |                          |
          v                         v                          v
 observed sensor data      hidden validation truth       unit outcomes
          |                         |                          |
          +-------------+-----------+                          |
                        v                                      |
             src/virtual_sensor.py                             |
          spatial regression / Kalman                          |
                        |                                      |
                        v                                      |
           src/feature_engineering.py <------------------------+
      as-of rolling mean/std/slope + fills + confidence
                        |
                        v
              src/defect_model.py
         XGBoost risk + calibrated confidence
                        |
                        v
             src/effective_trust.py
         input trust x confidence -> action
                        |
                        v
       artifacts/ weights, metadata, Optuna state

                            LIVE-DEMO SHAPE

  data/generate_demo_data.py              Production replacement
  unseen simulated shift                  PLC / SCADA / historian / MES
             |                                      |
             +------------------+-------------------+
                                v
                 data/build_demo_inference.py
        virtual fills + features + unit assessments + timestamps
                                |
                                v
                       data/demo_live/
                                |
                                v
                           app2.py
      current-time filtering -> KPIs -> charts -> alerts -> actions
                                |
                                v
                 optional grounded LLM explanation
```

## 4. Simulation and data model

### 4.1 Assembly-line mechanics

`src/line_sim.py` uses SimPy to represent 12 serial stations connected by
finite buffers. A unit must leave one station and enter the next buffer before
production continues. Each station can be:

- `WORKING`: processing a unit.
- `BLOCKED`: finished but unable to release into a full downstream buffer.
- `STARVED`: waiting because no upstream work is available.
- `DOWN`: unavailable because of a simulated breakdown.

The simulation logs state transitions, buffer levels, station health, sensor
readings, unit visits, and eventual defect outcomes. Random seeds make a given
configuration reproducible.

### 4.2 Health mechanism

The simulation contains a hidden station condition `H(t)` between 0 and 1.
It drives the simulation and is retained only for validation. Live dashboard
decisions use an observable condition proxy derived from as-of sensor deviation,
cycle drift, downtime, and current operational state.

Health changes through gradual degradation episodes:

```text
normal -> ramp down -> degraded hold -> ramp up -> normal
```

Shared process factors can affect non-adjacent stations that use the same
tooling, calibration source, or material condition. A station's health is:

```text
clip(idiosyncratic_episode_health - shared_factor_penalty, 0.02, 1.00)
```

Lower health increases cycle time, failure probability, sensor drift, and the
health-driven component of defect probability. This shared mechanism creates
a coherent precursor instead of generating unrelated random alerts.

Important interpretation: `health_true` is available because this is a
simulation. It is hidden from model training and operational scoring and appears
only in the explicitly labelled validation chart. A production deployment would
calibrate the observable proxy against maintenance outcomes or replace it with a
dedicated condition model.

Dashboard state labels use the observable condition estimate at the playback
clock:

- Running and health >= 0.90: `RUNNING`.
- Running and 0.82 <= health < 0.90: `WARNING`.
- Running and health < 0.82: `DEGRADED`.
- Simulated `DOWN`: `FAULT`.
- Blocking/starvation retain their operational state labels.

### 4.3 Instrumentation tiers

| Tier | Availability | Inference behavior |
|---|---|---|
| A | Dense physical measurements | Use measured temperature, torque, and vibration |
| B | No local sensor but correlated process peers exist | Fit spatial regressions from measured correlated stations |
| C | Sparse local manual checks and little cross-station correlation | Track state through a temporal Kalman filter |

The inference method is selected from training evidence rather than blindly
trusting the tier label. If neither correlation nor local history is adequate,
the station is marked unrecoverable instead of receiving an invented value.

### 4.4 Files generated for historical training

| File | Role | Model input? |
|---|---|---:|
| `station_registry.csv` | Static names, tiers, cycle assumptions, inspection flags | Configuration |
| `channel_registry.csv` | Marks real and deliberately irrelevant decoy channels | Feature selection |
| `health_log.csv` | Hidden simulated health truth | No; validation/demo only |
| `sensor_log.csv` | Measurements that would be observable | Yes |
| `unit_features.csv` | Per-unit observed snapshots with realistic missingness | Yes |
| `unit_features_true.csv` | Complete values behind missing sensors | No; imputation validation only |
| `unit_visit_times.csv` | When each unit reached each station | Yes; as-of joins |
| `manifest.json` | Seeds and chronological fit/validation/test sessions | Split control |
| `model_features.csv` | Final trend/imputation feature table | Yes |

The true-value table must never be joined into defect-model training. It exists
only to score virtual-sensor reconstruction.

## 5. Virtual-sensor pipeline

`src/virtual_sensor.py` chooses a method independently for every missing
station/channel combination.

### 5.1 Spatial branch

1. Measure cross-station correlation on historical truth using training
   sessions only.
2. Identify candidate source stations that are physically measured.
3. Fit linear regression on the candidate's observed readings.
4. Produce an estimate and validation-derived confidence.
5. Carry both columns downstream, for example `S9_torque_est` and
   `S9_torque_conf`.

This is transfer across process-related stations, not an assumption that the
nearest station is similar.

### 5.2 Temporal branch

1. Initialize a Kalman state from sparse historical checks.
2. Predict the state forward between checks.
3. Correct the estimate whenever a real check arrives.
4. Derive confidence from uncertainty/staleness and validation behavior.
5. Produce estimate/confidence pairs, such as the S4 channels in the demo.

The trend can appear nearly flat over short windows because the filter's
purpose is to provide stable estimates between sparse checks. The dashboard
therefore distinguishes measured samples, missing periods, inferred fills,
confidence, and the downstream handoff instead of claiming that a flat line is
a directly measured sensor trace.

### 5.3 Validation-only Optuna tuning

The virtual study searches correlation thresholds, minimum pair counts,
Kalman process/measurement noise scales, numeric floors, rolling windows, and
staleness limits. The objective combines spatial and temporal validation so a
configuration cannot win by improving only one branch.

The chronological split has 40 fit sessions, 10 validation sessions, and 10
final test sessions. Both Optuna studies use only fit/validation evidence. Study
names were versioned after this correction so an older test-contaminated study
cannot silently resume. The final test sessions are evaluated once after model
and threshold selection.

Best-trial parameters and checkpoint summaries are written below
`artifacts/tuning/`. Optuna's SQLite study lets interrupted runs resume under
the same study names.

## 6. Feature engineering

`src/feature_engineering.py` converts event streams into one feature row per
unit while preserving time order.

For each available channel it computes trailing:

- mean: recent operating level;
- standard deviation: local instability;
- slope: drift direction and rate.

As-of joins use only sensor records timestamped at or before the unit's station
visit. This prevents future leakage. A maximum-staleness window stops an old
measurement from masquerading as current evidence.

Virtual estimates are included together with their confidence columns. Decoy
channels are generated to test feature selection but excluded from the final
production feature set when ablation shows they add no value.

## 7. Defect model

### 7.1 Model and imbalance handling

`src/defect_model.py` uses a regularized XGBoost binary classifier. The problem
is a rare-event ranking task, so ordinary accuracy is misleading. A model that
always predicts "good" could exceed 99% accuracy while detecting zero defects.

The implementation therefore focuses on:

- ROC AUC for ranking quality;
- recall for missed-defect exposure;
- precision for alert workload;
- Matthews correlation coefficient for imbalanced classification;
- recall at top-risk fractions in the tuning objective.

The operating risk threshold is selected by searching score quantiles and
maximizing MCC on calibration evidence rather than assuming 0.5.

### 7.2 Model confidence

Raw boosted-tree probabilities are not treated as confidence. The model uses a
split-conformal-style calibration score representing how decisively a unit's
risk lies away from the learned decision boundary relative to calibration
examples. This produces `model_confidence` separately from `risk_score`.

### 7.3 Latest local artifact snapshot

The latest local metadata reports:

| Metric | Value |
|---|---:|
| Final chronological test ROC AUC | 0.7153 |
| Recall | 0.2105 |
| Precision | 0.0350 |
| MCC | 0.0737 |
| Learned threshold | 0.6053 |
| Held-out rows | 13,659 |
| Held-out positives | 76 |

These numbers belong to the packaged leakage-free baseline artifact generated
on the stated simulated configuration. Regeneration can change the result if
seeds, sessions, code, or an optional validation-only search budget changes.

## 8. Effective Trust and decisions

### 8.1 Input Trust

`src/effective_trust.py` assigns direct measured features trust 1.0. An inferred
`*_est` feature receives the confidence from its paired `*_conf` feature.
Input Trust is an importance-weighted average over evidence present for the
unit:

```text
Input Trust = sum(feature_present * feature_trust * feature_importance)
              --------------------------------------------------------
                 sum(feature_present * feature_importance)
```

A weak inferred feature only reduces trust in proportion to how much the model
actually relies on it.

### 8.2 Effective Trust

```text
Effective Trust = clip(Input Trust, 0, 1)
                  x clip(Model Confidence, 0, 1)
```

Risk and trust remain separate. Risk says how likely the defect model believes
a unit is to fail; trust says how safe it is to rely on that statement.

### 8.3 Operational policy

The model threshold determines high/low risk. A configurable trust threshold
determines high/low trust. The action matrix then routes the unit. In a real
deployment `AUTO-ACT` should initially mean an automatic hold request, not an
irreversible physical action, until plant safety validation is complete.

## 9. Bottleneck intelligence

Twinly deliberately separates a current bottleneck from a forming bottleneck.

### 9.1 Current constraint

The dashboard reconstructs state duration only from events at or before the
playback clock. Its live score mirrors the backend active-period reasoning:

```text
current score = utilization
                + blocked fraction
                - starved fraction
                + 1.5 x mean downstream starvation
                + 0.12 x (1 - health)
```

The highest score is the current sustained constraint. Queue growth is not
included in this score because source buffers can be structurally full; using
it here could incorrectly select the first station.

### 9.2 Emerging constraint

An emerging candidate must have either:

- queue slope greater than `0.02` units per 100 simulated seconds; or
- recent cycle drift greater than `12%` versus its local baseline.

Candidates are ranked using positive queue slope, positive cycle drift, and
queue pressure. This is an evidence warning, not a guaranteed time-to-failure
forecast.

## 10. Demo inference and time honesty

`data/generate_demo_data.py` creates a fresh shift with a different seed from
the historical sessions. `data/build_demo_inference.py` then:

1. loads tuned virtual-sensor parameters, or safe defaults;
2. fits virtual sensors on historical training data;
3. builds demo features using only observable evidence;
4. fits the production defect pipeline on historical training rows;
5. scores demo units;
6. writes timestamped assessments and virtual-sensor events.

The Streamlit app replays those files. It does not reveal an assessment until
its `latest_t` is at or before the current clock. Likewise, state, health,
sensor, buffer, visit, and virtual-sensor views are filtered to current time.

This file-backed replay is an implementation convenience for a hackathon demo.
In production, the same output tables would be incrementally produced by a
stream processor or inference service as new plant events arrive.

## 11. Dashboard guide

The left Control Deck is the only navigation mechanism. Playback controls stay
in the same sidebar so page content starts below Streamlit's toolbar and does
not jump when switching workspaces.

### 11.1 Playback controls

- **Play/Pause:** starts or stops advancement of the simulation clock.
- **Restart:** returns to the first timestamp and pauses playback.
- **Auto-play:** enables periodic reruns and time advancement.
- **Loop playback:** wraps to the beginning after the final timestamp.
- **Simulation timeline:** manually selects current evidence time.
- **Playback speed:** controls simulated seconds advanced per update.

Changing time clears stale generated Copilot answers so an explanation from a
future or previous timestamp is not shown as current.

### 11.2 Header and KPI cards

The header shows the current simulation clock and number of stations. The KPI
grid is reconstructed from current-time evidence:

| KPI | Interpretation |
|---|---|
| Units completed | Unique units observed at the final station by now |
| Throughput | Final-station completions extrapolated from a 5-15 minute rolling window |
| Work in progress | Units started minus units completed |
| Current bottleneck | Highest live sustained-constraint score |
| High-risk units | Available unit assessments above the learned risk threshold |
| Degraded stations | Current health below 0.82 |
| Average health | Mean current simulated station health |
| Line status | `CRITICAL`, `WATCH`, or `STABLE` from current supported conditions |

`CRITICAL` means a degraded station exists or the current non-source constraint
has at least 90% queue pressure. `WATCH` means a high-risk unit or emerging
bottleneck exists. Otherwise the display is `STABLE`.

### 11.3 Live Line

Purpose: answer "what is happening on the line now?"

Each station card shows state, active unit, utilization, health, recent cycle,
input buffer, sensor coverage, and a normalized operational risk indicator.
Animated buffer connectors show direction of flow and queue trend. The alert
feed prioritizes faults/degradation, high-risk units, growing queues, and blind
sensor conditions.

The station operational risk shown here is not defect probability. It is a
normalized presentation of the station constraint score.

### 11.4 Defect Intelligence

Purpose: answer "which units may fail, and can we trust the prediction?"

- Scatter x-axis: Effective Trust.
- Scatter y-axis: defect risk.
- Bubble size: model confidence.
- Bubble color: action policy.
- Horizontal line: learned high-risk threshold.
- Vertical line: human/automatic trust gate.

The unit selector opens the path that the selected unit has actually completed
by the current time. Suspected origin is based on abnormal observed signals or
the weakest health along that observed path. It is explicitly a diagnostic
lead, not proven causality. The watch cohort lists later units that traversed
the same suspected station and may deserve inspection.

### 11.5 Bottleneck Intelligence

Purpose: keep sustained constraints separate from early warnings.

The first card names the current constraint and its utilization, queue,
starvation, and cycle drift. The second card shows the strongest emerging
candidate only if queue or cycle evidence clears its trigger. Charts provide a
recent queue timeline and a station ranking of composite constraint evidence.

### 11.6 Station Health

Purpose: compare station condition and investigate one station deeply.

The overview table combines current state, health, utilization, queue pressure,
sensor status, channel coverage, and risk band. A station selector changes the
detail charts for health, cycles, buffers, and sensor readings over a trailing
window. No future timestamps are included.

The Sensor Coverage section explains whether evidence is measured, partial,
virtual, or unknown. It also shows the latest inferred values and confidence
actually passed downstream.

### 11.7 AI Copilot

Purpose: explain the dashboard state in operator language without becoming a
source of metrics.

The page visualizes four stages:

```text
Twin engine -> structured JSON context -> LLM explanation -> local fallback
```

The context contains compact KPIs, station states, bottleneck summaries,
high-risk units, sensor coverage, and recent alerts. It does not send raw CSV
files. The prompt instructs the model to use only supplied context and to state
when evidence is insufficient.

If no key exists, deterministic local answers are used. If the API returns a
temporary 429, the app retries once with a short delay. Quota, authentication,
permission, network, and empty-response failures are converted into clear
messages, and local answers remain available. LLM failure cannot change line
metrics, risk scores, or actions.

### 11.8 Model Analytics

Purpose: show historical validation without confusing it with live operations.

This page reads saved model metadata and displays AUC, MCC, recall, precision,
threshold, holdout size, and saved feature importance. It labels these as
held-out historical metrics. They do not describe the current line condition.

The dashboard does not fabricate confusion matrices or ROC/PR curves when
individual held-out predictions are unavailable.

### 11.9 Operational Perspectives

Purpose: project one shared state into role-specific summaries.

- Supervisor: immediate alerts and next actions.
- Manager: shift-level throughput, WIP, state mix, and decision distribution.
- Leadership/governance: automation boundaries, coverage, and trust posture.

These views consume the same live state; they are not separate models.

### 11.10 Business & Scale

Purpose: connect technical capability to deployment value without inventing
currency savings.

It summarizes production exposure, direct observability, virtual coverage, and
governed high-risk actions. Plant-specific ROI is intentionally left dependent
on real scrap cost, downtime, labor, throughput, and intervention efficacy.

## 12. LLM layer boundaries

The LLM is optional and non-authoritative.

### It can

- summarize current structured evidence;
- explain why a unit needs human review;
- name the current and emerging bottleneck from supplied fields;
- restate coverage and trust limitations;
- provide role-oriented wording.

### It cannot

- calculate production KPIs;
- retrain or tune any model;
- alter risk, confidence, trust, or actions;
- read future replay data;
- infer a causal root cause unsupported by context;
- control equipment.

The API key is read from the server environment and passed as a bearer token by
the Streamlit server. It must never be placed in source, browser code, screenshots,
or committed configuration.

## 13. Hyperparameter optimization and checkpoints

The command below tunes both model families:

```bash
python src/optuna_tune.py --study all --virtual-trials 30 --defect-trials 50
```

The virtual study tunes correlation selection, Kalman noise, feature windows,
and staleness. The defect study tunes XGBoost regularization/capacity plus the
threshold search range. Both use seeded multivariate TPE and median pruning.

Local outputs:

| Path | Purpose |
|---|---|
| `artifacts/tuning/optuna_studies.db` | Resumable study history |
| `artifacts/tuning/checkpoints/*.json` | Best-so-far trial payloads |
| `artifacts/tuning/virtual_sensor_best.json` | Final virtual/feature parameters |
| `artifacts/tuning/defect_model_best.json` | Final XGBoost/threshold parameters |
| `artifacts/defect_model.json` | Saved XGBoost weights |
| `artifacts/defect_model_meta.json` | Feature schema, metrics, threshold, and parameters |

The final model and metadata are packaged for immediate dashboard startup.
Tuning studies, checkpoints, and large historical data remain local and ignored.

## 14. Reproduction commands

### 14.1 Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 14.2 Full defensible run

```bash
python data/generate_training_data.py \
  --sessions 60 --duration 100000 --seed 100 \
  --validation-sessions 10 --test-sessions 10

python src/optuna_tune.py \
  --study all --virtual-trials 30 --defect-trials 50

python data/generate_demo_data.py --duration 8000 --seed 999
python data/build_demo_inference.py --use-tuned
streamlit run app2.py
```

### 14.3 Train without Optuna

```bash
python data/generate_training_data.py \
  --sessions 60 --duration 100000 --seed 100 \
  --validation-sessions 10 --test-sessions 10
python src/feature_engineering.py
python src/train_defect_model.py
python data/generate_demo_data.py --duration 8000 --seed 999
python data/build_demo_inference.py
streamlit run app2.py
```

This path deliberately uses source defaults. Tuned parameters are loaded only
when `--use-tuned` is explicitly supplied, preventing stale local search output
from silently changing a reproducible baseline.

### 14.4 Submission verification

```bash
python scripts/verify_submission.py
git status --short
git check-ignore -v data/simulated/model_features.csv
git check-ignore -v data/demo_live/model_features.csv
git check-ignore -v artifacts/defect_model.json
```

## 15. Production migration

The prototype can be migrated incrementally:

1. Define adapters that convert PLC/SCADA/historian/MES messages into the
   current station, buffer, sensor, and unit-visit schemas.
2. Replace hidden simulation health with a calibrated condition estimator.
3. Train virtual sensors only on approved historical periods and validate by
   temporarily masking known-good sensors.
4. Train the defect model with chronological plant splits and leakage review.
5. Calibrate risk and trust thresholds against plant costs and human workload.
6. Begin with advisory alerts, then human-approved holds, then limited
   automation only after safety and efficacy validation.
7. Add drift monitoring, model/version registry, audit logs, access control,
   and rollback.

## 16. Known limitations and honest claims

- The prototype is a digital shadow/replay, not a bidirectionally connected
  production digital twin.
- Simulation assumptions are not measured plant parameters.
- Demo `health_true` is hidden simulation truth, excluded from live decisions,
  and shown only in the explicitly labelled validation chart.
- Defect and bottleneck logic are related through simulated health dynamics;
  this is a designed test environment, not proof of universal causality.
- The defect model has useful ranking signal, not industry-grade proof of
  deployment performance.
- Rare-event accuracy is not a meaningful headline metric.
- Suspected origin is a review lead, not causal root-cause confirmation.
- Emerging bottleneck signals do not provide a calibrated time-to-failure.
- Virtual-sensor confidence requires ongoing validation under plant drift.
- The LLM is a language interface only; no RAG maintenance corpus is claimed.
- ROI is not calculated without plant-specific cost and intervention data.

## 17. Suggested judging/demo sequence

1. Open **Live Line** and start playback.
2. Point out station movement, buffer growth, state changes, and the clock.
3. Open **Bottleneck Intelligence** and contrast current versus emerging.
4. Open **Defect Intelligence** and explain risk versus Effective Trust.
5. Select a unit and show its observed route and suspected-origin evidence.
6. Open **Station Health**, then show measured versus virtual coverage.
7. Show S4 temporal/Kalman fills and S9 spatial/regression fills when available.
8. Open **AI Copilot** and ask which station is the current bottleneck.
9. Explain that API failure falls back locally and does not affect analytics.
10. Open **Model Analytics** and state the historical/evaluation boundary.

The strongest closing message is: Twinly does not only predict risk. It also
measures whether the evidence behind that prediction is trustworthy enough to
act on.
