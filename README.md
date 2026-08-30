# Twinly - Trust-Aware Production Digital Twin

Twinly is a Streamlit prototype for replaying a vehicle assembly line as a
live digital twin. It combines bottleneck detection, defect-risk prediction,
virtual sensors for poorly instrumented stations, and an Effective Trust layer
that decides whether a prediction can be acted on automatically or needs human
verification.

> **Submission entry point:** `app2.py`
>
> A legacy `app.py` may remain in a developer's local worktree, but it is
> ignored and is not part of the submission. Run and deploy `app2.py`.

## What the prototype demonstrates

| Capability | What it does | Main implementation |
|---|---|---|
| Assembly-line digital shadow | Replays station states, queues, unit movement, health, and sensors against a simulation clock | `src/line_sim.py`, `app2.py` |
| Current bottleneck detection | Ranks sustained constraints from utilization, blocking, starvation, queue pressure, cycle drift, and health | `src/bottleneck_detect.py`, `app2.py` |
| Emerging bottleneck warning | Detects recent queue growth and cycle-time drift before a stable constraint is fully formed | `src/bottleneck_detect.py`, `app2.py` |
| Virtual sensors | Uses spatial regression for correlated blind stations and temporal Kalman filtering for sparse isolated stations | `src/virtual_sensor.py` |
| Defect intelligence | Produces unit-level defect risk and calibrated model confidence with XGBoost | `src/defect_model.py`, `src/train_defect_model.py` |
| Trust-aware action gating | Multiplies input trust by model confidence and maps Risk x Trust to an operational action | `src/effective_trust.py` |
| Hyperparameter optimization | Tunes virtual-sensor, feature-window, XGBoost, and threshold parameters with persistent Optuna studies and checkpoints | `src/optuna_tune.py` |
| Operator dashboard | Separates live operations, defect intelligence, bottlenecks, health, analytics, personas, and business framing | `app2.py` |
| Optional AI Copilot | Explains a compact structured snapshot; local deterministic answers remain available if the API is disabled or unavailable | `app2.py` |

## Architecture

```text
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

The simulator is the prototype data source, not the trained model. In a plant,
the source would be replaced by PLC, SCADA, historian, IoT, or MES events while
the downstream schemas and decision logic remain the same.

## Run the final dashboard

If this machine already has `data/demo_live/` and the local artifacts:

```bash
cd /path/to/DigitalTwin.AI
source venv/bin/activate
streamlit run app2.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Reproduce from a clean clone

Generated data and trained weights are deliberately not committed. The full
reproducible path is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Historical offline corpus used for tuning/training.
python data/generate_training_data.py \
  --sessions 60 --duration 100000 --seed 100 --test-sessions 10

# Tune virtual sensors, feature windows, model parameters, and thresholds.
# Best-trial JSON files, SQLite studies, checkpoints, and final weights are
# written under artifacts/ and remain local.
python src/optuna_tune.py \
  --study all --virtual-trials 30 --defect-trials 50

# Create a fresh shift that was not used as training history, then run the
# trained inference mechanisms over it.
python data/generate_demo_data.py --duration 8000 --seed 999
python data/build_demo_inference.py

streamlit run app2.py
```

The 60-session run is the defensible evaluation configuration and can take
substantial time and disk space. The generator defaults are suitable only for
a quick engineering smoke test; do not quote smoke-test metrics in a pitch.

## Optional AI Copilot

The Copilot is an explanation layer, not a prediction engine. Without an API
key, all simulations, models, charts, alerts, and deterministic Copilot answers
continue to work.

```bash
export OPENAI_API_KEY="your_rotated_server_side_key"
export OPENAI_MODEL="gpt-5-mini"
streamlit run app2.py
```

Never commit an API key. `.env`, Streamlit secrets, and key files are ignored.
The example in `.env.example` contains placeholders only.

## Repository layout

```text
DigitalTwin.AI/
|-- app2.py                         # final submission dashboard
|-- requirements.txt
|-- README.md
|-- data/
|   |-- generate_training_data.py   # offline historical data generation
|   |-- generate_demo_data.py       # fresh demo shift generation
|   |-- build_demo_inference.py     # virtual fills + live defect assessments
|   |-- fetch_bosch.py              # optional real Bosch sample acquisition
|   |-- simulated/                  # generated locally; ignored by Git
|   `-- demo_live/                  # generated locally; ignored by Git
|-- src/
|   |-- line_sim.py                 # discrete-event line and health dynamics
|   |-- bottleneck_detect.py        # current and forming constraints
|   |-- virtual_sensor.py           # spatial/Kalman inference + confidence
|   |-- feature_engineering.py      # as-of rolling and inferred features
|   |-- defect_model.py             # XGBoost risk + calibrated confidence
|   |-- effective_trust.py          # trust fusion and action matrix
|   |-- optuna_tune.py              # resumable hyperparameter studies
|   |-- tuning_config.py            # local artifact/checkpoint paths
|   `-- personas.py                 # role-specific operational views
|-- artifacts/                      # local weights/tuning/checkpoints; ignored
|-- docs/
|   `-- SUBMISSION_REPORT.md        # detailed technical and dashboard guide
`-- scripts/
    `-- verify_submission.py        # repository hygiene and compile check
```

## Local artifact policy

The following remain on the developer machine but are excluded from Git:

- `data/simulated/`: large historical training and holdout tables.
- `data/demo_live/`: generated replay CSV/JSON files.
- `artifacts/`: model weights, metadata, Optuna best parameters, studies, and checkpoints.
- `docs/*.html`: generated visualization exports.
- virtual environments, caches, logs, databases, secrets, and key files.

The Python generators, model code, dependency list, documentation, and empty
directory placeholders are committed. This keeps the submission reproducible
without publishing generated data, model binaries, or credentials.

## Latest local validation snapshot

The latest local tuned artifact reports a chronological held-out ROC AUC of
`0.7286`, recall `0.3333`, precision `0.0294`, MCC `0.0815`, and a learned risk
threshold of `0.6127` on 13,678 held-out rows containing 84 positive examples.
These are ranking and rare-event metrics; ordinary accuracy is intentionally
not used because a model that predicts every unit as non-defective would look
accurate at this class imbalance while catching no defects.

The artifact itself is not committed. The report records the observed result,
and the complete command above reproduces the evaluation from source.

## Verify before pushing

```bash
python scripts/verify_submission.py
git status --short
```

The verifier compiles repository Python files and fails if generated data,
weights, tuning output, or common secret files are tracked.

For the complete system explanation and a page-by-page dashboard walkthrough,
read [docs/SUBMISSION_REPORT.md](docs/SUBMISSION_REPORT.md).
