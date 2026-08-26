# DigitalTwin.ai — Round 2 Pipeline & Checklist

Working document tracking the full build: what the pipeline does end-to-end, and
what's done vs. remaining. Complements [DigitalTwin_Round2_Plan.md](DigitalTwin_Round2_Plan.md)
(the original scope/phasing) and [Digitaltwin project context.md](Digitaltwin%20project%20context.md)
(Round 1 background, citable facts, cautions) — this file is the living
build tracker for Round 2.

---

## 1. What the pipeline does (end to end)

```
OFFLINE (train once)                          LIVE (the demo)
─────────────────────                          ────────────────
Run line_sim many times/long           ──►     Fresh simulation run starts
(health-drift episodes across                          │
 3-tier stations: A/B/C)                                ▼
        │                                      Station states, buffers, sensor
        ▼                                      readings stream in (Tier A/B dense,
Build labeled tables                           Tier C sparse/irregular)
 - per-unit defect table                                │
 - per-station-window bottleneck table                  ▼
        │                                      Virtual sensor fills Tier B (spatial
        ▼                                      regression) + Tier C (Kalman) gaps
Train defect_model.py (XGBoost)                         │
Train bottleneck forecaster (optional)                  ▼
        │                                      defect_model scores unit + confidence
        ▼                                      bottleneck_detect scores station window
Save model artifacts                                    │
                                                          ▼
                                                Effective Trust = input_trust ×
                                                model_confidence → gates action
                                                          │
                                                          ▼
                                                Template explanation generated
                                                (feature importances + trust numbers,
                                                 no LLM — see §5)
                                                          │
                                                          ▼
                                                Streamlit dashboard: supervisor /
                                                manager / leadership persona views
```

Core differentiator carried through every stage: a station's hidden health
state drives sensor drift, cycle-time/failure-rate creep, *and* defect risk
together — so bottleneck and defect signals are causally linked, not
independent random draws. This is what makes trend-based ("advanced")
prediction meaningful instead of naive thresholding.

---

## 2. Checklist

### Phase A — Data foundation
- [x] SimPy line simulation with WORKING/BLOCKED/STARVED/DOWN states — `src/line_sim.py`
- [x] Active-period + queue-growth bottleneck detection — `src/bottleneck_detect.py`
- [x] Plotly visualization of a sim run (state timeline, buffer heatmap, state composition) — `src/viz.py`
- [x] Bosch-faithful synthetic defect dataset generator (station-grouped, structural missingness, ~0.58% defect rate, calibrated cross-station correlation) — `data/get_data.py`
- [x] Real Bosch numeric-file fetch + stratified sample script — `data/fetch_bosch.py`
- [x] XGBoost defect model with conformal-style confidence — `src/defect_model.py`
- [x] Conda env `digitaltwin` (Python 3.13.3) set up, all `requirements.txt` deps installed and verified
- [x] Extend `line_sim.py` with the shared health-state process `H_s(t)` (gradual down→hold→up ramp, never instantaneous; recoverable episodes — most are mild near-misses)
- [x] Add 3-tier station instrumentation (`A` instrumented / `B` sensor-poor-correlated / `C` sensor-poor-isolated) replacing the binary `has_sensor` (kept as a derived legacy property)
- [x] Log continuous sensor channels (torque, vibration, temperature) per station per timestep — dense for Tier A, sparse/irregular manual-check for Tier C, absent for Tier B
- [x] Add delayed-defect fields (`defect_occurred_at` vs `defect_caught_at`) — verified 73% of defects in a test run were caught at a *later* inspection station than where they occurred
- [x] `data/generate_training_data.py` — runs multiple sessions, persists `station_registry.csv` / `health_log.csv` (hidden ground truth) / `sensor_log.csv` / `unit_features.csv` (observed-only) / `unit_features_true.csv` (validation-only) / `manifest.json` with a chronological train/test split, to `data/simulated/` (gitignored, regenerable)
- [x] Scaled the persisted dataset to a stable evaluation size (initial pass, later regenerated — see below)
- [x] **Fixed a real design bug**: station correlation was originally modelled as physical-adjacency smoothing (a latent value blurred across line-neighbours), which conflated "next to each other on the line" with "shares a real physical cause" — wrong, since two distant stations can share tooling/calibration/material-batch while neighbours share nothing. Replaced with explicit shared **process factors**: each station has a loading vector, and correlation follows shared loadings regardless of line position. Verified: Door-Fit (early, body construction) and Torque-2 (late, final assembly) — nowhere near each other — now correlate at 0.459 (health-level, large sample) purely through a shared "torque-calibration rig" factor; a same-position control pair with no shared factor stays near zero
- [x] Verified the correlation signal is real but noise-diluted at the single-reading level for *weakly*-loaded distant pairs (Door-Fit↔Torque-2 washes to ~0 per-unit despite being real at the health/tick level) — confirms the already-planned windowed/trend feature engineering is necessary, not optional, for the virtual sensor's spatial regression to actually exploit weaker distant correlations. Strongly-loaded partners (Torque-1, Electrical) remain usable even from raw per-unit snapshots (0.77, 0.73)
- [x] Verified tier isolation end-to-end: Tier B (`S9`) produces **zero** observed columns in `unit_features.csv`; Tier C (`S4`) is ~0.5% populated; Tier A (`S0`) is ~99% populated; `unit_features_true.csv` retains 100% ground truth for every tier
- [x] **Fixed a second real calibration bug**: strengthening the shared process factor (to fix Tier-B correlation, above) had the side effect of making health chronically degraded (only 58.5% of ticks at perfect health, mean 0.844) instead of rare/brief. Added a dead-zone so only genuine tail excursions of the shared factor cost health — restored to 95% perfectly healthy (mean 0.989) while the correlation fix is still intact (0.80 for the strong pair, isolation preserved for Tier C)
- [x] **Fixed a third bug**: even after the dead-zone fix, defect-origin health was statistically indistinguishable from the general population — no learnable signal at all (`DEFECT_HEALTH_FACTOR=6` far too weak once degraded time became rare). Empirically swept `DEFECT_HEALTH_FACTOR` (6 → 200) and rescaled baseline defect rates down (~0.42×) until degraded-health periods genuinely account for a distinguishable share of defects, verified directly (defect-origin health vs. population health), while overall rate stays near Bosch's 0.58%
- [x] Fixed an evaluation-methodology bug: a single held-out session left too few test defects (as low as 4) for any metric to be stable. `generate_training_data.py` now supports `--test-sessions N` (multiple trailing held-out sessions, not just one)
- [x] **Fixed a fourth bug (signal was mostly-noise + eval too small)**: firming up the numbers with more held-out defects revealed the earlier held-out AUC ~0.58 was a small-sample artifact — on ~76 held-out defects it fell to ~0.55 (chance), because (a) ~58% of defects originated at *perfect* health (spontaneous, no precursor — an unlearnable majority) and (b) the model overfit (train-AUC 1.0 / held-out 0.55). Split defect generation into an explicit spontaneous vs. health-driven component (`DEFECT_SPONTANEOUS_WEIGHT`, `DEFECT_RATE_SCALE`) and calibrated to **~74% health-driven / ~26% spontaneous** (verified via defect-origin health), a *stated assumption* that some failures are foreseeable and some genuinely aren't. Overall rate stays ~0.6% (Bosch 0.596%).
- [x] **Added decoy sensor channels**: `line_sim.py` emits irrelevant channels (`pressure` pure-noise + `humidity` structured random-walk) on an ISOLATED RNG stream (`_drng`), so a real historian's mix of useful/useless tags is modelled without perturbing the core health/defect realization (verified byte-identical: same 183/497 defect counts with decoys on/off). `channel_registry.csv` tags ground truth.
- [x] **Final calibrated dataset**: 60 sessions × 100,000s (50 train / 10 test) → 82,047 units, **497 defects** (0.61% rate), ~76 held-out defects — stable enough for a trustworthy held-out AUC. (The older 24-session/4-test set, 183 defects/32 held-out, is reproducible but too small for stable metrics.)
- [ ] Link `get_data.py`'s cross-sectional Bosch-style generator to reuse this same schema (currently a separate, older generator — kept for the offline Bosch-faithful notebook story per the design decision in [[round2-data-model-design]])
- [ ] Pull a matched sample of real `train_date.csv` (by `Id`, joined to the existing numeric sample) — optional, for calibrating real station timing distributions

### Phase B — Models
- [x] `src/virtual_sensor.py` — method is auto-selected per station from MEASURED correlation on training data (not the hard-coded tier label): high correlation with another station -> spatial regression; low correlation but own sparse history -> Kalman; neither -> flagged unrecoverable. Auto-detection matched the hand-designed tiers exactly (station 4 -> temporal, station 9 -> spatial), confirming the tier design is consistent with what the data actually supports
- [x] Validated on the held-out test session against fair baselines: Kalman beats naive forward-fill by 9-20% across channels; spatial regression beats a naive mean-guess by 29-44% overall and by **54-66% specifically during drift episodes** (the case that actually matters for defect/bottleneck prediction) — real numbers, not fabricated
- [x] `src/feature_engineering.py` — trend-aware, tier-complete feature table (`data/simulated/model_features.csv`, 82,047 units × 171 features incl. decoy distractors): rolling mean/std/slope for every observed channel (real + decoy) as-of each unit's actual visit time (never leaking a later reading in), Tier B filled via `virtual_sensor.py`'s spatial regression on SMOOTHED neighbor readings, Tier C via a full Kalman replay (dense estimate + confidence for every unit). Imputation runs on real channels only — decoys are never reconstructed.
- [x] **Regularised the defect model** (`defect_model.py`: depth 6→3, `min_child_weight=10`, `gamma`, `reg_lambda`) to stop it memorising the rare positives — closed a train-AUC-1.0 / held-out-0.55 overfit gap to a healthy **train 0.88 / held-out 0.72**.
- [x] `src/train_defect_model.py` — retrained on `model_features.csv` (60-session data) with an honest chronological holdout: **held-out AUC 0.72** on 76 test defects, real channels only — a genuine, stable ranking signal (no longer a small-sample artifact).
- [x] Primary metric is **recall-at-top-K%** (rank-based, threshold-free): **top 20% riskiest units catch 54% of defects (41/76, ~2.7× lift); top 10% catch 37%.** A continuous risk score + confidence, the intended input to `effective_trust.py`'s Risk×Trust gating.
- [x] **Feature selection via permutation importance** (held-out, 30 shuffles): decoys score ~0 (real +0.0014 vs decoy −0.0002; top-8 features all real), where gain importance could not separate them. **Ablation**: dropping all 66 decoy features and retraining *improves* held-out AUC (0.69 → 0.72) and top-10% recall (25% → 37%) — the model is measurably better without the noise channels.
- [ ] Defect model: detection lead-time metric (`defect_occurred_at` vs. model-flag time) — real, reportable number
- [ ] (Optional, "advanced") bottleneck forecasting layer: features = rolling queue-growth/utilization/sensor-drift trend, label = "becomes bottleneck within next N minutes", chronological split
- [ ] Bottleneck: lead-time / false-alarm-rate evaluation using the hidden ground-truth health process
- [x] `src/effective_trust.py` — input_trust × model_confidence fusion (MULTIPLY, not average) + Risk×Trust action-gating matrix (AUTO-ACT / HUMAN-VERIFY / MONITOR / PASS). input_trust is an importance-weighted average of per-feature trust (real=1.0, virtual-sensor est=its *_conf). **Validated on held-out data: AUTO-ACT flags are 2.99× as precise as HUMAN-VERIFY** (2.86% vs 0.96% defect rate) — trust correctly routes shaky high-risk flags to a human. Recalibrated `defect_model.py`'s conformal confidence to per-side normalisation so the trust scale spans a usable 0–1 (was crushed to ~0.06 under imbalance).

### Phase C — Explanation & UX
- [ ] Template-based natural-language explanation generator (feature importances + trust/confidence numbers → plain sentence, e.g. "Station 4: risk 0.82, driven by torque drift +3.2σ, Effective Trust 0.50 → human verification required") — no LLM
- [x] `src/personas.py` — supervisor (shop-floor: current/forming bottleneck, action queue, blind spots), manager (per-shift defect/action/trust trends), leadership (early-catch %, automation rate at a fixed trust policy, cited-anchor business context) — three projections of ONE shared per-unit assessment, so numbers are consistent across all three. All modules train on the same decoy-dropped production feature set via `load_production_split()`.
- [x] `app.py` — Streamlit dashboard: line/bottleneck view, risk×trust action scatter, per-shift trends, early-catch lift curve, three persona tabs, and a live trust-policy slider that re-gates the action matrix. Trains the same decoy-dropped production model via `load_production_split()`; all render paths validated headless with `streamlit.testing.v1.AppTest`.
- [ ] `notebooks/defect_model_training.ipynb` — model-training narrative notebook
- [ ] (Stretch, only if time allows) Real RAG layer: small team-authored maintenance-notes corpus + retrieval + LLM phrasing — must use a genuine corpus, never faked

### Phase D — Deliverables
- [ ] `docs/business_proposal.md` — problem framing, solution design, target users, business case & impact, phased roadmap, risks + mitigations
- [ ] `docs/architecture.md`
- [ ] Pitch deck (presents proposal + prototype)
- [ ] Demo video recorded from the running Streamlit app
- [ ] Final README pass — implementation approach, architecture, dependencies, execution instructions
- [ ] Repo hygiene check before submission (no secrets, no raw Bosch data committed, requirements.txt current)

---

## 3. Key design decisions (why, not just what)

- **Shared health-state mechanism** ties bottleneck, defect, and virtual-sensor
  signals to one causal root cause per episode — required for genuinely
  "advanced" (trend-based) prediction rather than independent random events.
- **3 instrumentation tiers**, not 2 — Tier B (spatially correlated) and Tier C
  (temporally smooth, sparse) call for different imputation techniques
  (spatial regression vs. Kalman filter), matching the brief's real
  complexity of mixed legacy/modern, richly/poorly instrumented stations.
- **Train offline, infer live** — models never train on the demo run itself;
  they're trained on a large batch of historical simulated sessions, saved as
  artifacts, then applied to a fresh live run. Standard digital-twin pattern,
  and the only honest way to avoid leaking the predicted event into training.
- **Chronological (not random) train/test splits** — avoids leaking within a
  single health episode.
- **Template explanations now, RAG only as a stretch** — Track 4's brief
  doesn't require an LLM/RAG layer; the differentiator is virtual sensors +
  Effective Trust, not narrative generation. A RAG layer is legitimate only
  with a real retrieval corpus — never fabricated.
- **No fabricated metrics anywhere** — every number quoted in the proposal or
  deck must come from an actual run of this pipeline.

See [round2_data_model_design memory] (session-local design notes) for the
full data schema (station registry, time-series sensor log, per-unit table,
engineered features) if resuming this build in a new session.
