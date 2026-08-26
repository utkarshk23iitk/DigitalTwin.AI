# Session notes — DigitalTwin.ai (2026-08-23 → 08-26)

Working log of what was built, decided, and discovered this session. Complements
[PIPELINE.md](PIPELINE.md) (the living build tracker) and
[README.md](README.md) (how to run it). Read this to resume with full context.

---

## What got done this session

1. **Environment fixed so the pipeline runs on this machine.**
   - `.venv` was nearly empty; installed all of `requirements.txt`.
   - xgboost couldn't load `libomp.dylib` (no Homebrew here). Vendored a copy
     into `.venv/.../xgboost/lib/` with an `@loader_path` rpath, so `.venv` is
     self-contained and the shared conda base is left untouched.
     ⚠️ A `pip install --force-reinstall xgboost` would wipe the vendored copy —
     re-copy `libomp.dylib` next to `libxgboost.dylib` if that happens.

2. **Decoy sensor channels added** (`line_sim.py`): `pressure` (pure noise) +
   `humidity` (useless random walk), emitted like real channels but on an
   **isolated RNG stream** (`_drng`) so they never perturb the core signal.
   `channel_registry.csv` tags ground truth. Models the real-historian problem
   of many irrelevant tags.

3. **Feature selection** (`train_defect_model.py`): gain vs **permutation
   importance** on held-out data, plus a **drop-decoys-and-retrain ablation**.

4. **Effective Trust** (`effective_trust.py`): input_trust × model_confidence
   (multiply, not average) → Risk×Trust action matrix (AUTO-ACT / HUMAN-VERIFY /
   MONITOR / PASS).

5. **Personas** (`personas.py`) and the **Streamlit dashboard** (`app.py`) —
   three lenses over one shared model state, with a live trust-policy slider.

6. **Docs updated** to the firmed-up numbers + honesty notes; everything pushed.

---

## The big finding (most important — carry into the pitch)

**"Firming up the numbers" exposed that the earlier held-out AUC ~0.58 was a
small-sample artifact.** With more held-out defects (32 → 76) it fell to ~0.55
(≈chance). Two root causes, both fixed:

- **Overfitting:** an unregularised depth-6 XGBoost hit train-AUC **1.0** /
  held-out **0.55**. → Regularised (`defect_model.py`: depth 6→3,
  `min_child_weight=10`, `gamma`, `reg_lambda`).
- **Signal was mostly noise:** ~58% of defects originated at *perfect* health
  (spontaneous, no precursor → unlearnable). → Split defect generation into an
  explicit spontaneous vs. health-driven component and calibrated to **~70/30**
  (verified ~74% originate at degraded health). *This is a stated modelling
  assumption*, not a tuned-to-look-good knob — some failures are foreseeable,
  some genuinely aren't.

**Result after fixes (60 sessions, 76 held-out defects, real channels only):**
held-out **AUC 0.72** (train 0.88 — healthy gap), **top-20% recall 54%**,
top-10% 37%. These are stable and defensible.

---

## Key decisions & their rationale

- **Isolated decoy RNG.** Decoys must be purely additive; verified byte-identical
  defect counts (183 / 497) with decoys on vs off. Without this, adding decoys
  silently changed the whole realization and tanked the signal.
- **Signal strength = 70/30 (user's call, "moderate").** Alternatives were
  "strong 90/10" (higher AUC, less defensible) and "keep 42%, model-only"
  (most conservative, AUC stays ~0.55).
- **Per-side confidence recalibration** (`defect_model.py`): the conformal score
  was crushed to ~0.06 on the high-risk side under imbalance. Normalising by each
  side's own range makes it span a usable 0–1 for the trust fusion / dashboard.
  Affects the `confidence` column only — risk/AUC unchanged.
- **`load_production_split()`** (in `effective_trust.py`): drops decoy features so
  personas / effective_trust / app all use the SAME real-only model → numbers
  consistent everywhere. `train_defect_model.py` keeps decoys in, on purpose, to
  run the ablation.
- **Fixed trust policy (0.5) for operational KPIs.** The leadership "automation
  rate" was tautological when the threshold = median-of-high-risk; a fixed bar
  gives a real measured rate.

---

## Honest caveats to preserve (a judge could probe these)

- **No lead-time-in-minutes yet.** "Predicts in advance" currently means
  build-time prediction *before* downstream inspection catch (measured in
  stations, not minutes). The minutes metric is a TODO — do NOT claim "15 min".
- **Line-sim constants are assumptions**, not measured (unlike the Bosch-
  calibrated defect-data constants). Say so.
- **Input-trust is underpowered on current data.** Only S4 (tier-C Kalman) is
  imputed; **S9 (tier-B) is a total blind spot** (unrecoverable — no sensor, no
  correlated partner cleared the bar → zero features). So effective-trust
  variation comes mostly from model_confidence, not input_trust. Honest, and
  on-theme ("knows its limits"), but a richer line would exercise input-trust more.
- **Spatial virtual-sensor branch is flaky** — underperformed its baseline on
  some runs; the "29–44% spatial improvement" figure in older docs needs a
  re-check before quoting.
- **Small-sample everywhere.** 76 test defects → AUC ±~0.03; top-K recall still
  wobbles. AUC is the number to lead with.

---

## Reproduce (the numbers above)

```bash
source .venv/bin/activate
python data/generate_training_data.py --sessions 60 --duration 100000 --seed 100 --test-sessions 10
python src/feature_engineering.py
python src/train_defect_model.py     # AUC, permutation importance, decoy ablation
python src/effective_trust.py        # 2.99x gating validation
python src/personas.py               # three persona views
streamlit run app.py                 # the dashboard (localhost:8501)
```

`data/simulated/` is gitignored (regenerable from seed). The real Bosch sample
`data/bosch_numeric_sample.csv` (99,741 rows) is also gitignored.

---

## Build state

Complete: line sim · bottleneck detection · virtual sensor · feature engineering
(+decoys) · defect model (AUC 0.72) · Effective Trust (AUTO-ACT flags 2.99× more
precise than HUMAN-VERIFY) · personas · Streamlit dashboard.

## Next steps (not yet done)

- Defect-model **lead-time metric** (turn "advance prediction" into a real number).
- Template **explanation generator** (importances + trust → plain sentence).
- Optional: bottleneck **forecasting** layer + lead-time/false-alarm eval.
- Phase D: `docs/business_proposal.md`, pitch deck, demo video (record `app.py`).
- Consider fixing / re-validating the spatial virtual-sensor branch.
