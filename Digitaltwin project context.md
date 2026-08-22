# DigitalTwin.ai — Project Context & Handoff

*Paste this into a Claude Project's instructions/knowledge so any new chat has full context.*

## The competition
- **Accenture Innovation Challenge 2026**, Round 1. Chosen problem: **#4 DigitalTwin.ai** — design a digital twin prototype of a vehicle assembly line that (a) shows bottlenecks forming and (b) predicts defects before they happen, and specifically handles **stations with little or no sensor data**.
- **Team:** technical/ML strength, 2–3 people. Goal: maximise chance of reaching Round 2 (prototype-ready stage).
- **Round 1 deliverables:** a submission deck in Accenture's official template (team details + ~200-word problem slide + ~200-word solution slide + a video slide) and a **video ≤3 minutes**. Rules require the official template, Arial font, file named `TeamName_IdeaName.pptx`, spell-check, and removing the instructions slide.

## The core idea (what makes this entry distinctive)
A live digital twin of the assembly line with **two engines**:
1. **Simulation engine** (discrete-event, SimPy) → predicts the next bottleneck using queue-growth signals + the **active-period method** (Roser & Nakano).
2. **ML defect engine** (gradient boosting, trained on the public **Bosch Production Line Performance** dataset) → flags parts likely to fail downstream.

**The differentiator — "a twin that knows its limits" (honesty under partial observability):**
- A line is unevenly instrumented; a naive twin is most confident exactly where it's most blind.
- Where a station has no sensor, **adaptive virtual sensors** infer its behaviour from surrounding data — **transfer learning** from similar stations (spatial) OR a **Kalman filter** from the line's own flow (temporal). Method-agnostic on the slide ("surrounding data") so it doesn't privilege one method.
- Every inferred value carries a **confidence score** (a *reliance index*, borrowed from semiconductor **virtual metrology**; **conformal prediction** for defect flags).
- **Effective Trust** = fuse *input trust* (is the data reliable?) × *model confidence* (is the prediction sure?). **Multiply, not average** — if either is weak, trust drops; averaging would hide bad data. Effective Trust **gates the action**: high-risk + low-trust → human verification; high-risk + high-trust → can auto-act.
- **Scope discipline:** model station flow, defect propagation, confidence-tagged state; deliberately **skip** part geometry, tool-wear physics, upstream supply chain.

## Verified facts / citable numbers (use only these — no fabricated metrics)
- Stopped automotive line ≈ **$2.3M/hour** (~$600/second), up ~50% since 2019 — **Siemens True Cost of Downtime 2024**. (Do NOT use the old $22k/min 2005 figure or the weakly-sourced $33k/min.)
- Defect escalation ≈ **1:10:100 rule** (design:assembly:field) — illustrative rule of thumb, say "roughly an order of magnitude at each stage," not a measured constant.
- A single recall can exceed **$10M** (2024 recall-cost survey).
- **Bosch Production Line Performance** dataset: ~1.18M parts, ~4,264 features, 52 stations, ~0.58% failures (extreme imbalance), ~81% missing values (missing-not-at-random). Metric: MCC.

## Key research anchors (for Round 2 depth)
- **Digital twin maturity:** Kritzinger et al. 2018 — Digital Model → Digital Shadow → Digital Twin. (Be honest: Round 2 build is a *shadow* with a path to full twin.)
- **Bottleneck:** Roser & Nakano (active-period method); Subramaniyan et al. 2018 (ARIMA + active-period *prediction* on automotive data).
- **Defect:** Bosch dataset; XGBoost/LightGBM, MCC objective; beware "leakage/magic" features (inflate Kaggle score, NOT deployable).
- **Sparse-sensor / honesty layer:** soft sensors/virtual metrology (Kadlec 2009; Cheng's Reliance Index + Global Similarity Index, IEEE T-SM 2008); transfer learning (Yao et al. 2023; Gao et al. CIRP 2020); conformal prediction (MAPIE library; Angelopoulos & Bates 2021 tutorial); imputation (BRITS, SAITS).
- **Honest novelty claim:** confidence-aware prediction is proven in *semiconductor* VM, NOT yet standard on *automotive assembly lines* — "we're transferring it," not "we invented it."

## Cautions to preserve (things a judge could probe)
- Kalman + flow-conservation for structural missingness = a Round-2 *hypothesis to test*, not a settled fact.
- XGBoost probabilities are miscalibrated — "model confidence" needs calibration (Platt/isotonic) or a bootstrap/conformal wrapper, not raw softmax.
- RAG/LLM explanation layer = nice demo polish, not the core; only claim RAG if a real retrieval corpus exists.
- Don't over-claim "no one does this" — say "rarely done explicitly for a production-line twin."
- **Never put fabricated performance numbers on slides** (e.g. 12–18% throughput, 0.94 AUC, 0.89 MAE). Only real numbers you can defend. Generate real ones in Round 2, then use them.

## Deck design decisions (the visual pitch deck — used for the VIDEO, not the template submission)
- 3 slides, industrial palette (deep navy #10233F, amber #F5A623 accent, red for defect/risk). Minimal text, one idea per slide.
- Slide 1 (problem): two cost cards ($2.3M/hr; ~100× costlier), hook "The problem isn't just the failure — it's not knowing where you're blind to it."
- Slide 2 (how it works): architecture diagram — physical line with a dashed sensorless Station 3, data up / alerts down, two engines with ML sublabels, "Our Edge" box led by tagline **"Virtual sensors + Effective Trust = a twin that knows its limits."**
- Slide 3 (why it matters + buildable): impact cards ("Catch it forming," "Contain the defect," "Act on trust, not just risk"), a "what we model / (dimmed) skip" line, three feasibility cards (Bosch data, free tooling, enterprise-relevant), closing "…and knowing when to say 'I'm not sure.'"

## Status of deliverables
- **Template submission deck** (`AIC_DigitalTwin_filled.pptx`): DONE — instructions slide removed; problem (181 words) and solution (190 words) filled in Arial. TODO by team: fill Team details slide (names/photos/college/stream/grad year — mandatory), rename file, spell-check.
- **Visual pitch deck** (`DigitalTwin_AI_Pitch.pptx` / video visuals): DONE — used as the video's on-screen slides + Q&A backup.
- **Video script** (`DigitalTwin_Video_Teleprompter.md`): DONE, but ran ~3:40 spoken — a **shorter ~2:50 version** was produced; use that (needs saving as final).
- **Framing/tone:** comparisons point at *existing technology* ("most digital twins stop here"), never at other participant teams; blame framed on the *system* (warning comes too late) not the plant teams.

## What's left / next steps
- Finalise the shortened video script and record (one clear speaker; show the 3 visuals in sync).
- Fill the Team details slide; rename; spell-check; submit.
- **Round 2 (if shortlisted):** build the prototype — SimPy line sim + live dashboard → active-period bottleneck detection → XGBoost defect model on Bosch (leakage-free, calibrated) → virtual-sensor + Effective Trust fusion → confidence-gated action matrix. Generate REAL metrics. The full 8-stage pipeline / fusion formula / action matrix (from teammate's work) is the Round-2 architecture, deliberately kept off Round-1 slides.