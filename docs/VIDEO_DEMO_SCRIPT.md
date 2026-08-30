# Twinly Video Demonstration Script

Target length: 3 minutes 20 seconds

## Before Recording

1. Launch from the repository root with the Gemini key exported in the shell:

   ```bash
   set -a
   source .env
   set +a
   streamlit run app2.py
   ```

   Do not show the terminal or API key in the recording. If Gemini is unavailable,
   Twinly's grounded local fallback can deliver the same dashboard-specific answers.

2. Open the dashboard, pause playback, and move the simulation timeline to `3000` seconds.
3. Keep the browser at 100% zoom and the sidebar open.
4. Clear any alert workflow state by refreshing the browser session before recording.
5. Record at 1080p. Move the cursor slowly and pause briefly after each click.

## Scene 1 - The Problem and Promise (0:00-0:18)

**On screen:** Hero, KPI cards, and the top of Live Digital Twin at `t=3000s`.

**Narration:**

> Manufacturing teams usually discover a bottleneck after production has already
> slowed, or a defect after it has travelled downstream. Twinly is a trust-aware
> digital twin that reconstructs the line in real time, detects operational risk,
> and decides whether the evidence is reliable enough for automation.

## Scene 2 - Live Production State (0:18-0:48)

**On screen actions:**

- Point to the simulation clock.
- Point to `34` completed production units, throughput near `52/hour`, and `25` work in progress.
- Scroll through the station pipeline and pause on S06 Topcoat.

**Narration:**

> This is not a static dashboard. At exactly 3,000 simulation seconds, Twinly only
> uses events that have already occurred. We have 34 completed production units,
> a rolling throughput of about 52 per hour, and 25 units currently in progress.
> Every station card combines its operational state, utilization, condition
> estimate, cycle evidence, buffer pressure, sensor coverage, and risk.

## Scene 3 - Bottleneck Intelligence (0:48-1:18)

**On screen actions:**

- Open **Bottleneck Intelligence**.
- Point to S06 Topcoat in the Current Bottleneck card.
- Point to utilization near 96%, the full `4/4` input queue, and the queue evidence chart.
- Briefly show the station ranking.

**Narration:**

> Twinly identifies S06 Topcoat as the sustained constraint. It is operating near
> 96 percent utilization with a full input queue. This is not chosen from one noisy
> threshold. The sustained ranking combines utilization, blocking, starvation,
> downstream starvation, and observable condition evidence. A separate emerging
> score uses queue growth, cycle drift, and buffer pressure, giving operators both
> the current constraint and the next place to watch.

## Scene 4 - Risk x Effective Trust (1:18-2:02)

**On screen actions:**

- Open **Defect Intelligence**.
- Pause on the four-quadrant Risk x Effective Trust matrix.
- Select **Production Unit U012**.
- Point to its risk, Model Confidence, Input Trust, Effective Trust, and action.
- Point to the Recommended Next Action panel.

**Narration:**

> Defect risk alone is not enough for a safe decision. Twinly separates Model
> Confidence from Input Trust, then multiplies them into Effective Trust. The matrix
> makes the policy explicit: high risk with strong trust can be auto-held; high risk
> with weak trust requires human verification; low-trust evidence remains under
> monitoring instead of receiving a confident pass.
>
> For Production Unit U012, risk is about 41 percent, below the 61 percent operating
> threshold. Input Trust is high, but the model is not decisive, so Effective Trust
> falls to about 7 percent. Twinly recommends continued monitoring and a targeted
> inspection lead rather than inventing a defect alarm. The suspected origin remains
> clearly labelled as evidence for review, not a proven root cause.

## Scene 5 - Partial Observability (2:02-2:28)

**On screen actions:**

- Open **Station Health**.
- Scroll to **Sensor coverage & trust**.
- Point to S04 and S09.
- Show the virtual-sensor table and trust calculation flow.

**Narration:**

> Real production lines are unevenly instrumented. S04 is reconstructed from its own
> sparse history using a temporal estimator, while S09 is inferred spatially from
> stations sharing the same physical process factor. Every inferred value carries
> confidence, and that confidence travels downstream into Input Trust. Twinly never
> presents an inferred reading as if it were directly measured.

## Scene 6 - Operator Workflow (2:28-2:50)

**On screen actions:**

- Return to **Live Line**.
- In the alert feed select the critical S03 alert.
- Assign it to **Maintenance**.
- Click **Under review**.
- Pause on the updated workflow status.

**Narration:**

> Intelligence becomes useful only when a team can act on it. The live alert feed
> supports ownership and workflow directly. Here, the critical S03 condition is
> assigned to Maintenance and moved under review, while the underlying evidence and
> simulation clock remain unchanged.

## Scene 7 - Grounded AI Copilot (2:50-3:13)

**On screen actions:**

- Open **AI Copilot**.
- Click the suggested question: **Why is S06 the current bottleneck?**
- Show the answer.
- Turn on **Show the exact structured context sent to the LLM** briefly.

**Narration:**

> The Copilot does not calculate plant metrics. Those are produced locally by the
> twin engine first. Gemini receives only this compact structured snapshot and turns
> it into operator language. The exact context is inspectable, and if the cloud call
> fails, Twinly automatically uses its deterministic grounded fallback without
> affecting the dashboard.

## Scene 8 - Closing Impact (3:13-3:25)

**On screen:** Return to the hero and KPI area, or hold on Operational Perspectives.

**Narration:**

> Twinly does more than predict risk. It shows where the line is constrained, where
> evidence is incomplete, what action is supported, and when a human must remain in
> control. That is how a digital twin becomes operationally useful and responsibly
> automatable.

**Final title card:**

> Twinly - See the line. Understand the risk. Act with confidence.

## Recording Notes

- Do not call the simulation a real factory deployment.
- Do not claim U012 is defective; it is a monitored production unit.
- Do not describe the suspected origin as confirmed causation.
- Say "condition estimate," not "measured health."
- Avoid displaying `.env`, API keys, source terminals, or internal notes.
- If time is limited, remove Scene 6 and shorten Scene 5 for a 2 minute 45 second cut.

## Optional Technical Architecture Deep-Dive

Use this approximately 2.5-minute paragraph as an appendix when the judging format
allows a longer technical video. For a strict three-minute limit, extract only the
sentences about the chronological split, virtual sensors, and Effective Trust.

> Technically, Twinly is organized as a layered event-processing architecture. A
> SimPy discrete-event simulation currently acts as the PLC, SCADA, and MES source,
> producing timestamped station states, finite-buffer levels, production-unit visits,
> sensor readings, and quality outcomes through schemas that can later be populated by
> real plant adapters. Every live calculation is an as-of calculation: a value is used
> only when its timestamp is earlier than or equal to the playback clock. The
> bottleneck engine separates sustained constraints from emerging ones. Its sustained
> score adds utilization and blocked time, subtracts starvation, adds one-and-a-half
> times downstream starvation, and applies a small observable-condition penalty. A
> separate emerging score combines positive queue slope, recent cycle-time drift, and
> buffer pressure, so a transient growing queue is not confused with the line's true
> long-run constraint. For defect prediction, trailing sensor mean, standard deviation,
> and slope features capture operating level, instability, and drift. Sensor-poor S09
> is reconstructed through spatial linear regression from stations sharing the same
> process factor, while isolated S04 uses a Kalman filter that alternates prediction and
> correction over sparse observations. Each inferred value is paired with confidence.
> A regularized XGBoost model consumes the measured and inferred evidence because tree
> boosting handles nonlinear interactions, rare events, and structured missingness
> without pretending that every absent sensor is zero. The data is split chronologically
> into 40 fitting sessions, 10 validation sessions, and 10 untouched final-test sessions;
> the final model reaches a ROC AUC of 0.715 on 13,659 test production units containing
> 76 defects. Accuracy is deliberately not used as the headline because defects are
> rare. Input Trust is an importance-weighted average of the reliability of the evidence
> actually present for each production unit. Model Confidence measures how decisively the
> risk score lies away from the learned operating threshold. Twinly then computes
> Effective Trust as Input Trust multiplied by Model Confidence. Multiplication ensures
> that one weak factor cannot be hidden by one strong factor, and the resulting Risk x
> Trust matrix deterministically selects AUTO-ACT, HUMAN-VERIFY, MONITOR, or PASS. Only
> after these calculations are complete does Gemini receive a compact JSON snapshot to
> explain the result; it never creates a plant metric, changes a score, or overrides an
> operational action.
