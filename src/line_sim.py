"""
line_sim.py — Discrete-event simulation of a vehicle assembly line.

Models the line as a series of stations connected by finite buffers.
Each station has a (health-modulated) cycle time and can be in one of four
states at any moment:

    WORKING  — actively processing a unit
    BLOCKED  — finished a unit but the downstream buffer is full
    STARVED  — idle because the upstream buffer is empty
    DOWN     — random breakdown (optional)

These four states are exactly what the active-period bottleneck method needs.

Every station also carries a hidden **health state** H(t) in [0, 1] that
drifts down in rare, gradual episodes (never instantaneously) and back up
again. Health is the single shared cause behind three observable effects:
cycle time creeping up and breakdown risk rising (-> a *forming* bottleneck),
sensor readings drifting together across channels (-> what a virtual sensor
has to infer), and defect risk rising for units processed during the dip
(-> a *delayed* defect, often only caught several stations later). One
episode leaves a time-aligned trail across all three signals, which is what
makes trend-based prediction meaningful instead of independent coin-flips.

Stations are assigned one of three instrumentation tiers:

    A — fully instrumented: dense, regular sensor readings
    B — sensor-poor but correlated with OTHER stations that share a real
        physical/process cause (the same tooling rig, calibration source, or
        material batch) -- deliberately NOT assumed to be its line-neighbours.
        Being next to each other on the line doesn't imply a shared cause,
        and two stations far apart can share one (e.g. every station using
        the same torque-calibration rig). Correlation is expressed as shared
        loadings onto a small set of process factors (see PROCESS_FACTORS
        below), so its state can be inferred from whichever stations
        actually load on the same factor(s), wherever they sit on the line.
    C — sensor-poor and has ~zero loading on every process factor (no other
        station's readings help), but its own health is smooth over time and
        observed only sparsely/irregularly (a manual check), so its state
        must be tracked via a temporal filter (e.g. Kalman) instead

The simulation records several logs, consumed by downstream modules:

    log            — station state-change events (existing)
    buffer_log     — fixed-interval buffer occupancy samples (existing)
    health_log     — ground-truth H(t) per station (hidden from all models;
                     kept only to score detection lead time / false alarms)
    sensor_log     — observable channel readings (dense for tier A, sparse
                     for tier C, absent for tier B) -> virtual_sensor.py
    unit_log       — per-unit, per-station, per-channel snapshot (true value
                     + observed-or-None) -> the per-unit defect feature table
    unit_summary   — per-unit defect outcome: where it occurred vs. where it
                     was caught (an inspection station, possibly much later)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import simpy


# ----------------------------- configuration ----------------------------- #

@dataclass
class StationConfig:
    """Static description of one station on the line."""
    index: int
    name: str
    mean_cycle: float                 # mean processing time (seconds), at full health
    cv: float = 0.15                  # coefficient of variation of cycle time
    tier: str = "A"                   # "A" instrumented / "B" sensor-poor-correlated
                                      # / "C" sensor-poor-isolated (sparse manual checks)
    is_inspection: bool = False       # can this station catch/reveal a defect?
    failure_rate: float = 0.0         # prob. per unit of a random breakdown, at full health
    repair_time: float = 30.0         # mean repair duration (seconds)
    base_defect_rate: float = 0.01    # prob. per unit of a defect, at full health
    # Loading onto shared process factors (see PROCESS_FACTORS) -- what makes
    # this station's health correlate with OTHER stations that share a real
    # cause (tooling, calibration, material batch), regardless of where
    # either sits on the line. Empty/all-zero means no shared cause: this
    # station's health is driven purely by its own idiosyncratic episodes.
    factor_loadings: tuple[float, ...] = ()

    @property
    def has_sensor(self) -> bool:
        """Legacy convenience flag: True only for fully instrumented (tier A) stations."""
        return self.tier == "A"


@dataclass
class LineConfig:
    """Whole-line configuration."""
    stations: list[StationConfig]
    buffer_capacity: int = 5          # capacity of each inter-station buffer
    warmup: float = 200.0             # seconds discarded before logging
    seed: int = 42
    sample_interval: float = 5.0      # buffer-level polling period (0 = off)


# ------------------------------ state model ------------------------------- #

WORKING, BLOCKED, STARVED, DOWN = "WORKING", "BLOCKED", "STARVED", "DOWN"

# --------------------------- health/sensor dynamics ------------------------ #
# Every constant below is a chosen assumption, not a measured fact -- unlike
# the Bosch-calibrated constants in data/get_data.py, there is no real "live
# assembly line" telemetry to calibrate against. State them as such in the
# README/proposal rather than presenting them as measured.

HEALTH_TICK = 10.0          # seconds between health-process updates

# Shared process factors: each is an independent slow random walk. A
# station's exposure to factor k is StationConfig.factor_loadings[k] -- two
# stations correlate exactly to the extent they load on the same factor(s),
# with NO assumption that they sit near each other on the line. This is what
# makes tier-B stations (no sensor of their own) inferable from whichever
# *other* stations genuinely share a cause with them.
SEG_RHO = 0.985
SEG_SIGMA = 0.06
SEG_HEALTH_FACTOR = 1.5
# Dead zone: a station only takes a health penalty from its shared factor
# once the factor's adverse excursion clears this bar -- otherwise ordinary
# (non-degrading) factor wobble would chronically depress health for ~40% of
# time, destroying the "healthy almost always, rare sharp episodes" property.
# The factor itself is unchanged, so two same-loading stations still enter
# and exit a penalised state together -- correlation during a real dip is
# preserved; only how often "dip" triggers at all is being reined in here.
SEG_DEADZONE = 0.55

# Idiosyncratic degradation episodes: rare, and ramped -- never instantaneous.
# down -> hold -> up, each phase spanning many ticks, so a fault "builds"
# across dozens of readings instead of flipping in one step. Severity is
# drawn per episode, so many episodes are mild near-misses that never cross
# a failure/defect threshold -- not every dip becomes a labelled event.
EPISODE_PROB_PER_TICK = 0.00015
EPISODE_DROP_RANGE = (0.20, 0.60)     # fraction of health lost at the trough
EPISODE_DOWN_TICKS = (15, 45)         # ramp-down duration, in ticks
EPISODE_HOLD_TICKS = (5, 20)          # sustained-degraded duration, in ticks
EPISODE_UP_TICKS = (15, 45)           # recovery duration, in ticks

# How much degraded health (1 - H) inflates cycle time / breakdown / defect
# risk. Failure and defect scale with the square of (1 - H) so risk stays low
# until a station is genuinely unwell, then rises steeply near the trough.
CYCLE_HEALTH_FACTOR = 0.6
FAIL_HEALTH_FACTOR = 4.0
# Empirically tuned (not guessed): with only ~3% of tick-time genuinely
# degraded, a mild multiplier here left defects statistically
# indistinguishable from the healthy population (verified: defect-origin
# health was ~equal to population health, no learnable signal at all). This
# value was swept until degraded-health exposure accounts for a clear
# majority of degraded-origin defects (~45% of defects at health<0.85 vs
# ~3% of all time spent there) while the overall rate stays near Bosch's
# 0.58%. See PIPELINE.md for the before/after numbers.
DEFECT_HEALTH_FACTOR = 200.0

# Tier-C stations are read only via an irregular manual check -- this is the
# probability of a check happening on any given tick (~ every 2500s / 40min
# on average), matching the brief's "some stations rely entirely on manual
# checklists."
TIER_C_CHECK_PROB = 0.004

# Sensor channels: all three co-drift with the same (1 - health) term, so a
# fault shows up as correlated movement across channels, never a single
# giveaway column.
SENSOR_PARAMS = {
    "torque":      {"baseline": 50.0, "std": 1.5, "loading": 18.0},
    "vibration":   {"baseline": 0.20, "std": 0.03, "loading": 0.35},
    "temperature": {"baseline": 70.0, "std": 2.0, "loading": 15.0},
}


@dataclass
class StationRuntime:
    """Mutable per-station runtime state + accumulated statistics."""
    cfg: StationConfig
    state: str = STARVED
    state_since: float = 0.0
    produced: int = 0
    health: float = 1.0
    # accumulated time spent in each state (for utilisation / bottleneck calc)
    time_in_state: dict = field(default_factory=lambda: {
        WORKING: 0.0, BLOCKED: 0.0, STARVED: 0.0, DOWN: 0.0})
    # active-period tracking: current uninterrupted "active" (non-idle) run
    active_start: Optional[float] = None
    longest_active: float = 0.0


class AssemblyLine:
    """A SimPy model of a serial assembly line with finite buffers."""

    def __init__(self, cfg: LineConfig):
        self.cfg = cfg
        self.env = simpy.Environment()
        random.seed(cfg.seed)
        # Health/sensor dynamics use their own RNG stream so they don't
        # perturb the sequencing of the existing state/breakdown randomness.
        self._nrng = np.random.default_rng(cfg.seed + 1000)

        n = len(cfg.stations)
        # buffers[i] feeds station i (buffers[0] is the raw-material source).
        # A source process keeps buffers[0] topped up so station 0 never starves.
        self.buffers: list[simpy.Store] = [
            simpy.Store(self.env, capacity=cfg.buffer_capacity)
            for _ in range(n)
        ]
        self.output_store = simpy.Store(self.env)  # finished units

        self.rt: list[StationRuntime] = [
            StationRuntime(cfg=s) for s in cfg.stations
        ]

        # event log: list of dicts {t, station, state, buffer_in, buffer_out}
        self.log: list[dict] = []
        # buffer log: fixed-interval samples {t, levels: [...]}. The event log's
        # buffer readings are taken at state changes — i.e. just after a station
        # pulls a unit — so they are a biased sample of queue length. Anything
        # that reasons about buffer occupancy must use this instead.
        self.buffer_log: list[dict] = []
        # ground-truth health per station -- never fed to any model, kept only
        # to score detection lead time and false-alarm rate offline.
        self.health_log: list[dict] = []
        # observable sensor readings: dense (tier A), sparse (tier C), absent (tier B)
        self.sensor_log: list[dict] = []
        # per-unit, per-station, per-channel snapshot: true value always
        # present, observed value is None where the tier hides it.
        self.unit_log: list[dict] = []
        # per-unit defect outcome: where it happened vs. where it was caught.
        self.unit_summary: list[dict] = []
        self._unit_counter = 0

        # health-process bookkeeping. Loadings matrix (n_stations, n_factors):
        # row i is station i's exposure to each shared process factor.
        n_factors = max((len(s.factor_loadings) for s in cfg.stations), default=0)
        self._n_factors = n_factors
        self._loadings = np.zeros((n, n_factors))
        for i, s in enumerate(cfg.stations):
            if s.factor_loadings:
                self._loadings[i, :len(s.factor_loadings)] = s.factor_loadings
        self._factors = np.zeros(n_factors)
        self._episode = [
            {"phase": "none", "health": 1.0, "tick": 0, "trough": 1.0,
             "start_health": 1.0, "down_ticks": 0, "hold_ticks": 0, "up_ticks": 0}
            for _ in range(n)
        ]
        self._last_check_t = np.full(n, -1e9)
        self._inspection_stations = sorted(
            i for i, s in enumerate(cfg.stations) if s.is_inspection)
        if not self._inspection_stations:
            self._inspection_stations = [n - 1]

    # -- helpers ----------------------------------------------------------- #

    def _cycle_time(self, i: int) -> float:
        """Sample a positive, noisy cycle time (normal, clamped at 0.1).

        The mean inflates as health falls, so a degrading station visibly
        slows down before it breaks -- the leading indicator a bottleneck
        forecaster should learn to pick up on."""
        s = self.cfg.stations[i]
        health = self.rt[i].health
        eff_mean = s.mean_cycle * (1 + CYCLE_HEALTH_FACTOR * (1 - health))
        sd = eff_mean * s.cv
        return max(0.1, random.gauss(eff_mean, sd))

    def _true_sensor_value(self, i: int, channel: str, health: float) -> float:
        """The physical value a sensor at station i would read right now,
        regardless of whether this station's tier actually exposes it."""
        p = SENSOR_PARAMS[channel]
        return float(p["baseline"] + p["loading"] * (1 - health)
                     + self._nrng.normal(0, p["std"]))

    def _set_state(self, i: int, new_state: str):
        """Transition station i to a new state, accumulating elapsed time."""
        r = self.rt[i]
        now = self.env.now
        elapsed = now - r.state_since
        if now >= self.cfg.warmup:
            # only accumulate the portion after warmup
            start = max(r.state_since, self.cfg.warmup)
            r.time_in_state[r.state] += now - start
        # active-period bookkeeping: "active" = not STARVED (waiting for work)
        was_active = r.state != STARVED
        will_be_active = new_state != STARVED
        if will_be_active and not was_active:
            r.active_start = now
        if was_active and not will_be_active and r.active_start is not None:
            r.longest_active = max(r.longest_active, now - r.active_start)
            r.active_start = None
        r.state = new_state
        r.state_since = now
        if now >= self.cfg.warmup:
            self.log.append({
                "t": round(now, 2),
                "station": i,
                "state": new_state,
                "buffer_in": len(self.buffers[i].items),
                "buffer_out": (len(self.buffers[i + 1].items)
                               if i + 1 < len(self.buffers) else 0),
            })

    # -- health/sensor process ---------------------------------------------- #

    def _advance_episode(self, i: int):
        """Step station i's degradation episode state machine by one tick.

        none -> down -> hold -> up -> none. Each phase spans many ticks, so
        health moves gradually; there is no path that changes it in one step."""
        ep = self._episode[i]
        if ep["phase"] == "none":
            if self._nrng.random() < EPISODE_PROB_PER_TICK:
                ep["trough"] = 1.0 - self._nrng.uniform(*EPISODE_DROP_RANGE)
                ep["down_ticks"] = int(self._nrng.integers(*EPISODE_DOWN_TICKS))
                ep["hold_ticks"] = int(self._nrng.integers(*EPISODE_HOLD_TICKS))
                ep["up_ticks"] = int(self._nrng.integers(*EPISODE_UP_TICKS))
                ep["start_health"] = ep["health"]
                ep["phase"] = "down"
                ep["tick"] = 0
            return
        ep["tick"] += 1
        if ep["phase"] == "down":
            frac = min(1.0, ep["tick"] / max(1, ep["down_ticks"]))
            ep["health"] = ep["start_health"] + (ep["trough"] - ep["start_health"]) * frac
            if ep["tick"] >= ep["down_ticks"]:
                ep["phase"], ep["tick"] = "hold", 0
        elif ep["phase"] == "hold":
            ep["health"] = ep["trough"]
            if ep["tick"] >= ep["hold_ticks"]:
                ep["phase"], ep["tick"] = "up", 0
        elif ep["phase"] == "up":
            frac = min(1.0, ep["tick"] / max(1, ep["up_ticks"]))
            ep["health"] = ep["trough"] + (1.0 - ep["trough"]) * frac
            if ep["tick"] >= ep["up_ticks"]:
                ep["phase"], ep["tick"], ep["health"] = "none", 0, 1.0

    def _log_sensors(self, i: int, now: float, health: float):
        for ch in SENSOR_PARAMS:
            val = self._true_sensor_value(i, ch, health)
            self.sensor_log.append({"t": round(now, 2), "station": i,
                                    "channel": ch, "value": round(val, 4)})

    def _health_process(self):
        """Evolve every station's hidden health each tick and log what the
        line's actual instrumentation would observe of it."""
        n = len(self.rt)
        while True:
            now = self.env.now
            if self._n_factors:
                innovation = self._nrng.normal(0, SEG_SIGMA, self._n_factors)
                self._factors = SEG_RHO * self._factors + innovation
                # station i's shared-cause condition = its loadings . factor values,
                # NOT a function of its position i -- two stations with identical
                # loadings correlate perfectly regardless of how far apart they are.
                condition = self._loadings @ self._factors
            else:
                condition = np.zeros(n)

            for i in range(n):
                self._advance_episode(i)
                seg_penalty = max(0.0, -condition[i] - SEG_DEADZONE) * SEG_HEALTH_FACTOR
                health = float(np.clip(self._episode[i]["health"] - seg_penalty, 0.02, 1.0))
                self.rt[i].health = health

                if now >= self.cfg.warmup:
                    self.health_log.append({"t": round(now, 2), "station": i,
                                            "health_true": round(health, 4)})
                    tier = self.cfg.stations[i].tier
                    if tier == "A":
                        self._log_sensors(i, now, health)
                    elif tier == "C" and self._nrng.random() < TIER_C_CHECK_PROB:
                        self._log_sensors(i, now, health)
                        self._last_check_t[i] = now
            yield self.env.timeout(HEALTH_TICK)

    def _finalize_unit(self, unit: dict):
        """Resolve a finished unit's defect outcome: where it happened
        (possibly at a non-inspection station) vs. where it was first caught
        by an inspection station at or after that point."""
        flags = sorted(unit["defect_flags"])
        occurred = flags[0] if flags else None
        caught = None
        if occurred is not None:
            caught = next((s for s in self._inspection_stations if s >= occurred),
                         self._inspection_stations[-1])
        if self.env.now >= self.cfg.warmup:
            self.unit_summary.append({
                "unit_id": unit["id"],
                "response": int(bool(flags)),
                "defect_occurred_at": occurred,
                "defect_caught_at": caught,
            })

    # -- SimPy processes --------------------------------------------------- #

    def _source(self):
        """Keep the first buffer supplied with raw units, indefinitely."""
        buf = self.buffers[0]
        while True:
            if len(buf.items) < buf.capacity:
                self._unit_counter += 1
                yield buf.put({"id": self._unit_counter, "defect_flags": []})
            else:
                yield self.env.timeout(1.0)

    def _monitor(self):
        """Poll every buffer on a fixed interval, the way real SCADA would."""
        while True:
            if self.env.now >= self.cfg.warmup:
                self.buffer_log.append({
                    "t": round(self.env.now, 2),
                    "levels": [len(b.items) for b in self.buffers],
                })
            yield self.env.timeout(self.cfg.sample_interval)

    def _station_proc(self, i: int):
        """Main loop for station i: take a unit, process it, pass it on."""
        s = self.cfg.stations[i]
        r = self.rt[i]
        in_buf = self.buffers[i]
        out_buf = self.buffers[i + 1] if i + 1 < len(self.buffers) else self.output_store
        is_last = (i == len(self.cfg.stations) - 1)

        while True:
            # --- wait for an input unit (STARVED while empty) ---
            # Ask for the unit first: SimPy triggers the event synchronously if
            # one is already available, so `.triggered` distinguishes "got it
            # instantly" from "about to wait" exactly. Testing the buffer
            # beforehand instead would miss waits that begin inside the yield.
            get_evt = in_buf.get()
            if not get_evt.triggered:
                self._set_state(i, STARVED)
            unit = yield get_evt

            # --- optional random breakdown, more likely while unhealthy ---
            if s.failure_rate:
                eff_failure_rate = min(0.9, s.failure_rate
                                       * (1 + FAIL_HEALTH_FACTOR * (1 - r.health) ** 2))
                if random.random() < eff_failure_rate:
                    self._set_state(i, DOWN)
                    yield self.env.timeout(max(1.0, random.gauss(
                        s.repair_time, s.repair_time * 0.3)))

            # --- process the unit (WORKING) ---
            self._set_state(i, WORKING)
            health = r.health
            now = self.env.now
            if now >= self.cfg.warmup:
                for ch in SENSOR_PARAMS:
                    true_v = self._true_sensor_value(i, ch, health)
                    if s.tier == "A":
                        obs_v = true_v
                    elif s.tier == "C" and (now - self._last_check_t[i]) <= HEALTH_TICK:
                        obs_v = true_v
                    else:
                        obs_v = None
                    self.unit_log.append({
                        "unit_id": unit["id"], "station": i, "t": round(now, 2),
                        "channel": ch, "value_true": round(true_v, 4),
                        "value_observed": (round(obs_v, 4) if obs_v is not None else None),
                    })
            yield self.env.timeout(self._cycle_time(i))
            r.produced += 1

            # synthesise a ground-truth defect event for this station,
            # more likely while unhealthy -- never a flat, health-blind rate
            eff_defect_rate = min(0.9, s.base_defect_rate
                                  * (1 + DEFECT_HEALTH_FACTOR * (1 - health) ** 2))
            if random.random() < eff_defect_rate:
                unit["defect_flags"].append(i)

            # --- hand off downstream (BLOCKED if the buffer is full) ---
            put_evt = out_buf.put(unit)
            if not put_evt.triggered:
                self._set_state(i, BLOCKED)
            yield put_evt
            if is_last:
                self._finalize_unit(unit)

    # -- public API -------------------------------------------------------- #

    def run(self, until: float) -> "AssemblyLine":
        """Run the simulation until the given sim-time and return self."""
        self.env.process(self._source())
        self.env.process(self._health_process())
        if self.cfg.sample_interval > 0:
            self.env.process(self._monitor())
        for i in range(len(self.cfg.stations)):
            self.env.process(self._station_proc(i))
        self.env.run(until=until)
        # close out any open active periods at end-of-run
        for r in self.rt:
            if r.active_start is not None:
                r.longest_active = max(r.longest_active,
                                       self.env.now - r.active_start)
        return self

    def station_stats(self) -> list[dict]:
        """Summary stats per station over the (post-warmup) horizon."""
        horizon = max(1e-9, self.env.now - self.cfg.warmup)
        out = []
        for i, r in enumerate(self.rt):
            tis = r.time_in_state
            out.append({
                "index": i,
                "name": r.cfg.name,
                "tier": r.cfg.tier,
                "has_sensor": r.cfg.has_sensor,
                "produced": r.produced,
                "utilisation": round(tis[WORKING] / horizon, 3),
                "blocked_frac": round(tis[BLOCKED] / horizon, 3),
                "starved_frac": round(tis[STARVED] / horizon, 3),
                "down_frac": round(tis[DOWN] / horizon, 3),
                "longest_active": round(r.longest_active, 2),
            })
        return out


# --------------------------- default line factory ------------------------- #

def default_line(seed: int = 42) -> AssemblyLine:
    """
    A readable 12-station line spanning body / paint / final assembly.

    Three shared process factors drive correlated health, assigned by real
    shared cause rather than physical adjacency:

        factor 0 "body material batch"    -- stations 0, 1, 3 (full), 2 (partial)
        factor 1 "paint-booth environment" -- stations 5, 6, 7
        factor 2 "torque-calibration rig"  -- stations 8, 9, 10, and 2 (partial)

    Factor 2 deliberately links station 2 (Door-Fit, early in body
    construction) with stations 8-10 (Torque-1/Torque-2/Electrical, in final
    assembly) -- they are nowhere near each other on the line, but share the
    same torque-tooling calibration source, so their health genuinely
    correlates. This is the point: shared cause, not shared position.

    Station 4 ('Paint-Inspect') is tier C: ~zero loading on every factor
    (nothing correlates with it) and sensor-poor, read only via sparse manual
    checks -- it doubles as a mid-line inspection point, so its state must
    come from its own smooth-over-time behaviour (Kalman), not from anyone
    else's readings. Station 9 ('Torque-2') is tier B: sensor-poor but
    loaded on factor 2 along with 8, 10, and 2 -- inferable from THEM.
    Station 11 ('Final-QC') is the terminal inspection point, catching
    anything not already caught earlier. A deliberate slow station
    (station 6) creates a bottleneck to detect.
    """
    # base_defect_rate values are deliberately tiny (~0.34% combined per unit
    # at full health, summed across all 12 stations) and scaled DOWN together
    # with DEFECT_HEALTH_FACTOR raised way up (see that constant's comment) --
    # empirically swept so degraded-health periods, not baseline noise,
    # account for the visible defect rate. Overall rate lands near Bosch's
    # 0.58% once episodes are folded in.
    #                    name,       mean_cycle, cv,   tier, fail,  repair, defect,   inspect, loadings(f0,f1,f2)
    specs = [
        ("Frame-Weld",    50, 0.12, "A", 0.002, 40, 0.00025, False, (1.0, 0.0, 0.0)),
        ("Body-Weld",     52, 0.12, "A", 0.002, 40, 0.00029, False, (1.0, 0.0, 0.0)),
        ("Door-Fit",      48, 0.15, "A", 0.001, 30, 0.00021, False, (0.7, 0.0, 0.5)),  # shares 2 factors
        ("Seal",          46, 0.15, "A", 0.001, 30, 0.00017, False, (1.0, 0.0, 0.0)),
        ("Paint-Inspect", 55, 0.20, "C", 0.001, 30, 0.00063, True,  (0.0, 0.0, 0.0)),  # isolated, tier C
        ("Primer",        50, 0.15, "A", 0.001, 30, 0.00021, False, (0.0, 1.0, 0.0)),
        ("Topcoat",       72, 0.18, "A", 0.003, 60, 0.00038, False, (0.0, 1.0, 0.0)),  # BOTTLENECK (slow)
        ("Cure",          49, 0.12, "A", 0.001, 30, 0.00013, False, (0.0, 1.0, 0.0)),
        ("Torque-1",      50, 0.14, "A", 0.002, 35, 0.00025, False, (0.0, 0.0, 1.0)),
        ("Torque-2",      51, 0.16, "B", 0.002, 35, 0.00050, False, (0.0, 0.0, 1.0)),  # tier B, correlated
        ("Electrical",    53, 0.15, "A", 0.002, 35, 0.00025, False, (0.0, 0.0, 0.6)),
        ("Final-QC",      47, 0.12, "A", 0.001, 30, 0.00017, True,  (0.0, 0.0, 0.0)),  # terminal inspection
    ]
    stations = [
        StationConfig(index=i, name=n, mean_cycle=mc, cv=cv, tier=tier,
                      failure_rate=fr, repair_time=rt, base_defect_rate=dr,
                      is_inspection=insp, factor_loadings=load)
        for i, (n, mc, cv, tier, fr, rt, dr, insp, load) in enumerate(specs)
    ]
    return AssemblyLine(LineConfig(stations=stations, buffer_capacity=4,
                                   warmup=200.0, seed=seed))


if __name__ == "__main__":
    line = default_line().run(until=3000)
    print(f"{'#':>2} {'station':<14} {'tier':<5} {'util':>6} "
          f"{'block':>6} {'starve':>7} {'prod':>5} {'longActive':>10}")
    for s in line.station_stats():
        print(f"{s['index']:>2} {s['name']:<14} {s['tier']:<5} "
              f"{s['utilisation']:>6} {s['blocked_frac']:>6} "
              f"{s['starved_frac']:>7} {s['produced']:>5} {s['longest_active']:>10}")
    print(f"\nhealth_log: {len(line.health_log)} rows  "
          f"sensor_log: {len(line.sensor_log)} rows  "
          f"unit_log: {len(line.unit_log)} rows  "
          f"units finished: {len(line.unit_summary)}  "
          f"defects: {sum(u['response'] for u in line.unit_summary)}")
