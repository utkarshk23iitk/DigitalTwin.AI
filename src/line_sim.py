"""
line_sim.py — Discrete-event simulation of a vehicle assembly line.

Models the line as a series of stations connected by finite buffers.
Each station has a (noisy) cycle time and can be in one of four states at
any moment:

    WORKING  — actively processing a unit
    BLOCKED  — finished a unit but the downstream buffer is full
    STARVED  — idle because the upstream buffer is empty
    DOWN     — random breakdown (optional)

These four states are exactly what the active-period bottleneck method needs.
Some stations are flagged `has_sensor=False` to simulate uneven instrumentation
— the digital twin must infer their behaviour rather than read it directly.

The simulation records a time-series log of every station's state and buffer
level, which downstream modules (bottleneck.py, virtual_sensor.py) consume.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import simpy


# ----------------------------- configuration ----------------------------- #

@dataclass
class StationConfig:
    """Static description of one station on the line."""
    index: int
    name: str
    mean_cycle: float                 # mean processing time (seconds)
    cv: float = 0.15                  # coefficient of variation of cycle time
    has_sensor: bool = True           # False => sensor-poor station
    failure_rate: float = 0.0         # prob. per unit of a random breakdown
    repair_time: float = 30.0         # mean repair duration (seconds)
    # ground-truth defect tendency for this station (used to synthesise labels)
    base_defect_rate: float = 0.01


@dataclass
class LineConfig:
    """Whole-line configuration."""
    stations: list[StationConfig]
    buffer_capacity: int = 5          # capacity of each inter-station buffer
    warmup: float = 200.0             # seconds discarded before logging
    seed: int = 42


# ------------------------------ state model ------------------------------- #

WORKING, BLOCKED, STARVED, DOWN = "WORKING", "BLOCKED", "STARVED", "DOWN"


@dataclass
class StationRuntime:
    """Mutable per-station runtime state + accumulated statistics."""
    cfg: StationConfig
    state: str = STARVED
    state_since: float = 0.0
    produced: int = 0
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

        # buffers[i] feeds station i (buffers[0] is the raw-material source).
        # A source process keeps buffers[0] topped up so station 0 never starves.
        self.buffers: list[simpy.Store] = [
            simpy.Store(self.env, capacity=cfg.buffer_capacity)
            for _ in range(len(cfg.stations))
        ]
        self.output_store = simpy.Store(self.env)  # finished units

        self.rt: list[StationRuntime] = [
            StationRuntime(cfg=s) for s in cfg.stations
        ]

        # event log: list of dicts {t, station, state, buffer_in, buffer_out}
        self.log: list[dict] = []
        self._unit_counter = 0

    # -- helpers ----------------------------------------------------------- #

    def _cycle_time(self, s: StationConfig) -> float:
        """Sample a positive, noisy cycle time (normal, clamped at 0.1)."""
        sd = s.mean_cycle * s.cv
        return max(0.1, random.gauss(s.mean_cycle, sd))

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

    def _station_proc(self, i: int):
        """Main loop for station i: take a unit, process it, pass it on."""
        s = self.cfg.stations[i]
        r = self.rt[i]
        in_buf = self.buffers[i]
        out_buf = self.buffers[i + 1] if i + 1 < len(self.buffers) else self.output_store

        while True:
            # --- wait for an input unit (STARVED while empty) ---
            if not in_buf.items:
                self._set_state(i, STARVED)
            unit = yield in_buf.get()

            # --- optional random breakdown ---
            if s.failure_rate and random.random() < s.failure_rate:
                self._set_state(i, DOWN)
                yield self.env.timeout(max(1.0, random.gauss(s.repair_time,
                                                             s.repair_time * 0.3)))

            # --- process the unit (WORKING) ---
            self._set_state(i, WORKING)
            yield self.env.timeout(self._cycle_time(s))
            r.produced += 1

            # synthesise a ground-truth defect event for this station
            if random.random() < s.base_defect_rate:
                unit["defect_flags"].append(i)

            # --- hand off downstream (BLOCKED if the buffer is full) ---
            if hasattr(out_buf, "capacity") and len(out_buf.items) >= out_buf.capacity:
                self._set_state(i, BLOCKED)
            yield out_buf.put(unit)

    # -- public API -------------------------------------------------------- #

    def run(self, until: float) -> "AssemblyLine":
        """Run the simulation until the given sim-time and return self."""
        self.env.process(self._source())
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
    Station 4 ('Paint-Inspect') and station 9 ('Torque-2') are sensor-poor.
    A deliberate slow station (station 6) creates a bottleneck to detect.
    """
    specs = [
        # idx, name,           mean_cycle, cv,   sensor, fail,  repair, defect
        ("Frame-Weld",    50, 0.12, True,  0.002, 40, 0.010),
        ("Body-Weld",     52, 0.12, True,  0.002, 40, 0.012),
        ("Door-Fit",      48, 0.15, True,  0.001, 30, 0.008),
        ("Seal",          46, 0.15, True,  0.001, 30, 0.006),
        ("Paint-Inspect", 55, 0.20, False, 0.001, 30, 0.020),  # sensor-poor
        ("Primer",        50, 0.15, True,  0.001, 30, 0.009),
        ("Topcoat",       72, 0.18, True,  0.003, 60, 0.014),  # BOTTLENECK (slow)
        ("Cure",          49, 0.12, True,  0.001, 30, 0.005),
        ("Torque-1",      50, 0.14, True,  0.002, 35, 0.011),
        ("Torque-2",      51, 0.16, False, 0.002, 35, 0.013),  # sensor-poor
        ("Electrical",    53, 0.15, True,  0.002, 35, 0.010),
        ("Final-QC",      47, 0.12, True,  0.001, 30, 0.007),
    ]
    stations = [
        StationConfig(index=i, name=n, mean_cycle=mc, cv=cv, has_sensor=hs,
                      failure_rate=fr, repair_time=rt, base_defect_rate=dr)
        for i, (n, mc, cv, hs, fr, rt, dr) in enumerate(specs)
    ]
    return AssemblyLine(LineConfig(stations=stations, buffer_capacity=4,
                                   warmup=200.0, seed=seed))


if __name__ == "__main__":
    line = default_line().run(until=3000)
    print(f"{'#':>2} {'station':<14} {'sensor':<7} {'util':>6} "
          f"{'block':>6} {'starve':>7} {'prod':>5} {'longActive':>10}")
    for s in line.station_stats():
        print(f"{s['index']:>2} {s['name']:<14} "
              f"{'yes' if s['has_sensor'] else 'NO':<7} "
              f"{s['utilisation']:>6} {s['blocked_frac']:>6} "
              f"{s['starved_frac']:>7} {s['produced']:>5} {s['longest_active']:>10}")
