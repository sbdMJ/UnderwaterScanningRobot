# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Trajectory logging + scan-quality metrics for the PKRC wallscan task.

Pure torch/numpy (no isaaclab / pxr imports), so the metric math is unit-testable
without the sim app — same convention as geometry.py / sensors.py /
scan_state_machine.py, and the reason ``__init__.py`` keeps those lazy.

Metric definitions mirror the env's own semantics, so the numbers mean the same
thing the reward/termination logic means:

* ``tilt_deg`` — ``arccos(body_up_world_z)``, exactly the quantity ``cfg.tilt_scale``
  penalises (wallscan_env ``"tilt"`` reward term). Reported separately for the heave
  legs and the sway legs because sidestep thrust heels the hull via the TAM My
  coupling, so the two legs are expected to differ.
* ``crab_deg`` — ``|wrap_to_pi(yaw - theta)|``. After the spin search the env sets
  ``_yaw_ref_cur = theta_gt`` (a cylinder's wall normal IS the outward radial), so
  this is precisely the heading error the reward tracks. It grows if and only if the
  vehicle translates along the wall without rotating with the curvature — crab-walking.
* ``s_hat_err`` — ``|s - s_gt|``: the DVL-dead-reckoned / UKF-M-corrected estimate
  against the geometric arc length. This is drift, so it is reported in cm.
* heave speed — ``|dz/dt|`` inside DESCEND/ASCEND. Config target is
  ``scan.ref_step / step_dt`` (the reference ramp rate the policy is paced to).
* sway speed — ``|ds_gt/dt|`` inside SWAY_A/SWAY_B. Config target is
  ``scan.ref_step_s / step_dt``, falling back to ``ref_step`` when unset.

Two exclusions keep the numbers honest:

1. Steps with ``searching`` set are dropped from every metric — during the initial
   spin search no scan is in progress, the estimate is frozen, and z is pinned at
   the operating ceiling.
2. Speeds only use step pairs that stay inside one phase AND one episode, so a
   phase advance or an auto-reset never contributes a bogus difference. Isaac Lab
   resets terminated envs inside ``step()``, so the row after a ``done`` already
   belongs to the next episode — ``episode`` is derived accordingly.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .scan_state_machine import ASCEND, DESCEND, SWAY_A, SWAY_B

HEAVE_PHASES: tuple[int, ...] = (DESCEND, ASCEND)
SWAY_PHASES: tuple[int, ...] = (SWAY_A, SWAY_B)

#: Per-step, per-env channels the logger stores. ``episode`` is derived, not recorded.
FIELDS: tuple[str, ...] = (
    "phase",
    "cycles",
    "searching",
    "done",
    "terminated",
    "time_out",
    "term_collided",
    "term_oob",
    "term_tilted",
    "x",
    "y",
    "z",
    "yaw",
    "theta",
    "tilt_deg",
    "s",
    "s_gt",
    "s_ref",
    "z_ref",
    "wall_dist",
    "clearance",
)


def wrap_to_pi(x):
    """Wrap radians to [-pi, pi). Works for torch tensors and numpy arrays alike."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


class TrajectoryLog:
    """Accumulates per-step ``[N]`` channels on the CPU, then yields ``[T, N]`` arrays.

    Kept deliberately dumb: the caller is responsible for handing over values that
    cost no extra RNG draws. In particular ``play.py`` must NOT call the env's
    ``_read_state()`` to fill these in — ``apply_sensors`` draws from the global
    torch RNG, so an extra call would shift the noise stream and change the run.
    """

    def __init__(self) -> None:
        self._cols: dict[str, list[torch.Tensor]] = {f: [] for f in FIELDS}

    def __len__(self) -> int:
        return len(self._cols["z"])

    def record(self, **channels: torch.Tensor) -> None:
        missing = set(FIELDS) - set(channels)
        extra = set(channels) - set(FIELDS)
        if missing or extra:
            raise ValueError(f"TrajectoryLog.record: missing={sorted(missing)} unexpected={sorted(extra)}")
        for name, value in channels.items():
            self._cols[name].append(torch.as_tensor(value).detach().to("cpu", torch.float32).clone())

    def as_arrays(self, step_dt: float) -> dict[str, Any]:
        """Stack into ``[T, N]`` numpy arrays, adding derived ``t`` and ``episode``."""
        import numpy as np

        if len(self) == 0:
            raise ValueError("TrajectoryLog is empty — nothing was recorded")
        out: dict[str, Any] = {name: torch.stack(col).numpy() for name, col in self._cols.items()}

        out["episode"], out["ended_episode"] = episode_indices(out["done"])
        out["t"] = (np.arange(len(self), dtype=np.float64) * step_dt)[:, None] * np.ones(
            (1, out["z"].shape[1])
        )
        return out


def episode_indices(done):
    """``(episode, ended_episode)`` from a ``[T, N]`` done channel.

    Two different indices because a done row answers two different questions:

    * ``episode`` labels the STATE in that row. Isaac Lab resets terminated envs inside
      ``step()``, and both ``play.py`` and ``run_wallscan_mpc.py`` read the state after
      ``step()`` returns, so the row carrying ``done`` already holds the POST-reset pose. It
      therefore belongs to the NEXT episode: ``cumsum(done)``.
    * ``ended_episode`` labels the episode that FINISHED at that row, which is what
      ``terminated``/``time_out``/``term_*`` describe: ``cumsum(done) - done``.

    Until 2026-07-31 only the second form existed and was used for both, so one post-reset row
    leaked into every scored episode. Measured effect on a 9000-row log: negligible for means
    (a 148 deg crab spike contributed 0.016 deg) but it made max-type statistics meaningless,
    and it grows with the number of resets in the window.
    """
    import numpy as np

    d = np.asarray(done) > 0.5
    cum = np.cumsum(d, axis=0)
    return cum.astype(np.int64), (cum - d).astype(np.int64)


def _with_episode_indices(traj: dict[str, Any]) -> dict[str, Any]:
    """Shallow copy of ``traj`` with both episode indices recomputed from ``done``.

    Recomputed rather than trusted so that ``.npz`` files written before the fix score
    correctly without being regenerated.
    """
    out = dict(traj)
    out["episode"], out["ended_episode"] = episode_indices(traj["done"])
    return out


def _rows_since_episode_start(traj: dict[str, Any]):
    """``[T, N]`` count of rows elapsed since the current episode began."""
    import numpy as np

    ep = traj["episode"]
    n_t, n_env = ep.shape
    out = np.zeros((n_t, n_env), np.int64)
    idx = np.arange(n_t)
    for e in range(n_env):
        starts = np.concatenate(([0], np.flatnonzero(np.diff(ep[:, e]) != 0) + 1))
        first = starts[np.searchsorted(starts, idx, side="right") - 1]
        out[:, e] = idx - first
    return out


def _select(traj: dict[str, Any], episode: int | None, settle_s: float = 0.0,
            step_dt: float = 0.02):
    """Boolean ``[T, N]`` mask of rows that count toward the metrics.

    ``settle_s`` drops that many seconds from the START of each episode. The spawn hands the
    vehicle an arbitrary heading error (up to 180 deg, since Stage3 spawns level at a random
    bearing), and correcting it is a transient, not scan performance. Measured 2026-07-31 on
    three seeds: full-window crab was 0.064 / 0.556 / 2.109 deg, but the last 50% of the same
    runs was 0.080 / 0.029 / 0.042 — the entire seed-to-seed spread was the approach. Seed 2
    alone averaged 18.7 deg over its first 20 s and 0.000-0.140 deg thereafter.

    It also removes an asymmetry against the RL baseline: ``play.py`` excludes the policy's
    ~10 s spin search through the ``searching`` flag, while an NMPC run that skips the search
    has nothing excluded, so its worst seconds were being averaged in.
    """
    import numpy as np

    sel = traj["searching"] < 0.5
    if episode is not None:
        sel = sel & (traj["episode"] == episode)
    sel = sel.astype(bool) if not isinstance(sel, np.ndarray) else sel
    if settle_s > 0.0:
        sel = sel & (_rows_since_episode_start(traj) * step_dt >= settle_s)
    return sel


def _phase_mask(traj: dict[str, Any], phases: tuple[int, ...]):
    import numpy as np

    ph = np.rint(traj["phase"]).astype(np.int64)
    m = np.zeros(ph.shape, dtype=bool)
    for p in phases:
        m |= ph == p
    return m


def _mean(values, mask) -> float:
    return float(values[mask].mean()) if mask.any() else float("nan")


def _legs(traj: dict[str, Any], phases: tuple[int, ...], sel):
    """Yield ``(env, start, stop)`` for each COMPLETE leg of one of ``phases``.

    A leg is a maximal run of consecutive rows with the same phase in the same episode. It
    counts as complete only when BOTH ends are phase transitions inside that episode, which
    drops the two cases that otherwise poison a speed average:

    * the first leg of the log, which began before logging started, and
    * the last leg, cut off by the log window or by the episode ending.

    Why this matters (measured 2026-07-31 on ``trajectory_s0_gt.npz``): four complete vertical
    legs ran at 0.1954-0.1957 m/s against a 0.20 target, but the truncated fifth leg had a net
    rate of 0.066 m/s against a path-length rate of 0.452 — the vehicle was oscillating while
    settling at a phase boundary. Averaging ``|dz/dt|`` over every vertical row therefore
    reported 0.233 m/s, a 16% "overshoot" that did not exist. Four separate control changes
    were tried against that phantom before the metric itself was checked.
    """
    import numpy as np

    ph = np.rint(traj["phase"]).astype(np.int64)
    ep = traj["episode"]
    want = _phase_mask(traj, phases)
    n_t, n_env = ph.shape
    for e in range(n_env):
        cuts = np.flatnonzero((ph[1:, e] != ph[:-1, e]) | (ep[1:, e] != ep[:-1, e])) + 1
        edges = np.concatenate(([0], cuts, [n_t]))
        for i in range(len(edges) - 1):
            a, b = int(edges[i]), int(edges[i + 1])
            if b - a < 2 or not want[a, e] or not sel[a:b, e].all():
                continue
            # Completeness is about the KIND of boundary, not the run's index. Both ends must
            # be phase transitions inside one episode: the log start/end and any auto-reset
            # leave a leg unfinished. Indexing by "first and last run" is not equivalent and
            # was measured to fail on exactly the case it was meant to catch -- a time-out on
            # the final row splits off a 1-row run, so the genuinely truncated leg before it
            # was no longer "last" and slipped through, dragging the aggregate heave rate from
            # 0.1957 to 0.1706 m/s.
            left_ok = a > 0 and ep[a, e] == ep[a - 1, e]
            right_ok = b < n_t and ep[b, e] == ep[b - 1, e]
            if not (left_ok and right_ok):
                continue
            yield e, a, b


def _leg_speed(traj: dict[str, Any], channel: str, phases: tuple[int, ...], sel, step_dt: float,
               path_length: bool = False) -> float:
    """Scan speed over complete legs: total distance travelled / total leg time.

    ``path_length=False`` (default) uses each leg's NET displacement, which is the scan rate
    the mission cares about. ``path_length=True`` sums ``|d channel|`` instead, counting
    back-and-forth motion as progress; the ratio of the two is a useful oscillation indicator,
    which is why both are reported.
    """
    import numpy as np

    dist = 0.0
    secs = 0.0
    for e, a, b in _legs(traj, phases, sel):
        col = traj[channel][a:b, e]
        dist += float(np.abs(np.diff(col)).sum()) if path_length else float(abs(col[-1] - col[0]))
        secs += (b - a - 1) * step_dt
    return dist / secs if secs > 0 else float("nan")


def _tangential_drift(traj: dict[str, Any], sel) -> float:
    """Mean |arc displacement| inside a complete VERTICAL leg, in metres.

    Controller-independent coverage measure. During DESCEND/ASCEND the scan plan says the
    vehicle holds its arc position and only changes depth, so the ideal is exactly zero for
    every run, whatever reference its own state machine happened to generate. That matters:
    ``|s_gt - s_ref|`` compares a controller against ITS OWN plan, and the state machine only
    advances a phase once the error is small, so a lagging controller gets a lagging reference
    and is flattered. This quantity has no such loophole.

    Measured 2026-08-08 on a stress-DR run whose every other metric looked healthy (wall 4.7 cm,
    both leg speeds on target, 2.00 cycles): the first DESCEND moved 1.36 m along the wall.
    """
    import numpy as np

    d = [abs(float(traj["s_gt"][b - 1, e] - traj["s_gt"][a, e]))
         for e, a, b in _legs(traj, HEAVE_PHASES, sel)]
    return float(np.mean(d)) if d else float("nan")


def _sway_step_error(traj: dict[str, Any], sel, sway_step: float | None) -> float:
    """Mean |net arc advance over a complete SWAY leg - sway_step|, in metres.

    The other half of coverage: a sway leg is supposed to advance the scan by exactly one
    ``sway_step``. Overshooting duplicates wall, undershooting misses it. Also independent of
    the controller's own reference.
    """
    import numpy as np

    if sway_step is None:
        return float("nan")
    d = [abs(abs(float(traj["s_gt"][b - 1, e] - traj["s_gt"][a, e])) - sway_step)
         for e, a, b in _legs(traj, SWAY_PHASES, sel)]
    return float(np.mean(d)) if d else float("nan")


def _leg_count(traj: dict[str, Any], phases: tuple[int, ...], sel) -> int:
    return sum(1 for _ in _legs(traj, phases, sel))


def _rate_metrics(traj: dict[str, Any], sel, step_dt: float, d_ref: float | None,
                  sway_step: float | None = None) -> dict[str, Any]:
    """The metrics that are averages/rates over the selected rows.

    Split out so the settled window reports exactly the same quantities as the full window
    rather than a hand-picked subset. Cumulative and per-episode figures (cycles, episode
    length, terminations) are deliberately NOT here: they need the whole episode.
    """
    import numpy as np

    heave = sel & _phase_mask(traj, HEAVE_PHASES)
    sway = sel & _phase_mask(traj, SWAY_PHASES)
    crab = np.degrees(np.abs(wrap_to_pi(traj["yaw"] - traj["theta"])))
    s_err = np.abs(traj["s"] - traj["s_gt"])
    # ARC-LENGTH TRACKING error: how far the vehicle actually is from the arc position the scan
    # asked for. Distinct from `s_hat_err`, which compares the ESTIMATE to ground truth -- a
    # perfect estimator still leaves this untouched. Added 2026-08-08 after a trajectory plot
    # showed stress-DR runs drifting tangentially through the vertical legs (first DESCEND moved
    # 1.36 m along the wall against a reference of 0.00) while every existing metric looked
    # healthy: wall distance 4.7 cm, both leg speeds on target, 2.00 cycles.
    #
    # For a wall INSPECTION this is coverage error, so it belongs next to standoff rather than in
    # a diagnostic footnote: scanning a metre away from the planned arc is missed or duplicated
    # wall either way. Reported with RMS/p95/max as well, because the repo has been bitten before
    # by a mean that hid the outliers (see the episode-indexing note above).
    s_track = np.abs(traj["s_gt"] - traj["s_ref"])
    out = {
        "scored_steps": int(sel.sum()),
        "tilt_heave_deg": _mean(traj["tilt_deg"], heave),
        "tilt_sway_deg": _mean(traj["tilt_deg"], sway),
        "tilt_deg": _mean(traj["tilt_deg"], sel),
        "crab_deg": _mean(crab, sel),
        "s_hat_err_cm": _mean(s_err, sel) * 100.0,
        "s_track_err_cm": _mean(s_track, sel) * 100.0,
        "s_track_rms_cm": float(np.sqrt(np.mean(np.square(s_track[sel])))) * 100.0 if sel.any() else float("nan"),
        "s_track_p95_cm": float(np.percentile(s_track[sel], 95)) * 100.0 if sel.any() else float("nan"),
        "s_track_max_cm": float(s_track[sel].max()) * 100.0 if sel.any() else float("nan"),
        # Reference-free coverage pair; prefer these when comparing controllers.
        "heave_drift_cm": _tangential_drift(traj, sel) * 100.0,
        "sway_step_err_cm": _sway_step_error(traj, sel, sway_step) * 100.0,
        "heave_speed_mps": _leg_speed(traj, "z", HEAVE_PHASES, sel, step_dt),
        "sway_speed_mps": _leg_speed(traj, "s_gt", SWAY_PHASES, sel, step_dt),
        # Path-length rates: equal to the net rates when the leg is monotone, larger when the
        # vehicle oscillates. A big gap means the net figure is hiding ripple, not that the
        # scan is fast.
        "heave_path_speed_mps": _leg_speed(traj, "z", HEAVE_PHASES, sel, step_dt, path_length=True),
        "sway_path_speed_mps": _leg_speed(traj, "s_gt", SWAY_PHASES, sel, step_dt, path_length=True),
        "heave_legs_scored": _leg_count(traj, HEAVE_PHASES, sel),
        "sway_legs_scored": _leg_count(traj, SWAY_PHASES, sel),
    }
    if d_ref is not None:
        out["wall_dist_err_cm"] = _mean(np.abs(traj["wall_dist"] - d_ref), sel) * 100.0
    return out


def compute_metrics(
    traj: dict[str, Any],
    *,
    step_dt: float,
    d_ref: float | None = None,
    heave_target: float | None = None,
    sway_target: float | None = None,
    sway_step: float | None = None,
    episode: int | None = 0,
    episode_length_s: float | None = None,
    settle_s: float | None = None,
) -> dict[str, Any]:
    """Scan-quality metrics for a logged run.

    ``episode`` selects which per-env episode to score; ``None`` pools every logged step.

    Beware episode 0. ``wallscan_env._reset_idx`` randomises ``episode_length_buf`` over
    ``[0, max_episode_length)`` on the initial full reset (the standard Isaac Lab trick to
    decorrelate episode ends across envs), so every env's FIRST episode is a partial one
    of expected length ``episode_length_s / 2``. Rate-like metrics (tilt, speeds, crab,
    s_hat error) are unaffected, but a cumulative one — ``cycles_mean`` — is roughly
    halved. To compare against a full-episode figure, log past the first reset and score
    ``episode=1``. Passing ``episode_length_s`` makes the truncation explicit in the
    output instead of silently biasing it.
    """
    import numpy as np

    traj = _with_episode_indices(traj)
    sel = _select(traj, episode)

    # Cycles credited per env at its last in-scope row (the counter is monotonic
    # within an episode and zeroed on reset, so the last row is the episode total).
    n_env = traj["z"].shape[1]
    cycles = np.full(n_env, np.nan)
    for i in range(n_env):
        rows = np.flatnonzero(sel[:, i])
        if rows.size:
            cycles[i] = traj["cycles"][rows[-1], i]

    metrics: dict[str, Any] = {
        "num_envs": int(n_env),
        "num_steps": int(traj["z"].shape[0]),
        "step_dt": float(step_dt),
        "scored_episode": episode,
        **_rate_metrics(traj, sel, step_dt, d_ref, sway_step),
        "cycles_mean": float(np.nanmean(cycles)) if np.isfinite(cycles).any() else float("nan"),
        "cycles_per_env": [None if not np.isfinite(c) else float(c) for c in cycles],
    }
    if settle_s is not None and settle_s > 0.0:
        # Same quantities over the settled window. Reported ALONGSIDE the full-window figures,
        # never instead of them, so what was excluded stays visible.
        metrics["settle_s"] = float(settle_s)
        metrics["settled"] = _rate_metrics(
            traj, _select(traj, episode, settle_s=settle_s, step_dt=step_dt), step_dt,
            d_ref, sway_step
        )
    if heave_target is not None:
        metrics["heave_speed_target_mps"] = float(heave_target)
    if sway_target is not None:
        metrics["sway_speed_target_mps"] = float(sway_target)
    if d_ref is not None:
        metrics["d_ref_m"] = float(d_ref)

    # How long the scored episode actually lasted per env. Worth reporting because the
    # cycle count is only meaningful against the time the env was given.
    if episode is not None:
        # `ended_episode`: how many STEPS the episode ran, which includes the step that ended
        # it. (`episode` would drop that row, because the state logged there is post-reset.)
        lengths = (traj["ended_episode"] == episode).sum(axis=0).astype(np.float64)
    else:
        lengths = np.full(n_env, float(traj["z"].shape[0]))
    metrics["episode_steps_mean"] = float(lengths.mean())
    metrics["episode_s_mean"] = float(lengths.mean() * step_dt)
    metrics["episode_s_per_env"] = [float(v * step_dt) for v in lengths]
    if episode_length_s is not None:
        metrics["episode_s_nominal"] = float(episode_length_s)
        # 5% slack: the last episode in a log is usually cut off by the step budget too.
        metrics["scored_episode_truncated"] = bool(metrics["episode_s_mean"] < 0.95 * episode_length_s)

    # Termination breakdown, on the rows that ended an episode.
    #
    # The `searching` filter is deliberately NOT applied here. Isaac Lab resets
    # terminated envs INSIDE step(), so on a done row `searching`/`z`/`cycles` are
    # already the next episode's post-reset values, while `terminated`/`time_out`/
    # `term_*` still describe the episode that just ended. Filtering on searching
    # hides every done row (the search always restarts after a reset).
    ended = traj["done"] > 0.5
    if episode is not None:
        # `ended_episode`, not `episode`: the done row's flags describe the episode that just
        # finished, while its state already belongs to the next one. See episode_indices.
        ended = ended & (traj["ended_episode"] == episode)
    term = ended & (traj["terminated"] > 0.5)
    collided = traj["term_collided"] > 0.5
    oob = traj["term_oob"] > 0.5
    tilted = traj["term_tilted"] > 0.5
    metrics["terminations"] = {
        "total": int(ended.sum()),
        "time_out": int((ended & (traj["time_out"] > 0.5)).sum()),
        "collided": int((term & collided).sum()),
        "out_of_bounds": int((term & oob).sum()),
        "tilted": int((term & tilted).sum()),
        # `_get_dones` builds `terminated` as collided|oob|tilted|success but records no
        # mask for success, so a termination with no cause flag set IS the success
        # termination (cycles >= cfg.success_cycles).
        "success": int((term & ~collided & ~oob & ~tilted).sum()),
    }
    return metrics


def format_metrics(metrics: dict[str, Any]) -> str:
    """Render the README's results table for a metrics dict."""

    def num(key: str, scale: float = 1.0, digits: int = 2) -> str:
        v = metrics.get(key)
        return "n/a" if v is None or v != v else f"{v * scale:.{digits}f}"

    heave_t = metrics.get("heave_speed_target_mps")
    sway_t = metrics.get("sway_speed_target_mps")
    target = ""
    if heave_t is not None and sway_t is not None:
        target = f" (target {heave_t:.2f} / {sway_t:.2f})"

    lines = [
        "",
        f"scan metrics — {metrics['num_envs']} envs x {metrics['num_steps']} steps"
        f" @ {metrics['step_dt'] * 1000:.0f} ms"
        + (f", episode {metrics['scored_episode']} only" if metrics["scored_episode"] is not None else ", all episodes")
        + f", {metrics['scored_steps']} steps scored",
        "-" * 72,
        f"{'tilt (heave / sway)':<34}{num('tilt_heave_deg')} / {num('tilt_sway_deg')} deg",
        f"{'scan speed (heave / sway)':<34}{num('heave_speed_mps', digits=3)} /"
        f" {num('sway_speed_mps', digits=3)} m/s{target}",
        f"{'crab audit (yaw - theta)':<34}{num('crab_deg')} deg",
        f"{'arc-length estimate error (s_hat)':<34}{num('s_hat_err_cm')} cm",
        f"{'coverage: heave tangential drift':<34}{num('heave_drift_cm')} cm",
        f"{'coverage: sway step error':<34}{num('sway_step_err_cm')} cm",
        f"{'arc-length TRACKING error (own ref)':<34}{num('s_track_err_cm')} cm"
        f"  (p95 {num('s_track_p95_cm')}, max {num('s_track_max_cm')})",
        f"{'scan cycles completed':<34}{num('cycles_mean')}"
        + (f"  (in {metrics['episode_s_mean']:.0f} s/episode)" if "episode_s_mean" in metrics else ""),
    ]
    if "wall_dist_err_cm" in metrics:
        lines.append(f"{'wall distance error vs d_ref':<34}{num('wall_dist_err_cm')} cm")
    t = metrics["terminations"]
    lines += [
        f"{'terminations':<34}{t['total']}"
        f"  (success {t.get('success', 0)}, time-out {t.get('time_out', 0)},"
        f" collided {t['collided']}, out-of-bounds {t['out_of_bounds']}, tilted {t['tilted']})",
        "-" * 72,
    ]
    if metrics.get("scored_episode_truncated"):
        lines += [
            f"WARNING: the scored episode averaged {metrics['episode_s_mean']:.0f} s vs a nominal"
            f" {metrics['episode_s_nominal']:.0f} s.",
            "         wallscan_env randomises episode_length_buf on the first reset, so episode 0",
            "         is a PARTIAL episode — 'scan cycles completed' is biased low. Log past the",
            "         first reset and pass --score_episode 1 for a full-episode number.",
            "-" * 72,
        ]
    lines.append("")
    return "\n".join(lines)


def plot_trajectory(
    traj: dict[str, Any],
    path: str,
    *,
    episode: int | None = 0,
    max_envs: int = 8,
    title: str | None = None,
) -> str:
    """Per-env s-vs-z scan trace with the reference overlaid; saves a PNG at ``path``.

    Matches the layout of the committed ``results/trajectory_*.png``: one panel per
    env, ground-truth arc length on x, depth on y, reference as a red dashed zig-zag.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = min(max_envs, traj["z"].shape[1])
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), squeeze=False)

    for i in range(n):
        ax = axes[i // ncols][i % ncols]
        keep = np.ones(traj["z"].shape[0], dtype=bool)
        if episode is not None:
            keep &= traj["episode"][:, i] == episode
        ax.plot(traj["s_gt"][keep, i], traj["z"][keep, i], lw=0.8, color="tab:blue", label="traj")
        ax.plot(traj["s_ref"][keep, i], traj["z_ref"][keep, i], "r--", lw=0.8, label="ref")
        ax.set_title(f"env{i}")
        ax.set_xlabel("s [m]")
        ax.set_ylabel("z [m]")
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
