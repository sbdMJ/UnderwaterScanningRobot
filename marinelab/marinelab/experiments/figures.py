# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""General comparison figures F1–F6 (plan §10) from results/ artifacts — no sim needed.

Every function takes file paths / pre-collected data and writes a figure (png + pdf).
Statistics style follows the plan: mean ± SD with per-trial points overlaid; no min/max
bars. Method order and labels are shared with the LaTeX tables (``aggregate``).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .aggregate import METHOD_LABELS, _method_key  # noqa: E402

_DOT = dict(s=14, zorder=3, alpha=0.75, color="black")
_ERR = dict(fmt="_", ms=18, capsize=4, lw=1.5, zorder=2)


def _save(fig, out: str) -> list[str]:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    stem = str(Path(out).with_suffix(""))
    paths = [f"{stem}.png", f"{stem}.pdf"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return paths


def _methods_sorted(summary: dict) -> list[str]:
    return sorted({m for (m, _c) in summary}, key=_method_key)


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _traj_2d(arr: np.ndarray) -> np.ndarray:
    """(steps,) or (steps, n_env) -> (steps, n_env)."""
    a = np.asarray(arr)
    return a[:, None] if a.ndim == 1 else a


# ---------------------------------------------------------------------------
# F1 — main-comparison statistics: mean ± SD + per-trial points, one panel/metric
# ---------------------------------------------------------------------------


def fig_overlay(summary: dict, metrics: list[str], out: str, *, cond: str | None = None,
                labels: dict[str, str] | None = None) -> list[str]:
    labels = labels or {}
    conds = sorted({c for (_m, c) in summary})
    cond = cond if cond is not None else conds[0]
    methods = [m for m in _methods_sorted(summary) if (m, cond) in summary]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.2 * len(metrics) + 1, 3.0),
                             squeeze=False)
    rng = np.random.default_rng(0)
    for j, metric in enumerate(metrics):
        ax = axes[0, j]
        for i, m in enumerate(methods):
            stat = summary[(m, cond)][metric]
            values = [v for v in stat["values"] if np.isfinite(v)]
            if np.isfinite(stat["mean"]):
                ax.errorbar(i, stat["mean"], yerr=stat["sd"], **_ERR)
            if values:
                ax.scatter(i + rng.uniform(-0.12, 0.12, len(values)), values, **_DOT)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([_label(m) for m in methods], rotation=30, ha="right", fontsize=8)
        ax.set_title(labels.get(metric, metric), fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"condition: {cond}", fontsize=9)
    fig.tight_layout()
    return _save(fig, out)


# ---------------------------------------------------------------------------
# F2 — representative trajectory: unwrapped wall plane (s–z), path vs reference
# ---------------------------------------------------------------------------


def fig_trajectory(cases: list[tuple[str, str]], out: str, *, env: int = 0) -> list[str]:
    """cases: [(label, npz_path)], one panel per case; plots (s_gt, z) vs (s_ref, z_ref)."""
    fig, axes = plt.subplots(1, len(cases), figsize=(3.2 * len(cases), 3.2),
                             sharey=True, squeeze=False)
    for j, (label, npz_path) in enumerate(cases):
        traj = np.load(npz_path)
        s, z = _traj_2d(traj["s_gt"])[:, env], _traj_2d(traj["z"])[:, env]
        s_ref, z_ref = _traj_2d(traj["s_ref"])[:, env], _traj_2d(traj["z_ref"])[:, env]
        ax = axes[0, j]
        ax.plot(s_ref, z_ref, "--", color="tab:gray", lw=1.2, label="reference")
        ax.plot(s, z, color="tab:blue", lw=1.0, label="trajectory")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("arc length s [m]")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_ylabel("depth z [m]")
            ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, out)


# ---------------------------------------------------------------------------
# F3 — robustness curve: metric vs perturbation level, one line per method
# ---------------------------------------------------------------------------


def cond_level(cond: str) -> float:
    """'dr25' -> 25.0; falls back to the trailing number in the name."""
    digits = "".join(ch for ch in cond if ch.isdigit() or ch == ".")
    if not digits:
        raise ValueError(f"cannot parse a level from condition {cond!r}; pass level_map")
    return float(digits)


def fig_sweep(summary: dict, metric: str, out: str, *, level_map: dict[str, float] | None = None,
              ylabel: str | None = None, xlabel: str = "perturbation level [%]") -> list[str]:
    level_map = level_map or {c: cond_level(c) for (_m, c) in summary}
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    rng = np.random.default_rng(0)
    for m in _methods_sorted(summary):
        levels = sorted({level_map[c] for (mm, c) in summary if mm == m})
        means, sds = [], []
        for lv in levels:
            cond = next(c for (mm, c) in summary if mm == m and level_map[c] == lv)
            stat = summary[(m, cond)][metric]
            means.append(stat["mean"])
            sds.append(stat["sd"])
            values = [v for v in stat["values"] if np.isfinite(v)]
            ax.scatter(lv + rng.uniform(-0.8, 0.8, len(values)), values, s=8, alpha=0.4,
                       zorder=1)
        ax.errorbar(levels, means, yerr=sds, marker="o", ms=4, capsize=3, lw=1.4,
                    label=_label(m), zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or metric)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out)


# ---------------------------------------------------------------------------
# F4 — disturbance-response time series, event time marked
# ---------------------------------------------------------------------------


def fig_timeseries(cases: list[tuple[str, str, str]], out: str, *, env: int = 0,
                   t_event: float | None = None) -> list[str]:
    """cases: [(label, npz_path, metrics_json_path)] — wall-distance error and tilt vs time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 4.2), sharex=True)
    for label, npz_path, metrics_path in cases:
        traj = np.load(npz_path)
        with open(metrics_path) as fh:
            meta = json.load(fh)
        dt, d_ref = float(meta["step_dt"]), float(meta["d_ref_m"])
        wall = _traj_2d(traj["wall_dist"])[:, env]
        tilt = _traj_2d(traj["tilt_deg"])[:, env]
        t = np.arange(len(wall)) * dt
        ax1.plot(t, np.abs(wall - d_ref), lw=0.9, label=label)
        ax2.plot(t, tilt, lw=0.9, label=label)
    for ax, ylab in ((ax1, "|wall dist err| [m]"), (ax2, "tilt [deg]")):
        if t_event is not None:
            ax.axvline(t_event, color="black", ls=":", lw=1.0)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(ylab, fontsize=9)
    ax1.legend(fontsize=8)
    ax2.set_xlabel("time [s]")
    fig.tight_layout()
    return _save(fig, out)


# ---------------------------------------------------------------------------
# F5 — zero-shot vs fine-tuned grouped bars
# ---------------------------------------------------------------------------


def fig_zeroshot_ft(named_summaries: dict[str, dict], metric: str, out: str, *,
                    cond: str | None = None, ylabel: str | None = None) -> list[str]:
    """named_summaries: {"zero-shot": summary_a, "fine-tuned": summary_b, ...}."""
    methods = sorted({m for s in named_summaries.values() for (m, _c) in s}, key=_method_key)
    width = 0.8 / len(named_summaries)
    fig, ax = plt.subplots(figsize=(1.1 * len(methods) + 2, 3.2))
    rng = np.random.default_rng(0)
    for k, (name, summary) in enumerate(named_summaries.items()):
        xs, means, sds = [], [], []
        for i, m in enumerate(methods):
            key = next(((mm, c) for (mm, c) in summary
                        if mm == m and (cond is None or c == cond)), None)
            if key is None:
                continue
            stat = summary[key][metric]
            x = i + (k - (len(named_summaries) - 1) / 2) * width
            xs.append(x)
            means.append(stat["mean"] if np.isfinite(stat["mean"]) else 0.0)
            sds.append(stat["sd"])
            values = [v for v in stat["values"] if np.isfinite(v)]
            ax.scatter(x + rng.uniform(-width / 4, width / 4, len(values)), values, **_DOT)
        ax.bar(xs, means, width=width * 0.9, yerr=sds, capsize=3, label=name, zorder=1,
               alpha=0.85)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([_label(m) for m in methods], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel or metric)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out)


# ---------------------------------------------------------------------------
# F6 — cost comparison: offline (training/tuning) and online (per-step) cost
# ---------------------------------------------------------------------------


def fig_cost(offline: dict[str, dict], summary: dict, out: str, *,
             inference_metric: str = "controller_cost.solve_ms_mean") -> list[str]:
    """offline: method -> {"env_steps": .., "wall_clock_s": ..} (tuning budget / training
    cost); summary: an E1 summary for the per-step inference cost column."""
    methods = sorted(set(offline) | {m for (m, _c) in summary}, key=_method_key)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.0))
    xs = np.arange(len(methods))
    wall = [offline.get(m, {}).get("wall_clock_s", 0.0) for m in methods]
    ax1.bar(xs, wall, width=0.6)
    ax1.set_ylabel("offline cost: wall-clock [s]", fontsize=9)
    positive = [w for w in wall if w > 0]
    if positive and max(positive) / min(positive) > 50:  # spans decades (PPO vs BO): log axis
        ax1.set_yscale("log")
    inf_means, inf_sds = [], []
    for m in methods:
        key = next(((mm, c) for (mm, c) in summary if mm == m), None)
        stat = summary[key][inference_metric] if key else {"mean": 0.0, "sd": 0.0}
        inf_means.append(stat["mean"] if np.isfinite(stat["mean"]) else 0.0)
        inf_sds.append(stat["sd"])
    ax2.bar(xs, inf_means, width=0.6, yerr=inf_sds, capsize=3)
    ax2.set_ylabel("per-step compute [ms]", fontsize=9)
    for ax in (ax1, ax2):
        ax.set_xticks(xs)
        ax.set_xticklabels([_label(m) for m in methods], rotation=20, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out)
