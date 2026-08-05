# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""General comparison figures F1–F6 (plan §10), styled after the SSI-MPC paper.

Style contract (docs/SSI-MPC.pdf, T-RO / MATLAB conventions):
- serif (Times) text with STIX math, boxed axes with inward ticks on all four sides,
  light gray grid, framed white legend;
- one fixed color+marker per method across every figure (MATLAB default color order:
  Nominal blue, BO orange, PPO yellow, SSI green, ours purple — "Ours" is purple in
  the reference paper as well);
- time series are mean-over-runs curves with a ±SD band and 1 s rolling smoothing,
  not raw single-run traces (their Figs. 3/10–12);
- statistics keep the plan's per-trial point overlay (mean ± SD, no min/max bars).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .aggregate import METHOD_LABELS, _method_key  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "0.7",
    "legend.fancybox": False,
    "figure.dpi": 200,
    "savefig.dpi": 300,
})

# MATLAB default color order, fixed per method across every figure.
METHOD_COLORS = {"nominal": "#0072BD", "bo": "#D95319", "ppo": "#EDB120",
                 "ssi": "#77AC30", "diff": "#7E2F8E"}
METHOD_MARKERS = {"nominal": "o", "bo": "s", "ppo": "^", "ssi": "d", "diff": "v"}


def _color(method: str) -> str:
    return METHOD_COLORS.get(method, "0.4")


def _marker(method: str) -> str:
    return METHOD_MARKERS.get(method, "x")


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _save(fig, out: str) -> list[str]:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    stem = str(Path(out).with_suffix(""))
    paths = [f"{stem}.png", f"{stem}.pdf"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return paths


def _methods_sorted(summary: dict) -> list[str]:
    return sorted({m for (m, _c) in summary}, key=_method_key)


def _traj_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    return a[:, None] if a.ndim == 1 else a


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


# ---------------------------------------------------------------------------
# F1 — main-comparison statistics: mean ± SD + per-trial points, one panel/metric
# ---------------------------------------------------------------------------


def fig_overlay(summary: dict, metrics: list[str], out: str, *, cond: str | None = None,
                labels: dict[str, str] | None = None) -> list[str]:
    labels = labels or {}
    conds = sorted({c for (_m, c) in summary})
    cond = cond if cond is not None else conds[0]
    methods = [m for m in _methods_sorted(summary) if (m, cond) in summary]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.1 * len(metrics) + 0.8, 2.7),
                             squeeze=False)
    rng = np.random.default_rng(0)
    for j, metric in enumerate(metrics):
        ax = axes[0, j]
        for i, m in enumerate(methods):
            stat = summary[(m, cond)][metric]
            values = [v for v in stat["values"] if np.isfinite(v)]
            if np.isfinite(stat["mean"]):
                ax.errorbar(i, stat["mean"], yerr=stat["sd"], fmt=_marker(m), ms=6,
                            color=_color(m), mfc=_color(m), mec="black", mew=0.5,
                            capsize=3, lw=1.3, zorder=3)
            if values:
                ax.scatter(i + rng.uniform(-0.14, 0.14, len(values)), values, s=9,
                           color="0.25", alpha=0.65, zorder=2, linewidths=0)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([_label(m) for m in methods], rotation=35, ha="right", fontsize=8)
        ax.set_title(labels.get(metric, metric))
        ax.set_xlim(-0.6, len(methods) - 0.4)
    fig.tight_layout(w_pad=1.2)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# F2 — representative trajectory: unwrapped wall plane (s–z), path vs reference
# ---------------------------------------------------------------------------


def fig_trajectory(cases: list[tuple[str, str, str]], out: str, *, env: int = 0) -> list[str]:
    """cases: [(method, label, npz_path)], one panel per case."""
    fig, axes = plt.subplots(1, len(cases), figsize=(2.6 * len(cases), 2.9),
                             sharey=True, squeeze=False)
    for j, (method, label, npz_path) in enumerate(cases):
        traj = np.load(npz_path)
        s, z = _traj_2d(traj["s_gt"])[:, env], _traj_2d(traj["z"])[:, env]
        s_ref, z_ref = _traj_2d(traj["s_ref"])[:, env], _traj_2d(traj["z_ref"])[:, env]
        ax = axes[0, j]
        ax.plot(s, z, color=_color(method), lw=0.8, alpha=0.9, zorder=2, label=label)
        ax.plot(s_ref, z_ref, "--", color="black", lw=1.4, zorder=3, label="Reference")
        ax.set_title(label)
        ax.set_xlabel("Arc length $s$ [m]")
        if j == 0:
            ax.set_ylabel("Depth $z$ [m]")
            ax.legend(loc="best")
    fig.tight_layout(w_pad=0.8)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# F3 — robustness curve: metric vs perturbation level, one line per method
# ---------------------------------------------------------------------------


def cond_level(cond: str) -> float:
    digits = "".join(ch for ch in cond if ch.isdigit() or ch == ".")
    if not digits:
        raise ValueError(f"cannot parse a level from condition {cond!r}; pass level_map")
    return float(digits)


def fig_sweep(summary: dict, metric: str, out: str, *, level_map: dict[str, float] | None = None,
              ylabel: str | None = None, xlabel: str = "Perturbation level [%]") -> list[str]:
    level_map = level_map or {c: cond_level(c) for (_m, c) in summary}
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
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
            ax.scatter(lv + rng.uniform(-0.9, 0.9, len(values)), values, s=7,
                       color=_color(m), alpha=0.35, zorder=1, linewidths=0)
        means, sds = np.asarray(means, float), np.asarray(sds, float)
        finite = np.isfinite(means)
        ax.plot(np.asarray(levels)[finite], means[finite], marker=_marker(m), ms=5,
                color=_color(m), mec="black", mew=0.4, label=_label(m), zorder=3)
        ax.fill_between(np.asarray(levels)[finite], (means - sds)[finite],
                        (means + sds)[finite], color=_color(m), alpha=0.15,
                        linewidth=0, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or metric)
    ax.legend(loc="best")
    fig.tight_layout()
    return _save(fig, out)


# ---------------------------------------------------------------------------
# F4 — disturbance-response time series: mean over runs ± SD band, smoothed
# ---------------------------------------------------------------------------


def fig_timeseries(cases: list[tuple[str, str, list[str], str]], out: str, *,
                   t_event: float | None = None, smooth_s: float = 1.0) -> list[str]:
    """cases: [(method, label, [npz_path, ...], metrics_json_path)] — every npz (seed)
    contributes all its envs; curves are the across-run mean with a ±SD band."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.8, 3.8), sharex=True)
    for method, label, npz_paths, metrics_path in cases:
        with open(metrics_path) as fh:
            meta = json.load(fh)
        dt, d_ref = float(meta["step_dt"]), float(meta["d_ref_m"])
        window = max(1, int(round(smooth_s / dt)))
        wall_runs, tilt_runs = [], []
        for p in npz_paths:
            traj = np.load(p)
            wall_runs.append(np.abs(_traj_2d(traj["wall_dist"]) - d_ref))
            tilt_runs.append(_traj_2d(traj["tilt_deg"]))
        n = min(w.shape[0] for w in wall_runs)
        wall = np.concatenate([w[:n] for w in wall_runs], axis=1)
        tilt = np.concatenate([w[:n] for w in tilt_runs], axis=1)
        t = np.arange(n) * dt
        for ax, data in ((ax1, wall), (ax2, tilt)):
            mean = _smooth(data.mean(axis=1), window)
            sd = _smooth(data.std(axis=1), window)
            ax.plot(t, mean, color=_color(method), label=label, zorder=3)
            if data.shape[1] > 1:
                ax.fill_between(t, mean - sd, mean + sd, color=_color(method),
                                alpha=0.18, linewidth=0, zorder=2)
    for ax, ylab in ((ax1, r"Wall-distance error [m]"), (ax2, r"Tilt [deg]")):
        if t_event is not None:
            ax.axvline(t_event, color="black", ls=":", lw=1.0, zorder=1)
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
    ax1.legend(loc="best", ncols=1)
    ax2.set_xlabel("Time [s]")
    fig.tight_layout(h_pad=0.6)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# F5 — zero-shot vs fine-tuned grouped bars
# ---------------------------------------------------------------------------

_GROUP_FACE = ["#FFFFFF", "0.55"]  # zero-shot: hollow; fine-tuned: filled gray


def fig_zeroshot_ft(named_summaries: dict[str, dict], metric: str, out: str, *,
                    cond: str | None = None, ylabel: str | None = None) -> list[str]:
    methods = sorted({m for s in named_summaries.values() for (m, _c) in s}, key=_method_key)
    width = 0.8 / len(named_summaries)
    fig, ax = plt.subplots(figsize=(0.95 * len(methods) + 1.6, 2.8))
    rng = np.random.default_rng(0)
    for k, (name, summary) in enumerate(named_summaries.items()):
        for i, m in enumerate(methods):
            key = next(((mm, c) for (mm, c) in summary
                        if mm == m and (cond is None or c == cond)), None)
            if key is None:
                continue
            stat = summary[key][metric]
            x = i + (k - (len(named_summaries) - 1) / 2) * width
            mean = stat["mean"] if np.isfinite(stat["mean"]) else 0.0
            ax.bar(x, mean, width=width * 0.88, yerr=stat["sd"], capsize=3,
                   facecolor=_GROUP_FACE[k % len(_GROUP_FACE)], edgecolor=_color(m),
                   linewidth=1.3, zorder=2,
                   error_kw={"elinewidth": 1.0, "zorder": 4})
            values = [v for v in stat["values"] if np.isfinite(v)]
            ax.scatter(x + rng.uniform(-width / 4, width / 4, len(values)), values, s=9,
                       color="0.2", alpha=0.7, zorder=3, linewidths=0)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([_label(m) for m in methods], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel or metric)
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=_GROUP_FACE[k % len(_GROUP_FACE)], edgecolor="0.3",
                     linewidth=1.2) for k in range(len(named_summaries))]
    ax.legend(handles, list(named_summaries), loc="best")
    fig.tight_layout()
    return _save(fig, out)


# ---------------------------------------------------------------------------
# F6 — cost comparison: offline (training/tuning) and online (per-step) cost
# ---------------------------------------------------------------------------


def fig_cost(offline: dict[str, dict], summary: dict, out: str, *,
             inference_metric: str = "controller_cost.solve_ms_mean") -> list[str]:
    methods = sorted(set(offline) | {m for (m, _c) in summary}, key=_method_key)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    xs = np.arange(len(methods))
    colors = [_color(m) for m in methods]
    wall = [offline.get(m, {}).get("wall_clock_s", 0.0) for m in methods]
    ax1.bar(xs, wall, width=0.62, color=colors, edgecolor="black", linewidth=0.6)
    ax1.set_ylabel("Offline cost [s]")
    positive = [w for w in wall if w > 0]
    if positive and max(positive) / min(positive) > 50:
        ax1.set_yscale("log")
    inf_means, inf_sds = [], []
    for m in methods:
        key = next(((mm, c) for (mm, c) in summary if mm == m), None)
        stat = summary[key][inference_metric] if key else {"mean": 0.0, "sd": 0.0}
        inf_means.append(stat["mean"] if np.isfinite(stat["mean"]) else 0.0)
        inf_sds.append(stat["sd"])
    ax2.bar(xs, inf_means, width=0.62, yerr=inf_sds, capsize=3, color=colors,
            edgecolor="black", linewidth=0.6)
    ax2.set_ylabel("Computation time [ms]")
    for ax in (ax1, ax2):
        ax.set_xticks(xs)
        ax.set_xticklabels([_label(m) for m in methods], rotation=20, ha="right", fontsize=8)
        ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.5)
    return _save(fig, out)
