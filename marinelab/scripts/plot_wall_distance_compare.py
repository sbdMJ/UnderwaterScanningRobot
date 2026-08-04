# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Wall-standoff comparison plot: RL policy vs diff-WMPC, from saved ``results/trajectory_*.npz``.

The scan requirement is a CONSTANT sonar standoff (``d_ref`` = 1.5 m). Every other metric in this
project is a proxy for that, so it is worth seeing directly rather than as a single RMS number.

Pure numpy + matplotlib(Agg), so it runs natively -- no Isaac Sim, no acados, no container.

Two things the figure has to be honest about:

* ``wall_dist`` is the sonar range from the NOMINAL mount, identical in both runs, so the two
  panels are on the same footing. ``clearance`` (radial, body origin) is NOT the same quantity
  and is deliberately not mixed in.
* The RL trace includes its ~10 s spin search and its spawn transient; the NMPC skips the search
  by construction. The settle window is drawn so the reader can discount both, and the summary
  table reports settled and full-window side by side.

Usage:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python marinelab/scripts/plot_wall_distance_compare.py \\
        --rl nominal_repro --mpc dw8_s0 dw8_s1 dw8_s2 --out results/wall_distance_compare.png
"""

from __future__ import annotations

import argparse
import os
import sys
import types

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RESULTS = os.path.abspath(os.path.join(_ROOT, "..", "results"))

# Bare parent packages so marinelab/__init__.py (which pulls in isaaclab) never runs;
# eval_metrics is pure torch/numpy. Same trick as tests/conftest.py and rescore_trajectories.py.
sys.path.insert(0, _ROOT)
for _name, _sub in (("marinelab", ""), ("marinelab.tasks", "tasks"),
                    ("marinelab.tasks.pkrc_wallscan", "tasks/pkrc_wallscan")):
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.__path__ = [os.path.join(_ROOT, "marinelab", _sub) if _sub
                       else os.path.join(_ROOT, "marinelab")]
        sys.modules[_name] = _m

from marinelab.tasks.pkrc_wallscan import eval_metrics as em  # noqa: E402


def load(tag: str, results: str) -> dict[str, np.ndarray]:
    path = os.path.join(results, f"trajectory_{tag}.npz")
    if not os.path.isfile(path):
        raise SystemExit(f"missing {path}")
    return em._with_episode_indices(dict(np.load(path)))


def select(traj: dict[str, np.ndarray], settle_s: float, step_dt: float) -> np.ndarray:
    """Boolean (T, E) row mask, delegating to the scoring module rather than reimplementing it.

    Reimplementing it here got the episode index wrong on the first attempt: Isaac Lab resets
    inside ``step()`` and both loggers read state afterwards, so the row carrying ``done``
    already holds the POST-reset pose. Using ``cumsum(done) - done`` kept that row, and a single
    across-the-tank sonar reading (8.64 m vs a 1.5 m target) took the standoff RMS from 0.7 cm
    to 11.0 cm. ``eval_metrics`` already distinguishes the two indices and is tested for it, so
    the plot and the metrics table now agree by construction.
    """
    return em._select(traj, episode=0, settle_s=settle_s, step_dt=step_dt)


def standoff_error(traj: dict[str, np.ndarray], d_ref: float, settle_s: float,
                   step_dt: float) -> tuple[np.ndarray, np.ndarray]:
    """(full, settled) flattened |sonar range - d_ref| in cm, first episode only."""
    err = np.abs(traj["wall_dist"] - d_ref) * 100.0
    full = select(traj, 0.0, step_dt)
    sett = select(traj, settle_s, step_dt)
    return err[full], err[sett]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", default="nominal_repro", help="tag of the RL trajectory npz")
    ap.add_argument("--mpc", nargs="+", default=["dw8_s0", "dw8_s1", "dw8_s2"])
    ap.add_argument("--results", default=_RESULTS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--d_ref", type=float, default=1.5)
    ap.add_argument("--step_dt", type=float, default=0.02)
    ap.add_argument("--settle_s", type=float, default=20.0)
    ap.add_argument("--ymax", type=float, default=4.0, help="y clip [m] for the time-series panels")
    args = ap.parse_args()
    out = args.out or os.path.join(args.results, "wall_distance_compare.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rl = load(args.rl, args.results)
    mpcs = [(t, load(t, args.results)) for t in args.mpc]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))

    # -- top left: RL time series -----------------------------------------
    ax = axes[0, 0]
    keep = select(rl, 0.0, args.step_dt)
    t = np.arange(rl["wall_dist"].shape[0]) * args.step_dt
    for e in range(rl["wall_dist"].shape[1]):
        y = np.where(keep[:, e], rl["wall_dist"][:, e], np.nan)
        ax.plot(t, y, lw=0.6, alpha=0.75)
    ax.set_title(f"RL policy  ({args.rl}, {rl['wall_dist'].shape[1]} envs)")

    # -- top right: diff-WMPC time series ---------------------------------
    ax2 = axes[0, 1]
    for tag, tr in mpcs:
        k = select(tr, 0.0, args.step_dt)
        tt = np.arange(tr["wall_dist"].shape[0]) * args.step_dt
        for e in range(tr["wall_dist"].shape[1]):
            y = np.where(k[:, e], tr["wall_dist"][:, e], np.nan)
            ax2.plot(tt, y, lw=0.6, alpha=0.75)
    n_env = sum(tr["wall_dist"].shape[1] for _, tr in mpcs)
    ax2.set_title(f"diff-WMPC  ({len(mpcs)} seeds x {mpcs[0][1]['wall_dist'].shape[1]} envs = {n_env})")

    for a in (ax, ax2):
        a.axhline(args.d_ref, color="r", ls="--", lw=1.2, label=f"target {args.d_ref} m")
        a.axvline(args.settle_s, color="k", ls=":", lw=1.0, label=f"settle {args.settle_s:.0f} s")
        a.set_xlabel("t [s]")
        a.set_ylabel("sonar range to wall [m]")
        a.set_ylim(0.0, args.ymax)
        a.grid(alpha=0.25)
        a.legend(fontsize=8, loc="upper right")

    # -- bottom left: error CDF -------------------------------------------
    ax3 = axes[1, 0]
    rows = []
    for label, trs in (("RL policy", [(args.rl, rl)]), ("diff-WMPC", mpcs)):
        full = np.concatenate([standoff_error(tr, args.d_ref, args.settle_s, args.step_dt)[0]
                               for _, tr in trs])
        sett = np.concatenate([standoff_error(tr, args.d_ref, args.settle_s, args.step_dt)[1]
                               for _, tr in trs])
        for nm, arr in (("full", full), ("settled", sett)):
            xs = np.sort(arr)
            ax3.plot(xs, np.linspace(0, 100, xs.size), lw=1.6 if nm == "settled" else 1.0,
                     ls="-" if nm == "settled" else "--",
                     label=f"{label} ({nm})  RMS {np.sqrt(np.mean(arr ** 2)):.2f} cm")
            rows.append((label, nm, arr))
    ax3.set_xscale("log")
    ax3.set_xlabel("|sonar range - target|  [cm, log]")
    ax3.set_ylabel("percent of steps below")
    ax3.set_title("standoff error distribution (first episode)")
    ax3.grid(alpha=0.25, which="both")
    ax3.legend(fontsize=8, loc="lower right")

    # -- bottom right: settled box ----------------------------------------
    ax4 = axes[1, 1]
    sel = [(f"{lb}\n{nm}", arr) for lb, nm, arr in rows]
    ax4.boxplot([a for _, a in sel], tick_labels=[n for n, _ in sel], showfliers=False, whis=(5, 95))
    ax4.set_ylabel("|sonar range - target|  [cm]")
    ax4.set_title("box: median / quartiles / 5-95 pct (outliers hidden)")
    ax4.grid(alpha=0.25, axis="y")

    fig.suptitle("Wall standoff: RL policy vs diff-WMPC   (both nominal conditions, "
                 f"target {args.d_ref} m sonar range)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=120)
    print(f"-> {out}")

    print(f"\n{'group':<22}{'RMS cm':>9}{'mean cm':>9}{'median':>9}{'p95':>9}{'max':>9}{'N':>10}")
    for lb, nm, arr in rows:
        print(f"{lb + ' / ' + nm:<22}{np.sqrt(np.mean(arr ** 2)):>9.2f}{arr.mean():>9.2f}"
              f"{np.median(arr):>9.2f}{np.percentile(arr, 95):>9.2f}{arr.max():>9.2f}{arr.size:>10d}")


if __name__ == "__main__":
    sys.exit(main())
