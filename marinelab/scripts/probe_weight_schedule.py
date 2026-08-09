# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Does a Diff-WMPC policy actually SCHEDULE its weights, or has it collapsed to a constant?

This is the Fig. 5 analogue of Jahncke et al., RA-L 11(3) 2026 (the paper `diff_wmpc.py` ports).
Their weights-varying result rests on the policy adapting along the reference — the velocity-error
weight rising on straights, the steering-rate weight tightening *before* turn-in. If our policy
emits the same vector everywhere, "weights-varying" is a label with nothing behind it, and any
downstream comparison is really a comparison of fixed weight vectors.

Measured on `checkpoints/dw_ekf/policy_final.pt` (the legacy `error_phase` policy) on
2026-08-06: <±10% spread across five very different states, and a frozen copy of its output
reproduced the live policy's closed-loop numbers to within noise on every metric.

Native — no acados, no Isaac Sim. Usage::

    python marinelab/scripts/probe_weight_schedule.py checkpoints/dw_preview/policy_final.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_pure(name: str, relpath: str):
    """Import a pure-torch module by path, bypassing `marinelab.__init__` (which pulls isaaclab)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dw = _load_pure("_dw", "marinelab/marinelab/algorithms/diff_wmpc.py")

ERROR_NAMES = ("radial", "z", "s", "v_rad", "v_tan", "v_z",
               "head_x", "head_y", "roll", "pitch", "w_x", "w_y")
NE = len(ERROR_NAMES)

# The four scan phases as the policy sees them through the reference preview. v_z_des and
# v_tan_des are the scan speeds from ScanCfg (0.20 m/s heave, 0.10 m/s sway); the sign of
# v_z_des is what distinguishes DESCEND from ASCEND.
PHASE_REF = {
    "DESCEND":  (-0.20, 0.00),
    "SWAY_A":   (0.00, +0.10),
    "ASCEND":   (+0.20, 0.00),
    "SWAY_B":   (0.00, -0.10),
}
PHASE_IDX = {"DESCEND": 0, "SWAY_A": 1, "ASCEND": 2, "SWAY_B": 3}


def make_ref(v_z: float, v_tan: float, n_stages: int, switch_to=None, switch_at=None):
    """[1, n_stages+1] reference. With `switch_to`, the leg changes partway through the horizon,
    which is the only input that can reveal ANTICIPATION rather than reaction."""
    vz = torch.full((1, n_stages + 1), float(v_z))
    vt = torch.full((1, n_stages + 1), float(v_tan))
    if switch_to is not None:
        vz[0, switch_at:] = switch_to[0]
        vt[0, switch_at:] = switch_to[1]
    return {"v_z_des": vz, "v_tan_des": vt,
            "z_ref": torch.zeros(1, n_stages + 1), "s_ref": torch.zeros(1, n_stages + 1)}


def weights_for(policy, mode, ref, phase_idx, n_stages, e_now=None):
    feat = dw.build_features(mode, e_now=torch.zeros(NE) if e_now is None else e_now,
                             phase=torch.tensor(float(phase_idx)), ref=ref, n_stages=n_stages)
    policy.reset_history()
    with torch.no_grad():
        return policy(feat)[:NE].numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n_stages", type=int, default=None,
                    help="Horizon. Default: the value stamped in the checkpoint.")
    ap.add_argument("--switch_at", type=int, default=15,
                    help="Stage where the anticipation probe switches leg (default: mid-horizon).")
    args = ap.parse_args()

    st = torch.load(args.ckpt, map_location="cpu")
    mode = st.get("feat_mode", "error_phase") if isinstance(st, dict) else "error_phase"
    n_stages = args.n_stages or (st.get("horizon", 30) if isinstance(st, dict) else 30)
    sd = st["policy"] if isinstance(st, dict) and "policy" in st else st
    nu = int(sd["head.bias"].shape[0]) - NE

    policy = dw.WeightPolicy(dw.feature_dim(mode, NE), NE, nu)
    policy.load_state_dict(sd)
    policy.eval()
    print(f"{args.ckpt}\n  feat_mode={mode}  horizon={n_stages}  updates={st.get('n_updates', '?')}")

    names = list(PHASE_REF)
    cols = {n: weights_for(policy, mode, make_ref(*PHASE_REF[n], n_stages), PHASE_IDX[n], n_stages)
            for n in names}

    print(f"\n{'weight':8}" + "".join(f"{n:>10}" for n in names) + f"{'spread':>9}")
    print("-" * (8 + 10 * len(names) + 9))
    spreads = []
    for i, w in enumerate(ERROR_NAMES):
        v = np.array([cols[n][i] for n in names])
        rel = (v.max() - v.min()) / max(abs(v.mean()), 1e-9)
        spreads.append(rel)
        print(f"{w:8}" + "".join(f"{cols[n][i]:10.1f}" for n in names) + f"{100 * rel:8.0f}%")

    print(f"\nmax spread across phases: {100 * max(spreads):.0f}%   "
          f"median: {100 * float(np.median(spreads)):.0f}%")
    if max(spreads) < 0.20:
        print("  -> COLLAPSED to a near-constant vector. A frozen vector will reproduce this "
              "policy, and 'weights-varying' is not doing work.")
    else:
        print("  -> the weights DO vary with the reference. Whether the variation HELPS is a "
              "separate, closed-loop question.")

    if mode in ("preview", "both"):
        # Anticipation: same current leg, different UPCOMING leg. A reactive policy cannot tell
        # these apart -- for `error_phase` the features are literally identical.
        print(f"\nanticipation probe (current leg DESCEND, switch at stage {args.switch_at} of "
              f"{n_stages} = {args.switch_at * 0.05:.2f} s ahead)")
        base = weights_for(policy, mode, make_ref(*PHASE_REF["DESCEND"], n_stages), 0, n_stages)
        ahead = weights_for(policy, mode,
                            make_ref(*PHASE_REF["DESCEND"], n_stages,
                                     switch_to=PHASE_REF["SWAY_A"], switch_at=args.switch_at),
                            0, n_stages)
        rel = np.abs(ahead - base) / np.maximum(np.abs(base), 1e-9)
        order = np.argsort(-rel)[:4]
        for i in order:
            print(f"  {ERROR_NAMES[i]:8} {base[i]:9.1f} -> {ahead[i]:9.1f}  ({100 * rel[i]:+.0f}%)")
        print(f"  max {100 * rel.max():.0f}% — this is the quantity that is structurally ZERO "
              f"for feat_mode=error_phase.")


if __name__ == "__main__":
    main()
