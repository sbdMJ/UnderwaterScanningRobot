#!/usr/bin/env python3
"""Read TensorBoard event file and extract key metrics with trends."""

import os
import sys
from tensorboard.backend.event_processing import event_accumulator


LOG_DIR = "/workspace/isaaclab/logs/rsl_rl/hero_agent_sac_mpc/2026-02-22_11-29-54/"

# Tags we want to extract (partial matching supported via prefixes)
TARGET_TAGS = [
    "Reward/tracking",
    "Reward/total",
    "Reward/total_normalized",
    "Reward/action_rate",
    "Reward/action_magnitude",
    "SAC/alpha",
    "SAC/q1_mean",
    "SAC/q2_mean",
    "SAC/target_q_mean",
    "Loss/actor",
    "Loss/critic",
    "Loss/dynamics",
    "Encoder/grad_norm",
    "Encoder/z_mean",
    "Environment/attitude_error_roll_deg",
    "Environment/attitude_error_pitch_deg",
    "Train/episode_length",
]

MPC_PREFIX = "MPC/"


def compute_trend(values):
    """Determine trend from a list of scalar values."""
    if len(values) < 2:
        return "insufficient_data"

    # Compare last 20% vs first 20%
    n = len(values)
    chunk = max(1, n // 5)
    first_avg = sum(values[:chunk]) / chunk
    last_avg = sum(values[-chunk:]) / chunk

    if first_avg == 0:
        if last_avg == 0:
            return "flat"
        pct_change = float("inf") if last_avg > 0 else float("-inf")
    else:
        pct_change = (last_avg - first_avg) / abs(first_avg) * 100

    if abs(pct_change) < 5:
        return "flat"
    elif pct_change > 0:
        return f"increasing (+{pct_change:.1f}%)"
    else:
        return f"decreasing ({pct_change:.1f}%)"


def format_value(v):
    """Format a float value for display."""
    if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0):
        return f"{v:.4e}"
    return f"{v:.6f}"


def print_metric(tag, events):
    """Print summary for a single metric."""
    steps = [e.step for e in events]
    values = [e.value for e in events]

    first_val = values[0]
    last_val = values[-1]
    min_val = min(values)
    max_val = max(values)
    trend = compute_trend(values)

    print(f"  Tag: {tag}")
    print(f"    Steps logged : {len(values)} (step {steps[0]} -> {steps[-1]})")
    print(f"    First value  : {format_value(first_val)}")
    print(f"    Last value   : {format_value(last_val)}")
    print(f"    Min          : {format_value(min_val)}")
    print(f"    Max          : {format_value(max_val)}")
    print(f"    Trend        : {trend}")
    print()


def main():
    if not os.path.isdir(LOG_DIR):
        print(f"ERROR: Directory not found: {LOG_DIR}")
        sys.exit(1)

    # Load event accumulator
    print(f"Loading TensorBoard events from:\n  {LOG_DIR}\n")
    ea = event_accumulator.EventAccumulator(
        LOG_DIR,
        size_guidance={event_accumulator.SCALARS: 0},  # 0 = load all
    )
    ea.Reload()

    # Get all available scalar tags
    all_tags = sorted(ea.Tags().get("scalars", []))
    print(f"Total scalar tags found: {len(all_tags)}")
    print("=" * 72)

    # Determine global step range
    global_max_step = 0
    global_min_step = float("inf")
    for tag in all_tags:
        events = ea.Scalars(tag)
        if events:
            global_min_step = min(global_min_step, events[0].step)
            global_max_step = max(global_max_step, events[-1].step)

    if global_max_step > 0:
        print(f"Global step range: {global_min_step} -> {global_max_step}")
    print("=" * 72)
    print()

    # Match and print target tags
    matched_tags = set()

    # Exact / prefix match for TARGET_TAGS
    print("=" * 72)
    print("REQUESTED METRICS")
    print("=" * 72)
    print()

    for target in TARGET_TAGS:
        found = False
        for tag in all_tags:
            if tag == target or tag.startswith(target + "/"):
                events = ea.Scalars(tag)
                if events:
                    print_metric(tag, events)
                    matched_tags.add(tag)
                    found = True
        if not found:
            # Try case-insensitive or partial match
            for tag in all_tags:
                if target.lower() in tag.lower():
                    events = ea.Scalars(tag)
                    if events:
                        print_metric(tag, events)
                        matched_tags.add(tag)
                        found = True
                        break
            if not found:
                print(f"  Tag: {target}")
                print(f"    ** NOT FOUND **")
                print()

    # MPC/ prefixed tags
    print("=" * 72)
    print("MPC/ PREFIXED METRICS")
    print("=" * 72)
    print()
    mpc_found = False
    for tag in all_tags:
        if tag.startswith(MPC_PREFIX) and tag not in matched_tags:
            events = ea.Scalars(tag)
            if events:
                print_metric(tag, events)
                matched_tags.add(tag)
                mpc_found = True
    if not mpc_found:
        print("  No MPC/ prefixed tags found.")
        print()

    # Show all remaining unmatched tags for awareness
    remaining = [t for t in all_tags if t not in matched_tags]
    if remaining:
        print("=" * 72)
        print(f"OTHER AVAILABLE TAGS ({len(remaining)}):")
        print("=" * 72)
        for tag in remaining:
            events = ea.Scalars(tag)
            if events:
                print(f"  {tag}  ({len(events)} points, last={format_value(events[-1].value)})")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
