#!/usr/bin/env python3
"""
Analysiert Greifer-Positionen aus aufgezeichneten Episoden.

Liest direct_frames.json jeder Episode und extrahiert den minimalen
gripper_r_outer_joint1-Wert im letzten Drittel der Episode (= Greifphase).

Aufruf:
    python scripts/analyze_gripper_positions.py
    python scripts/analyze_gripper_positions.py --recording_root /pfad/zu/recording_data
"""

import argparse
import json
from pathlib import Path

GRIPPER_JOINT = "idx81_gripper_r_outer_joint1"


def analyze_episode(episode_dir: Path) -> float | None:
    frames_path = episode_dir / "direct_frames.json"
    if not frames_path.exists():
        return None
    with open(frames_path) as f:
        frames = json.load(f)
    if not frames:
        return None

    # Letztes Drittel = Greif- und Hebephase
    start = len(frames) * 2 // 3
    grasp_frames = frames[start:]

    positions = []
    for frame in grasp_frames:
        names = frame.get("joint_names", [])
        positions_raw = frame.get("joint_positions", [])
        if GRIPPER_JOINT in names:
            idx = names.index(GRIPPER_JOINT)
            positions.append(positions_raw[idx])

    if not positions:
        return None
    return min(positions)  # Minimum = am weitesten geschlossen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recording_root",
        type=Path,
        default=Path(__file__).parents[2] / "recording_data",
    )
    args = parser.parse_args()

    recording_root = args.recording_root.resolve()
    if not recording_root.exists():
        print(f"[ERROR] Nicht gefunden: {recording_root}")
        return

    episodes = sorted(
        [e for e in recording_root.iterdir() if e.is_dir()],
        key=lambda p: p.name,
    )

    results = []
    skipped = 0
    for ep in episodes:
        val = analyze_episode(ep)
        if val is None:
            skipped += 1
            continue
        results.append((ep.name, val))

    if not results:
        print("Keine auswertbaren Episoden gefunden.")
        return

    values = [v for _, v in results]
    values_sorted = sorted(values)

    print(f"Episoden ausgewertet : {len(results)}")
    print(f"Übersprungen          : {skipped}")
    print()
    print(f"  Minimum  : {min(values):.4f}  (Greifer am weitesten geschlossen)")
    print(f"  Maximum  : {max(values):.4f}  (Greifer am weitesten offen)")
    print(f"  Median   : {values_sorted[len(values_sorted)//2]:.4f}")
    print(f"  P10      : {values_sorted[len(values_sorted)//10]:.4f}")
    print(f"  P25      : {values_sorted[len(values_sorted)//4]:.4f}")
    print(f"  P75      : {values_sorted[3*len(values_sorted)//4]:.4f}")
    print(f"  P90      : {values_sorted[9*len(values_sorted)//10]:.4f}")
    print()
    print("Verteilung (Histogramm):")

    bucket_size = 0.05
    buckets: dict[float, int] = {}
    for v in values:
        bucket = round(int(v / bucket_size) * bucket_size, 2)
        buckets[bucket] = buckets.get(bucket, 0) + 1
    for b in sorted(buckets):
        bar = "█" * buckets[b]
        print(f"  {b:.2f}–{b+bucket_size:.2f}  {bar} ({buckets[b]})")

    print()
    print("Empfehlung für min_position:")
    p25 = values_sorted[len(values_sorted) // 4]
    suggestion = round(p25 - 0.05, 2)
    print(f"  P25={p25:.3f} → Vorschlag: {suggestion:.2f}")
    print(f"  (25% der Episoden haben Greifer offener als P25 → diese wären False Positives)")


if __name__ == "__main__":
    main()
