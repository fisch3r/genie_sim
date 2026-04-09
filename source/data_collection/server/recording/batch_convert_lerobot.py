#!/usr/bin/env python3
# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
"""
Post-Processing: Konvertiert alle aufgezeichneten Episoden → LeRobot Format.

Ausführen NACH dem Sim-Ende:
    python batch_convert_lerobot.py [--recording_root PATH]

Standard-Suchpfad: ../recording_data/  (relativ zu diesem Script)

Erkennung:
  - "bereit":         direct_frames.json + recording_info.json vorhanden
  - "bereits fertig": task_result.json vorhanden (von direct_to_lerobot.py geschrieben)
  - "übersprungen":   keines der obigen → unvollständige Episode
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def find_episodes(recording_root: Path) -> list[Path]:
    """Gibt alle Episode-Verzeichnisse zurück die konvertierbar sind."""
    episodes = []
    if not recording_root.exists():
        print(f"[ERROR] Recording root nicht gefunden: {recording_root}")
        return episodes
    for entry in sorted(recording_root.iterdir()):
        if not entry.is_dir():
            continue
        has_frames = (entry / "direct_frames.json").exists()
        has_info = (entry / "recording_info.json").exists()
        if has_frames and has_info:
            episodes.append(entry)
    return episodes


def convert_episode(episode_dir: Path, lerobot_dir: Path, script_path: Path) -> bool:
    task_info_path = episode_dir / "recording_info.json"
    log_path = episode_dir / "lerobot_convert.log"
    print(f"  → Konvertiere: {episode_dir.name} ... ", end="", flush=True)
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--recording_path", str(episode_dir),
                "--task_info_path", str(task_info_path),
                "--output_dir", str(lerobot_dir),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    if result.returncode == 0:
        print("OK")
        return "ok"
    elif result.returncode == 2:
        print("gefiltert (metric)")
        return "filtered"
    else:
        print(f"FEHLER (rc={result.returncode}) — Log: {log_path}")
        return "error"


def main():
    parser = argparse.ArgumentParser(description="Batch-Konvertierung: Recording → LeRobot")
    parser.add_argument(
        "--recording_root",
        type=Path,
        default=Path(__file__).parents[2] / "recording_data",
        help="Pfad zum recording_data Verzeichnis (default: ../recording_data/)",
    )
    args = parser.parse_args()

    recording_root = args.recording_root.resolve()
    lerobot_dir = recording_root.parent / "lerobot_dataset"
    script_path = Path(__file__).parent / "direct_to_lerobot.py"

    print(f"Recording root: {recording_root}")
    print(f"LeRobot output: {lerobot_dir}")
    print()

    episodes = find_episodes(recording_root)
    if not episodes:
        print("Keine konvertierbaren Episoden gefunden.")
        return

    pending = [e for e in episodes if not (e / "task_result.json").exists()]
    done = [e for e in episodes if (e / "task_result.json").exists()]

    print(f"Episoden gesamt:      {len(episodes)}")
    print(f"Bereits konvertiert:  {len(done)}")
    print(f"Zu konvertieren:      {len(pending)}")
    print()

    if not pending:
        print("Alle Episoden bereits konvertiert.")
        return

    success = 0
    filtered = 0
    failed = 0
    total = len(pending)
    durations = []
    run_start = time.time()

    for i, ep in enumerate(pending):
        ep_start = time.time()

        if durations:
            avg_dur = sum(durations) / len(durations)
            remaining = avg_dur * (total - i)
            h, rem = divmod(int(remaining), 3600)
            m, s = divmod(rem, 60)
            eta_str = f"{h:02d}:{m:02d}:{s:02d}"
            print(f"  [{i + 1}/{total}] ETA {eta_str} | ø {avg_dur:.1f}s/ep", flush=True)
        else:
            print(f"  [{i + 1}/{total}]", flush=True)

        result = convert_episode(ep, lerobot_dir, script_path)

        ep_dur = time.time() - ep_start
        durations.append(ep_dur)

        if result == "ok":
            success += 1
        elif result == "filtered":
            filtered += 1
        else:
            failed += 1

    elapsed = time.time() - run_start
    h_e, rem_e = divmod(int(elapsed), 3600)
    m_e, s_e = divmod(rem_e, 60)
    print()
    print(f"Fertig: {success} konvertiert, {filtered} gefiltert (metric), {failed} fehlgeschlagen")
    print(f"Gesamtdauer: {h_e:02d}:{m_e:02d}:{s_e:02d}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
