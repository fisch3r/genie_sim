#!/usr/bin/env python3
# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
"""
Konvertiert direkt aufgezeichnete Genie Sim Daten → LeRobot v2.1 Format.
Kein ROS bag, kein HDF5 Zwischenschritt.

Input:
  {recording_path}/direct_frames.json     — Joint States + Timestamps pro Frame
  {recording_path}/camera_direct/N/       — JPEG Frames pro Kamera
  {recording_path}/recording_info.json    — Task Metadaten

Output:
  {output_dir}/data/chunk-000/episode_XXXXXX.parquet
  {output_dir}/videos/chunk-000/observation.images.*/episode_XXXXXX.mp4
  {output_dir}/meta/{info,tasks,episodes}.jsonl + stats.json
"""
import argparse
import fcntl
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Joint Config ─────────────────────────────────────────────────────────────

def load_joint_config(robot_name: str = "G2") -> dict:
    config_path = Path(__file__).parents[2] / "config" / "robot_cfg" / "robot_joint_names.json"
    with open(config_path) as f:
        cfg = json.load(f)
    key = "G2" if "G2" in robot_name else "G1"
    return cfg[key]


def omnipicker_sim_to_real(g_pos: float) -> float:
    if g_pos > 0.75:
        return 0.0
    elif g_pos < 0.6:
        return 1.0
    return min(1.0, max(0.0, (g_pos - 0.6) / 0.15))


def build_state_vector(frame_data: dict, config: dict) -> np.ndarray:
    """Baut 21D State-Vektor: arm(14) + waist(5) + left_eff(1) + right_eff(1)."""
    name_to_pos = dict(zip(frame_data["joint_names"], frame_data["joint_positions"]))

    arm_pos = [name_to_pos.get(n, 0.0) for n in config["arm_joint_names"]]
    waist_pos = [name_to_pos.get(n, 0.0) for n in config["waist_joint_names"]]
    left_eff = omnipicker_sim_to_real(name_to_pos.get(config["left_effector_joint_name"], 0.0))
    right_eff = omnipicker_sim_to_real(name_to_pos.get(config["right_effector_joint_name"], 0.0))

    return np.array(arm_pos + waist_pos + [left_eff, right_eff], dtype=np.float32)


# ── Camera Utils ─────────────────────────────────────────────────────────────

def find_cameras(cam_dir: Path) -> list:
    """Ermittelt verfügbare Kamera-Namen aus dem ersten Frame-Ordner."""
    frame_dirs = sorted(
        [p for p in cam_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    if not frame_dirs:
        return []
    return sorted(jpg.stem for jpg in frame_dirs[0].glob("*_color.jpg"))


# ── Video Encoding ────────────────────────────────────────────────────────────

def encode_video(cam_dir: Path, cam_name: str, out_path: Path, fps: int = 30) -> bool:
    """Enkodiert JPEG-Sequenz → MP4 (H.264) mit ffmpeg."""
    frames = sorted(
        cam_dir.glob(f"*/{cam_name}.jpg"),
        key=lambda p: int(p.parent.name),
    )
    if not frames:
        logger.warning(f"Keine Frames für {cam_name}")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / f"_tmp_{cam_name}"
    tmp_dir.mkdir(exist_ok=True)
    try:
        for i, src in enumerate(frames):
            dst = tmp_dir / f"{i:06d}.jpg"
            if not dst.exists():
                os.symlink(src.resolve(), dst)

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmp_dir / "%06d.jpg"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg failed for {cam_name}: {result.stderr.decode()}")
            return False
        logger.info(f"Video encoded: {out_path} ({len(frames)} frames)")
        return True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── LeRobot Meta ──────────────────────────────────────────────────────────────

def load_or_init_meta(output_dir: Path, cameras: list, fps: int = 30):
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info_path = meta_dir / "info.json"
    tasks_path = meta_dir / "tasks.jsonl"
    episodes_path = meta_dir / "episodes.jsonl"

    state_dim = 21  # 14 arm + 5 waist + 2 effectors

    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        next_ep = info["total_episodes"]
        next_frame = info["total_frames"]
    else:
        info = {
            "codebase_version": "v2.1",
            "robot_type": "G2A",
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 1,
            "fps": fps,
            "features": {
                "observation.state": {"dtype": "float32", "shape": [state_dim], "names": None},
                "action": {"dtype": "float32", "shape": [state_dim], "names": None},
                "timestamp": {"dtype": "float32", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
                **{
                    f"observation.images.{cam}": {
                        "dtype": "video",
                        "shape": [3, 480, 640],
                        "names": ["channel", "height", "width"],
                        "video_info": {
                            "video.fps": float(fps),
                            "video.codec": "h264",
                            "video.pix_fmt": "yuv420p",
                            "video.is_depth_map": False,
                        },
                    }
                    for cam in cameras
                },
            },
        }
        next_ep = 0
        next_frame = 0

    return info, next_ep, next_frame, meta_dir, tasks_path, episodes_path


def append_meta(info: dict, meta_dir: Path, tasks_path: Path, episodes_path: Path,
                ep_meta: dict, task_name: str):
    if not tasks_path.exists():
        with open(tasks_path, "w") as f:
            f.write(json.dumps({"task_index": 0, "task": task_name}) + "\n")
    with open(episodes_path, "a") as f:
        f.write(json.dumps(ep_meta) + "\n")
    # info.json wird bereits in Phase 1 von convert() atomar geschrieben — kein erneuter Write


def update_stats(output_dir: Path, states: np.ndarray, actions: np.ndarray):
    stats_path = output_dir / "meta" / "stats.json"
    new_stats = {
        "observation.state": {
            "min": states.min(0).tolist(),
            "max": states.max(0).tolist(),
            "mean": states.mean(0).tolist(),
            "std": states.std(0).tolist(),
        },
        "action": {
            "min": actions.min(0).tolist(),
            "max": actions.max(0).tolist(),
            "mean": actions.mean(0).tolist(),
            "std": actions.std(0).tolist(),
        },
    }
    if stats_path.exists():
        with open(stats_path) as f:
            old = json.load(f)
        for key in ["observation.state", "action"]:
            new_stats[key]["min"] = np.minimum(old[key]["min"], new_stats[key]["min"]).tolist()
            new_stats[key]["max"] = np.maximum(old[key]["max"], new_stats[key]["max"]).tolist()
    with open(stats_path, "w") as f:
        json.dump(new_stats, f, indent=2)


# ── Hauptkonvertierung ────────────────────────────────────────────────────────

def convert(recording_path: str, task_info_path: str, output_dir: str, task_name: str = ""):
    recording_path = Path(recording_path)
    output_dir = Path(output_dir)

    # Task-Metadaten
    with open(task_info_path) as f:
        task_info = json.load(f)
    robot_name = task_info.get("robot_name", "G2")
    fps = int(task_info.get("fps", 30))
    if not task_name:
        task_name = task_info.get("task_name", "pick workpiece")

    # Joint Config
    config = load_joint_config(robot_name)

    # Direkte Frames laden
    frames_path = recording_path / "direct_frames.json"
    if not frames_path.exists():
        logger.error(f"direct_frames.json nicht gefunden: {frames_path}")
        sys.exit(1)
    with open(frames_path) as f:
        direct_frames = json.load(f)
    n_frames = len(direct_frames)
    logger.info(f"Frames geladen: {n_frames}")
    if n_frames == 0:
        logger.error("Keine Frames — Episode wird übersprungen")
        sys.exit(1)

    # Camera-Frames prüfen
    cam_dir = recording_path / "camera_direct"
    cameras = find_cameras(cam_dir) if cam_dir.exists() else []
    logger.info(f"Kameras: {cameras}")

    # State / Action aufbauen (in Sim: commanded ≈ actual)
    states = np.stack([build_state_vector(f, config) for f in direct_frames])
    actions = states.copy()

    # Lock-Datei für atomare Meta-Updates (verhindert Race Condition bei parallelen Konvertierungen)
    lock_path = output_dir / "meta" / ".convert.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Episode-Index reservieren + Parquet schreiben (unter Lock) ---
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)

        info, ep_idx, start_frame_idx, meta_dir, tasks_path, episodes_path = \
            load_or_init_meta(output_dir, cameras, fps)

        data_dir = output_dir / "data" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for fi, (state, action) in enumerate(zip(states, actions)):
            timestamp = float(fi) / fps
            row = {
                "observation.state": state.tolist(),
                "action": action.tolist(),
                "timestamp": timestamp,
                "frame_index": fi,
                "episode_index": ep_idx,
                "index": start_frame_idx + fi,
                "task_index": 0,
            }
            for cam in cameras:
                row[f"observation.images.{cam}"] = {
                    "path": f"videos/chunk-000/observation.images.{cam}/episode_{ep_idx:06d}.mp4",
                    "timestamp": timestamp,
                }
            rows.append(row)

        parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
        logger.info(f"Parquet gespeichert: {parquet_path}")

        # ep_idx sofort reservieren — kein anderer Prozess bekommt denselben Index
        info["total_episodes"] = ep_idx + 1
        info["total_frames"] = start_frame_idx + n_frames
        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)
    # Lock freigegeben — Video-Encoding läuft parallel (jeder Prozess hat einzigartigen ep_idx)

    # --- Phase 2: Videos enkodieren (außerhalb des Locks, dauert länger) ---
    for cam in cameras:
        video_dir = output_dir / "videos" / "chunk-000" / f"observation.images.{cam}"
        video_dir.mkdir(parents=True, exist_ok=True)
        encode_video(cam_dir, cam, video_dir / f"episode_{ep_idx:06d}.mp4", fps=fps)

    # --- Phase 3: Abschließende Meta-Aktualisierung (unter Lock) ---
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)

        ep_meta = {"episode_index": ep_idx, "tasks": [task_name], "length": n_frames}
        append_meta(info, meta_dir, tasks_path, episodes_path, ep_meta, task_name)
        update_stats(output_dir, states, actions)

    # task_result.json (wird von run_data_collection.py erwartet)
    result = {"task_name": task_name, "task_status": True, "return_code": 0, "metric_status": {}}
    with open(recording_path / "task_result.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"✅ Episode {ep_idx} fertig: {n_frames} Frames → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direkte Genie Sim Aufnahme → LeRobot v2.1")
    parser.add_argument("--recording_path", required=True, help="Pfad zur Episode (recording_data/...)")
    parser.add_argument("--task_info_path", required=True, help="Pfad zu recording_info.json")
    parser.add_argument("--output_dir", required=True, help="LeRobot Dataset Ausgabeverzeichnis")
    parser.add_argument("--task_name", default="", help="Task-Beschreibung")
    args = parser.parse_args()
    convert(args.recording_path, args.task_info_path, args.output_dir, args.task_name)
