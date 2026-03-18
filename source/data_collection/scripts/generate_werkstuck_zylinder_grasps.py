#!/usr/bin/env python3
"""
Generates grasp poses for benchmark_werkstuck_zylinder_000.

Object geometry (from object_parameters.json):
  - Shape: cylinder, Z-axis up (standing upright)
  - Diameter: 0.03 m  (radius 0.015 m)
  - Height:   0.02 m
  - Origin:   centroid = [0, 0, 0] in object frame

Grasp pose convention (GraspNet / AgiBot):
  - 4x4 matrix in object coordinates
  - Column 0 (X): finger closing direction
  - Column 1 (Y): finger opening direction
  - Column 2 (Z): approach direction (pre-grasp → grasp)
  - Column 3:     translation (gripper center between fingers)

robot_gripper_2_grasp_gripper for omnipicker:
  [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]

After that transform the disable_upside_down check (right arm) keeps poses
where the world Y-axis of the pose has Z-component > 0 — which corresponds
to approach direction having negative Z (coming from above or horizontally
with a downward component).

Two grasp families are generated:
  1. Top-down  — approach = [0,0,-1], fingers horizontal (36 angles × N heights)
  2. Side      — approach horizontal, finger opening vertical (36 angles)
"""

import os
import pickle
import numpy as np

# ── Object geometry ───────────────────────────────────────────────────────────
RADIUS = 0.015   # half of 0.03 m diameter
HEIGHT = 0.02    # 0.02 m tall
GRASP_WIDTH = RADIUS * 2 + 0.004  # gripper opening: diameter + 4 mm clearance

OUTPUT_PATH = os.path.join(
    os.environ.get("SIM_ASSETS", os.path.expanduser("~/genie-sim/genie_sim/assets")),
    "interaction",
    "benchmark_werkstuck_zylinder_000",
    "grasp_pose",
    "grasp_pose.pkl",
)


def make_pose(x_axis, y_axis, z_axis, translation):
    """Assemble a 4×4 grasp pose from column axes and translation."""
    T = np.eye(4)
    T[:3, 0] = x_axis / np.linalg.norm(x_axis)
    T[:3, 1] = y_axis / np.linalg.norm(y_axis)
    T[:3, 2] = z_axis / np.linalg.norm(z_axis)
    T[:3, 3] = translation
    return T


grasp_poses = []
widths = []

# ── Side grasps — fingers horizontal, approach with 10° downward tilt ────────
#
# Approach (Z in grasp DB): horizontal + 10° downward tilt.
# After robot_gripper_2_grasp_gripper: Robot Y = -Z_grasp = mostly horizontal
# with slight upward Z component = sin(10°) > 0 → passes disable_upside_down ✓
#
# Finger opening (Y in grasp DB): horizontal, perpendicular to approach.
# → Both fingers at the same height, one on each side of the cylinder.
# → No finger goes below the table.
#
# Translation: [0, 0, 0] — USD is "Aligned.usd" so origin is already at centroid.

n_angles = 36
tilt = np.radians(10.0)

for i in range(n_angles):
    theta = i * 2.0 * np.pi / n_angles

    # Approach: horizontal direction + 10° downward tilt
    z_axis = np.array([np.cos(theta) * np.cos(tilt),
                       np.sin(theta) * np.cos(tilt),
                       -np.sin(tilt)])
    z_axis /= np.linalg.norm(z_axis)

    # Finger opening: horizontal, perpendicular to approach direction
    y_axis = np.array([-np.sin(theta), np.cos(theta), 0.0])

    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # USD origin is at centroid (Aligned.usd), so no Z offset needed
    t = np.array([0.0, 0.0, 0.0])
    grasp_poses.append(make_pose(x_axis, y_axis, z_axis, t))
    widths.append(GRASP_WIDTH)

print(f"Side grasps (horizontal fingers, 10° tilt, centroid height): {n_angles}")

# ── Save ──────────────────────────────────────────────────────────────────────
grasp_poses = np.array(grasp_poses, dtype=np.float64)
widths = np.array(widths, dtype=np.float64)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump({"grasp_pose": grasp_poses, "width": widths}, f)

print(f"\nSaved {len(grasp_poses)} grasp poses → {OUTPUT_PATH}")
print(f"  grasp_pose.shape = {grasp_poses.shape}")
print(f"  width range: [{widths.min():.4f}, {widths.max():.4f}] m")
