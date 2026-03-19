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
where the world Y-axis of the pose has Z-component > 0.

For top-down grasps:
  Z_grasp = [0, 0, -1]  →  Robot_Y = -Z_grasp = [0, 0, 1]  →  Z-component = 1 > 0  ✓

Top-down approach:
  - Gripper descends from above, fingers close horizontally around the cylinder.
  - No table collision risk regardless of cylinder height.
  - Translation Z = HEIGHT/2: gripper center targets the upper half of the cylinder,
    ensuring the finger tips are well above the table surface.
  - 36 approach angles around the vertical axis.
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

# ── Top-down grasps — approach from above, fingers close horizontally ─────────
#
# Approach (Z in grasp DB): [0, 0, -1] — straight down.
# After robot_gripper_2_grasp_gripper:
#   Robot_Y = -Z_grasp = [0, 0, 1]  →  Z-component = 1 > 0  → passes disable_upside_down ✓
#
# Finger opening (Y in grasp DB): horizontal [cos(θ), sin(θ), 0].
# → Both fingers at the same height, closing from the sides of the cylinder.
# → Gripper body is above the cylinder — no table collision possible.
#
# Translation Z = HEIGHT/2 = 0.01 m:
#   Gripper center targets the upper half of the cylinder (1 cm above centroid).
#   Combined with pre_grasp_distance the approach starts well above the object.
#   Even if the cylinder centroid sits at table height (z=0.75), the gripper
#   center ends up at z=0.76 — 1 cm clear of the table.

n_angles = 36
Z_OFFSET = HEIGHT / 2  # 0.01 m — target upper half to stay clear of table

z_axis = np.array([0.0, 0.0, -1.0])  # approach straight down

for i in range(n_angles):
    theta = i * 2.0 * np.pi / n_angles

    # Finger opening: horizontal, rotates around vertical axis
    y_axis = np.array([np.cos(theta), np.sin(theta), 0.0])

    # X = Y × Z  (right-handed)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # Gripper center at upper half of cylinder — clear of table
    t = np.array([0.0, 0.0, Z_OFFSET])
    grasp_poses.append(make_pose(x_axis, y_axis, z_axis, t))
    widths.append(GRASP_WIDTH)

print(f"Top-down grasps (horizontal fingers, approach=[0,0,-1], z_offset={Z_OFFSET:.3f}m): {n_angles}")

# ── Save ──────────────────────────────────────────────────────────────────────
grasp_poses = np.array(grasp_poses, dtype=np.float64)
widths = np.array(widths, dtype=np.float64)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump({"grasp_pose": grasp_poses, "width": widths}, f)

print(f"\nSaved {len(grasp_poses)} grasp poses → {OUTPUT_PATH}")
print(f"  grasp_pose.shape = {grasp_poses.shape}")
print(f"  width range: [{widths.min():.4f}, {widths.max():.4f}] m")
