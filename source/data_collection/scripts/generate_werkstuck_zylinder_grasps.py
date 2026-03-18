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
    os.path.expanduser("~"),
    "genie-sim/genie_sim/assets/objects/benchmark/werkstuck"
    "/benchmark_werkstuck_zylinder_000/grasp_pose/grasp_pose.pkl",
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

# ── 1. Top-down grasps ────────────────────────────────────────────────────────
# Approach from above: Z = [0, 0, -1]
# Finger opening (Y) rotates around the cylinder Z-axis every 10°
# Grasp center at cylinder centroid.
#
# After robot_gripper_2_grasp_gripper transform, Y_robot = -Z_grasp = [0,0,1]
# → Z-component of robot Y-axis = 1 > 0  → passes disable_upside_down ✓

n_angles = 36
for i in range(n_angles):
    theta = i * 2.0 * np.pi / n_angles
    y_axis = np.array([np.cos(theta), np.sin(theta), 0.0])   # finger opening
    z_axis = np.array([0.0, 0.0, -1.0])                      # approach downward
    x_axis = np.cross(y_axis, z_axis)                         # closing direction

    # Grasp center at cylinder centroid (object frame origin)
    t = np.array([0.0, 0.0, 0.0])
    grasp_poses.append(make_pose(x_axis, y_axis, z_axis, t))
    widths.append(GRASP_WIDTH)

print(f"Top-down grasps: {n_angles}")

# ── 2. Side grasps — fingers vertical ────────────────────────────────────────
# Approach horizontally from every 10° direction.
# Finger opening (Y) = [0, 0, 1] (vertical) → one finger above, one below
# the cylinder's equator, consistent with its 2 cm height.
#
# After robot_gripper_2_grasp_gripper: Y_robot = -Z_grasp = -horizontal
# Z-component of Y_robot ≈ 0 → borderline for disable_upside_down.
# We tilt Z slightly downward (-5°) so the approach has a small downward
# component, guaranteeing it passes the right-arm upside-down filter.

tilt = np.radians(5.0)   # 5° downward tilt on the approach

for i in range(n_angles):
    theta = i * 2.0 * np.pi / n_angles
    # Approach: horizontal direction + slight downward tilt
    z_axis = np.array([np.cos(theta) * np.cos(tilt),
                       np.sin(theta) * np.cos(tilt),
                       -np.sin(tilt)])
    z_axis /= np.linalg.norm(z_axis)

    y_axis = np.array([0.0, 0.0, 1.0])   # finger opening = vertical
    x_axis = np.cross(y_axis, z_axis)

    # Grasp center at cylinder centroid
    t = np.array([0.0, 0.0, 0.0])
    grasp_poses.append(make_pose(x_axis, y_axis, z_axis, t))
    widths.append(GRASP_WIDTH)

print(f"Side grasps (vertical fingers, 5° tilt): {n_angles}")

# ── Save ──────────────────────────────────────────────────────────────────────
grasp_poses = np.array(grasp_poses, dtype=np.float64)
widths = np.array(widths, dtype=np.float64)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump({"grasp_pose": grasp_poses, "width": widths}, f)

print(f"\nSaved {len(grasp_poses)} grasp poses → {OUTPUT_PATH}")
print(f"  grasp_pose.shape = {grasp_poses.shape}")
print(f"  width range: [{widths.min():.4f}, {widths.max():.4f}] m")
