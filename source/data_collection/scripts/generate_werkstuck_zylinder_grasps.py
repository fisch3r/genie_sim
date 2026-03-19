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
  [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]   (right-multiplied onto R_grasp)

After that transform the pre-grasp is computed from col 2 of the TRANSFORMED pose:
  pre_grasp = grasp - col2_world * pre_grasp_distance

  col2_after_R2G = X_grasp_original

  → For pre_grasp to be ABOVE the grasp:  X_grasp must point DOWNWARD (Z < 0).

  disable_upside_down check (right arm): col1_world[2] > 0
  col1_after_R2G = -Z_grasp_original[2]

  → For the filter to pass:  Z_grasp[2] < 0  (downward approach component).

Side-grasp design with downward tilt satisfies both:
  - X_grasp_z = -cos(tilt) << 0   → pre_grasp is above the grasp     ✓
  - Z_grasp_z = -sin(tilt) < 0    → disable_upside_down passes        ✓

Translation z = HEIGHT / 2:
  Gripper center targets the UPPER HALF of the cylinder (not the centroid).
  If the cylinder centroid sits at z=0.75 (table surface), the gripper center
  ends up at z=0.76 — 1 cm above the table — avoiding table collisions.
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

# ── Side grasps — fingers horizontal, approach with downward tilt ─────────────
#
# This design is compatible with the codebase's pre_grasp computation which
# uses col2 of the R_r2g-transformed pose (= X_grasp_original) as the retreat
# direction.
#
# Z_grasp: approach direction, mostly horizontal with `tilt` downward component.
#   After R_r2g: col1[2] = -Z_grasp[2] = sin(tilt) > 0 → passes disable_upside_down ✓
#
# X_grasp: finger closing direction, mostly DOWNWARD.
#   After R_r2g: col2 = X_grasp, col2[2] = -cos(tilt) < 0
#   → pre_grasp = grasp + cos(tilt) * pre_grasp_distance upward ✓
#
# Y_grasp: finger opening direction, horizontal.
#   → Both fingers at the same height, one on each side of the cylinder.
#
# Translation z = HEIGHT / 2 = 0.01 m:
#   Gripper center targets the UPPER half of the cylinder.
#   Even if the cylinder centroid is at z=0.75 (table surface), the gripper
#   center ends up at z=0.76 — 1 cm clearance above the table.

n_angles = 36
tilt = np.radians(10.0)

for i in range(n_angles):
    theta = i * 2.0 * np.pi / n_angles

    # Approach: horizontal + `tilt` downward
    z_axis = np.array([np.cos(theta) * np.cos(tilt),
                       np.sin(theta) * np.cos(tilt),
                       -np.sin(tilt)])

    # Finger opening: horizontal, perpendicular to approach
    y_axis = np.array([-np.sin(theta), np.cos(theta), 0.0])

    # Finger closing: X = Y × Z  (right-handed, mostly downward)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # Gripper center at the UPPER HALF of the cylinder — avoids table
    t = np.array([0.0, 0.0, HEIGHT / 2])
    grasp_poses.append(make_pose(x_axis, y_axis, z_axis, t))
    widths.append(GRASP_WIDTH)

print(f"Side grasps (horizontal fingers, {np.degrees(tilt):.0f}° tilt, "
      f"z_offset={HEIGHT/2:.3f}m = upper half): {n_angles}")

# ── Save ──────────────────────────────────────────────────────────────────────
grasp_poses = np.array(grasp_poses, dtype=np.float64)
widths = np.array(widths, dtype=np.float64)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump({"grasp_pose": grasp_poses, "width": widths}, f)

print(f"\nSaved {len(grasp_poses)} grasp poses → {OUTPUT_PATH}")
print(f"  grasp_pose.shape = {grasp_poses.shape}")
print(f"  width range: [{widths.min():.4f}, {widths.max():.4f}] m")
