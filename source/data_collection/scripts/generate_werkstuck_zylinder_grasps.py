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
  - Column 1 (Y): finger opening direction  (= baseline: finger tips at ±Y·half_width)
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

Diametral grasp design — tangential approach, radial finger baseline:

  Z_grasp (approach): tangential direction with `tilt` downward component.
    After R_r2g: col1[2] = -Z_grasp[2] = sin(tilt) > 0 → disable_upside_down ✓

  Y_grasp (finger opening = baseline): inward radial  [-cos θ, -sin θ, 0]
    Finger tips at ±Y · half_width straddle the cylinder DIAMETRICALLY.
    When closing, both fingers press from opposite sides simultaneously — no
    net lateral force, cylinder stays centered.  (Old tangential-Y design had
    the closing force push the cylinder sideways along the table.)

  X_grasp = Y × Z  →  X_z = -cos(tilt) ≈ -0.985  (mostly downward)
    After R_r2g: col2 = X_grasp, col2[2] ≈ -0.985
    → pre_grasp = grasp + 0.985 · pre_grasp_distance upward ✓

Translation z = HEIGHT / 2 = 0.01 m:
  Gripper center targets the UPPER half of the cylinder.
  Cylinder centroid at z≈0.75 → gripper center at z≈0.76 (1 cm above table).
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

# ── Diametral grasps — tangential approach, fingers straddle diameter ─────────
#
# For each approach angle theta, the arm approaches tangentially while the two
# finger tips sit on OPPOSITE sides of the cylinder diameter (radial direction).
# Closing the gripper squeezes the cylinder symmetrically — no net push force.
#
# Z_grasp = tangential + tilt:  [-sin θ·cos t,  cos θ·cos t,  -sin t]
# Y_grasp = inward radial:      [-cos θ,        -sin θ,        0     ]
# X_grasp = Y × Z              (normalised, mostly downward)
#
#   X_z = -cos(tilt) ≈ -0.985  → pre_grasp ~9.85 cm ABOVE grasp ✓
#   col1[2] after R_r2g = sin(tilt) = 0.174 > 0 → disable_upside_down ✓

n_angles = 36
tilt = np.radians(10.0)

for i in range(n_angles):
    theta = i * 2.0 * np.pi / n_angles

    # Approach: tangential + `tilt` downward
    z_axis = np.array([-np.sin(theta) * np.cos(tilt),
                        np.cos(theta) * np.cos(tilt),
                        -np.sin(tilt)])

    # Finger baseline: inward radial → tips on opposite sides of diameter
    y_axis = np.array([-np.cos(theta), -np.sin(theta), 0.0])

    # Finger closing: X = Y × Z  (right-handed, mostly downward)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # Gripper center at the UPPER HALF of the cylinder — avoids table
    t = np.array([0.0, 0.0, HEIGHT / 2])
    grasp_poses.append(make_pose(x_axis, y_axis, z_axis, t))
    widths.append(GRASP_WIDTH)

print(f"Diametral grasps (tangential approach, {np.degrees(tilt):.0f}° tilt, "
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
