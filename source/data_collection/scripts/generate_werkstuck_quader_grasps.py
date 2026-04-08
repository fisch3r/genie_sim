#!/usr/bin/env python3
"""
Generates grasp poses for benchmark_werkstuck_quader_000.

Object geometry:
  - Shape: cube, all sides 0.08 m (80 mm)
  - Origin: centroid = [0, 0, 0] in object frame  (cube sits on table,
            bottom face at Z = -0.04 m, top face at Z = +0.04 m)

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
  → For pre_grasp to be ABOVE the grasp:  X_grasp[2] must be < 0 (downward).

  disable_upside_down check (right arm): col1_world[2] > 0
  col1_after_R2G = -Z_grasp_original[2]
  → For the filter to pass:  Z_grasp[2] < 0  → need tilt > 0°.

Grasp design — side approach, flat-face baseline:

  Z_grasp (approach): outward radial + `tilt` downward component.
    [cos θ·cos t, sin θ·cos t, -sin t]
    After R_r2g: Y_gripper = -Z_grasp → inward radial ✓
    col1[2] = sin(tilt) > 0 → disable_upside_down ✓

  Y_grasp (baseline, tangential):
    [-sin θ, cos θ, 0]

  X_grasp = Y × Z:
    [-cos θ·sin t, -sin θ·sin t, -cos t]
    X_grasp[2] = -cos(tilt) ≈ -0.985 → pre_grasp ~9.85 cm above grasp ✓

Translation: gripper center at cube centroid [0, 0, 0].

Approach angles: 8 angles at 45° intervals.  For best contact against flat faces
use the 4 face-perpendicular angles (0°, 90°, 180°, 270°).  The 45° corner
angles are also included to give the IK solver more options.
"""

import os
import pickle
import numpy as np

# ── Object geometry ───────────────────────────────────────────────────────────
SIDE = 0.08          # 80 mm cube side length
GRASP_WIDTH = SIDE + 0.004   # gripper opening: side + 4 mm clearance = 0.084 m

OUTPUT_PATH = os.path.join(
    os.environ.get("SIM_ASSETS", os.path.expanduser("~/genie-sim/genie_sim/assets")),
    "interaction",
    "benchmark_werkstuck_quader_000",
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

# ── Side grasps — outward radial approach, 10° downward tilt ─────────────────
#
# 8 approach directions at 45° intervals.
# Face-perpendicular angles (0°, 90°, 180°, 270°) give the most stable flat-face
# contact; 45° corner angles provide additional IK candidates.
#
# Z_grasp = outward radial + tilt:  [cos θ·cos t,  sin θ·cos t,  -sin t]
# Y_grasp = tangential:             [-sin θ,        cos θ,         0    ]
# X_grasp = Y × Z                  [-cos θ·sin t, -sin θ·sin t,  -cos t]
#
#   X_z = -cos(tilt) ≈ -0.985  → pre_grasp ~9.85 cm ABOVE grasp ✓
#   col1[2] after R_r2g = sin(tilt) = 0.174 > 0 → disable_upside_down ✓

n_angles = 8
tilt = np.radians(10.0)

for i in range(n_angles):
    theta = i * 2.0 * np.pi / n_angles

    # Approach: outward radial + `tilt` downward
    z_axis = np.array([np.cos(theta) * np.cos(tilt),
                       np.sin(theta) * np.cos(tilt),
                       -np.sin(tilt)])

    # Finger baseline: tangential
    y_axis = np.array([-np.sin(theta), np.cos(theta), 0.0])

    # Finger closing: X = Y × Z  (right-handed, mostly downward)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # Gripper center above centroid — upper quarter of cube avoids table
    t = np.array([0.0, 0.0, SIDE * 0.25])
    grasp_poses.append(make_pose(x_axis, y_axis, z_axis, t))
    widths.append(GRASP_WIDTH)

print(f"Side grasps ({n_angles} angles, {np.degrees(tilt):.0f}° tilt, "
      f"z_offset={SIDE * 0.25:.3f}m = upper quarter): {n_angles}")

# ── Save ──────────────────────────────────────────────────────────────────────
grasp_poses = np.array(grasp_poses, dtype=np.float64)
widths = np.array(widths, dtype=np.float64)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump({"grasp_pose": grasp_poses, "width": widths}, f)

print(f"\nSaved {len(grasp_poses)} grasp poses → {OUTPUT_PATH}")
print(f"  grasp_pose.shape = {grasp_poses.shape}")
print(f"  width range: [{widths.min():.4f}, {widths.max():.4f}] m")
