# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.logger import logger
from common.data_filter.runtime_checker.checker_base import SyncChecker
from common.data_filter.runtime_checker.checker_factory import register_checker


@register_checker("gripper_not_fully_closed")
class GripperNotFullyClosedChecker(SyncChecker):
    """
    Verifies that the gripper control joint has NOT reached its fully-closed
    position after the grasp attempt.  When an object is properly held between
    the fingers the joint is blocked by the object and stays above a threshold.
    When the gripper closes on air (object beside/outside the fingers) it
    reaches near-zero — this checker returns False in that case.

    Parameters
    ----------
    arm : str
        "right" or "left"
    min_position : float
        Minimum joint value that counts as "something is between the fingers".
        Default 0.15 (gripper open=0.95, closed=0.0).
    """

    def __init__(self, arm: str = "right", min_position: float = 0.15, **kwargs):
        super().__init__(name="gripper_not_fully_closed", **kwargs)
        self.arm = arm
        self.min_position = min_position

    def check_impl(self) -> bool:
        cc = self.command_controller
        # Prim path like "/G2/joints/idx81_gripper_r_outer_joint1"
        joint_prim = cc.gripper_controll_joint.get(self.arm)
        if joint_prim is None:
            logger.warning(f"gripper_not_fully_closed: no control joint for arm={self.arm}")
            return False

        joint_name = joint_prim.split("/")[-1]
        dof_names = list(cc.dof_names)
        if joint_name not in dof_names:
            logger.warning(f"gripper_not_fully_closed: joint '{joint_name}' not in dof_names")
            return False

        idx = dof_names.index(joint_name)
        articulation = cc._initialize_articulation()
        joint_positions = articulation.get_joint_positions()
        current_pos = float(joint_positions[idx])

        logger.info(
            f"gripper_not_fully_closed: arm={self.arm}, joint={joint_name}, "
            f"pos={current_pos:.4f}, min={self.min_position}"
        )
        return current_pos >= self.min_position
