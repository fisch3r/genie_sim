# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from common.base_utils.logger import logger
from common.data_filter.runtime_checker.checker_base import SyncChecker
from common.data_filter.runtime_checker.checker_factory import register_checker


@register_checker("object_attached")
class ObjectAttachedChecker(SyncChecker):
    """
    Checks whether the target object is physically attached to the gripper
    (i.e. a FixedJoint was created between gripper and object mesh).

    This is more reliable than distance_to_target for verifying a successful
    grasp: it passes only when the object is actually between the fingers,
    not when the gripper merely touches or pushes the object.

    Parameters
    ----------
    object_id : str
        Semantic object ID as used in the task JSON, e.g.
        "geniesim_2025_target_building_block".  The checker looks for any
        prim path in attach_states that contains this string.
    """

    def __init__(self, object_id: str, **kwargs):
        super().__init__(**kwargs)
        self.object_id = object_id

    def check_impl(self) -> bool:
        attach_states: dict = self.command_controller.attach_states
        if not attach_states:
            logger.debug(f"object_attached: attach_states is empty — {self.object_id} not attached")
            return False
        for prim_path in attach_states:
            if self.object_id in prim_path:
                logger.info(f"object_attached: {self.object_id} attached via {prim_path}")
                return True
        logger.debug(f"object_attached: {self.object_id} not found in attach_states={list(attach_states.keys())}")
        return False
