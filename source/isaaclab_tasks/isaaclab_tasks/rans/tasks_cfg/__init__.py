# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# isort: off
from .task_core_cfg import TaskCoreCfg  # noqa: F401, F403

# isort: on
from isaaclab_tasks.rans.utils.misc import factory

from .go_through_poses_6DoF_cfg import GoThroughPoses3DCfg  # noqa: F401, F403
from .go_through_poses_cfg import GoThroughPosesCfg  # noqa: F401, F403
from .go_through_positions_6DoF_cfg import GoThroughPositions3DCfg  # noqa: F401, F403
from .go_through_positions_cfg import GoThroughPositionsCfg  # noqa: F401, F403
from .go_to_pose_6DoF_cfg import GoToPose3DCfg  # noqa: F401, F403
from .go_to_pose_cfg import GoToPoseCfg  # noqa: F401, F403
from .go_to_position_6DoF_cfg import GoToPosition3DCfg  # noqa: F401, F403
from .go_to_position_cfg import GoToPositionCfg  # noqa: F401, F403
from .go_to_position_6DoF_with_obstacles_cfg import GoToPosition3DWithObstaclesCfg  # noqa: F401, F403
from .go_to_position_with_obstacles_cfg import GoToPositionWithObstaclesCfg
from .push_block_cfg import PushBlockCfg  # noqa: F401, F403
from .race_gates_cfg import RaceGatesCfg  # noqa: F401, F403
from .track_velocities_6DoF_cfg import TrackVelocities3DCfg  # noqa: F401, F403
from .track_velocities_cfg import TrackVelocitiesCfg  # noqa: F401, F403
from .rendezvous_cfg import RendezvousCfg  # noqa: F401, F403
from .stabilization_6DoF_cfg import Stabilization3DCfg  # noqa: F401, F403
from .dock_to_station_cfg import DockInStationWithObstaclesCfg  # noqa: F401, F403
from .go_to_pose_6DoF_with_obstacles_cfg import GoToPose3DWithObstaclesCfg  # noqa: F401, F403
from .go_to_pose_6DoFBox_cfg import GoToPose3DBoxCfg  # noqa: F401, F403
from .velo_stabilization_6DoF_cfg import VeloStabilization3DCfg  # noqa: F401, F403

TASK_CFG_FACTORY = factory()
TASK_CFG_FACTORY.register("GoThroughPoses", GoThroughPosesCfg)
TASK_CFG_FACTORY.register("GoThroughPoses3D", GoThroughPoses3DCfg)
TASK_CFG_FACTORY.register("GoThroughPositions", GoThroughPositionsCfg)
TASK_CFG_FACTORY.register("GoToPose", GoToPoseCfg)
TASK_CFG_FACTORY.register("GoToPose3D", GoToPose3DCfg)
TASK_CFG_FACTORY.register("GoToPosition", GoToPositionCfg)
TASK_CFG_FACTORY.register("GoToPosition3D", GoToPosition3DCfg)
TASK_CFG_FACTORY.register("PushBlock", PushBlockCfg)
TASK_CFG_FACTORY.register("TrackVelocities", TrackVelocitiesCfg)
TASK_CFG_FACTORY.register("TrackVelocities3D", TrackVelocities3DCfg)
TASK_CFG_FACTORY.register("GoToPositionWithObstacles", GoToPositionWithObstaclesCfg)
TASK_CFG_FACTORY.register("GoToPosition3DWithObstacles", GoToPosition3DWithObstaclesCfg)
TASK_CFG_FACTORY.register("RaceGates", RaceGatesCfg)
TASK_CFG_FACTORY.register("Rendezvous", RendezvousCfg)
TASK_CFG_FACTORY.register("Stabilization3D", Stabilization3DCfg)
TASK_CFG_FACTORY.register("DockInStationWithObstacles", DockInStationWithObstaclesCfg)
TASK_CFG_FACTORY.register("GoToPose3DWithObstacles", GoToPose3DWithObstaclesCfg)
TASK_CFG_FACTORY.register("GoToPose3DBox", GoToPose3DBoxCfg)
TASK_CFG_FACTORY.register("VeloStabilization3D", VeloStabilization3DCfg)
