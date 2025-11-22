# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import NoisyObservationsCfg

from .go_to_position_6DoF_cfg import GoToPosition3DCfg


@configclass
class GoToPosition3DWithObstaclesCfg(GoToPosition3DCfg):
    """Configuration for the GoToPosition with obstacles task in 3D space."""

    # Obstacles
    obstacles_storage_height_pos: float = -3.0
    """Height where to store the obstacles. Defaults to -2.0 m."""

    max_obstacle_distance_from_target: float = 10.0
    """Maximal distance between the target and the obstacles. Defaults to 10 m."""

    max_num_vis_obstacles: int = 8
    """Max number of obstacles visible in the environment. Defaults to 8."""

    minimum_point_distance: float = 0.05
    """The minimum distance between the points sampled to create the obstacles grid. Should be between 0 and 1. Smaller values can create more complex env."""

    max_obstacle_distance_from_target: float = 10.0
    """Maximal distance between the target and the obstacles. Defaults to 10 m."""

    obstacle_radius: float = 0.2
    """Radius of the obstacles. Defaults to 0.2 m."""

    obstacles_height: float = 0.5
    """Height of the obstacles. Defaults to 0.5 m."""


    # Spaces
    observation_space: int = 24  # pos err xyz + orientn err rpy + root lin vel + root ang vel + 3 x closest obstacle dists xyz, hight, radius
