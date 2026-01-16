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

    # Tolerance
    minimum_obstacle_distance_to_target: float = 1.0
    """Minimal distance between the target and the obstacles. Defaults to 0.5 m."""
    minimum_obstacle_distance_to_robot: float = 0.5
    """Minimal distance between the robot and the obstacles. Defaults to 0.1 m."""
    collision_threshold: float = 3.0
    """Collision threshold. Defaults to 10.0"""

    # Obstacles
    obstacles_height: float = 0.1
    """Height of the obstacles. Defaults to 0.1 m."""
    min_obstacle_height: float = -2.0
    """Minimal height of the obstacles. Defaults to 0.0 m."""
    max_obstacle_height: float = 2.0
    """Maximal height of the obstacles. Defaults to 0.0 m."""
    minimum_point_distance = 0.05
    """The minimum distance between the points sampled to create the obstacles grid. Should be between 0 and 1. Smaller values can create more complex env."""
    max_num_vis_obstacles: int = 8
    """Max number of obstacles visible in the environment. Defaults to 8."""
    min_num_obstacles: int = 3
    """Min number of obstacles visible in the environment. Defaults to 3."""
    obstacle_radius: float = 0.1
    """Radius of the obstacles. Defaults to 0.1 m."""
    obstacles_storage_height_pos: float = -500.0
    """Height where to store the obstacles. Defaults to -2.0 m."""
    max_obstacle_distance_from_target: float = 10
    """Maximal distance between the target and the obstacles. Defaults to 10 m."""
    min_obstacle_distance_from_target: float = 1.0
    """Minimal distance between the target and the obstacles. Defaults to 1 m."""
    min_obstacle_distance_from_robot: float = 0.8
    """Minimal distance between the robot and the obstacles. Defaults to 1 m."""
    min_distance_between_obstacle: float = 1.5
    """Minimal distance between the obstacles. Defaults to 0.5 m."""
    collision_penalty: float = -50.0
    """Penalty for colliding with an obstacle. Defaults to -10.0."""
    collision_penalty_weight: float = 1.0
    """Weight for the collision penalty. Defaults to 1.0."""


    # ISS box
    iss_box_storage_height_pos: float = -500.0
    """Height where to store the ISS box walls. Defaults to -500.0 m."""

    # Spaces
    observation_space: int = 31  # pos err xyz + orientn err rpy + root lin vel + root ang vel + collision signal + 3 x (xyz distance and magnitude [4]), TODO height, radius

    gen_space: int = 8
