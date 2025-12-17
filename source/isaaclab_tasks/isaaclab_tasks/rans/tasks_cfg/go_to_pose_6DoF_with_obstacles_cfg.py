# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import NoisyObservationsCfg

from .go_to_pose_6DoF_cfg import GoToPose3DCfg


@configclass
class GoToPose3DWithObstaclesCfg(GoToPose3DCfg):
    """Configuration for the GoToPosition with obstacles task in 3D space."""

    # Initial conditions
    spawn_min_dist: float = 3.0
    """Minimal distance between the spawn pose and the target pose in m. Defaults to 0.5 m."""
    spawn_max_dist: float = 7.0
    """Maximal distance between the spawn pose and the target pose in m. Defaults to 5.0 m."""
    
    # Tolerance
    minimum_obstacle_distance_to_target: float = 1.0
    """Minimal distance between the target and the obstacles. Defaults to 0.5 m."""
    minimum_obstacle_distance_to_robot: float = 0.5
    """Minimal distance between the robot and the obstacles. Defaults to 0.1 m."""
    collision_threshold: float = 3.0
    """Collision threshold. Defaults to 10.0"""
    maximum_robot_distance: float = 8.0
    """Maximal distance between the robot and the target pose. Defaults to 10 m."""
    

    # Obstacles
    obstacles_height: float = 0.5
    """Height of the obstacles. Defaults to 0.5 m."""
    min_obstacle_height: float = -2.0
    """Minimal height of the obstacles. Defaults to 0.0 m."""
    max_obstacle_height: float = 2.0
    """Maximal height of the obstacles. Defaults to 0.0 m."""
    minimum_point_distance = 0.05
    """The minimum distance between the points sampled to create the obstacles grid. Should be between 0 and 1. Smaller values can create more complex env."""
    max_num_vis_obstacles: int = 100
    """Max number of obstacles visible in the environment. Defaults to 8."""
    obstacle_radius: float = 0.2
    """Radius of the obstacles. Defaults to 0.2 m."""
    obstacles_storage_height_pos: float = -500.0
    """Height where to store the obstacles. Defaults to -2.0 m."""
    max_obstacle_distance_from_target: float = 10
    """Maximal distance between the target and the obstacles. Defaults to 10 m."""
    min_obstacle_distance_from_target: float = 0.3
    """Minimal distance between the target and the obstacles. Defaults to 1 m."""
    min_obstacle_distance_from_robot: float = 1
    """Minimal distance between the robot and the obstacles. Defaults to 1 m."""
    min_distance_between_obstacle: float = 1.5
    """Minimal distance between the obstacles. Defaults to 0.5 m."""
    collision_penalty: float = -50.0
    """Penalty for colliding with an obstacle. Defaults to -10.0."""
    collision_penalty_weight: float = 1.0
    """Weight for the collision penalty. Defaults to 1.0."""

    # Spaces
    observation_space: int = 33  # pos err xyz + orientn err rpy + root lin vel + root ang vel + collided signal +3 x closest obstacle dists xyz, TODO hight, radius

    gen_space: int = 8

