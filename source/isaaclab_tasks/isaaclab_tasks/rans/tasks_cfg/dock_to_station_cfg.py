# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math

from isaaclab.utils import configclass

from isaaclab_tasks.rans.domain_randomization import NoisyObservationsCfg

from .go_to_pose_6DoF_cfg import GoToPose3DCfg

@configclass
class DockInStationWithObstaclesCfg(GoToPose3DCfg):

    # Collision Avoidance switch
    enable_collision_avoidance: bool = True
    
    # Easy evaluation toggle (spawns robot in front of docking station with upright orientation)
    easy_eval_inits: bool = False
    easy_spawn_min_dist: float = 0.5
    easy_spawn_max_dist: float = 1.0
    easy_transverse_jitter: float = 0.05

    # Tolerance
    collision_threshold: float = 3.0
    """Collision threshold. Defaults to 10.0"""
    max_num_vis_obstacles: int = 10
    """Maximum number of visible obstacles. Defaults to 10"""
    position_tolerance: float = 0.01
    """Tolerance for the position of the robot. Defaults to 1cm."""
    orientation_tolerance: float = math.pi / 180  # 1 degree
    """Tolerance for the orientation of the robot. Defaults to 5 degrees."""

    # Reward
    position_exponential_reward_coeff: float = 0.5
    orientation_exponential_reward_coeff: float = 1.0
    linear_velocity_min_value: float = 0.5
    linear_velocity_max_value: float = 2.0
    angular_velocity_min_value: float = 0.5
    angular_velocity_max_value: float = 20.0
    boundary_exponential_reward_coeff: float = 1.0
    pose_weight: float = 2.0
    linear_velocity_weight: float = -0.32 # -0.08
    angular_velocity_weight: float = -0.12 # -0.05
    boundary_weight: float = -10.0
    progress_weight: float = 1.5
    cuboid_violation_penalty: float = -0.075
    collision_penalty: float = -10.0
    negative_progress_penalty: float = -5.0

    # Randomization
    noisy_observation_cfg: NoisyObservationsCfg = NoisyObservationsCfg(
        enable=True,
        randomization_modes=["uniform"],
        slices=[(0, 3), (3, 9), (9, 12), (12, 15)],
        max_delta=[0.03, 0.01, 0.03, 0.03],
    )

    # Obstacles
    obstacles_storage_height_pos: float = -3.0
    """Height where to store the obstacles. Defaults to -2.0 m."""

    max_obstacle_distance_from_target: float = 2.0
    """Maximal distance between the target and the obstacles. Defaults to 10 m."""

    min_num_vis_obstacles: int = 1
    """Min number of obstacles visible in the environment. Defaults to 3."""

    max_num_vis_obstacles: int = 4
    """Max number of obstacles visible in the environment. Defaults to 8."""

    minimum_point_distance: float = 0.05
    """The minimum distance between the points sampled to create the obstacles grid. Should be between 0 and 1. Smaller values can create more complex env."""

    max_obstacle_distance_from_target: float = 0.5
    """Maximal distance between the target and the obstacles. Defaults to 10 m."""

    obstacle_radius: float = 0.1
    """Radius of the obstacles. Defaults to 0.2 m."""

    obstacles_height: float = 0.2
    """Height of the obstacles. Defaults to 0.5 m."""

    collision_threshold: float = 3.0
    """Threshold of contact force to consider a collision has happened. Defaults to 3.0"""

    collision_penalty: float = -10.0
    """Penalty applied when a collision with an obstacle is detected. Defaults to -10.0"""

    collision_penalty_weight: float = 1.0
    """Weight for the collision penalty in the total reward calculation. Defaults to 1.0"""

    min_obstacle_distance_from_target: float = 0.3
    """Minimal distance between the target and the obstacles. Defaults to 1.0 m."""

    min_obstacle_distance_from_robot: float = 0.3
    """Minimal distance between the robot and the obstacles. Defaults to 0.5 m."""

    max_hight_from_target: float = 0.5
    """Max hight of the obstacles from the target position. Defaults to 2.0 m. Positive and negative."""


    
    # Spaces
    observation_space: int = 24 # pos err xyz + orientn err rpy + root lin vel + root ang vel + 3 x closest obstacle dists xyz, TODO hight, radius
