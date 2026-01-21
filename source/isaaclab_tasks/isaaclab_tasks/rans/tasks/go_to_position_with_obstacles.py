# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import math
import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.markers import SPHERE_CFG, VisualizationMarkers, VisualizationMarkersCfg, PIN_SPHERE_CFG

from isaaclab.utils import math as math_utils

from isaaclab.scene import InteractiveScene

from isaaclab_tasks.rans import GoToPositionWithObstaclesCfg
from isaaclab_tasks.rans.utils import ObjectStorage

from .task_core import TaskCore

import torch.nn.functional as F

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class GoToPositionWithObstaclesTask(TaskCore):
    """
    Implements the GoToPosition task. The robot has to reach a target position and keep it.
    """

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        task_cfg: GoToPositionWithObstaclesCfg = GoToPositionWithObstaclesCfg(),
        task_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        env_ids: torch.Tensor | None = None,
        decimation: int = 1,
        num_tasks: int = 1,
    ) -> None:
        """
        Initializes the GoToPosition task.

        Args:
            task_cfg: The configuration of the task.
            task_uid: The unique id of the task.
            num_envs: The number of environments.
            device: The device on which the tensors are stored.
            task_id: The id of the task.
            env_ids: The ids of the environments used by this task."""

        super().__init__(
            scene=scene,
            task_uid=task_uid,
            num_envs=num_envs,
            device=device,
            env_ids=env_ids,
            decimation=decimation,
            num_tasks=num_tasks,
        )
        
        self._task_cfg = task_cfg

        # Defines the observation and actions space sizes for this task
        self._dim_task_obs = self._task_cfg.observation_space
        self._dim_gen_act = self._task_cfg.gen_space

        # Buffers
        self.initialize_buffers(env_ids=env_ids)

        # Obstacles
        self.obstacles_generator = ObjectStorage(
            num_envs=num_envs,
            max_num_vis_objects_in_env=self._task_cfg.max_num_vis_obstacles,
            store_height=self._task_cfg.obstacles_storage_height_pos, 
            rng=self._rng,
            device=device,
        )
        self.batch_indices = (
            torch.arange(num_envs, device=self._device).unsqueeze(1).expand(-1, self._task_cfg.max_num_vis_obstacles)
        )

        self._num_cells = int(1.0 / (self._task_cfg.minimum_point_distance * 2))

        self.design_scene()

    @property
    def eval_data_keys(self) -> list[str]:
        """
        Returns the keys of the data used for evaluation.

        Returns:
            list[str]: The keys of the data used for evaluation."""
        
        return [
            "target_position", 
            "pos_obstacles_in_env",
            "position_distance",
            "cos_heading_to_target_error",
            "sin_heading_to_target_error",
            "obstacles_distance",
            "cos_obstacles_heading_error",
            "sin_obstacles_heading_error",
        ]
    
    @property
    def eval_data_specs(self)->dict[str, list[str]]:
        return {
            "target_position": [".x.m", ".y.m"],
            "pos_obstacles_in_env": [coord for i in range(self._task_cfg.max_num_vis_obstacles) for coord in (f".x_{i}.m", f".y_{i}.m", f".z_{i}.m", f".qx_{i}.u", f".qy_{i}.u", f".qz_{i}.u", f".qw_{i}.u")],
            "position_distance": [".distance.m"],
            "cos_heading_to_target_error": [".cos(heading).u"],
            "sin_heading_to_target_error": [".sin(heading).u"],
            "obstacles_distance": [".distance.m"],
            "cos_obstacles_heading_error": [".cos(heading).u"],
            "sin_obstacles_heading_error": [".sin(heading).u"],
        }
    
    @property
    def eval_data(self) -> dict:
        """
        Returns the data used for evaluation.

        Returns:
            dict: The data used for evaluation."""
        
        return {
            "target_position": self._target_positions,
            "pos_obstacles_in_env": self._pos_obstacles_in_env,
            "position_distance": self._task_data[:, 0],
            "cos_heading_to_target_error": self._task_data[:, 1],
            "sin_heading_to_target_error": self._task_data[:, 2],
            "obstacles_distance": self._task_data[:, 6:9],
            "cos_obstacles_heading_error": self._task_data[:, 9:12],
            "sin_obstacles_heading_error": self._task_data[:, 12:15],
        }

    def register_robot(self, robot) -> None:
        self._robot = robot

    def register_sensors(self) -> None:
        filters = [f"/World/envs/env_.*/Obstacles/cylinder_{i}" for i in range(self._task_cfg.max_num_vis_obstacles)]
        self._robot.activateSensors("contacts", filters)
        self._robot.register_sensors()
        
    def design_scene(self) -> None:
        """
        Initializes the obstacles for the task.
        """
        
        opacity = 1
        self.wall_thickness = 0.05
        self.wall_height = 2.0
        self.wall_width = 3.0
        self.wall_length = 6.4
        self.x_shift = -0.05
        self.height_from_floor = 0.3

        prim_utils.create_prim("/World/envs/env_0/Obstacles", "Xform")

        rigid_objects = {}
        MIN_MASS = 0.5
        MAX_MASS = 70.0
        MIN_BRIGHTNESS_SCALE = 0.2  # Darkest shade (prevents pure black)
        MAX_BRIGHTNESS_SCALE = 1.0  # Lightest shade
        low, high = -self.wall_length, self.wall_length  # X and Y range
        min_z, max_z = self.wall_thickness + self.height_from_floor, self.wall_height + self.height_from_floor  # Z range (height variation)

        for i in range(self._task_cfg.max_num_vis_obstacles):

            position = low + (high - low) * torch.rand(3)
            position[2] = self._task_cfg.obstacles_storage_height_pos #min_z + (max_z - min_z) * torch.rand(1) 
            mass = torch.randint(int(MIN_MASS), int(MAX_MASS), (1,)).item()
            # Color depending on mass. Lighter color red for low mass, darker blue for high mass
            mass_tensor = torch.tensor([mass], dtype=torch.float32)
            normalized_mass = (mass_tensor - MIN_MASS) / (MAX_MASS - MIN_MASS)
            normalized_mass = torch.clamp(normalized_mass, 0.0, 1.0).item()
            R = 1.0 - normalized_mass  # Red is strong for low mass
            G = 0.0                    # Keep green at zero for a pure red-blue transition
            B = normalized_mass        # Blue is strong for high mass
            color_tuple = (R, G, B)

            rigid_objects[f"obstacle_{i}"] = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Obstacles/cylinder_{i}",
                spawn=sim_utils.CylinderCfg(
                    radius=self._task_cfg.obstacle_radius,
                    height=self._task_cfg.obstacles_height,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color_tuple),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
            )

        obstacles_cfg = RigidObjectCollectionCfg(rigid_objects=rigid_objects)

        self.obstacles = RigidObjectCollection(obstacles_cfg)

    def initialize_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Initializes the buffers used by the task.

        Args:
            env_ids: The ids of the environments used by this task."""
        super().initialize_buffers(env_ids)
        self._position_error = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self._position_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._previous_position_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._target_positions = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self._markers_pos = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.initial_velocity = torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)
        self._half_init_lin_vel_x = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)
        self._half_init_lin_vel_y = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)
        self._half_init_ang_vel = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)
        self._previous_position_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._pos_obstacles_in_env = torch.zeros(
            (self._num_envs, self._task_cfg.max_num_vis_obstacles, 7), device=self._device, dtype=torch.float32) # [x, y, z, qx, qy, qz, qw]

    def create_logs(self) -> None:
        """
        Creates a dictionary to store the training statistics for the task."""
        super().create_logs()
        self.scalar_logger.add_log("task_state", "GoToPositionWithObstacles/AVG/normed_linear_velocity", "mean")
        self.scalar_logger.add_log("task_state", "GoToPositionWithObstacles/AVG/absolute_angular_velocity", "mean")
        self.scalar_logger.add_log("task_state", "GoToPositionWithObstacles/EMA/position_distance", "ema")
        self.scalar_logger.add_log("task_state", "GoToPositionWithObstacles/EMA/boundary_distance", "ema")
        self.scalar_logger.add_log("task_state", "GoToPositionWithObstacles/AVG/target_heading_error", "mean")
        self.scalar_logger.add_log("task_state", "GoToPositionWithObstacles/AVG/curriculum_level", "mean")
        self.scalar_logger.add_log("task_state", "GoToPositionWithObstacles/AVG/num_visible_obstacles", "mean")


        self.scalar_logger.add_log("task_reward", "GoToPositionWithObstacles/AVG/position", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPositionWithObstacles/AVG/heading", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPositionWithObstacles/AVG/linear_velocity", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPositionWithObstacles/AVG/angular_velocity", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPositionWithObstacles/AVG/boundary", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPositionWithObstacles/EMA/action_rate_at_target", "ema")
        self.scalar_logger.add_log("task_reward", "GoToPositionWithObstacles/AVG/individual_reward", "mean")
        self.scalar_logger.set_ema_coeff(self._task_cfg.ema_coeff)
        self.scalar_logger.add_log("task_reward", "GoToPositionWithObstacles/SUM/num_collisions", "sum")
        self.scalar_logger.add_log("task_reward", "GoToPositionWithObstacles/AVG/progress", "mean")

    
    def get_observations(self) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.

        Args:
            robot_data: The current state of the robot.

        self._task_data[:, 0] = The distance between the robot and the target position.
        self._task_data[:, 1] = The cosine of the angle between the robot heading and the target position.
        self._task_data[:, 2] = The sine of the angle between the robot heading and the target position.
        self._task_data[:, 3] = The linear velocity of the robot along the x-axis.
        self._task_data[:, 4] = The linear velocity of the robot along the y-axis.
        self._task_data[:, 5] = The angular velocity of the robot.
        self._task_data[:, 6:6 + self._task_cfg.num_obstacles] = The distance between the robot and the obstacles.
        self.task_data[:, 6 + self._task_cfg.num_obstacles: 6 + 2 * self._task_cfg.num_obstacles] = The cosine of the angle between the robot and the obstacles.
        self.task_data[:, 6 + 2 * self._task_cfg.num_obstacles: 6 + 3 * self._task_cfg.num_obstacles] = The sine of the angle between the robot and the obstacles.

        Returns:
            torch.Tensor: The observation tensor."""
        # position error
        position_error = self._target_positions[:, :2] - self._robot.root_link_pos_w[self._env_ids, :2]
        position_dist = torch.norm(position_error, dim=-1)
        
        # position error expressed as distance and angular error (to the position)
        heading = self._robot.heading_w[self._env_ids]
        target_heading_w = torch.atan2(
            self._target_positions[:, 1] - self._robot.root_link_pos_w[self._env_ids, 1],
            self._target_positions[:, 0] - self._robot.root_link_pos_w[self._env_ids, 0],
        )
        target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

        # Obstacles positions
        # Filter obstacles by height
        obstacles_positions = self.obstacles.data.object_link_pos_w[self._env_ids]
        filtered_obstacles = obstacles_positions.clone()

        mask = obstacles_positions[:, :, 2] < 0
        filtered_obstacles[mask] = 2 * self._task_cfg.max_obstacle_distance_from_target
        
        self.scalar_logger.log("task_state", "GoToPositionWithObstacles/AVG/num_visible_obstacles", torch.sum(filtered_obstacles[:, :, 2] < abs(self._task_cfg.obstacles_storage_height_pos), dim=-1))

        self.scalar_logger.log("task_state", "GoToPositionWithObstacles/AVG/curriculum_level", self._gen_actions[0, 5].unsqueeze(0).repeat(len(self._env_ids), 1).squeeze())


        # Calculate distances and angles for the filtered obstacles
        obstacles_error = filtered_obstacles[:, :, :2] - self._robot.root_link_pos_w[self._env_ids, :2].unsqueeze(1)
        obstacles_dist = torch.norm(obstacles_error, dim=-1)

        # Get the 3 closest obstacles
        closest_distances, closest_indices = torch.topk(obstacles_dist, k=3, dim=1, largest=False)
        closest_distances = torch.clamp(closest_distances, min=0.0, max=1.0)
        closest_obstacles = torch.gather(
            filtered_obstacles, 1, closest_indices.unsqueeze(-1).expand(-1, -1, filtered_obstacles.size(-1))
        )

        obstacles_heading = torch.atan2(
            closest_obstacles[:, :, 1] - self._robot.root_link_pos_w[self._env_ids, 1].unsqueeze(1),
            closest_obstacles[:, :, 0] - self._robot.root_link_pos_w[self._env_ids, 0].unsqueeze(1),
        )
        obstacles_heading_error = torch.atan2(
            torch.sin(obstacles_heading - heading.unsqueeze(1)), torch.cos(obstacles_heading - heading.unsqueeze(1))
        )

        # Store in buffer [distance, cos(angle), sin(angle), lin_vel_x, lin_vel_y, ang_vel, obstacles_dist, obstacles_cos_angle, obstacles_sin_angle]
        self._task_data[:, 0] = position_dist
        self._task_data[:, 1] = torch.cos(target_heading_error)
        self._task_data[:, 2] = torch.sin(target_heading_error)
        self._task_data[:, 3:5] = self._robot.root_com_lin_vel_b[self._env_ids, :2]
        self._task_data[:, 5] = self._robot.root_com_ang_vel_w[self._env_ids, -1]
        # self._task_data[:, 6:9] = closest_distances
        # self._task_data[:, 9:12] = torch.cos(obstacles_heading_error)
        # self._task_data[:, 12:15] = torch.sin(obstacles_heading_error)
        
        task_id_one_hot = F.one_hot(torch.tensor([self._task_uid], device=self._device), num_classes=self._num_tasks).squeeze(0).repeat(self._num_envs, 1)
        semantic_emb = torch.zeros((self._num_envs, 5), device=self._device)
        semantic_emb[:, 0] = self._rng.sample_uniform_torch(low=0.8, high=1.0, shape=1, ids=self._env_ids)
        semantic_emb[:, 3] = self._rng.sample_uniform_torch(low=0.8, high=1.0, shape=1, ids=self._env_ids)
        semantic_emb[:, 4] = self._rng.sample_uniform_torch(low=0.8, high=1.0, shape=1, ids=self._env_ids)
        
        
        # print(closest_distances[:5])
        

        # Concatenate the task observations with the robot observations
        return torch.concat((self._robot.get_observations(env_ids=self._env_ids), self._task_data), dim=-1), task_id_one_hot, semantic_emb

    def compute_rewards(self) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot.

        The observation is given in the robot's frame. The task provides 3 elements:
        - The position of the object in the robot's frame. It is expressed as the distance between the robot and
            the target position, and the angle between the robot's heading and the target position.
        - The linear velocity of the robot in the robot's frame.
        - The angular velocity of the robot in the robot's frame.

        Angle measurements are converted to a cosine and a sine to avoid discontinuities in 0 and 2pi.
        This provides a continuous representation of the angle.

        The observation tensor is composed of the following elements:
        - self._task_data[:, 0]: The distance between the robot and the target position.
        - self._task_data[:, 1]: The cosine of the angle between the robot's heading and the target position.
        - self._task_data[:, 2]: The sine of the angle between the robot's heading and the target position.
        - self._task_data[:, 3]: The linear velocity of the robot along the x-axis.
        - self._task_data[:, 4]: The linear velocity of the robot along the y-axis.
        - self._task_data[:, 5]: The angular velocity of the robot.
        - self._task_data[:, 6:10] = The distance between the robot and the obstacles.

        Args:
            current_state (torch.Tensor): The current state of the robot.
            actions (torch.Tensor): The actions taken by the robot.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            torch.Tensor: The reward for the current state of the robot."""

        # position error
        self._position_error = self._target_positions[:, :2] - self._robot.root_link_pos_w[self._env_ids, :2]
        self._position_dist = torch.norm(self._position_error, dim=-1)
        # boundary distance
        boundary_dist = torch.abs(self._task_cfg.maximum_robot_distance - self._position_dist)
        # normed linear velocity
        linear_velocity = torch.norm(self._robot.root_com_vel_w[self._env_ids, :2], dim=-1)
        # normed angular velocity
        angular_velocity = torch.abs(self._robot.root_com_vel_w[self._env_ids, -1])
        
        # position error expressed as distance and angular error (to the position)
        heading = self._robot.heading_w[self._env_ids]
        target_heading_w = torch.atan2(
            self._target_positions[:, 1] - self._robot.root_link_pos_w[self._env_ids, 1],
            self._target_positions[:, 0] - self._robot.root_link_pos_w[self._env_ids, 0],
        )
        # heading reward + distance scaling
        dist_scaling = (
            torch.clamp(
                self._position_dist, self._task_cfg.min_heading_dist_scaler, self._task_cfg.max_heading_dist_scaler
            )
            - self._task_cfg.min_heading_dist_scaler
        ) / (self._task_cfg.max_heading_dist_scaler - self._task_cfg.min_heading_dist_scaler)
        target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        heading_rew = (
            torch.exp(-torch.abs(target_heading_error) / self._task_cfg.heading_exponential_reward_coeff) * dist_scaling
        )


        # Update logs
        self.scalar_logger.log("task_state", "GoToPositionWithObstacles/EMA/position_distance", self._position_dist)
        self.scalar_logger.log("task_state", "GoToPositionWithObstacles/EMA/boundary_distance", boundary_dist)
        self.scalar_logger.log("task_state", "GoToPositionWithObstacles/AVG/normed_linear_velocity", linear_velocity)
        self.scalar_logger.log("task_state", "GoToPositionWithObstacles/AVG/absolute_angular_velocity", angular_velocity)

        # position reward
        position_rew = torch.exp(-self._position_dist / self._task_cfg.position_exponential_reward_coeff)
        # progress
        progress_rew = self._previous_position_dist - self._position_dist
        # linear velocity reward
        linear_velocity_rew = linear_velocity - self._task_cfg.linear_velocity_min_value
        linear_velocity_rew[linear_velocity_rew < 0] = 0
        linear_velocity_rew[
            linear_velocity_rew > (self._task_cfg.linear_velocity_max_value - self._task_cfg.linear_velocity_min_value)
        ] = (self._task_cfg.linear_velocity_max_value - self._task_cfg.linear_velocity_min_value)
        # angular velocity reward
        angular_velocity_rew = angular_velocity - self._task_cfg.angular_velocity_min_value
        angular_velocity_rew[angular_velocity_rew < 0] = 0
        angular_velocity_rew[
            angular_velocity_rew
            > (self._task_cfg.angular_velocity_max_value - self._task_cfg.angular_velocity_min_value)
        ] = (self._task_cfg.angular_velocity_max_value - self._task_cfg.angular_velocity_min_value)
        # boundary rew
        boundary_rew = torch.exp(-boundary_dist / self._task_cfg.boundary_exponential_reward_coeff)

        # Checks if the goal is reached
        goal_is_reached = (self._position_dist < self._task_cfg.position_tolerance).int()
        reached_ids = goal_is_reached.nonzero(as_tuple=False).squeeze(-1)
        self._goal_reached *= goal_is_reached  # if not set the value to 0
        self._goal_reached += goal_is_reached  # if it is add 1

        # Check for collision with obstacles
        collisions = torch.squeeze(
            torch.max(torch.norm(self._robot.contacts.data.force_matrix_w[self._env_ids], dim=-1), dim=-1)[0], dim=-1
        )  # first max is for the 3 forces (x,y,z), second max is for obstacles

        num_collisions = 1 * (collisions > self._task_cfg.collision_threshold)
        collision_penalty_rew = self._task_cfg.collision_penalty * num_collisions

        # If goal is reached make next progress null
        self._previous_position_dist[reached_ids] = 0

        # Update logs for rewards
        self.scalar_logger.log("task_reward", "GoToPositionWithObstacles/AVG/position", position_rew)
        self.scalar_logger.log("task_reward", "GoToPositionWithObstacles/AVG/linear_velocity", linear_velocity_rew)
        self.scalar_logger.log("task_reward", "GoToPositionWithObstacles/AVG/angular_velocity", angular_velocity_rew)
        self.scalar_logger.log("task_reward", "GoToPositionWithObstacles/AVG/boundary", boundary_rew)
        self.scalar_logger.log("task_reward", "GoToPositionWithObstacles/AVG/progress", progress_rew)
        self.scalar_logger.log("task_reward", "GoToPositionWithObstacles/SUM/num_collisions", num_collisions)

        # Return the reward by combining the different components and adding the robot rewards
        return (
            # progress_rew * self._task_cfg.progress_weight
            + position_rew * self._task_cfg.position_weight
            # + heading_rew * self._task_cfg.heading_weight
            + linear_velocity_rew * self._task_cfg.linear_velocity_weight
            + angular_velocity_rew * self._task_cfg.angular_velocity_weight
            + boundary_rew * self._task_cfg.boundary_weight
            + collision_penalty_rew
        ) + self._robot.compute_rewards(env_ids=self._env_ids)

    def reset(
        self,
        env_ids: torch.Tensor,
        gen_actions: torch.Tensor | None = None,
        env_seeds: torch.Tensor | None = None,
    ) -> None:
        """
        Resets the task to its initial state.

        If gen_actions is None, then the environment is generated at random. This is the default mode.
        If env_seeds is None, then the seed is generated at random. This is the default mode.

        The environment actions for this task are the following all belong to the [0,1] range:
        - gen_actions[0]: The value used to sample the distance between the spawn position and the goal.
        - gen_actions[1]: The value used to sample the angle between the spawn heading and the heading required to be looking at the goal.
        - gen_actions[2]: The value used to sample the linear velocity of the robot at spawn.
        - gen_actions[3]: The value used to sample the angular velocity of the robot at spawn.
        - gen_actions[4]: The probability of enabling thrusters (1 = easy with more thrusters enabled, 0 = hard with fewer thrusters enabled).

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            gen_actions (torch.Tensor | None): The actions for the task. Defaults to None.
            env_seeds (torch.Tensor | None): The seeds for the environments. Defaults to None.
        """
        super().reset(env_ids, gen_actions=gen_actions, env_seeds=env_seeds)

        # Randomizes goals and initial conditions
        self.set_goals(env_ids)
        self.set_initial_conditions(env_ids)

        # Resets the goal reached flag
        self._goal_reached[env_ids] = 0

        # Make sure the position error and position dist are up to date after the reset
        self._position_error[env_ids] = (
            self._target_positions[env_ids] - self._robot.root_link_pos_w[self._env_ids, :2][env_ids]
        )
        self._position_dist[env_ids] = torch.linalg.norm(self._position_error[env_ids], dim=-1)
        self._previous_position_dist[env_ids] = self._position_dist[env_ids].clone()
        
    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates if the platforms should be killed or not.

        Returns:
            torch.Tensor: Whether the platforms should be killed or not."""
        self._previous_position_dist = self._position_dist.clone()
        
        self._position_error = self._target_positions[:, :2] - self._robot.root_link_pos_w[self._env_ids, :2]
        self._position_dist = torch.linalg.norm(self._position_error, dim=-1)
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.where(
            self._position_dist > self._task_cfg.maximum_robot_distance,
            ones,
            task_failed,
        )

        task_completed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        # task_completed = torch.where(
        #     self._goal_reached > self._task_cfg.reset_after_n_steps_in_tolerance,
        #     ones,
        #     task_completed,
        # )
        return task_failed, task_completed
    
    def set_goals(self, env_ids: torch.Tensor) -> None:
        """
        Generates a random goal for the task.
        These goals are generated in a way allowing to precisely control the difficulty of the task through the
        environment action. In this task, there is no specific actions related to the goals.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The target positions and orientations."""

        # # The position is picked randomly in a square centered on the origin
        # self._target_positions[env_ids] = (
        #     self._rng.sample_uniform_torch(
        #         -self._task_cfg.goal_max_dist_from_origin, self._task_cfg.goal_max_dist_from_origin, 2, ids=env_ids
        #     )
        #     + self._env_origins[env_ids, :2]
        # )
        # Spawn positions within defined bounding box determined from manual experiments inside the JEM
        x_range = (-0.7, 0.56)
        y_range = (-2.7, 2.7)
        z_range = (0.0, 0.0)

        self._target_positions[env_ids, 0] = self._rng.sample_uniform_torch(*x_range, 1, ids=env_ids).squeeze(-1) + self._env_origins[env_ids, 0]
        self._target_positions[env_ids, 1] = self._rng.sample_uniform_torch(*y_range, 1, ids=env_ids).squeeze(-1) + self._env_origins[env_ids, 1]
        # self._target_positions[env_ids, 2] = self._rng.sample_uniform_torch(*z_range, 1, ids=env_ids).squeeze(-1)

        # Update the visual markers
        self._markers_pos[env_ids, :2] = self._target_positions[env_ids]

    def set_initial_conditions(self, env_ids: torch.Tensor) -> None:
        """
        Generates the initial conditions for the robots. The initial conditions are randomized based on the
        environment actions. The generation of the initial conditions is done so that if the environment actions are
        close to 0 then the task is the easiest, if they are close to 1 then the task is hardest. The configuration of
        the task defines the ranges within which the initial conditions are randomized.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The initial position,
            orientation and velocity of the robot.
        """

        num_resets = len(env_ids)

        # Randomizes the initial pose of the platform
        initial_pose = torch.zeros((num_resets, 7), device=self._device, dtype=torch.float32)

        # Position
        r = (
            self._gen_actions[env_ids, 0] * (self._task_cfg.spawn_max_dist - self._task_cfg.spawn_min_dist)
            + self._task_cfg.spawn_min_dist
        )
        theta = self._rng.sample_uniform_torch(-math.pi, math.pi, 1, ids=env_ids)
        initial_pose[:, 0] = r * torch.cos(theta) + self._target_positions[env_ids, 0]
        initial_pose[:, 1] = r * torch.sin(theta) + self._target_positions[env_ids, 1]

        # chunck_size = self.scene.num_envs // self._num_tasks
        # start_indx = (self._task_uid - 1) * chunck_size
        # shifted_env_ids = env_ids + start_indx
        initial_pose[:, 2] = self._robot_origins[self._env_ids[env_ids], 2]
        # x_range = (-0.7, 0.56)
        # y_range = (-2.7, 2.7)
        # initial_pose[:, 0] = self._rng.sample_uniform_torch(*x_range, 1, ids=env_ids).squeeze(-1) + self._env_origins[env_ids, 0]
        # initial_pose[:, 1] = self._rng.sample_uniform_torch(*y_range, 1, ids=env_ids).squeeze(-1) + self._env_origins[env_ids, 1]
        # initial_pose[:, 2] = self._robot_origins[self._env_ids[env_ids], 2]

        # Orientation
        # Compute the heading to the target
        target_heading = torch.arctan2(
            self._target_positions[env_ids, 1] - initial_pose[:, 1],
            self._target_positions[env_ids, 0] - initial_pose[:, 0],
        )
        # Randomizes the heading of the platform
        delta_heading = (
            self._gen_actions[env_ids, 1]
            * (self._task_cfg.spawn_max_heading_dist - self._task_cfg.spawn_min_heading_dist)
            + self._task_cfg.spawn_min_heading_dist
        ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)
        # The spawn heading is the delta heading + the target heading
        theta = delta_heading + target_heading
        initial_pose[:, 3] = torch.cos(theta * 0.5)
        initial_pose[:, 6] = torch.sin(theta * 0.5)

        # Randomizes the velocity of the platform
        self.initial_velocity[env_ids] = torch.zeros((num_resets, 6), device=self._device, dtype=torch.float32)

        # Linear velocity
        velocity_norm = (
            self._gen_actions[env_ids, 2] * (self._task_cfg.spawn_max_lin_vel - self._task_cfg.spawn_min_lin_vel)
            + self._task_cfg.spawn_min_lin_vel
        )
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        self.initial_velocity[env_ids, 0] = velocity_norm * torch.cos(theta)
        self.initial_velocity[env_ids, 1] = velocity_norm * torch.sin(theta)

        # Angular velocity of the platform
        angular_velocity = (
            self._gen_actions[env_ids, 3] * (self._task_cfg.spawn_max_ang_vel - self._task_cfg.spawn_min_ang_vel)
            + self._task_cfg.spawn_min_ang_vel
        )
        self.initial_velocity[env_ids, 5] = angular_velocity

        # Apply to articulation
        self._robot.set_pose(initial_pose, self._env_ids[env_ids])
        self._robot.set_velocity(self.initial_velocity[env_ids], self._env_ids[env_ids])
        
        # Randomize obstacles positions
        obstacles_positions, mask = self.randomize_obstacles_positions(env_ids)
        # set the hight of the obstacles in the mask that are false to storage height
        for i in range(obstacles_positions.shape[0]):
            obstacles_positions[i, ~mask[i], 2] -= self._task_cfg.obstacles_storage_height_pos
        self.obstacles.write_object_link_pose_to_sim(obstacles_positions, env_ids=self._env_ids[env_ids])
        
    def randomize_obstacles_positions(self, env_ids: torch.tensor) -> tuple:
        """
        This function randomizes the positions of obstacles within a specified environment in a grid-based layout, ensuring that they are not placed too close to the target or robot. It also generates random orientations for the obstacles and creates a mask indicating which obstacles are visible.

        Args:
            env_ids (torch.tensor): The ids of the environments to randomize the obstacles in.

        Returns:
            tuple: A tuple containing the positions of the obstacles and a mask indicating which obstacles are visible.
        """

        # Randomize obstacles positions
        number_obstacles_to_generate = self._task_cfg.max_num_vis_obstacles * 4
        indices_of_obstacles_to_activate = self._rng.sample_unique_integers_torch(
            min=0, max=self._task_cfg.max_num_vis_obstacles**2, num=number_obstacles_to_generate, ids=env_ids
        )

        # x = (indices_of_obstacles_to_activate % self._num_cells) - self._num_cells / 2
        # y = (indices_of_obstacles_to_activate // self._num_cells) - self._num_cells / 2
        # # Scale to world coordinates
        # cell_size = self._task_cfg.maximum_robot_distance / self._num_cells  # Calculate the size of a grid cell
        # x = self._rng.sample_sign_torch("int", (number_obstacles_to_generate), ids=env_ids) * x * cell_size / 2
        # y = self._rng.sample_sign_torch("int", (number_obstacles_to_generate), ids=env_ids) * y * cell_size / 2
        # z = torch.ones_like(x) * 6.0

        x_range = (-0.7, 0.56)
        y_range = (-2.7, 2.7)
        z_range = (self._task_cfg.obstacles_height, self._task_cfg.obstacles_height)

        x = self._rng.sample_uniform_torch(*x_range, (number_obstacles_to_generate,), ids=env_ids)
        y = self._rng.sample_uniform_torch(*y_range, (number_obstacles_to_generate,), ids=env_ids)
        z = self._rng.sample_uniform_torch(*z_range, (number_obstacles_to_generate,), ids=env_ids)

        xyz = torch.stack((x, y, z), dim=2)
        xyz[..., :2] += self._env_origins[env_ids].unsqueeze(1)[..., :2]

        # **Step 5: Compute distances from obstacles to robot and target**
        distance_obstacle_to_target = torch.norm(xyz[:, :, :2] - self._target_positions[env_ids].unsqueeze(1)[..., :2], dim=-1)
        distance_obstacle_to_robot = torch.norm(xyz[:, :, :2] - self._robot.root_link_pos_w[env_ids].unsqueeze(1)[..., :2], dim=-1)

        # **Step 6: Mask obstacles that are too close**
        obstacles_mask = (
            (distance_obstacle_to_target < self._task_cfg.min_obstacle_distance_from_target) |
            (distance_obstacle_to_robot < self._task_cfg.min_obstacle_distance_from_robot)
        )

        # **Step 7: Extract valid and invalid obstacles**
        valid_indices = (~obstacles_mask).nonzero(as_tuple=True)
        valid_envs, valid_obs = valid_indices  # Separate environment and obstacle indices
        invalid_indices = obstacles_mask.nonzero(as_tuple=True)

        # **Step 8: Ensure valid obstacles exist for each environment**
        valid_obstacles = torch.zeros((len(env_ids), self._task_cfg.max_num_vis_obstacles, 3), device=self._device)  

        for env in range(len(env_ids)):
            env_mask = valid_envs == env
            valid_obs_for_env = valid_obs[env_mask]

            if valid_obs_for_env.numel() > 0:
                num_valid = min(valid_obs_for_env.numel(), self._task_cfg.max_num_vis_obstacles)
                valid_obstacles[env, :num_valid] = xyz[env, valid_obs_for_env[:num_valid]]

                # **Handle missing values by repeating valid ones**
                if num_valid < self._task_cfg.max_num_vis_obstacles:
                    extra_indices = torch.randint(0, num_valid, (self._task_cfg.max_num_vis_obstacles - num_valid,), device=self._device)
                    valid_obstacles[env, num_valid:] = valid_obstacles[env, extra_indices]

        # **Step 9: Replace invalid obstacles using proper indexing**
        for env in range(len(env_ids)):
            env_invalid_mask = invalid_indices[0] == env
            env_invalid_obs = invalid_indices[1][env_invalid_mask]

            if env_invalid_obs.numel() > 0:
                num_invalid = env_invalid_obs.numel()
                valid_obs_count = valid_obstacles.shape[1]

                # Ensure we don't exceed available valid obstacles
                sampled_indices = torch.randint(0, valid_obs_count, (num_invalid,), device=self._device)

                # Replace invalid obstacles with valid ones from the same environment
                xyz[env, env_invalid_obs] = valid_obstacles[env, sampled_indices]


        # Generate quats and concatenate with xyz
        xyzw = self.obstacles.data.object_com_quat_w[env_ids].clone()
        obstacles_positions = torch.cat((xyz[:, : self._task_cfg.max_num_vis_obstacles], xyzw), dim=-1)

        # Create visible obstacles
        # num_visible_obstacles_per_env = self._rng.sample_integer_torch(
        #     low=1, high=self._task_cfg.max_num_vis_obstacles, shape=(1,), ids=env_ids
        # )
        num_visible_obstacles_per_env = torch.round(self._gen_actions[env_ids, 5] * (self._task_cfg.max_num_vis_obstacles - self._task_cfg.min_num_obstacles) + self._task_cfg.min_num_obstacles)

        mask = torch.arange(self._task_cfg.max_num_vis_obstacles, device=self._device).unsqueeze(
            0
        ) < num_visible_obstacles_per_env.unsqueeze(1)
        
        return obstacles_positions, mask
    
    def create_task_visualization(self) -> None:
        """
        Adds the visual marker to the scene.

        There are two markers: one for the goal and one for the robot.

        - The goal marker is a sphere.
        - The robot also has a sphere at it's centre just as a placeholder and not used for visualization.
        """

        # Define visual markers: sphere for the goal and pose marker for the robot
        goal_marker_cfg = PIN_SPHERE_CFG.copy()
        # goal_marker_cfg.markers["sphere"].radius = 0.05
        # goal_marker_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
        robot_marker_cfg = SPHERE_CFG.copy()
        robot_marker_cfg.markers["sphere"].radius = 0.01

        # Update prim paths to match task ID
        goal_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/goal_pose"
        robot_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/robot_pose"
        # Create the visualization markers
        self.goal_pos_visualizer = VisualizationMarkers(goal_marker_cfg)
        self.robot_pos_visualizer = VisualizationMarkers(robot_marker_cfg)

        # Configuration for the obstacle lines
        obstacle_line_cfg = VisualizationMarkersCfg(
            prim_path=f"/Visuals/Command/task_{self._task_uid}/obstacle_lines",
            markers={
                "line": sim_utils.CylinderCfg(
                    radius=0.005,  # Small radius
                    height=1.0,   # Base height, will be scaled
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                ),
            },
        )
        self.obstacle_lines_visualizer = VisualizationMarkers(obstacle_line_cfg)

    def update_task_visualization(self) -> None:
        """Updates the visual marker to the scene."""

        self.goal_pos_visualizer.visualize(self._markers_pos)

        self._robot_marker_pos = self._robot.root_link_pos_w[self._env_ids, :3]
        self.robot_pos_visualizer.visualize(self._robot_marker_pos, self._robot.root_link_quat_w[self._env_ids])

        # Obstacles lines
        # Get robot position
        robot_pos = self._robot.root_link_pos_w[self._env_ids] # (N, 3)

        # Get obstacles
        obstacles_positions = self.obstacles.data.object_link_pos_w[self._env_ids] # (N, M, 3)
        
        # Filter obstacles
        filtered_obstacles = obstacles_positions.clone()
        mask = obstacles_positions[:, :, 2] < self._task_cfg.obstacles_storage_height_pos
        filtered_obstacles[mask] = 100.0

        # Calculate distances
        obstacles_error_w = filtered_obstacles - robot_pos.unsqueeze(1)
        obstacles_dist = torch.norm(obstacles_error_w, dim=-1)

        # Get the 3 closest obstacles
        closest_distances, closest_indices = torch.topk(obstacles_dist, k=3, dim=1, largest=False)
        
        # Get the world frame error for the closest obstacles
        closest_obstacles_error_w = torch.gather(
            obstacles_error_w, 1, closest_indices.unsqueeze(-1).expand(-1, -1, obstacles_error_w.size(-1))
        ) # (N, 3, 3)

        # Flatten
        vecs = closest_obstacles_error_w.view(-1, 3) # (N*3, 3)
        dists = closest_distances.view(-1) # (N*3)

        # Midpoints
        # robot_pos needs to be repeated
        robot_pos_expanded = robot_pos.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)
        midpoints = robot_pos_expanded + vecs / 2

        # Orientations
        # Align Z to vec
        z_col = torch.nn.functional.normalize(vecs, dim=-1)
        up = torch.tensor([0.0, 1.0, 0.0], device=self._device).expand_as(z_col)
        mask = torch.abs(torch.sum(z_col * up, dim=-1)) > 0.99
        up[mask] = torch.tensor([1.0, 0.0, 0.0], device=self._device)
        x_col = torch.cross(up, z_col, dim=-1)
        x_col = torch.nn.functional.normalize(x_col, dim=-1)
        y_col = torch.cross(z_col, x_col, dim=-1)
        rot_mat = torch.stack([x_col, y_col, z_col], dim=-1)
        quats = math_utils.quat_from_matrix(rot_mat)

        # Scales
        scales = torch.ones((len(dists), 3), device=self._device)
        scales[:, 2] = dists

        self.obstacle_lines_visualizer.visualize(midpoints, quats, scales=scales)