# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from isaaclab.markers import ARROW_CFG, PIN_ARROW_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveScene

from isaaclab_tasks.rans import RendezvousCfg

from .task_core import TaskCore
import torch.nn.functional as F

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class RendezvousTask(TaskCore):
    """
    Implements the Rendezvous task for space ship rendezvous simulation.
    
    This task creates a half-circle trajectory where the robot must navigate through a series of waypoints
    that form a rendezvous path. Each waypoint's orientation points toward a central rendezvous point,
    simulating a space ship approaching a target for docking or rendezvous.
    
    Key features:
    - Goals follow a half-circle arc from left to right
    - Each goal's orientation points toward the center (rendezvous point)
    - Environment actions control trajectory radius and spread
    - Supports variable number of waypoints along the trajectory
    """

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        task_cfg: RendezvousCfg = RendezvousCfg(),
        task_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        env_ids: torch.Tensor | None = None,
        decimation: int = 1,
        num_tasks: int = 1,
    ) -> None:
        """
        Initializes the Rendezvous task.

        Args:
            task_cfg: The configuration of the task.
            task_uid: The unique id of the task.
            num_envs: The number of environments.
            device: The device on which the tensors are stored.
            task_id: The id of the task.
            env_ids: The ids of the environments used by this task."""

        super().__init__(
            scene=scene, task_uid=task_uid, num_envs=num_envs, device=device, env_ids=env_ids, decimation=decimation, num_tasks=num_tasks
        )

        # Task and reward parameters
        self._task_cfg = task_cfg

        # Defines the observation and actions space sizes for this task
        self._dim_task_obs = self._task_cfg.observation_space
        self._dim_gen_act = self._task_cfg.gen_space

        # Buffers
        self.initialize_buffers(env_ids=env_ids)

    @property
    def eval_data_keys(self) -> list[str]:
        """
        Returns the keys of the data used for evaluation.

        Returns:
            list[str]: The keys of the data used for evaluation."""
        
        return [
            "target_positions", 
            "target_headings", 
            "target_index", 
            "num_goals", 
            "trajectory_completed",
            "position_distance",
            "cos_heading_to_target_error",
            "sin_heading_to_target_error",
            "cos_target_heading_error",
            "sin_target_heading_error",
            "subsequent_goals_distance",
            "cos_heading_to_subsequent_goals_error",
            "sin_heading_to_subsequent_goals_error",
            "cos_target_heading_to_subsequent_goals_error",
            "sin_target_heading_to_subsequent_goals_error",
            "total_goals_reached"
        ]
    
    @property
    def eval_data_specs(self)->dict[str, list[str]]:
        return {
            "target_positions": [coord for i in range(self._task_cfg.max_num_goals) for coord in (f".x_{i}.m", f".y_{i}.m")],
            "target_headings": [f".target_heading_{i}.rad" for i in range(self._task_cfg.max_num_goals)],
            "target_index": [".u"],
            "num_goals": [".u"],
            "trajectory_completed": [".u"],
            "position_distance": [".distance.m"],
            "cos_heading_to_target_error": [".cos(heading).u"],
            "sin_heading_to_target_error": [".sin(heading).u"],
            "cos_target_heading_error": [".cos(target_heading_error).u"],
            "sin_target_heading_error": [".sin(target_heading_error).u"],
            "subsequent_goals_distance": [f".distance_sub_goal_{i}.m" for i in range(self._task_cfg.num_subsequent_goals - 1)],
            "cos_heading_to_subsequent_goals_error": [f".cos(heading)_sub_goal_{i}.u" for i in range(self._task_cfg.num_subsequent_goals - 1)],
            "sin_heading_to_subsequent_goals_error": [f".sin(heading)_sub_goal_{i}.u" for i in range(self._task_cfg.num_subsequent_goals - 1)],
            "cos_target_heading_to_subsequent_goals_error": [f".cos(target_heading)_sub_goal_{i}.u" for i in range(self._task_cfg.num_subsequent_goals - 1)],
            "sin_target_heading_to_subsequent_goals_error": [f".sin(target_heading)_sub_goal_{i}.u" for i in range(self._task_cfg.num_subsequent_goals - 1)],
            "total_goals_reached": [".u"]
        }
    
    @property
    def eval_data(self) -> dict:
        """
        Returns the data used for evaluation.

        Returns:
            dict: The data used for evaluation."""
        
        subsequent_goals_distance = []
        cos_heading_to_subsequent_goals_error = []
        sin_heading_to_subsequent_goals_error = []
        cos_target_heading_to_subsequent_goals_error = []
        sin_target_heading_to_subsequent_goals_error = []

        for i in range(self._task_cfg.num_subsequent_goals - 1):
            subsequent_goals_distance.append(self._task_data[:, 8 + 5 * i])
            cos_heading_to_subsequent_goals_error.append(self._task_data[:, 9 + 5 * i])
            sin_heading_to_subsequent_goals_error.append(self._task_data[:, 10 + 5 * i])
            cos_target_heading_to_subsequent_goals_error.append(self._task_data[:, 11 + 5 * i])
            sin_target_heading_to_subsequent_goals_error.append(self._task_data[:, 12 + 5 * i])

        subsequent_goals_distance = torch.stack(subsequent_goals_distance, dim=-1)
        cos_heading_to_subsequent_goals_error = torch.stack(cos_heading_to_subsequent_goals_error, dim=-1)
        sin_heading_to_subsequent_goals_error = torch.stack(sin_heading_to_subsequent_goals_error, dim=-1)
        cos_target_heading_to_subsequent_goals_error = torch.stack(cos_target_heading_to_subsequent_goals_error, dim=-1)
        sin_target_heading_to_subsequent_goals_error = torch.stack(sin_target_heading_to_subsequent_goals_error, dim=-1)

        reshaped_subsequent_goals_distance = subsequent_goals_distance.view(
            subsequent_goals_distance.shape[0], self._task_cfg.num_subsequent_goals - 1, -1
        ).permute(0, 2, 1).reshape(subsequent_goals_distance.shape)
        reshaped_cos_heading_to_subsequent_goals_error = cos_heading_to_subsequent_goals_error.view(
            cos_heading_to_subsequent_goals_error.shape[0], self._task_cfg.num_subsequent_goals - 1, -1
        ).permute(0, 2, 1).reshape(cos_heading_to_subsequent_goals_error.shape)
        reshaped_sin_heading_to_subsequent_goals_error = sin_heading_to_subsequent_goals_error.view(
            sin_heading_to_subsequent_goals_error.shape[0], self._task_cfg.num_subsequent_goals - 1, -1
        ).permute(0, 2, 1).reshape(sin_heading_to_subsequent_goals_error.shape)
        reshaped_cos_target_heading_to_subsequent_goals_error = cos_target_heading_to_subsequent_goals_error.view(
            cos_target_heading_to_subsequent_goals_error.shape[0], self._task_cfg.num_subsequent_goals - 1, -1
        ).permute(0, 2, 1).reshape(cos_target_heading_to_subsequent_goals_error.shape)
        reshaped_sin_target_heading_to_subsequent_goals_error = sin_target_heading_to_subsequent_goals_error.view(
            sin_target_heading_to_subsequent_goals_error.shape[0], self._task_cfg.num_subsequent_goals - 1, -1
        ).permute(0, 2, 1).reshape(sin_target_heading_to_subsequent_goals_error.shape)
        
        return {
            "target_positions": self._target_positions,
            "target_headings": self._target_heading,
            "target_index": self._target_index,
            "num_goals": self._num_goals,
            "trajectory_completed": self._trajectory_completed,
            "position_distance": self._task_data[:, 3],
            "cos_heading_to_target_error": self._task_data[:, 4],
            "sin_heading_to_target_error": self._task_data[:, 5],
            "cos_target_heading_error": self._task_data[:, 6],
            "sin_target_heading_error": self._task_data[:, 7],
            "subsequent_goals_distance": reshaped_subsequent_goals_distance,
            "cos_heading_to_subsequent_goals_error": reshaped_cos_heading_to_subsequent_goals_error,
            "sin_heading_to_subsequent_goals_error": reshaped_sin_heading_to_subsequent_goals_error,
            "cos_target_heading_to_subsequent_goals_error": reshaped_cos_target_heading_to_subsequent_goals_error,
            "sin_target_heading_to_subsequent_goals_error": reshaped_sin_target_heading_to_subsequent_goals_error,
            "total_goals_reached": self.total_goals_reached,
        }

    def initialize_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Initializes the buffers used by the task.

        Args:
            env_ids: The ids of the environments used by this task."""

        super().initialize_buffers(env_ids)
        self._position_error = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self._position_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._previous_position_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._target_positions = torch.zeros(
            (self._num_envs, self._task_cfg.max_num_goals, 2),
            device=self._device,
            dtype=torch.float32,
        )
        self._target_heading = torch.zeros(
            (self._num_envs, self._task_cfg.max_num_goals),
            device=self._device,
            dtype=torch.float32,
        )
        self._target_index = torch.zeros((self._num_envs,), device=self._device, dtype=torch.long)
        self._trajectory_completed = torch.zeros((self._num_envs,), device=self._device, dtype=torch.bool)
        self._num_goals = torch.zeros((self._num_envs,), device=self._device, dtype=torch.long)
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)
        self.initial_velocity = torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)
        self.total_goals_reached = torch.zeros((self._num_envs,), device=self._device, dtype=torch.long)

    def create_logs(self) -> None:
        """
        Creates a dictionary to store the training statistics for the task."""

        super().create_logs()

        self.scalar_logger.add_log("task_state", "Rendezvous/AVG/normed_linear_velocity", "mean")
        self.scalar_logger.add_log("task_state", "Rendezvous/AVG/absolute_angular_velocity", "mean")
        self.scalar_logger.add_log("task_state", "Rendezvous/AVG/position_distance", "mean")
        self.scalar_logger.add_log("task_state", "Rendezvous/AVG/boundary_distance", "mean")
        self.scalar_logger.add_log("task_state", "Rendezvous/AVG/heading_distance", "mean")
        self.scalar_logger.add_log("task_state", "Rendezvous/AVG/target_heading_distance", "mean")

        self.scalar_logger.add_log("task_reward", "Rendezvous/AVG/linear_velocity", "mean")
        self.scalar_logger.add_log("task_reward", "Rendezvous/AVG/angular_velocity", "mean")
        self.scalar_logger.add_log("task_reward", "Rendezvous/AVG/boundary", "mean")
        self.scalar_logger.add_log("task_reward", "Rendezvous/AVG/orientation", "mean")
        self.scalar_logger.add_log("task_reward", "Rendezvous/AVG/target_heading", "mean")
        self.scalar_logger.add_log("task_reward", "Rendezvous/AVG/position_orientation", "mean")
        self.scalar_logger.add_log("task_reward", "Rendezvous/AVG/position_rew", "mean")
        self.scalar_logger.add_log("task_reward", "Rendezvous/SUM/num_goals", "sum")
        self.scalar_logger.add_log("task_reward", "Rendezvous/EMA/action_rate_at_target", "mean")
        self.scalar_logger.add_log("task_reward", "Rendezvous/AVG/individual_reward", "mean")

    def get_observations(self) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.

        The observation is given in the robot's frame. The observation is composed of the following elements:
        - The linear velocity of the robot.
        - The angular velocity of the robot.
        - The distance to the target position.
        - The angle between the robot's heading and the target position.
        - The angle between the robot's heading and the target heading.
        - Depending on the task configuration, a number of subsequent poses are added to the observation. For each of
            them, the following elements are added:
            - The distance between the n th and the n+1 th goal.
            - The angle between the n th goal and the n+1 th goal's position (in the current's goal frame).
            - The angle between the n th goal and the n+1 th goal's heading.

        Angle measurements are converted to a cosine and a sine to avoid discontinuities in 0 and 2pi.
        This provides a continuous representation of the angle.

        self._task_data[:, 0] = The distance between the robot and the target position.
        self._task_data[:, 1] = The cosine of the angle between the robot's heading and the target position.
        self._task_data[:, 2] = The sine of the angle between the robot's heading and the target position.
        self._task_data[:, 3] = The cosine of the angle between the robot's heading and the target heading.
        self._task_data[:, 4] = The sine of the angle between the robot's heading and the target heading.
        self._task_data[:, 5] = The linear velocity of the robot along the x-axis.
        self._task_data[:, 6] = The linear velocity of the robot along the y-axis.
        self._task_data[:, 7] = The angular velocity of the robot.
        self._task_data[:, 8 + i*5] = The distance between the n th and the n+1 th goal.
        self._task_data[:, 9 + i*5] = The cosine of the angle between the n th goal and the n+1 th goal's position.
        self._task_data[:, 10 + i*5] = The sine of the angle between the n th goal and the n+1 th goal's position.
        self._task_data[:, 11 + i*5] = The cosine of the angle between the n th goal and the n+1 th goal's heading.
        self._task_data[:, 12 + i*5] = The sine of the angle between the n th goal and the n+1 th goal's heading.

        Returns:
            torch.Tensor: The observation tensor."""

        # position error
        position_error = (
            self._target_positions[self._ALL_INDICES, self._target_index]
            - self._robot.root_link_pos_w[self._env_ids, :2]
        )
        position_dist = torch.linalg.norm(position_error, dim=-1)

        # position error expressed as distance and angular error (to the position)
        heading = self._robot.heading_w[self._env_ids]
        target_heading_w = torch.atan2(
            self._target_positions[self._ALL_INDICES, self._target_index, 1]
            - self._robot.root_link_pos_w[self._env_ids, 1],
            self._target_positions[self._ALL_INDICES, self._target_index, 0]
            - self._robot.root_link_pos_w[self._env_ids, 0],
        )
        target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        heading_error = torch.atan2(
            torch.sin(self._target_heading[self._ALL_INDICES, self._target_index] - heading),
            torch.cos(self._target_heading[self._ALL_INDICES, self._target_index] - heading),
        )

        # Store in buffer
        # breakpoint()
        self._task_data[:, 0] = position_dist
        self._task_data[:, 1] = torch.cos(target_heading_error)
        self._task_data[:, 2] = torch.sin(target_heading_error)
        self._task_data[:, 3] = torch.cos(heading_error)
        self._task_data[:, 4] = torch.sin(heading_error)
        self._task_data[:, 5:7] = self._robot.root_com_lin_vel_b[self._env_ids, :2]
        self._task_data[:, 7] = self._robot.root_com_ang_vel_w[self._env_ids, -1]

        # We compute the observations of the subsequent goals in the previous goal's frame.
        for i in range(self._task_cfg.num_subsequent_goals - 1):
            # Check if the index is looking beyond the number of goals
            overflowing = (self._target_index + i + 1) >= self._num_goals
            # If it is, then set the next index to 0 (Loop around)
            indices = (self._target_index + i + 1) * torch.logical_not(overflowing)
            # Compute the distance between the nth-1 goal, and the nth goal
            goal_distance = torch.linalg.norm(
                self._target_positions[self._ALL_INDICES, indices - 1]
                - self._target_positions[self._ALL_INDICES, indices],
                dim=-1,
            )
            # Compute the angular distance between the nth-1 goal, and the nth goal (world goal frame)
            target_heading_w = torch.atan2(
                self._target_positions[self._ALL_INDICES, indices, 1]
                - self._target_positions[self._ALL_INDICES, indices - 1, 1],
                self._target_positions[self._ALL_INDICES, indices, 0]
                - self._target_positions[self._ALL_INDICES, indices - 1, 0],
            )
            # Compute the heading in the nth-1 goal frame
            target_heading_error = torch.atan2(
                torch.sin(target_heading_w - self._target_heading[self._ALL_INDICES, indices - 1]),
                torch.cos(target_heading_w - self._target_heading[self._ALL_INDICES, indices - 1]),
            )
            # Compute the heading delta between the nth-1 goal, and the nth goal
            heading_error = torch.atan2(
                torch.sin(
                    self._target_heading[self._ALL_INDICES, indices]
                    - self._target_heading[self._ALL_INDICES, indices - 1]
                ),
                torch.cos(
                    self._target_heading[self._ALL_INDICES, indices]
                    - self._target_heading[self._ALL_INDICES, indices - 1]
                ),
            )
            # If the task is not set to loop, we set the next goal to be 0.
            if not self._task_cfg.loop:
                goal_distance = goal_distance * torch.logical_not(overflowing)
                target_heading_error = target_heading_error * torch.logical_not(overflowing)
            # Add to buffer
            self._task_data[:, 8 + 5 * i] = goal_distance
            self._task_data[:, 9 + 5 * i] = torch.cos(target_heading_error)
            self._task_data[:, 10 + 5 * i] = torch.sin(target_heading_error)
            self._task_data[:, 11 + 5 * i] = torch.cos(heading_error)
            self._task_data[:, 12 + 5 * i] = torch.sin(heading_error)

        for randomizer in self.randomizers:
            randomizer.observations(observations=self._task_data)

        task_id_one_hot = F.one_hot(torch.tensor([self._task_uid], device=self._device), num_classes=self._num_tasks).squeeze(0).repeat(self._num_envs, 1)
        semantic_emb = torch.zeros((self._num_envs, 5), device=self._device)
        semantic_emb[:, 0] = self._rng.sample_uniform_torch(low=0.7, high=1.0, shape=1, ids=self._env_ids)
        semantic_emb[:, 1] = self._rng.sample_uniform_torch(low=0.8, high=1.0, shape=1, ids=self._env_ids)
        semantic_emb[:, 2] = self._rng.sample_uniform_torch(low=0.6, high=1.0, shape=1, ids=self._env_ids)
        
        # Concatenate task observations with robot's internal observations
        return torch.concat((self._robot.get_observations(env_ids=self._env_ids), self._task_data), dim=-1), task_id_one_hot, semantic_emb

    def compute_rewards(self) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot.

        Returns:
            torch.Tensor: The reward for the current state of the robot."""

        # position error expressed as distance and angular error (to the position)
        self._position_error = (
            self._target_positions[self._ALL_INDICES, self._target_index]
            - self._robot.root_link_pos_w[self._env_ids, :2]
        )
        self._position_dist = torch.linalg.norm(self._position_error, dim=-1)
        heading = self._robot.heading_w[self._env_ids]
        heading_error = torch.atan2(
            torch.sin(self._target_heading[self._ALL_INDICES, self._target_index] - heading),
            torch.cos(self._target_heading[self._ALL_INDICES, self._target_index] - heading),
        )
        heading_dist = torch.abs(heading_error)

        # position error expressed as distance and angular error (to the position)
        target_heading_w = torch.atan2(
            self._target_positions[self._ALL_INDICES, self._target_index, 1]
            - self._robot.root_link_pos_w[self._env_ids, 1],
            self._target_positions[self._ALL_INDICES, self._target_index, 0]
            - self._robot.root_link_pos_w[self._env_ids, 0],
        )
        target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        target_heading_dist = torch.abs(target_heading_error)
        # boundary distance
        boundary_dist = torch.abs(self._task_cfg.maximum_robot_distance - self._position_dist)
        # normed linear velocity
        linear_velocity = torch.linalg.norm(self._robot.root_com_vel_w[self._env_ids, :2], dim=-1)
        # normed angular velocity
        angular_velocity = torch.abs(self._robot.root_com_vel_w[self._env_ids, -1])
        # progress
        progress_rew = self._previous_position_dist - self._position_dist
        # position
        position_rew = torch.exp(-self._position_dist / self._task_cfg.position_exponential_reward_coeff)

        # Update logs
        self.scalar_logger.log("task_state", "Rendezvous/AVG/position_distance", self._position_dist)
        self.scalar_logger.log("task_state", "Rendezvous/AVG/heading_distance", heading_dist)
        self.scalar_logger.log("task_state", "Rendezvous/AVG/target_heading_distance", target_heading_dist)
        self.scalar_logger.log("task_state", "Rendezvous/AVG/boundary_distance", boundary_dist)
        self.scalar_logger.log("task_state", "Rendezvous/AVG/normed_linear_velocity", linear_velocity)
        self.scalar_logger.log("task_state", "Rendezvous/AVG/absolute_angular_velocity", angular_velocity)

        # orientation reward (encourages the robot to orient w/ the target orientation)
        orientation_rew = torch.exp(-heading_dist / self._task_cfg.heading_exponential_reward_coeff)

        # target heading reward (encourages the robot to face the target)
        heading_to_target_rew = torch.exp(-target_heading_dist / self._task_cfg.target_heading_exponential_reward_coeff)

        # linear velocity reward
        linear_velocity_rew = linear_velocity - self._task_cfg.linear_velocity_min_value
        linear_velocity_rew[linear_velocity_rew < 0] = 0
        linear_velocity_rew[
            linear_velocity_rew > (self._task_cfg.linear_velocity_max_value - self._task_cfg.linear_velocity_min_value)
        ] = (self._task_cfg.linear_velocity_max_value - self._task_cfg.linear_velocity_min_value)
        # max_possible_linear_rew = (
        #     self._task_cfg.linear_velocity_max_value - self._task_cfg.linear_velocity_min_value
        # )
        # # Normalize to 0-1 based on its own max possible value from the previous calculation
        # if max_possible_linear_rew > EPS:  # Avoid division by zero
        #     linear_velocity_rew = linear_velocity_rew / max_possible_linear_rew
        # else:
        #     linear_velocity_rew = torch.zeros_like(linear_velocity_rew)
        # angular velocity reward
        angular_velocity_rew = angular_velocity - self._task_cfg.angular_velocity_min_value
        angular_velocity_rew[angular_velocity_rew < 0] = 0
        angular_velocity_rew[
            angular_velocity_rew
            > (self._task_cfg.angular_velocity_max_value - self._task_cfg.angular_velocity_min_value)
        ] = (self._task_cfg.angular_velocity_max_value - self._task_cfg.angular_velocity_min_value)
        # Normalize to 0-1 based on its own max possible value from the previous calculation
        max_possible_angular_rew = (self._task_cfg.angular_velocity_max_value - self._task_cfg.angular_velocity_min_value)
        if max_possible_angular_rew > EPS: # Avoid division by zero
            angular_velocity_rew = angular_velocity_rew / max_possible_angular_rew
        else:
            angular_velocity_rew = torch.zeros_like(angular_velocity_rew)

        # boundary rew
        boundary_rew = torch.exp(-boundary_dist / self._task_cfg.boundary_exponential_reward_coeff)

        # Checks if the goal is reached
        goal_position_reached = self._position_dist < self._task_cfg.position_tolerance
        goal_orientation_reached = heading_dist < self._task_cfg.heading_tolerance
        goal_reached = goal_position_reached * goal_orientation_reached
        reached_ids = goal_reached.nonzero(as_tuple=False).squeeze(-1)
        # if the goal is reached, the target index is updated
        self._target_index = self._target_index + goal_reached
        # Check if the trajectory is completed
        self._trajectory_completed = self._target_index > self._num_goals
        self.total_goals_reached += goal_reached
        # To avoid out of bounds errors, set the target index to 0 if the trajectory is completed
        # If the task loops, then the target index is set to 0 which will make the robot go back to the first goal
        # The episode termination is handled in the get_dones method (looping or not)
        self._target_index = self._target_index * (~self._trajectory_completed)

        # If goal is reached make next progress null
        self._previous_position_dist[reached_ids] = 0

        # Update logs
        self.scalar_logger.log("task_reward", "Rendezvous/AVG/linear_velocity", linear_velocity_rew * self._task_cfg.linear_velocity_weight)
        self.scalar_logger.log("task_reward", "Rendezvous/AVG/angular_velocity", angular_velocity_rew * self._task_cfg.angular_velocity_weight)
        self.scalar_logger.log("task_reward", "Rendezvous/AVG/boundary", boundary_rew * self._task_cfg.boundary_weight)
        self.scalar_logger.log("task_reward", "Rendezvous/AVG/orientation", orientation_rew * self._task_cfg.orientation_weight)
        self.scalar_logger.log("task_reward", "Rendezvous/AVG/target_heading", heading_to_target_rew * self._task_cfg.target_heading_weight)
        self.scalar_logger.log("task_reward", "Rendezvous/AVG/position_orientation", (position_rew) * (orientation_rew) * self._task_cfg.progress_weight)
        self.scalar_logger.log("task_reward", "Rendezvous/AVG/position_rew", position_rew)
        self.scalar_logger.log("task_reward", "Rendezvous/SUM/num_goals", goal_reached)

        # position_tracking_precision = 1.0 - torch.tanh(self._position_dist / self._task_cfg.position_tolerance)
        # action_rate = self._robot.action_rate[self._env_ids]
        # reward_action_rate_at_target = (
        #     self._task_cfg.weight_action_rate_at_target * position_tracking_precision * (1.0 - torch.tanh(action_rate / self._task_cfg.tanh_std_action_rate_at_target))
        # )
        # self.scalar_logger.log("task_reward", "Rendezvous/EMA/action_rate_at_target", reward_action_rate_at_target)

        # Return the reward by combining the different components and adding the robot rewards
        reward = (
            
            (position_rew) * (orientation_rew) * self._task_cfg.progress_weight
            # position_rew * self._task_cfg.position_weight
            # + orientation_rew * self._task_cfg.orientation_weight
            + heading_to_target_rew * self._task_cfg.target_heading_weight
            + linear_velocity_rew * self._task_cfg.linear_velocity_weight
            + angular_velocity_rew * self._task_cfg.angular_velocity_weight
            # + reward_action_rate_at_target
            + boundary_rew * self._task_cfg.boundary_weight
            + self._task_cfg.time_penalty
            + self._task_cfg.reached_bonus * goal_reached
        ) + self._robot.compute_rewards(env_ids=self._env_ids)  # type: ignore[return-value]

        self.scalar_logger.log("task_reward", "Rendezvous/AVG/individual_reward", reward)
        return reward

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
        - gen_actions[0]: The lower bound of the range used to sample the difference in heading between the goals.
        - gen_actions[1]: The range used to sample the difference in heading between the goals.
        - gen_actions[2]: The lower bound of the range used to sample the distance between the goals.
        - gen_actions[3]: The range used to sample the distance between the goals.
        - gen_actions[4]: The lower bound of the range used to sample the spread of the cone in which the goals are.
        - gen_actions[5]: The range used to sample the spread of the cone in which the goals are.
        - gen_actions[6]: The value used to sample the distance between the spawn position and the first goal.
        - gen_actions[7]: The value used to sample the angle between the spawn position and the first goal.
        - gen_actions[8]: The value used to sample the angle between the spawn heading and the first goal's heading.
        - gen_actions[9]: The value used to sample the linear velocity of the robot at spawn.
        - gen_actions[10]: The value used to sample the angular velocity of the robot at spawn.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            gen_actions (torch.Tensor | None): The actions for the task. Defaults to None.
            env_seeds (torch.Tensor | None): The seeds for the environments. Defaults to None.
        """

        super().reset(env_ids, gen_actions=gen_actions, env_seeds=env_seeds)

        # The first 6 env actions define ranges, we need to make sure they don't exceed the [0,1] range.
        # They are given as [min, delta] we will convert them to [min, max] that is max = min + delta
        # Note that they are defined as [min, delta] to make sure the min is the min and the max is the max. This
        # is always true as they are strictly positive.
        self._gen_actions[env_ids, 1] = torch.clip(self._gen_actions[env_ids, 0] + self._gen_actions[env_ids, 1], max=1)
        self._gen_actions[env_ids, 3] = torch.clip(self._gen_actions[env_ids, 2] + self._gen_actions[env_ids, 3], max=1)
        self._gen_actions[env_ids, 5] = torch.clip(self._gen_actions[env_ids, 4] + self._gen_actions[env_ids, 5], max=1)

        # Randomizes goals and initial conditions
        self.set_goals(env_ids)
        self.set_initial_conditions(env_ids)

        # Resets the goal reached flag
        self._goal_reached[env_ids] = 0

        # Reset the target index and trajectory completed
        self._target_index[env_ids] = 0
        self._trajectory_completed[env_ids] = False

        # Make sure the position error and position dist are up to date after the reset
        self._position_error[env_ids] = (
            self._target_positions[env_ids, self._target_index[env_ids]]
            - self._robot.root_link_pos_w[self._env_ids, :2][env_ids]
        )
        self._position_dist[env_ids] = torch.linalg.norm(self._position_error[env_ids], dim=-1)
        self._previous_position_dist[env_ids] = self._position_dist[env_ids].clone()

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates if the platforms should be killed or not.

        Returns:
            torch.Tensor: Whether the platforms should be killed or not."""

        # Kill robots that would stray too far from the target.
        self._position_error = (
            self._target_positions[self._ALL_INDICES, self._target_index]
            - self._robot.root_link_pos_w[self._env_ids, :2]
        )
        self._previous_position_dist = self._position_dist.clone()
        self._position_dist = torch.linalg.norm(self._position_error, dim=-1)
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.where(
            self._position_dist > self._task_cfg.maximum_robot_distance,
            ones,
            task_failed,
        )

        task_completed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        # If the task is set to loop, don't terminate the episode early.
        # if not self._task_cfg.loop:
        #     task_completed = torch.where(self._trajectory_completed > 0, ones, task_completed)
        return task_failed, task_completed

    # def set_goals(self, env_ids: torch.Tensor):
    #     """
    #     Generates a sequence of goals on a half-circle trajectory.
    #     All goals are oriented to point towards a single, randomly chosen focus point in space.

    #     Args:
    #         env_ids (torch.Tensor): The ids of the environments to reset.
    #     """
    #     num_resets = len(env_ids)

    #     # 1. Sample trajectory-wide parameters ONCE per environment.
    #     # The radius of the half-circle trajectory, centered on the environment origin.
    #     circle_radius = (
    #         self._rng.sample_uniform_torch(
    #             self._gen_actions[env_ids, 2], self._gen_actions[env_ids, 3], (num_resets,), env_ids
    #         )
    #         * (self._task_cfg.goal_max_dist - self._task_cfg.goal_min_dist)
    #         + self._task_cfg.goal_min_dist
    #     )

    #     # Define a single random focus point that all goals will look at.
    #     # This point is chosen independently of the circle's center.
    #     # Note: You may need to add 'focus_point_distance_range' and 'focus_point_angle_range'
    #     # to your task configuration class (RendezvousCfg).
    #     focus_point_dist = self._rng.sample_uniform_torch(
    #         self._task_cfg.focus_point_distance_range[0], self._task_cfg.focus_point_distance_range[1], (1,), env_ids
    #     )
    #     focus_point_angle = self._rng.sample_uniform_torch(
    #         self._task_cfg.focus_point_angle_range[0], self._task_cfg.focus_point_angle_range[1], (1,), env_ids
    #     )

    #     # focus_point = torch.zeros((num_resets, 2), device=self._device)
    #     # focus_point[:, 0] = focus_point_dist * torch.cos(focus_point_angle)
    #     # focus_point[:, 1] = focus_point_dist * torch.sin(focus_point_angle)
    #     focus_point = torch.zeros((num_resets, 2), device=self._device)
    #     # Calculate the focus point relative to the circle's center
    #     focus_point[:, 0] = focus_point_dist * torch.cos(focus_point_angle) + self._env_origins[env_ids, 0]
    #     focus_point[:, 1] = focus_point_dist * torch.sin(focus_point_angle) + self._env_origins[env_ids, 1]

    #     # The number of goals to place on the trajectory.
    #     self._num_goals[env_ids] = self._rng.sample_integer_torch(
    #         self._task_cfg.min_num_goals, self._task_cfg.max_num_goals, (num_resets,), ids=env_ids
    #     ).to(torch.long)

    #     # 2. Vectorized calculation of all goal positions and headings.
    #     goal_indices = torch.arange(self._task_cfg.max_num_goals, device=self._device).expand(num_resets, -1)
    #     valid_goal_mask = goal_indices < self._num_goals[env_ids].unsqueeze(-1)

    #     last_goal_idx = torch.clamp(self._num_goals[env_ids] - 1, min=1).unsqueeze(-1)
    #     progress = goal_indices / last_goal_idx

    #     angles = -math.pi / 2 + progress * math.pi

    #     # Calculate goal positions on the circle.
    #     positions_x = circle_radius.unsqueeze(-1) * torch.cos(angles) + self._env_origins[env_ids, 0].unsqueeze(-1)
    #     positions_y = circle_radius.unsqueeze(-1) * torch.sin(angles) + self._env_origins[env_ids, 1].unsqueeze(-1)
    #     new_positions = torch.stack((positions_x, positions_y), dim=-1)

    #     # Calculate orientations to point towards the new random `focus_point`.
    #     new_headings = torch.atan2(
    #         focus_point[:, 1].unsqueeze(-1) - new_positions[..., 1],
    #         focus_point[:, 0].unsqueeze(-1) - new_positions[..., 0],
    #     )

    #     # 3. Apply the mask to fill the buffers.
    #     # self._target_positions[env_ids] = torch.where(valid_goal_mask.unsqueeze(-1), new_positions, 0.0)
    #     # self._target_heading[env_ids] = torch.where(valid_goal_mask, new_headings, 0.0)
    #     self._target_positions[env_ids] = torch.where(
    #         valid_goal_mask.unsqueeze(-1), new_positions, math.nan
    #     )
    #     self._target_heading[env_ids] = torch.where(valid_goal_mask, new_headings, 0.0)

    def set_goals(self, env_ids: torch.Tensor):
        """
        Generates a sequence of goals on a half-circle trajectory using a sequential loop.

        This method first defines trajectory-wide parameters for each environment, including the radius of a
        half-circle and a common focus point. It then iteratively generates goals along this half-circle.
        All goals are oriented to point towards the single, randomly chosen focus point. If an environment
        requires fewer than the maximum number of goals, its last valid goal is duplicated for the
        remaining slots in the tensor, following a similar padding logic as the reference implementation.

        Args:
            env_ids (torch.Tensor): The ids of the environments to reset.
        """
        num_resets = len(env_ids)

        # 1. Sample trajectory-wide parameters ONCE per environment before the loop.
        # The radius of the half-circle trajectory, centered on the environment origin.
        circle_radius = (
            self._rng.sample_uniform_torch(
                self._gen_actions[env_ids, 2], self._gen_actions[env_ids, 3], 1, env_ids
            )
            * (self._task_cfg.goal_max_dist - self._task_cfg.goal_min_dist)
            + self._task_cfg.goal_min_dist
        )

        # Define a single random focus point that all goals will look at.
        focus_point_dist = self._rng.sample_uniform_torch(self._task_cfg.focus_point_distance_range[0], self._task_cfg.focus_point_distance_range[1], (1,), env_ids)
        focus_point_angle = self._rng.sample_uniform_torch(self._task_cfg.focus_point_angle_range[0], self._task_cfg.focus_point_angle_range[1], (1,), env_ids)
        
        # Calculate the focus point's coordinates relative to the environment origin.
        focus_point = torch.zeros((num_resets, 2), device=self._device)
        focus_point[:, 0] = focus_point_dist * torch.cos(focus_point_angle) + self._env_origins[env_ids, 0]
        focus_point[:, 1] = focus_point_dist * torch.sin(focus_point_angle) + self._env_origins[env_ids, 1]

        # The number of goals to place on the trajectory.
        self._num_goals[env_ids] = self._rng.sample_integer_torch(
            self._task_cfg.min_num_goals, self._task_cfg.max_num_goals, 1, ids=env_ids
        ).to(torch.long)

        # Pre-calculate the index of the last goal for each environment to determine progress.
        # Clamp at a minimum of 1 to prevent division by zero for environments with 1 goal.
        last_goal_idx = torch.clamp(self._num_goals[env_ids] - 1, min=1).float()

        # 2. Iterate to generate goals sequentially for each step up to the maximum number of goals.
        for i in range(self._task_cfg.max_num_goals):
            # Calculate the progress along the half-circle (from 0.0 to 1.0).
            progress = i / last_goal_idx
            # Map progress to an angle on the half-circle (-pi/2 to +pi/2).
            angles = -math.pi / 2 + progress * math.pi

            # Calculate the goal's position on the circle for all environments.
            pos_x = circle_radius * torch.cos(angles) + self._env_origins[env_ids, 0]
            pos_y = circle_radius * torch.sin(angles) + self._env_origins[env_ids, 1]
            
            # Calculate the heading from the new position towards the focus point.
            heading = torch.atan2(
                focus_point[:, 1] - pos_y,
                focus_point[:, 0] - pos_x,
            )

            if i == 0:
                # For the first goal, directly assign the calculated values.
                self._target_positions[env_ids, i, 0] = pos_x
                self._target_positions[env_ids, i, 1] = pos_y
                self._target_heading[env_ids, i] = heading
            else:
                # For subsequent goals, first assign the newly calculated values.
                self._target_positions[env_ids, i, 0] = pos_x
                self._target_positions[env_ids, i, 1] = pos_y
                self._target_heading[env_ids, i] = heading
                
                # Then, for environments that have already reached their goal count,
                # overwrite the new goal with the previous one to pad the tensor.
                overwrite_mask = i >= self._num_goals[env_ids]

                # Apply the overwrite for positions.
                self._target_positions[env_ids, i] = torch.where(
                    overwrite_mask.unsqueeze(-1),  # Broadcast mask for x,y coordinates
                    self._target_positions[env_ids, i - 1],
                    self._target_positions[env_ids, i],
                )
                # Apply the overwrite for headings.
                self._target_heading[env_ids, i] = torch.where(
                    overwrite_mask,
                    self._target_heading[env_ids, i - 1],
                    self._target_heading[env_ids, i],
                )

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
            orientation and velocity of the robot."""

        num_resets = len(env_ids)

        # Randomizes the initial pose of the platform
        initial_pose = torch.zeros((num_resets, 7), device=self._device, dtype=torch.float32)

        # Position, the position is picked in a cone behind the first target.
        r = (
            self._gen_actions[env_ids, 6] * (self._task_cfg.spawn_max_dist - self._task_cfg.spawn_min_dist)
            + self._task_cfg.spawn_min_dist
        )
        # We add pi to make sure the robot is behind the first target
        # if the env_action is 0, then the robot is perfectly aligned with the target
        theta = (
            (
                self._gen_actions[env_ids, 7]
                * (self._task_cfg.spawn_max_cone_spread - self._task_cfg.spawn_min_cone_spread)
                + self._task_cfg.spawn_min_cone_spread
            )
            * self._rng.sample_sign_torch("float", 1, ids=env_ids)
            + self._target_heading[env_ids, 0]
            + math.pi
        )
        initial_pose[:, 0] = r * torch.cos(theta) + self._target_positions[env_ids, 0, 0]
        initial_pose[:, 1] = r * torch.sin(theta) + self._target_positions[env_ids, 0, 1]
        initial_pose[:, 2] = self._robot_origins[env_ids, 2]
        # Orientation
        delta_heading = (
            (
                self._gen_actions[env_ids, 8]
                * (self._task_cfg.spawn_max_heading_dist - self._task_cfg.spawn_min_heading_dist)
            )
            + self._task_cfg.spawn_min_heading_dist
        ) * self._rng.sample_sign_torch("float", 1, ids=env_ids)
        theta = delta_heading + self._target_heading[env_ids, 0]
        initial_pose[:, 3] = torch.cos(theta * 0.5)
        initial_pose[:, 6] = torch.sin(theta * 0.5)

        # Randomizes the velocity of the platform
        self.initial_velocity[env_ids] = torch.zeros((num_resets, 6), device=self._device, dtype=torch.float32)

        # Linear velocity
        theta = self._rng.sample_uniform_torch(0, 2 * math.pi, 1, ids=env_ids)
        velocity_norm = (
            self._gen_actions[env_ids, 9] * (self._task_cfg.spawn_max_lin_vel - self._task_cfg.spawn_min_lin_vel)
            + self._task_cfg.spawn_min_lin_vel
        )
        self.initial_velocity[env_ids, 0] = velocity_norm * torch.cos(theta)
        self.initial_velocity[env_ids, 1] = velocity_norm * torch.sin(theta)

        # Angular velocity of the platform
        angular_velocity = (
            self._gen_actions[env_ids, 10] * (self._task_cfg.spawn_max_ang_vel - self._task_cfg.spawn_min_ang_vel)
            + self._task_cfg.spawn_min_ang_vel
        )
        self.initial_velocity[env_ids, 5] = angular_velocity

        # Apply to articulation
        if self._env_ids is not None:
            self._robot.set_pose(initial_pose, self._env_ids[env_ids])
            self._robot.set_velocity(self.initial_velocity[env_ids], self._env_ids[env_ids])

    def create_task_visualization(self) -> None:
        """Adds the visual marker to the scene.

        There are 3 makers for the goals:
        - The next goal is marked in red.
        - The passed goals are marked in grey.
        - The current goals are marked in green.

        They are represented by a pin with an arrow on top of it. The arrow's orientation is the same as the goal's.
        The pin is here to precisely visualize the position of the goal.

        The robot is represented by a diamond with two colors. The colors are used to represent the orientation of the
        robot. The green color represents the front of the robot, and the red color represents the back of the robot.
        """

        # Define the visual markers and edit their properties
        goal_marker_cfg_green = PIN_ARROW_CFG.copy()
        goal_marker_cfg_green.markers["pin_arrow"].visual_material.diffuse_color = (
            0.0,
            1.0,
            0.0,
        )
        goal_marker_cfg_grey = PIN_ARROW_CFG.copy()
        goal_marker_cfg_grey.markers["pin_arrow"].visual_material.diffuse_color = (
            0.5,
            0.5,
            0.5,
        )
        goal_marker_cfg_red = PIN_ARROW_CFG.copy()
        robot_marker_cfg = ARROW_CFG.copy()
        goal_marker_cfg_red.prim_path = f"/Visuals/Command/task_{self._task_uid}/next_goal"
        goal_marker_cfg_grey.prim_path = f"/Visuals/Command/task_{self._task_uid}/passed_goals"
        goal_marker_cfg_green.prim_path = f"/Visuals/Command/task_{self._task_uid}/current_goals"
        robot_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/robot_pose"
        # We should create only one of them.
        self.next_goal_visualizer = VisualizationMarkers(goal_marker_cfg_red)
        self.passed_goals_visualizer = VisualizationMarkers(goal_marker_cfg_grey)
        self.current_goals_visualizer = VisualizationMarkers(goal_marker_cfg_green)
        self.robot_pos_visualizer = VisualizationMarkers(robot_marker_cfg)

    def update_task_visualization(self) -> None:
        """Updates the visual marker to the scene.
        This implements the logic to check to use the appropriate colors. It also discards duplicate goals.
        Since the number of goals is flexible, but the length of the tensor is fixed, we need to discard some of the
        goals."""

        # For the reminder of the goals, we need to check if they are passed or not.
        # We do this by iterating over the 2nd axis of the self._target_position tensor.
        # The update time scales linearly with the number of goals.
        passed_goals_pos_list = []
        passed_goals_heading_list = []
        next_goals_pos_list = []
        next_goals_heading_list = []
        for i in range(self._task_cfg.max_num_goals):
            ok_goals = self._num_goals >= i
            passed_goals = torch.logical_and(self._target_index > i, ok_goals)
            passed_goals_pos_list.append(self._target_positions[passed_goals, i])
            passed_goals_heading_list.append(self._target_heading[passed_goals, i])
            next_goals = torch.logical_and(self._target_index < i, ok_goals)
            next_goals_pos_list.append(self._target_positions[next_goals, i])
            next_goals_heading_list.append(self._target_heading[next_goals, i])
        passed_pos_goals = torch.cat(passed_goals_pos_list, dim=0)
        passed_heading_goals = torch.cat(passed_goals_heading_list, dim=0)
        next_pos_goals = torch.cat(next_goals_pos_list, dim=0)
        next_heading_goals = torch.cat(next_goals_heading_list, dim=0)

        # Assign the positions to the visual markers (They need to be dynamically allocated)
        # Under the hood, these are converted to numpy arrays, so that's definitely a waste, but since it's
        # only for visualization, it's not a big deal.
        current_goals_pos = torch.zeros(
            (self._target_positions[self._ALL_INDICES, self._target_index].shape[0], 3),
            device=self._device,
        )
        current_goals_pos[:, :2] = self._target_positions[self._ALL_INDICES, self._target_index]
        passed_goals_pos = torch.zeros((passed_pos_goals.shape[0], 3), device=self._device)
        passed_goals_pos[:, :2] = passed_pos_goals
        next_goals_pos = torch.zeros((next_pos_goals.shape[0], 3), device=self._device)
        next_goals_pos[:, :2] = next_pos_goals

        # Assign the orientations to the visual markers (They need to be dynamically allocated)
        current_goals_quat = torch.zeros(
            (self._target_heading[self._ALL_INDICES, self._target_index].shape[0], 4),
            device=self._device,
        )
        current_goals_quat[:, 0] = torch.cos(self._target_heading[self._ALL_INDICES, self._target_index] * 0.5)
        current_goals_quat[:, 3] = torch.sin(self._target_heading[self._ALL_INDICES, self._target_index] * 0.5)
        passed_goals_quat = torch.zeros((passed_heading_goals.shape[0], 4), device=self._device)
        passed_goals_quat[:, 0] = torch.cos(passed_heading_goals * 0.5)
        passed_goals_quat[:, 3] = torch.sin(passed_heading_goals * 0.5)
        next_goals_quat = torch.zeros((next_heading_goals.shape[0], 4), device=self._device)
        next_goals_quat[:, 0] = torch.cos(next_heading_goals * 0.5)
        next_goals_quat[:, 3] = torch.sin(next_heading_goals * 0.5)

        # If there are no goals of a given type, we should hide the markers.
        # Logic for NEXT goals (Red)
        if next_goals_pos.shape[0] == 0:
            self.next_goal_visualizer.set_visibility(False)
        else:
            self.next_goal_visualizer.set_visibility(True)
            self.next_goal_visualizer.visualize(next_goals_pos, orientations=next_goals_quat)

        # Logic for PASSED goals (Grey)
        if passed_goals_pos.shape[0] == 0:
            self.passed_goals_visualizer.set_visibility(False)
        else:
            self.passed_goals_visualizer.set_visibility(True)
            self.passed_goals_visualizer.visualize(passed_goals_pos, orientations=passed_goals_quat)
            
        # Logic for CURRENT goals (Green)
        if current_goals_pos.shape[0] == 0:
            self.current_goals_visualizer.set_visibility(False)
        else:
            self.current_goals_visualizer.set_visibility(True)
            self.current_goals_visualizer.visualize(current_goals_pos, orientations=current_goals_quat)

        # Update the robot visualization. TODO Ideally we should lift the diamond a bit.
        self._robot_marker_pos[:, :2] = self._robot.root_link_pos_w[self._env_ids, :2]
        self.robot_pos_visualizer.visualize(self._robot_marker_pos, self._robot.root_link_quat_w[self._env_ids])

        self._robot.update_robot_visualization()
