# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from isaaclab.markers import BICOLOR_DIAMOND_CFG, PIN_SPHERE_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveScene

from isaaclab_tasks.rans import GoToPositionCfg

from .task_core import TaskCore
import torch.nn.functional as F

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class GoToPositionTask(TaskCore):
    """
    Implements the GoToPosition task. The robot has to reach a target position and keep it.
    """

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        task_cfg: GoToPositionCfg = GoToPositionCfg(),
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

        super().__init__(scene=scene, task_uid=task_uid, num_envs=num_envs, device=device, env_ids=env_ids, decimation=decimation, num_tasks=num_tasks)

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

        target_position: The position of the target in the world frame.
        position_distance: The distance between the robot and the target position.
        cos_heading_to_target_error: The cosine of the angle between the robot's heading and the target position.
        sin_heading_to_target_error: The sine of the angle between the robot's heading and the target position.
        half_init_lin_vel_x: Whether the robot's linear velocity along the x-axis is less than half of the initial linear velocity.
        half_init_lin_vel_y: Whether the robot's linear velocity along the y-axis is less than half of the initial linear velocity.
        half_init_ang_vel: Whether the robot's angular velocity is less than half of the initial angular velocity.

        Returns:
            list[str]: The keys of the data used for evaluation."""
        
        return [
            "target_position", 
            "position_distance", 
            "cos_heading_to_target_error", 
            "sin_heading_to_target_error",
            "initial_lin_vel_x",
            "initial_lin_vel_y",
            "initial_ang_vel",
            "half_init_lin_vel_x",
            "half_init_lin_vel_y",
            "half_init_ang_vel",
        ]
    
    @property
    def eval_data_specs(self)-> dict[str, list[str]]:
        """
        Returns the specifications of the data used for evaluation.

        Returns:
            dict: The specifications of the data used for evaluation."""
        
        return {
            "target_position": [".x.m", ".y.m"],
            "position_distance": [".distance.m"],
            "cos_heading_to_target_error": [".cos(heading).u"],
            "sin_heading_to_target_error": [".sin(heading).u"],
            "initial_lin_vel_x": [".initial_lin_vel_x.m/s"],
            "initial_lin_vel_y": [".initial_lin_vel_y.m/s"],
            "initial_ang_vel": [".initial_ang_vel.rad/s"],
            "half_init_lin_vel_x": [".half_init_lin_vel_x.u"],
            "half_init_lin_vel_y": [".half_init_lin_vel_y.u"],
            "half_init_ang_vel": [".half_init_ang_vel.u"],
        }
    
    @property
    def eval_data(self) -> dict:
        """
        Returns the data used for evaluation.

        Returns:
            dict: The data used for evaluation."""
        return {
            "target_position": self._target_positions,
            "position_distance": self._position_dist,
            "cos_heading_to_target_error": self._task_data[:, 1],
            "sin_heading_to_target_error": self._task_data[:, 2],
            "initial_lin_vel_x": self.initial_velocity[:, 0],
            "initial_lin_vel_y": self.initial_velocity[:, 1],
            "initial_ang_vel": self.initial_velocity[:, 5],
            "half_init_lin_vel_x": self._half_init_lin_vel_x,
            "half_init_lin_vel_y": self._half_init_lin_vel_y,
            "half_init_ang_vel": self._half_init_ang_vel,
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
        self._target_positions = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self._markers_pos = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.initial_velocity = torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)
        self._half_init_lin_vel_x = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)
        self._half_init_lin_vel_y = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)
        self._half_init_ang_vel = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)
        
        

    def create_logs(self) -> None:
        """
        Creates a dictionary to store the training statistics for the task."""
        super().create_logs()
        self.scalar_logger.add_log("task_state", "GoToPosition/AVG/normed_linear_velocity", "mean")
        self.scalar_logger.add_log("task_state", "GoToPosition/AVG/absolute_angular_velocity", "mean")
        self.scalar_logger.add_log("task_state", "GoToPosition/EMA/position_distance", "ema")
        self.scalar_logger.add_log("task_state", "GoToPosition/EMA/boundary_distance", "ema")
        self.scalar_logger.add_log("task_state", "GoToPosition/AVG/target_heading_error", "mean")

        self.scalar_logger.add_log("task_reward", "GoToPosition/AVG/position", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPosition/AVG/heading", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPosition/AVG/linear_velocity", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPosition/AVG/angular_velocity", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPosition/AVG/boundary", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPosition/EMA/action_rate_at_target", "ema")
        self.scalar_logger.add_log("task_reward", "GoToPosition/AVG/individual_reward", "mean")
        self.scalar_logger.set_ema_coeff(self._task_cfg.ema_coeff)

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

        Returns:
            torch.Tensor: The observation tensor."""
        
        # print("$"*50)
        # print("GoToPositionTask get_observations")
        # print(self._target_positions[:, :2])
        # print(self._robot.root_link_pos_w[self._env_ids, :2])
        # position error
        position_error = self._target_positions[:, :2] - self._robot.root_link_pos_w[self._env_ids, :2]
        position_dist = torch.linalg.norm(position_error, dim=-1)
        # position error expressed as distance and angular error (to the position)
        heading = self._robot.heading_w[self._env_ids]
        target_heading_w = torch.atan2(
            self._target_positions[:, 1] - self._robot.root_link_pos_w[self._env_ids, 1],
            self._target_positions[:, 0] - self._robot.root_link_pos_w[self._env_ids, 0],
        )
        target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

        # Store in buffer [distance, cos(angle), sin(angle), lin_vel_x, lin_vel_y, ang_vel, prev_action]
        self._task_data[:, 0] = position_dist
        self._task_data[:, 1] = torch.cos(target_heading_error)
        self._task_data[:, 2] = torch.sin(target_heading_error)
        self._task_data[:, 3:5] = self._robot.root_com_lin_vel_b[self._env_ids, :2]
        self._task_data[:, 5] = self._robot.root_com_ang_vel_w[self._env_ids, -1]

        for randomizer in self.randomizers:
            randomizer.observations(observations=self._task_data)

        # Check if it reached half of the initial linear and angular velocity
        self._half_init_lin_vel_x = torch.abs(self.initial_velocity[:, 0] / 2) >= torch.abs(self._robot.root_com_vel_w[self._env_ids, 0])
        self._half_init_lin_vel_y = torch.abs(self.initial_velocity[:, 1] / 2) >= torch.abs(self._robot.root_com_vel_w[self._env_ids, 1])
        self._half_init_ang_vel = torch.abs(self.initial_velocity[:, 5] / 2) >= torch.abs(self._robot.root_com_vel_w[self._env_ids, -1])

        # Task specific observations
        # task_obs = torch.concat((
        #     normalized_masses.unsqueeze(-1), 
        #     nomalized_coms, 
        #     self._robot._thrusters_active_mask[self._env_ids]
        # ), dim=-1)  # [masses, com_x, com_y, thrusters_active_mask]
        task_id_one_hot = F.one_hot(torch.tensor([self._task_uid], device=self._device), num_classes=self._num_tasks).squeeze(0).repeat(self._num_envs, 1)
        task_obs = task_id_one_hot

        # Concatenate the task observations with the robot observations
        return torch.concat((self._task_data, self._robot.get_observations(env_ids=self._env_ids)), dim=-1), task_obs

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
        Args:
            current_state (torch.Tensor): The current state of the robot.
            actions (torch.Tensor): The actions taken by the robot.
            step (int, optional): The current step. Defaults to 0.
        Returns:
            torch.Tensor: The reward for the current state of the robot."""
        # position error
        self._position_error = self._target_positions[:, :2] - self._robot.root_link_pos_w[self._env_ids, :2]
        self._position_dist = torch.linalg.norm(self._position_error, dim=-1)
        # boundary distance
        boundary_dist = torch.abs(self._task_cfg.maximum_robot_distance - self._position_dist)
        # normed linear velocity
        linear_velocity = torch.linalg.norm(self._robot.root_com_vel_w[self._env_ids, :2], dim=-1)
        # normed angular velocity
        angular_velocity = torch.abs(self._robot.root_com_vel_w[self._env_ids, -1])
        # Compute the heading to the target
        heading = self._robot.heading_w[self._env_ids]
        target_heading_w = torch.atan2(
            self._target_positions[:, 1] - self._robot.root_link_pos_w[self._env_ids, 1],
            self._target_positions[:, 0] - self._robot.root_link_pos_w[self._env_ids, 0],
        )
        target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        # Update logs
        self.scalar_logger.log("task_state", "GoToPosition/EMA/position_distance", self._position_dist)
        self.scalar_logger.log("task_state", "GoToPosition/EMA/boundary_distance", boundary_dist)
        self.scalar_logger.log("task_state", "GoToPosition/AVG/normed_linear_velocity", linear_velocity)
        self.scalar_logger.log("task_state", "GoToPosition/AVG/absolute_angular_velocity", angular_velocity)
        self.scalar_logger.log("task_state", "GoToPosition/AVG/target_heading_error", target_heading_error)
        
        # position reward
        position_rew = torch.exp(-self._position_dist / self._task_cfg.position_exponential_reward_coeff)

        progress_rew = 1 - torch.clamp(
            torch.linalg.norm(self._target_positions[:, :2] - self._robot.root_link_pos_w[self._env_ids, :2], dim=-1)
            / (self._task_cfg.maximum_robot_distance + EPS),
            min=0.0,
            max=1.0,
        )
        # heading reward + distance scaling
        dist_scaling = (
            torch.clamp(
                self._position_dist, self._task_cfg.min_heading_dist_scaler, self._task_cfg.max_heading_dist_scaler
            )
            - self._task_cfg.min_heading_dist_scaler
        ) / (self._task_cfg.max_heading_dist_scaler - self._task_cfg.min_heading_dist_scaler)
        heading_rew = (
            torch.exp(-torch.abs(target_heading_error) / self._task_cfg.heading_exponential_reward_coeff) * dist_scaling
        )
        # linear velocity reward
        linear_velocity_rew = linear_velocity - self._task_cfg.linear_velocity_min_value
        linear_velocity_rew[linear_velocity_rew < 0] = 0
        linear_velocity_rew[
            linear_velocity_rew > (self._task_cfg.linear_velocity_max_value - self._task_cfg.linear_velocity_min_value)
        ] = (self._task_cfg.linear_velocity_max_value - self._task_cfg.linear_velocity_min_value)
        # proximity_scale = torch.exp(-self._position_dist / 0.1)
        # linear_velocity_rew = -proximity_scale * linear_velocity
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
        goal_is_reached = (self._position_dist < self._task_cfg.position_tolerance).int()
        self._goal_reached *= goal_is_reached  # if not set the value to 0
        self._goal_reached += goal_is_reached  # if it is add 1
        # Update logs for rewards
        self.scalar_logger.log("task_reward", "GoToPosition/AVG/position", position_rew * self._task_cfg.position_weight)
        self.scalar_logger.log("task_reward", "GoToPosition/AVG/heading", heading_rew * self._task_cfg.progress_weight)
        self.scalar_logger.log("task_reward", "GoToPosition/AVG/linear_velocity", linear_velocity_rew * self._task_cfg.linear_velocity_weight)
        self.scalar_logger.log("task_reward", "GoToPosition/AVG/angular_velocity", angular_velocity_rew * self._task_cfg.angular_velocity_weight)
        self.scalar_logger.log("task_reward", "GoToPosition/AVG/boundary", boundary_rew * self._task_cfg.boundary_weight)

        position_tracking_precision = 1.0 - torch.tanh(self._position_dist / self._task_cfg.position_tolerance)
        action_rate = self._robot.action_rate[self._env_ids]
        reward_action_rate_at_target = (
            self._task_cfg.weight_action_rate_at_target * position_tracking_precision * (1.0 - torch.tanh(action_rate / self._task_cfg.tanh_std_action_rate_at_target))
        )
        self.scalar_logger.log("task_reward", "GoToPosition/EMA/action_rate_at_target", reward_action_rate_at_target)


        # Return the reward by combining the different components and adding the robot rewards
        reward =  (
            position_rew * self._task_cfg.position_weight
            # + progress_rew * self._task_cfg.progress_weight
            + heading_rew * self._task_cfg.heading_weight
            + linear_velocity_rew * self._task_cfg.linear_velocity_weight
            + angular_velocity_rew * self._task_cfg.angular_velocity_weight
            + reward_action_rate_at_target
            # + boundary_rew * self._task_cfg.boundary_weight
        ) + self._robot.compute_rewards(env_ids=self._env_ids)

        self.scalar_logger.log("task_reward", "GoToPosition/AVG/individual_reward", reward)
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

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates if the platforms should be killed or not.

        Returns:
            torch.Tensor: Whether the platforms should be killed or not."""

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

        # The position is picked randomly in a square centered on the origin
        self._target_positions[env_ids] = (
            self._rng.sample_uniform_torch(
                -self._task_cfg.goal_max_dist_from_origin, self._task_cfg.goal_max_dist_from_origin, 2, ids=env_ids
            )
            + self._env_origins[env_ids, :2]
        )

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

    def create_task_visualization(self) -> None:
        """Adds the visual marker to the scene.

        There are two markers: one for the goal and one for the robot.

        The goal marker is a pin with a red sphere on top. The pin is here to precisely show the position of the goal.

        The robot is represented by a diamond with two colors. The colors are used to represent the orientation of the
        robot. The green color represents the front of the robot, and the red color represents the back of the robot.
        """

        # Define the visual markers and edit their properties
        goal_marker_cfg = PIN_SPHERE_CFG.copy()
        robot_marker_cfg = BICOLOR_DIAMOND_CFG.copy()
        goal_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/goal_pose"
        robot_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/robot_pose"
        # We should create only one of them.
        self.goal_pos_visualizer = VisualizationMarkers(goal_marker_cfg)
        self.robot_pos_visualizer = VisualizationMarkers(robot_marker_cfg)

    def update_task_visualization(self) -> None:
        """Updates the visual marker to the scene."""

        if self._num_tasks == 1:
            self.goal_pos_visualizer.visualize(self._markers_pos)
            self._robot_marker_pos[:, :2] = self._robot.root_link_pos_w[:, :2]
            self.robot_pos_visualizer.visualize(self._robot_marker_pos, self._robot.root_link_quat_w)
        else:
            # chunk_size = self.scene.num_envs // self._num_tasks
            # start_indx = (self._task_uid - 1) * chunk_size
            # end_indx = start_indx + chunk_size

            self.goal_pos_visualizer.visualize(self._markers_pos)
            self._robot_marker_pos[:, :2] = self._robot.root_link_pos_w[self._env_ids, :2]
            self.robot_pos_visualizer.visualize(self._robot_marker_pos, self._robot.root_link_quat_w[self._env_ids])

        # self._robot.update_robot_visualization()

        

