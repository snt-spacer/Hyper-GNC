# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
import isaacsim.core.utils.prims as prim_utils

from isaaclab.markers import SPHERE_CFG, VisualizationMarkers
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.markers import POSE_MARKER_3D_CFG, VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene
from isaaclab.utils import math as math_utils

from isaaclab_tasks.rans import VisualGoToPose3DCfg
from isaaclab_tasks.rans.utils import ObjectStorage

from .task_core import TaskCore
import torch.nn.functional as F

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class VisualGoToPose3DTask(TaskCore):
    """
    Implements the GoToPose task in 3D space. The robot has to reach a target position and keep it.
    """

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        task_cfg: VisualGoToPose3DCfg = VisualGoToPose3DCfg(),
        task_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        env_ids: torch.Tensor | None = None,
        num_tasks: int = 1,
    ) -> None:
        """
        Initializes the 3D GoToPose task.

        Args:
            scene: Interactive scene containing sim entities for the task.
            task_cfg: The configuration of the task.
            task_uid: The unique id of the task.
            num_envs: The number of environments.
            device: The device on which the tensors are stored.
            env_ids: The ids of the environments used by this task.
        """

        super().__init__(scene=scene, task_uid=task_uid, num_envs=num_envs, device=device, env_ids=env_ids, num_tasks=num_tasks)

        # Task and reward parameters
        self._task_cfg = task_cfg

        # Defines the observation and action space sizes for this task
        self._dim_task_obs = self._task_cfg.observation_space
        self._dim_gen_act = self._task_cfg.gen_space

        # Hardcoded docking port transform obtained from USD, scaled down by a factor of 100
        self._docking_port_pos = torch.tensor([-0.170, -3.1559212, 2.210], device=self._device)
        self._docking_port_rot = math_utils.quat_from_euler_xyz(
            torch.tensor([math.radians(90)], device=self._device),
            torch.tensor([math.radians(0)], device=self._device),
            torch.tensor([math.radians(90)], device=self._device)
        )[0]  # extract 1x4 -> [4]


        self._docking_pose_local = torch.zeros((7,), device=self._device)
        self._docking_pose_local[:3] = self._docking_port_pos
        self._docking_pose_local[3:] = self._docking_port_rot
        self._forward_offset = torch.tensor([0.25, 0.0, 0.0], device=self._device)  # meters, [0.15, 0.0, 0.0]

        # Buffers
        self.initialize_buffers(env_ids=env_ids)

        # ISS box
        self.first_reset = True


        self.design_scene()

    @property
    def eval_data_keys(self) -> list[str]:
        """
        Returns the keys of the data used for evaluation.

        Returns:
            list[str]: The keys of the data used for evaluation.
        """
        return [
            "position_error",
            "orientation_error",
            "position_dist",
            "target_positions",
            "target_orientations",
            "local_pos_error",
            "collision_signal",
        ]
    
    @property
    def eval_data_specs(self)->dict[str, list[str]]:

        return {
            "position_error": ["(N, 3)"],
            "orientation_error": ["(N,)"],
            "position_dist": ["(N,)"],
            "target_positions": ["(N, 3)"],
            "target_orientations": ["(N, 4)"],
            "local_pos_error": ["(N, 3)"],
            "collision_signal": ["(N, 1)"],
        }
    
    @property
    def eval_data(self) -> dict:
        """
        Returns the data used for evaluation.

        Returns:
            dict: The data used for evaluation.
        """
        return {
            "position_error": self._position_error,
            "orientation_error": self._orientation_error,
            "position_dist": self._position_dist,
            "target_positions": self._target_positions,
            "target_orientations": self._target_orientations,
            "local_pos_error": self._local_pos_error,
            "collision_signal": self.collided_signal,
        }

    def initialize_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Initializes the buffers used by the 3D task.

        Args:
            env_ids: The ids of the environments used by this task.
        """

        super().initialize_buffers(env_ids)
        self._position_error = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)  # (x, y, z)
        self._position_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._previous_position_dist = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)
        self._target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self._local_pos_error = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        # Orientation tracking (quaternion)
        self._target_orientations = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float32
        )  # (qw, qx, qy, qz)
        self._target_orientations[:, 0] = 1.0  # Initialize to no rotation
        self._orientation_error = torch.zeros(
            (self._num_envs,), device=self._device, dtype=torch.float32
        )  # Orientation error metric
        self._markers_pos = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self._markers_rot = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)

    def register_robot(self, robot) -> None:
        self._robot = robot

    def run_setup(self, robot, envs_origin):
        super().run_setup(robot, envs_origin)
        
    # def register_sensors(self) -> None:
        # Walls
        # filters = [f"/World/envs/env_.*/ISSBox/wall_{name}" for name in self.wall_box_names]
        # self._robot.activateSensors("contacts", filters)
        # self._robot.register_sensors()

    def create_logs(self) -> None:
        """
        Creates a dictionary to store the training statistics for the task.
        Tracks 3D velocity, angular velocity, and position distance.
        """

        super().create_logs()

        self.scalar_logger.add_log("task_state", "GoToPose6DoF/AVG/normed_linear_velocity", "mean")
        self.scalar_logger.add_log("task_state", "GoToPose6DoF/AVG/absolute_angular_velocity", "mean")
        self.scalar_logger.add_log("task_state", "GoToPose6DoF/EMA/position_distance", "ema")
        self.scalar_logger.add_log("task_state", "GoToPose6DoF/EMA/orientation_error", "ema")
        # self.scalar_logger.add_log("task_state", "GoToPose6DoF/EMA/boundary_distance", "ema")
        # self.scalar_logger.add_log("task_state", "GoToPose6DoF/AVG/collision_signal", "mean")

        self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/position", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/orientation", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/linear_velocity", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/angular_velocity", "mean")
        # self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/boundary", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/progress", "mean")

        self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/total_reward", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/weighted_position_orientation_reward", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/weighted_linear_velocity_reward", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/weighted_angular_velocity_reward", "mean")
        # self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/weighted_boundary_reward", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/weighted_progress_reward", "mean")


        self.scalar_logger.set_ema_coeff(self._task_cfg.ema_coeff)

    def design_scene(self) -> None:
        
        # ###########
        # # ISS JEM #
        # ###########
        from isaaclab.utils.assets import REPO_ROOT_PATH
        from isaaclab.sim.spawners import UsdFileCfg
        from isaaclab.assets import RigidObject
        iss_jem_cfg = RigidObjectCfg(
            prim_path="/World/JEM",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),  # wxyz format
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
            ),
            spawn=UsdFileCfg(
                usd_path=f"{REPO_ROOT_PATH}/assets/environments/KibouIsaac/Asset_KIBOU3.usd",
                scale=(0.01, 0.01, 0.01),
            ),
            #collision_group=-1,  # Shared global collision group
            debug_vis=False,
        )
        self.iss_jem = RigidObject(iss_jem_cfg)

        

    def get_observations(self) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.
        Tracks 6DoF (position + quaternion orientation).

        Returns:
            torch.Tensor: The observation tensor.
        """

        # position error in world frame
        self._position_error = self._target_positions - self._robot.root_link_pos_w[self._env_ids]
        # rotate into robot's local frame via inverse of current orientation
        current_quat_w = self._robot.root_link_quat_w[self._env_ids]
        current_quat_w[current_quat_w.sum(dim=-1) == 0.0] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

        self._local_pos_error = math_utils.quat_rotate_inverse(current_quat_w, self._position_error)

        # log the global distance for debugging
        self._position_dist = self._position_error.norm(dim=-1)  # shape [N]
        self.scalar_logger.log("task_state", "GoToPose6DoF/EMA/position_distance", self._position_dist)

        # robot orientation
        # Compute a relative quaternion from robot -> target
        target_quat_w = self._target_orientations
        # rel_quat = conj(current) * target
        rel_quat = math_utils.quat_mul(
            math_utils.quat_conjugate(current_quat_w), target_quat_w
        )  # rotation from robot's orientation to target's orientation in robot local frame

        # Rotation matrix magic:
        rel_mat = math_utils.matrix_from_quat(rel_quat)
        # Extract the first two columns
        col0 = rel_mat[:, :, 0]  # shape [N, 3]
        col1 = rel_mat[:, :, 1]  # shape [N, 3]
        # Re-orthonormalize using Gram-Schmidt:
        col0 = torch.nn.functional.normalize(col0, dim=-1)
        proj = (col1 * col0).sum(dim=-1, keepdim=True)
        col1 = col1 - proj * col0
        col1 = torch.nn.functional.normalize(col1, dim=-1)
        # Stack to get 6D representation
        rel_mat_6 = torch.cat([col0, col1], dim=-1)  # shape [N, 6]


        # Store in buffer [position_dist, rotation error, linear_vel_xyz, angular_vel_xyz]
        self._task_data[:, 0:3] = self._local_pos_error
        self._task_data[:, 3:9] = rel_mat_6
        self._task_data[:, 9:12] = self._robot.root_com_lin_vel_b[self._env_ids]
        self._task_data[:, 12:15] = self._robot.root_com_ang_vel_b[self._env_ids]
        

        # Make sure that the orientation error magnitude is also updated
        current_quat = self._robot.root_quat_w[self._env_ids]  # (w, x, y, z)
        target_quat = self._target_orientations  # (w, x, y, z)
        self._orientation_error = math_utils.quat_error_magnitude(target_quat, current_quat)
        
        task_id_one_hot = F.one_hot(torch.tensor([self._task_uid], device=self._device), num_classes=self._num_tasks).squeeze(0).repeat(self._num_envs, 1)
        semantic_emb = torch.zeros((self._num_envs, 5), device=self._device)
        semantic_emb[:, 0] = self._rng.sample_uniform_torch(low=0.8, high=1.0, shape=1, ids=self._env_ids)
        semantic_emb[:, 3] = self._rng.sample_uniform_torch(low=0.8, high=1.0, shape=1, ids=self._env_ids)
       
        # Concatenate task observations with robot's internal observations
        return torch.concat((self._robot.get_observations(env_ids=self._env_ids), self._task_data), dim=-1), task_id_one_hot, semantic_emb

    def compute_rewards(self) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot.

        Returns:
            torch.Tensor: The computed reward for the current state.
        """

        # boundary distance
        boundary_dist = torch.abs(self._task_cfg.maximum_robot_distance - self._position_dist)
        # normed linear velocity
        linear_velocity = torch.linalg.norm(self._robot.root_com_vel_w[self._env_ids], dim=-1)
        # normed angular velocity
        angular_velocity = torch.linalg.norm(self._robot.root_com_ang_vel_w[self._env_ids], dim=-1)

        # Position reward (exponential decay based on distance)
        position_rew = torch.exp(-self._position_dist / self._task_cfg.position_exponential_reward_coeff)

        # Quaternion-based orientation reward (smallest rotation angle to target orientation)
        orientation_rew = torch.exp(-self._orientation_error / self._task_cfg.orientation_exponential_reward_coeff)

        # Linear velocity reward
        linear_velocity_rew = linear_velocity - self._task_cfg.linear_velocity_min_value
        linear_velocity_rew[linear_velocity_rew < 0] = 0
        linear_velocity_rew[
            linear_velocity_rew > (self._task_cfg.linear_velocity_max_value - self._task_cfg.linear_velocity_min_value)
        ] = (self._task_cfg.linear_velocity_max_value - self._task_cfg.linear_velocity_min_value)

        # Angular velocity reward
        angular_velocity_rew = angular_velocity - self._task_cfg.angular_velocity_min_value
        angular_velocity_rew[angular_velocity_rew < 0] = 0
        angular_velocity_rew[
            angular_velocity_rew
            > (self._task_cfg.angular_velocity_max_value - self._task_cfg.angular_velocity_min_value)
        ] = (self._task_cfg.angular_velocity_max_value - self._task_cfg.angular_velocity_min_value)

        # boundary reward
        boundary_rew = torch.exp(-boundary_dist / self._task_cfg.boundary_exponential_reward_coeff)

        # progress reward
        progress = self._previous_position_dist - self._position_dist
        progress_rew = progress * (self._task_cfg.maximum_robot_distance - self._position_dist)

        # Check if goal is reached
        position_goal_reached = (self._position_dist < self._task_cfg.position_tolerance).int()
        orientation_goal_reached = (self._orientation_error < self._task_cfg.orientation_tolerance).int()
        goal_is_reached = position_goal_reached * orientation_goal_reached
        self._goal_reached *= goal_is_reached  # If not reached, reset count
        self._goal_reached += goal_is_reached  # If reached, count steps in goal state


        # Logging
        self.scalar_logger.log("task_state", "GoToPose6DoF/EMA/position_distance", self._position_dist)
        self.scalar_logger.log("task_state", "GoToPose6DoF/EMA/orientation_error", self._orientation_error)
        # self.scalar_logger.log("task_state", "GoToPose6DoF/EMA/boundary_distance", boundary_dist)
        self.scalar_logger.log("task_state", "GoToPose6DoF/AVG/normed_linear_velocity", linear_velocity)
        self.scalar_logger.log("task_state", "GoToPose6DoF/AVG/absolute_angular_velocity", angular_velocity)
        # self.scalar_logger.log("task_state", "GoToPose6DoF/AVG/collision_signal", self.collided_signal)
        # Logging rewards
        self.scalar_logger.log("task_reward", "GoToPose6DoF/AVG/position", position_rew)
        self.scalar_logger.log("task_reward", "GoToPose6DoF/AVG/orientation", orientation_rew)
        self.scalar_logger.log("task_reward", "GoToPose6DoF/AVG/linear_velocity", linear_velocity_rew)
        self.scalar_logger.log("task_reward", "GoToPose6DoF/AVG/angular_velocity", angular_velocity_rew)
        # self.scalar_logger.log("task_reward", "GoToPose6DoF/AVG/boundary", boundary_rew)
        self.scalar_logger.log("task_reward", "GoToPose6DoF/AVG/progress", progress_rew)

        # Compute final reward
        total_reward = (
            (position_rew * orientation_rew) * self._task_cfg.pose_weight
            + linear_velocity_rew * self._task_cfg.linear_velocity_weight
            + angular_velocity_rew * self._task_cfg.angular_velocity_weight
            # + boundary_rew * self._task_cfg.boundary_weight
            + progress_rew * self._task_cfg.progress_weight
        ) + self._robot.compute_rewards(env_ids=self._env_ids)


        # Logging rewards w/ weights
        self.scalar_logger.log(
            "task_reward", "GoToPose6DoF/AVG/total_reward", total_reward
        )
        self.scalar_logger.log(
            "task_reward", "GoToPose6DoF/AVG/weighted_position_orientation_reward", (position_rew * orientation_rew) * self._task_cfg.pose_weight
        )
        self.scalar_logger.log(
            "task_reward", "GoToPose6DoF/AVG/weighted_linear_velocity_reward", linear_velocity_rew * self._task_cfg.linear_velocity_weight
        )
        self.scalar_logger.log(
            "task_reward", "GoToPose6DoF/AVG/weighted_angular_velocity_reward", angular_velocity_rew * self._task_cfg.angular_velocity_weight
        )
        # self.scalar_logger.log(
        #     "task_reward", "GoToPose6DoF/AVG/weighted_boundary_reward", boundary_rew * self._task_cfg.boundary_weight
        # )
        self.scalar_logger.log(
            "task_reward", "GoToPose6DoF/AVG/weighted_progress_reward", progress_rew * self._task_cfg.progress_weight
        )

        return total_reward

    def reset(
        self, env_ids: torch.Tensor, gen_actions: torch.Tensor | None = None, env_seeds: torch.Tensor | None = None
    ) -> None:
        """
        If gen_actions is None, then the environment is generated at random. This is the default mode.
        If env_seeds is None, then the seed is generated at random. This is the default mode.

        The environment actions for this task are the following, all belonging to the [0,1] range:
        - gen_actions[0]: The value used to sample the distance between the spawn position and the goal.
        - gen_actions[1]: The value used to sample the yaw offset of the robot at spawn.
        - gen_actions[2]: The value used to sample the pitch offset of the robot at spawn.
        - gen_actions[3]: The value used to sample the roll offset of the robot at spawn.
        - gen_actions[4]: The value used to sample the linear velocity of the robot at spawn.
        - gen_actions[5]: The value used to sample the angular velocity of the robot at spawn.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            gen_actions (torch.Tensor | None): The actions for the task. Defaults to None.
            env_seeds (torch.Tensor | None): The seeds for the environments. Defaults to None.
        """
        super().reset(env_ids, gen_actions=gen_actions, env_seeds=env_seeds)

        # Randomizes goals and initial conditions
        self.set_goals(env_ids)
        self.set_initial_conditions(env_ids)

        # Make sure the position error and position dist are up to date after the reset
        self._position_error[env_ids] = (
            self._target_positions[env_ids] - self._robot.root_link_pos_w[self._env_ids][env_ids]
        )
        self._position_dist[env_ids] = torch.linalg.norm(self._position_error[env_ids], dim=-1)
        self._previous_position_dist[env_ids] = self._position_dist[env_ids].clone()
        # Update also the orientation error
        current_quat = self._robot.root_link_quat_w[self._env_ids][env_ids]  # Current robot orientation
        target_quat = self._target_orientations[env_ids]  # Target orientation
        self._orientation_error[env_ids] = math_utils.quat_error_magnitude(target_quat, current_quat)

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates if the agents should be killed or not.

        Returns:
            task_failed (torch.Tensor): Environments where the robot has exceeded max error in pose.
            task_completed (torch.Tensor): Environments where the goal has been reached for enough steps.
        """
        # Compute position error in world frame (extend to 3D)
        self._position_error = self._target_positions - self._robot.root_link_pos_w[self._env_ids]
        self._previous_position_dist = self._position_dist.clone()
        self._position_dist = torch.linalg.norm(self._position_error, dim=-1)
        current_quat = self._robot.root_quat_w[self._env_ids]  # (w, x, y, z)
        target_quat = self._target_orientations  # (w, x, y, z)
        self._orientation_error = math_utils.quat_error_magnitude(target_quat, current_quat)
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.where(
            self._position_dist > self._task_cfg.maximum_robot_distance,
            ones,
            task_failed,
        )
        task_completed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        # Task completion if goal is reached for required number of steps
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
            Tuple[torch.Tensor, torch.Tensor]: The target positions and orientations.
        """
        # Fixed goal orientation: rotate the docking orientation to align with robot
        q_x = math_utils.quat_from_euler_xyz(
            torch.tensor([math.radians(90)], device=self._device),
            torch.tensor([0.0], device=self._device),
            torch.tensor([0.0], device=self._device),
        )[0]

        q_z = math_utils.quat_from_euler_xyz(
            torch.tensor([0.0], device=self._device),
            torch.tensor([math.radians(-90)], device=self._device),
            torch.tensor([0.0], device=self._device),
        )[0]

        # Compose final goal orientation (rotate docking frame)
        rotated_goal_quat = math_utils.quat_mul(q_z, math_utils.quat_mul(q_x, self._docking_port_rot))

        # Offset goal position slightly in front of docking port
        offset_world = math_utils.quat_rotate(
            self._docking_port_rot,  # original docking orientation
            self._forward_offset  # (0.0, -0.15, -0.10)
        )
        goal_pos = self._docking_port_pos + offset_world  # single target pos

        # Apply to all specified environments
        self._target_positions[env_ids] = goal_pos.unsqueeze(0).expand(len(env_ids), 3)
        self._target_orientations[env_ids] = rotated_goal_quat.unsqueeze(0).expand(len(env_ids), 4)

        # Update visual markers
        self._markers_pos[env_ids] = self._target_positions[env_ids]
        self._markers_rot[env_ids] = self._target_orientations[env_ids]


    def set_initial_conditions(self, env_ids: torch.Tensor) -> None:
        """
        Spawns robots randomly within a cuboid region relative to docking port inside the JEM.
        Orientation is set to identity. Velocity randomized based on environment action difficulty.
        """
        num_resets = len(env_ids)

        # Spawn positions within defined bounding box determined from manual experiments inside the JEM
        # In docking port local frame (always use default ranges; easy-eval only affects orientation)
        x_range = (0.3, 6.0)
        y_range = (-0.85, -1.55)
        z_range = (-0.5, 0.75)

        # Uniform random samples inside the box
        local_offsets = torch.stack([
            self._rng.sample_uniform_torch(*x_range, 1, ids=env_ids).squeeze(-1),  # x
            self._rng.sample_uniform_torch(*y_range, 1, ids=env_ids).squeeze(-1),  # y
            self._rng.sample_uniform_torch(*z_range, 1, ids=env_ids).squeeze(-1),  # z
        ], dim=-1)  # shape: (num_resets, 3)


        # Rotate into world space
        # Ensure local_offsets is always 2D
        if local_offsets.ndim == 1:
            local_offsets = local_offsets.unsqueeze(0)

        # Ensure quat is always (N, 4)
        batched_rot = self._docking_port_rot.view(1, 4).expand(local_offsets.shape[0], 4)

        assert batched_rot.shape[0] == local_offsets.shape[0], f"Shape mismatch: {batched_rot.shape} vs {local_offsets.shape}"

        offset_world = math_utils.quat_rotate(batched_rot, local_offsets)

        # Final spawn position
        # spawn_pos = self._docking_port_pos.unsqueeze(0) + offset_world
        spawn_pos = torch.zeros_like(offset_world)
        spawn_pos[:, :2] = self._docking_port_pos[:2] + self._env_origins[env_ids][:, :2] + offset_world[:, :2]  # x, y
        spawn_pos[:, 2] = self._docking_port_pos[2] + offset_world[:, 2]  # z

        # Initial pose
        initial_pose = torch.zeros((num_resets, 7), device=self._device)
        initial_pose[:, :3] = spawn_pos
        if getattr(self._task_cfg, "easy_eval_inits", False):
            # Align robot orientation directly with the goal orientation (upright, facing dock)
            initial_pose[:, 3:] = self._target_orientations[env_ids]
        else:
            # Compute base yaw to face the target
            delta_pos = self._target_positions[env_ids, :2] - spawn_pos[:, :2]
            base_yaw = torch.atan2(delta_pos[:, 1], delta_pos[:, 0])

            # Random yaw/pitch/roll offsets based on difficulty
            yaw_offset = (
                self._gen_actions[env_ids, 1]
                * (self._task_cfg.spawn_max_heading_dist - self._task_cfg.spawn_min_heading_dist)
                + self._task_cfg.spawn_min_heading_dist
            ) * self._rng.sample_sign_torch(dtype="float", shape=1, ids=env_ids)

            pitch_offset = (
                self._gen_actions[env_ids, 2]
                * (self._task_cfg.spawn_max_pitch_dist - self._task_cfg.spawn_min_pitch_dist)
                + self._task_cfg.spawn_min_pitch_dist
            ) * self._rng.sample_sign_torch(dtype="float", shape=1, ids=env_ids)

            roll_offset = (
                self._gen_actions[env_ids, 3]
                * (self._task_cfg.spawn_max_roll_dist - self._task_cfg.spawn_min_roll_dist)
                + self._task_cfg.spawn_min_roll_dist
            ) * self._rng.sample_sign_torch(dtype="float", shape=1, ids=env_ids)

            # Final orientation
            final_yaw = base_yaw + yaw_offset.squeeze(-1)

            yaw_quat = math_utils.quat_from_euler_xyz(
                torch.zeros_like(final_yaw),
                torch.zeros_like(final_yaw),
                final_yaw,
            )
            pitch_quat = math_utils.quat_from_euler_xyz(
                pitch_offset.squeeze(-1),
                torch.zeros_like(pitch_offset.squeeze(-1)),
                torch.zeros_like(pitch_offset.squeeze(-1)),
            )
            roll_quat = math_utils.quat_from_euler_xyz(
                torch.zeros_like(roll_offset.squeeze(-1)),
                roll_offset.squeeze(-1),
                torch.zeros_like(roll_offset.squeeze(-1)),
            )

            # Expand if needed
            if yaw_quat.ndim == 1:
                yaw_quat = yaw_quat.unsqueeze(0).expand(num_resets, -1)
            if pitch_quat.ndim == 1:
                pitch_quat = pitch_quat.unsqueeze(0).expand(num_resets, -1)
            if roll_quat.ndim == 1:
                roll_quat = roll_quat.unsqueeze(0).expand(num_resets, -1)

            # Compose rotations
            adjusted_quat = math_utils.quat_mul(roll_quat, math_utils.quat_mul(pitch_quat, yaw_quat))
            initial_pose[:, 3:] = adjusted_quat

        # Initial Velocity (use default behavior regardless of easy-eval setting)
        initial_velocity = torch.zeros((num_resets, 6), device=self._device, dtype=torch.float32)
        velocity_norm = (
            self._gen_actions[env_ids, 4] * (self._task_cfg.spawn_max_lin_vel - self._task_cfg.spawn_min_lin_vel)
            + self._task_cfg.spawn_min_lin_vel
        )

        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        phi = torch.rand((num_resets,), device=self._device) * math.pi

        initial_velocity[:, 0] = velocity_norm * torch.sin(phi) * torch.cos(theta)
        initial_velocity[:, 1] = velocity_norm * torch.sin(phi) * torch.sin(theta)
        initial_velocity[:, 2] = velocity_norm * torch.cos(phi)

        # Angular velocity
        initial_velocity[:, 3:] = (
            self._gen_actions[env_ids, 5].unsqueeze(-1)
            * (self._task_cfg.spawn_max_ang_vel - self._task_cfg.spawn_min_ang_vel)
            + self._task_cfg.spawn_min_ang_vel
        ) * torch.randn((num_resets, 3), device=self._device)

        self._robot.set_pose(initial_pose, self._env_ids[env_ids])
        self._robot.set_velocity(initial_velocity, self._env_ids[env_ids])

    def create_task_visualization(self) -> None:
        """
        Adds the visual marker to the scene.

        There are two markers: one for the goal and one for the robot.

        - The goal marker is a pose marker.
        - The robot is represented by another smaller pose marker, and upon task completion, the robot and the goal markers must align together.
        """

        # Define visual markers: sphere for the goal and pose marker for the robot
        goal_marker_cfg = POSE_MARKER_3D_CFG.copy()
        robot_marker_cfg = POSE_MARKER_3D_CFG.copy()
        robot_marker_cfg.markers["pose_marker_3d"].arrow_body_length = 0.2
        robot_marker_cfg.markers["pose_marker_3d"].arrow_body_radius = 0.01

        # Update prim paths to match task ID
        goal_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/goal_pose"
        robot_marker_cfg.prim_path = f"/Visuals/Command/task_{self._task_uid}/robot_pose"
        # Create the visualization markers
        self.goal_pos_visualizer = VisualizationMarkers(goal_marker_cfg)
        self.robot_pos_visualizer = VisualizationMarkers(robot_marker_cfg)

    def update_task_visualization(self) -> None:
        """Updates the visual marker to the scene."""

        self.goal_pos_visualizer.visualize(self._markers_pos, self._markers_rot)

        self._robot_marker_pos = self._robot.root_link_pos_w[self._env_ids, :3]
        self.robot_pos_visualizer.visualize(self._robot_marker_pos, self._robot.root_link_quat_w[self._env_ids])