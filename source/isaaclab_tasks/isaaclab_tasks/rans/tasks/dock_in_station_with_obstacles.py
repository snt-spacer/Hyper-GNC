# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.markers import POSE_MARKER_3D_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveScene
from isaaclab.utils import math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.utils.assets import REPO_ROOT_PATH
from isaaclab.sim.spawners import UsdFileCfg

from isaaclab_tasks.rans import DockInStationWithObstaclesCfg
from isaaclab_tasks.rans.utils import ObjectStorage

from .task_core import TaskCore
import torch.nn.functional as F

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class DockInStationWithObstaclesTask(TaskCore):
    """
    Implements the GoToPose task in 3D space. The robot has to reach a target position and keep it.
    """

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        task_cfg: DockInStationWithObstaclesCfg = DockInStationWithObstaclesCfg(),
        task_uid: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        env_ids: torch.Tensor | None = None,
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

        super().__init__(scene=scene, task_uid=task_uid, num_envs=num_envs, device=device, env_ids=env_ids)

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
        self.initialize_buffers()

        # Obstacles
        self.obstacles_generator = ObjectStorage(
            num_envs=num_envs,
            max_num_vis_objects_in_env=self._task_cfg.max_num_vis_obstacles,
            store_height=self._task_cfg.obstacles_storage_height_pos,
            rng=self._rng,
            device=device,
        )
        self._num_cells = int(1.0 / (self._task_cfg.minimum_point_distance * 2))

        self.design_scene()

    def design_scene(self) -> None:
        """
        Sets up the ISS-JEM environment.
        Initializes the obstacles for the task.
        """

        ###########
        # ISS JEM #
        ###########
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
            collision_group=-1,  # Shared global collision group
            debug_vis=False,
        )
        self.iss_jem = RigidObject(iss_jem_cfg)

        #############
        # Obstacles #
        #############
        prim_utils.create_prim("/World/envs/env_0/Obstacles", "Xform")

        rigid_objects = {}
        low, high = 5, 15

        for i in range(self._task_cfg.max_num_vis_obstacles):

            position = low + (high - low) * torch.rand(3)
            position[2] = 0.5

            rigid_objects[f"obstacle_{i}"] = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Obstacles/cylinder_{i}",
                spawn=sim_utils.CylinderCfg(
                    radius=self._task_cfg.obstacle_radius,
                    height=self._task_cfg.obstacles_height,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=position),
            )

        obstacles_cfg = RigidObjectCollectionCfg(rigid_objects=rigid_objects)

        self.obstacles = RigidObjectCollection(obstacles_cfg)

    def register_robot(self, robot) -> None:
        self._robot = robot

    def register_sensors(self) -> None:
        filters = [f"/World/envs/env_.*/Obstacles/cylinder_{i}" for i in range(self._task_cfg.max_num_vis_obstacles)]
        self._robot.activateSensors("contacts", filters)
        self._robot.register_sensors()

    def get_docking_pose_world(self) -> torch.Tensor:
        """
        Computes the world pose of the docking station per environment using the JEM pose.

        Returns:
            torch.Tensor: (num_envs, 7) world-space pose of docking station [x, y, z, qw, qx, qy, qz]
        """
        # (num_envs, 3) and (num_envs, 4)
        jem_pos = self.iss_jem.data.root_state_w[:, :3]
        jem_quat = self.iss_jem.data.root_state_w[:, 3:7]

        # (1, 3), (1, 4) → broadcast to (num_envs, *)
        dock_rel_pos = self._docking_pose_local[:3].unsqueeze(0).expand_as(jem_pos)
        dock_rel_quat = self._docking_pose_local[3:].unsqueeze(0).expand_as(jem_quat)

        # Compose world-space transform: jem * docking_local
        docking_pos_w, docking_quat_w = math_utils.combine_frame_transforms(
            jem_pos, jem_quat,
            dock_rel_pos, dock_rel_quat,
        )

        return torch.cat((docking_pos_w, docking_quat_w), dim=-1)  # (num_envs, 7)


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
        self._orientation_error = torch.zeros(
            (self._num_envs,), device=self._device, dtype=torch.float32
        )  # Orientation error metric
        self._markers_pos = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self._markers_rot = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)
        

    def run_setup(self, robot, envs_origin):
        super().run_setup(robot, envs_origin)
        self.obstacles_generator.create_storage_buffer(env_origin=self._env_origins)

    def create_logs(self) -> None:
        """
        Creates a dictionary to store the training statistics for the task.
        Tracks 3D velocity, angular velocity, and position distance.
        """

        super().create_logs()

        self.scalar_logger.add_log("task_state", "AVG/normed_linear_velocity", "mean")
        self.scalar_logger.add_log("task_state", "AVG/absolute_angular_velocity", "mean")
        self.scalar_logger.add_log("task_state", "EMA/position_distance", "ema")
        self.scalar_logger.add_log("task_state", "EMA/orientation_error", "ema")
        self.scalar_logger.add_log("task_state", "EMA/boundary_distance", "ema")

        self.scalar_logger.add_log("task_reward", "AVG/position", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/orientation", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/linear_velocity", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/angular_velocity", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/boundary", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/progress", "mean")
        self.scalar_logger.add_log("task_reward", "AVG/cuboid_penalty", "mean")

        self.scalar_logger.set_ema_coeff(self._task_cfg.ema_coeff)

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
        self._local_pos_error = math_utils.quat_rotate_inverse(current_quat_w, self._position_error)

        # log the global distance for debugging
        self._position_dist = self._position_error.norm(dim=-1)  # shape [N]
        self.scalar_logger.log("task_state", "EMA/position_distance", self._position_dist)

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

        # Obstacles positions
        # Filter obstacles by height
        obstacles_positions = self.obstacles.data.object_link_pos_w[self._env_ids]
        filtered_obstacles = obstacles_positions.clone()

        mask = obstacles_positions[:, :, 2] < self._task_cfg.obstacles_storage_height_pos # Big negative value for obstacles in storage 
        filtered_obstacles[mask] = 2 * self._task_cfg.max_obstacle_distance_from_target

        # Calculate distances and angles for the filtered obstacles
        obstacles_error_w = filtered_obstacles - self._robot.root_link_pos_w[self._env_ids].unsqueeze(1)
        obstacles_dist = torch.norm(obstacles_error_w, dim=-1)

        # Get the 3 closest obstacles
        closest_distances, closest_indices = torch.topk(obstacles_dist, k=3, dim=1, largest=False)
        closest_obstacles_pos_w = torch.gather(
            filtered_obstacles, 1, closest_indices.unsqueeze(-1).expand(-1, -1, filtered_obstacles.size(-1))
        )
        # Get the world frame error for the closest obstacles
        closest_obstacles_error_w = torch.gather(
            obstacles_error_w, 1, closest_indices.unsqueeze(-1).expand(-1, -1, obstacles_error_w.size(-1))
        )
        # Get the current robot orientation (already computed for target)
        current_quat_w = self._robot.root_link_quat_w[self._env_ids]

        # Rotate the closest obstacle error vectors into the robot's local frame 
        # The quat_rotate_inverse function expects a [N, 4] quaternion and a [N, 3] vector.
        # We need to broadcast the quaternion for each of the M=3 obstacles.
        current_quat_w_expanded = current_quat_w.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 4) # [N*3, 4]
        closest_obstacles_error_w_flat = closest_obstacles_error_w.reshape(-1, 3) # [N*3, 3]
        
        # Perform the rotation
        closest_obstacles_error_local_flat = math_utils.quat_rotate_inverse(
            current_quat_w_expanded, closest_obstacles_error_w_flat
        ) # [N*3, 3]
        
        # Reshape back to [N, 3, 3]
        closest_obstacles_error_local = closest_obstacles_error_local_flat.reshape(-1, 3, 3)

        # Normalize the local obstacle position errors by max distance
        normalized_closest_obstacles_error_local = closest_obstacles_error_local / self._task_cfg.max_obstacle_distance_from_target
        # normalized_closest_obstacles_hight TODO
        # normalized_closes_obstacles_radius
        
        # Flatten the 3 closest obstacle 3D position errors for observation [N, 15]
        obstacles_observation = normalized_closest_obstacles_error_local.flatten(start_dim=1)

        # Store in buffer [position_dist, rotation error, linear_vel_xyz, angular_vel_xyz]
        self._task_data[:, 0:3] = self._local_pos_error
        self._task_data[:, 3:9] = rel_mat_6
        self._task_data[:, 9:12] = self._robot.root_com_lin_vel_b[self._env_ids]
        self._task_data[:, 12:15] = self._robot.root_com_ang_vel_b[self._env_ids]

        # Append obstacles info
        self._task_data[:, 15:24] = obstacles_observation

        # Make sure that the orientation error magnitude is also updated
        current_quat = self._robot.root_quat_w[self._env_ids]  # (w, x, y, z)
        target_quat = self._target_orientations  # (w, x, y, z)
        self._orientation_error = math_utils.quat_error_magnitude(target_quat, current_quat)

        task_id_one_hot = F.one_hot(torch.tensor([self._task_uid], device=self._device), num_classes=self._num_tasks).squeeze(0).repeat(self._num_envs, 1)
        semantic_emb = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0]], device=self._device).repeat(self._num_envs, 1)
        noise = self._rng.sample_uniform_torch(low=-0.1, high=0.1, shape=semantic_emb.shape[-1], ids=self._env_ids)
        semantic_emb += noise
        

        # Concatenate task observations with robot's internal observations
        return torch.concat((self._task_data, self._robot.get_observations(env_ids=self._env_ids)), dim=-1), task_id_one_hot, semantic_emb

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

        if self._task_cfg.enable_collision_avoidance:
            # Collision penalty with environment
            # collisions = torch.squeeze(
            #     torch.max(torch.norm(self._robot.contacts.data.force_matrix_w, dim=-1), dim=-1)[0], dim=-1
            # )
            # num_collisions = (collisions > self._task_cfg.collision_threshold).int()
            # collision_penalty_rew = self._task_cfg.collision_penalty * num_collisions

            # Check for collision with obstacles
            collisions = torch.squeeze(
                torch.max(torch.norm(self._robot.contacts.data.force_matrix_w[self._env_ids], dim=-1), dim=-1)[0], dim=-1
            )  # first max is for the 3 forces (x,y,z), second max is for obstacles

            num_collisions = 1 * (collisions > self._task_cfg.collision_threshold)
            collision_penalty_rew = self._task_cfg.collision_penalty * num_collisions


        
        # Boundaries of cuboid, relative to docking port (in world frame)
        # Expand those from what you used in set_initial_conditions
        x_min, x_max = 0.3, 6.0 # into / out of plane
        y_min, y_max = -1.55, -0.85 # up / down
        z_min, z_max = -0.5, 0.75 # left / right

        # Position relative to docking port
        rel_pos = self._robot.root_link_pos_w[self._env_ids] - self._docking_port_pos.unsqueeze(0)

        # # Check violations
        # out_of_bounds_x = (rel_pos[:, 0] < x_min) | (rel_pos[:, 0] > x_max)
        # out_of_bounds_y = (rel_pos[:, 1] < y_min) | (rel_pos[:, 1] > y_max)
        # out_of_bounds_z = (rel_pos[:, 2] < z_min) | (rel_pos[:, 2] > z_max)

        # any_violation = (out_of_bounds_x | out_of_bounds_y | out_of_bounds_z).float()

        # # Apply penalty
        # cuboid_penalty = self._task_cfg.cuboid_violation_penalty * any_violation

        # Define lower and upper limits
        x_bounds = torch.tensor([x_min, x_max], device=self._device)
        y_bounds = torch.tensor([y_min, y_max], device=self._device)
        z_bounds = torch.tensor([z_min, z_max], device=self._device)

        # Compute per-axis violation (distance beyond bounds)
        x_violation = torch.maximum(x_bounds[0] - rel_pos[:, 0], rel_pos[:, 0] - x_bounds[1])
        x_violation = torch.clamp(x_violation, min=0.0)

        y_violation = torch.maximum(y_bounds[0] - rel_pos[:, 1], rel_pos[:, 1] - y_bounds[1])
        y_violation = torch.clamp(y_violation, min=0.0)

        z_violation = torch.maximum(z_bounds[0] - rel_pos[:, 2], rel_pos[:, 2] - z_bounds[1])
        z_violation = torch.clamp(z_violation, min=0.0)

        # Total violation magnitude
        violation_magnitude = torch.sqrt(x_violation**2 + y_violation**2 + z_violation**2)

        # Scaled penalty (optional: square it to punish more harshly as it drifts further)
        cuboid_penalty = self._task_cfg.cuboid_violation_penalty * violation_magnitude
        
        # Add an overshoot penalty if the robot goes beyond the docking port pose (i.e. negative progress in a way where progress was defined as progress = self._previous_position_dist - self._position_dist)

        # Measure negative progress
        negative_progress = (self._position_dist - self._previous_position_dist).clamp(min=0)
        negative_progress_penalty = self._task_cfg.negative_progress_penalty * negative_progress

        # Logging
        self.scalar_logger.log("task_state", "EMA/position_distance", self._position_dist)
        self.scalar_logger.log("task_state", "EMA/orientation_error", torch.norm(self._orientation_error))
        self.scalar_logger.log("task_state", "EMA/boundary_distance", boundary_dist)
        self.scalar_logger.log("task_state", "AVG/normed_linear_velocity", linear_velocity)
        self.scalar_logger.log("task_state", "AVG/absolute_angular_velocity", angular_velocity)

        # Logging rewards
        self.scalar_logger.log("task_reward", "AVG/position", position_rew)
        self.scalar_logger.log("task_reward", "AVG/orientation", orientation_rew)
        self.scalar_logger.log("task_reward", "AVG/linear_velocity", linear_velocity_rew)
        self.scalar_logger.log("task_reward", "AVG/angular_velocity", angular_velocity_rew)
        self.scalar_logger.log("task_reward", "AVG/boundary", boundary_rew)
        self.scalar_logger.log("task_reward", "AVG/progress", progress_rew)
        self.scalar_logger.log("task_reward", "AVG/cuboid_penalty", cuboid_penalty)
        # self.scalar_logger.log("task_reward", "AVG/negative_progress", negative_progress_penalty)
        # if self._task_cfg.enable_collision_avoidance:
        #     self.scalar_logger.log("task_reward", "SUM/num_collisions", num_collisions)

        # Compute final reward
        total_reward = (
            (position_rew * orientation_rew) * self._task_cfg.pose_weight
            + linear_velocity_rew * self._task_cfg.linear_velocity_weight
            + angular_velocity_rew * self._task_cfg.angular_velocity_weight
            + boundary_rew * self._task_cfg.boundary_weight
            + progress_rew * self._task_cfg.progress_weight
            + (collision_penalty_rew if self._task_cfg.enable_collision_avoidance else 0)
            + cuboid_penalty

        ) + self._robot.compute_rewards(env_ids=self._env_ids)

        # Add negative progress penalty to the total reward
        # total_reward -= negative_progress_penalty

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
        task_completed = torch.where(
            self._goal_reached > self._task_cfg.reset_after_n_steps_in_tolerance,
            ones,
            task_completed,
        )
        return task_failed, task_completed

    def set_goals(self, env_ids: torch.Tensor) -> None:
        """
        Sets the same docking port goal pose for all specified environments.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
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
        spawn_pos = self._docking_port_pos.unsqueeze(0) + offset_world

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

        # Add obstacles to the scene
        obstacles_positions, mask = self.randomize_obstacles_positions(env_ids)
        self._pos_obstacles_in_env = self.obstacles_generator.get_positions_on_storage(obstacles_positions, mask, env_ids)
        self.obstacles.write_object_link_pose_to_sim(self._pos_obstacles_in_env, env_ids=env_ids)

    def randomize_obstacles_positions(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Randomizes the positions of the obstacles in the environment. It creates an extra obstacle position for the robot.
        Args:
            env_ids (torch.Tensor): The ids of the environments.
        Returns:
                tuple[torch.Tensor, torch.Tensor]: The positions of the obstacles and the mask indicating valid obstacles.
        """

        pos_1 = self._target_positions[env_ids].unsqueeze(1) + torch.tensor([0.0, 0.5, 0.0], device=self._device)
        pos_2 = self._target_positions[env_ids].unsqueeze(1) + torch.tensor([0.0, 0.5, 1.0], device=self._device)
        pos_3 = self._target_positions[env_ids].unsqueeze(1) + torch.tensor([0.0, 0.5, -1.0], device=self._device)
        pos_4 = self._target_positions[env_ids].unsqueeze(1) + torch.tensor([0.0, 0.5, -2.0], device=self._device)

        xyz = torch.cat((pos_1, pos_2, pos_3, pos_4), dim=1)
        # Randomize obstacles positions
        # number_obstacles_to_generate = self._task_cfg.max_num_vis_obstacles * 4
        # indices_of_obstacles_to_activate = self._rng.sample_unique_integers_torch(
        #     min=0, max=self._task_cfg.max_num_vis_obstacles**2, num=number_obstacles_to_generate, ids=env_ids
        # )

        # x = (indices_of_obstacles_to_activate % self._num_cells) - self._num_cells / 2
        # y = (indices_of_obstacles_to_activate // self._num_cells) - self._num_cells / 2
        # # Scale to world coordinates
        # cell_size = self._task_cfg.maximum_robot_distance / self._num_cells  # Calculate the size of a grid cell
        # x = self._rng.sample_sign_torch("int", (number_obstacles_to_generate), ids=env_ids) * x * cell_size / 2
        # y = self._rng.sample_sign_torch("int", (number_obstacles_to_generate), ids=env_ids) * y * cell_size / 2
        # z = self._rng.sample_uniform_torch(- self._task_cfg.max_hight_from_target, self._task_cfg.max_hight_from_target, x.shape[-1], ids=env_ids)#torch.ones_like(x) * self._task_cfg.obstacles_height / 2
        # xyz = torch.stack((x, y, z), dim=2)
        # xyz += self._target_positions[env_ids].unsqueeze(1)#self._env_origins[env_ids].unsqueeze(1)[..., :2]

        """Move the obstacle if it is too close to target position or the robot position"""
        # Calculate the distances between the obstacles and the target and robot positions
        # distance_obstacle_to_target = torch.norm(xyz - self._target_positions[env_ids].unsqueeze(1), dim=-1)
        # distance_obstacle_to_robot = torch.norm(
        #     xyz - self._robot.root_link_pos_w[env_ids].unsqueeze(1), dim=-1
        # )
        # # Create a mask to filter out obstacles that are too close to the target or robot
        # # True where obstacles are INVALID (too close)
        # obstacles_mask = (distance_obstacle_to_target < self._task_cfg.min_obstacle_distance_from_target) | (
        #     distance_obstacle_to_robot < self._task_cfg.min_obstacle_distance_from_robot
        # )

        # # Count the number of valid obstacles in each environment.
        # # ~obstacles_mask is True for VALID obstacles.
        # num_valid_per_env = (~obstacles_mask).sum(dim=1)

        # # Handle the edge case where an environment might have zero valid obstacles.
        # # We clamp to 1 to avoid errors in random sampling (e.g., division by zero).
        # # If an env has 0 valid obstacles, it will just sample from itself, which is fine as it's invalid anyway.
        # num_valid_per_env_safe = num_valid_per_env.clamp(min=1)

        # # 1. Sort obstacles to group valid ones first.
        # # We sort by the inverted mask (valid=True=1, invalid=False=0), descending.
        # # This puts all valid obstacles at the beginning of each row.
        # # `sorted_indices` stores the original indices of the obstacles after sorting.
        # sorted_indices = torch.argsort((~obstacles_mask).int(), dim=1, descending=True)
        # # `unsort_indices` is needed later to revert the sorting.
        # unsort_indices = torch.argsort(sorted_indices, dim=1)

        # # Gather the xyz positions into the new sorted order.
        # # Shape: (num_envs, num_obstacles, 3)
        # sorted_xyz = torch.gather(xyz, 1, sorted_indices.unsqueeze(-1).expand_as(xyz))

        # # 2. Create a pool of random valid replacements, in sorted order.
        # # For each environment, generate random indices within its valid range [0, num_valid - 1].
        # # Shape: (num_envs, num_obstacles)
        # rand_indices_into_valid_section = torch.floor(
        #     torch.rand(xyz.shape[:2], device=self._device) * num_valid_per_env_safe.unsqueeze(1)
        # ).long()

        # # Sample from the valid section of sorted_xyz to get replacement positions.
        # # `replacement_pool_sorted` now contains randomly selected valid positions for every slot.
        # replacement_pool_sorted = torch.gather(
        #     sorted_xyz, 1, rand_indices_into_valid_section.unsqueeze(-1).expand_as(xyz)
        # )

        # # 3. Unsort the replacements to match the original obstacle order.
        # # This maps the shuffled replacements back to their original positions.
        # replacement_xyz = torch.gather(
        #     replacement_pool_sorted, 1, unsort_indices.unsqueeze(-1).expand_as(xyz)
        # )

        # # 4. Apply the replacements using the original mask.
        # # Where the obstacle was invalid (mask is True), use the new replacement position.
        # # Otherwise, keep the original xyz position.
        # xyz = torch.where(obstacles_mask.unsqueeze(-1), replacement_xyz, xyz)

        # Generate quats and concatenate with xyz
        xyzw = self.obstacles.data.object_com_quat_w[env_ids].clone()
        obstacles_positions = torch.cat((xyz[:, : self._task_cfg.max_num_vis_obstacles], xyzw), dim=-1)

        # Create visible obstacles
        num_visible_obstacles_per_env = self._rng.sample_integer_torch(
            low=1, high=self._task_cfg.max_num_vis_obstacles, shape=(1,), ids=env_ids
        )
        num_visible_obstacles_per_env[0] = 4
        mask = torch.arange(self._task_cfg.max_num_vis_obstacles, device=self._device).unsqueeze(
            0
        ) < num_visible_obstacles_per_env.unsqueeze(1)


        return obstacles_positions, mask
    

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