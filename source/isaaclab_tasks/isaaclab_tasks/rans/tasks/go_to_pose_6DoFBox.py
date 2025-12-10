# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.markers import POSE_MARKER_3D_CFG, VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene
from isaaclab.utils import math as math_utils

from isaaclab_tasks.rans import GoToPose3DBoxCfg
from isaaclab_tasks.rans.utils import ObjectStorage

from .task_core import TaskCore
import torch.nn.functional as F

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class GoToPose3DBoxTask(TaskCore):
    """
    Implements the GoToPose task in 3D space. The robot has to reach a target position and keep it.
    """

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        task_cfg: GoToPose3DBoxCfg = GoToPose3DBoxCfg(),
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

        self.wall_box_names = ["floor", "ceiling", "back", "front", "left", "right", ]

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
        self.obstacles_generator.create_storage_buffer(env_origin=self._env_origins)

    def register_sensors(self) -> None:
        # Obstacles
        filters = [f"/World/envs/env_.*/Obstacles/cylinder_{i}" for i in range(self._task_cfg.max_num_vis_obstacles)]
        # Walls
        filters += [f"/World/envs/env_.*/ISSBox/wall_{name}" for name in self.wall_box_names]
        self._robot.activateSensors("contacts", filters)
        self._robot.register_sensors()

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
        self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/collision_penalty", "mean")

        self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/total_reward", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/weighted_position_orientation_reward", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/weighted_linear_velocity_reward", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/weighted_angular_velocity_reward", "mean")
        # self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/weighted_boundary_reward", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/weighted_progress_reward", "mean")
        self.scalar_logger.add_log("task_reward", "GoToPose6DoF/AVG/weighted_collision_penalty", "mean")


        self.scalar_logger.set_ema_coeff(self._task_cfg.ema_coeff)

    def design_scene(self) -> None:
        ############
        # ISS Box #
        ############
        prim_utils.create_prim("/World/envs/env_0/ISSBox", "Xform")

        light_cfg = sim_utils.CylinderLightCfg(
            intensity=2000.0, 
            color=(0.75, 0.75, 0.75), 
            # radius=1.0,
            length=6.0,
            treat_as_line=True,
        )
        light_cfg.func(
            "/World/envs/env_0/ISSBox/Env_Light", 
            light_cfg, 
            translation=(0.0, 0.0, 2.0),
            orientation=(0.707, 0.0, 0.0, 0.707)
        )

        opacity = 1
        self.wall_thickness = 0.05
        self.wall_height = 2.0
        self.wall_width = 2.0
        self.wall_length = 6.4
        self.x_shift = -0.05
        self.height_from_floor = 0.3
        wall_sizes = [
            [self.wall_width, self.wall_length, self.wall_thickness], 
            [self.wall_width, self.wall_length, self.wall_thickness], 
            [self.wall_width, self.wall_thickness, self.wall_height], 
            [self.wall_width, self.wall_thickness, self.wall_height], 
            [self.wall_thickness, self.wall_length, self.wall_height], 
            [self.wall_thickness, self.wall_length, self.wall_height]
        ]
        wall_positions = [
            [0 + self.x_shift , 0, self.height_from_floor + self._task_cfg.iss_box_storage_height_pos], 
            [0 + self.x_shift, 0, 2.0 + self.height_from_floor + self._task_cfg.iss_box_storage_height_pos], 
            [0 + self.x_shift, -self.wall_length/2, 1.0 + self.height_from_floor + self._task_cfg.iss_box_storage_height_pos], 
            [0 + self.x_shift, self.wall_length/2, 1.0 + self.height_from_floor + self._task_cfg.iss_box_storage_height_pos], 
            [-self.wall_width/2 + self.x_shift, 0, 1.0 + self.height_from_floor + self._task_cfg.iss_box_storage_height_pos], 
            [self.wall_width/2 + self.x_shift, 0, 1.0 + self.height_from_floor + self._task_cfg.iss_box_storage_height_pos]
        ]
        walls_rigid_objects = {}

        for name in self.wall_box_names:
            walls_rigid_objects[name] = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/ISSBox/wall_{name}",
                spawn=sim_utils.CuboidCfg(
                    size=wall_sizes[self.wall_box_names.index(name)],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(torch.rand(1).item(), 0.1, 1), 
                        emissive_color=(0.1, 0.1, 1),
                        opacity=opacity
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=wall_positions[self.wall_box_names.index(name)],
                )
            )


        iss_box_cfg = RigidObjectCollectionCfg(rigid_objects=walls_rigid_objects)
        self.iss_box = RigidObjectCollection(iss_box_cfg)

        # visual material
        visual_material_cfg = sim_utils.GlassMdlCfg(
            glass_color=(0.1, 0.1, 1.0), 
            glass_ior=1.0,
            frosting_roughness=0.3,
        )
        visual_material_cfg.func("/World/Looks/glassMaterial", visual_material_cfg)
        for name in self.wall_box_names:
            if name == "ceiling" or name == "front":
                sim_utils.bind_visual_material(f"/World/envs/env_0/ISSBox/wall_{name}", "/World/Looks/glassMaterial")

        
        #############
        # Obstacles #
        #############

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
        # ###########
        # # ISS JEM #
        # ###########
        # from isaaclab.utils.assets import REPO_ROOT_PATH
        # from isaaclab.sim.spawners import UsdFileCfg
        # from isaaclab.assets import RigidObject
        # iss_jem_cfg = RigidObjectCfg(
        #     prim_path="/World/JEM",
        #     init_state=RigidObjectCfg.InitialStateCfg(
        #         pos=(0.0, 0.0, 0.0),
        #         rot=(1.0, 0.0, 0.0, 0.0),  # wxyz format
        #         lin_vel=(0.0, 0.0, 0.0),
        #         ang_vel=(0.0, 0.0, 0.0),
        #     ),
        #     spawn=UsdFileCfg(
        #         usd_path=f"{REPO_ROOT_PATH}/assets/environments/KibouIsaac/Asset_KIBOU3.usd",
        #         scale=(0.01, 0.01, 0.01),
        #     ),
        #     #collision_group=-1,  # Shared global collision group
        #     debug_vis=False,
        # )
        # self.iss_jem = RigidObject(iss_jem_cfg)

        

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

        # Collision check
        collisions = torch.squeeze(
            torch.max(torch.norm(self._robot.contacts.data.force_matrix_w[self._env_ids], dim=-1), dim=-1)[0], dim=-1
        )
        self.collided_signal = torch.where(collisions > 0, 1.0, 0.0).unsqueeze(-1)

        # Obstacles positions
        # Filter obstacles by height
        obstacles_positions = self.obstacles.data.object_link_pos_w[self._env_ids]
        filtered_obstacles = obstacles_positions.clone()

        mask = obstacles_positions[:, :, 2] < self._task_cfg.obstacles_storage_height_pos # Big negative value for obstacles in storage 
        filtered_obstacles[mask] = 100.0

        # Calculate distances and angles for the filtered obstacles
        obstacles_error_w = filtered_obstacles - self._robot.root_link_pos_w[self._env_ids].unsqueeze(1)
        obstacles_dist = torch.norm(obstacles_error_w, dim=-1)

        # Get the 3 closest obstacles
        closest_distances, closest_indices = torch.topk(obstacles_dist, k=3, dim=1, largest=False)
        
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

        # Normalize distances by wall dimensions
        closest_obstacles_error_local[:, :, 0] = closest_obstacles_error_local[:, :, 0] / self.wall_width
        closest_obstacles_error_local[:, :, 1] = closest_obstacles_error_local[:, :, 1] / self.wall_length
        closest_obstacles_error_local[:, :, 2] = closest_obstacles_error_local[:, :, 2] / self.wall_height

        obstacles_observation = closest_obstacles_error_local.flatten(start_dim=1)


        # Store in buffer [position_dist, rotation error, linear_vel_xyz, angular_vel_xyz]
        self._task_data[:, 0:3] = self._local_pos_error
        self._task_data[:, 3:9] = rel_mat_6
        self._task_data[:, 9:12] = self._robot.root_com_lin_vel_b[self._env_ids]
        self._task_data[:, 12:15] = self._robot.root_com_ang_vel_b[self._env_ids]
        # self._task_data[:, 15:23] = self._robot.get_observations(env_ids=self._env_ids)
        # self._task_data[:, 23:24] = self.collided_signal
        # self._task_data[:, 24:34] = obstacles_observation
        
        
        self._task_data[:, 15:16] = self.collided_signal
        self._task_data[:, 16:25] = obstacles_observation
        self._task_data[:, 25:33] = self._robot.get_observations(env_ids=self._env_ids)
       

        # Make sure that the orientation error magnitude is also updated
        current_quat = self._robot.root_quat_w[self._env_ids]  # (w, x, y, z)
        target_quat = self._target_orientations  # (w, x, y, z)
        self._orientation_error = math_utils.quat_error_magnitude(target_quat, current_quat)

        task_id_one_hot = F.one_hot(torch.tensor([self._task_uid], device=self._device), num_classes=self._num_tasks).squeeze(0).repeat(self._num_envs, 1)
        semantic_emb = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.5]], device=self._device).repeat(self._num_envs, 1)
        # noise = self._rng.sample_uniform_torch(low=-0.1, high=0.1, shape=semantic_emb.shape[-1], ids=self._env_ids)
        # semantic_emb += noise

        # Concatenate task observations with robot's internal observations
        return self._task_data, task_id_one_hot, semantic_emb

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

        # Collision penalty
        # Collision check
        collisions = torch.squeeze(
            torch.max(torch.norm(self._robot.contacts.data.force_matrix_w[self._env_ids], dim=-1), dim=-1)[0], dim=-1
        )
        self.collided_signal = torch.where(collisions > 0, 1.0, 0.0).unsqueeze(-1)
        collision_penalty_rew = (self.collided_signal * self._task_cfg.collision_penalty).squeeze()

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
        self.scalar_logger.log("task_reward", "GoToPose6DoF/AVG/collision_penalty", collision_penalty_rew)

        # Compute final reward
        total_reward = (
            (position_rew * orientation_rew) * self._task_cfg.pose_weight
            + linear_velocity_rew * self._task_cfg.linear_velocity_weight
            + angular_velocity_rew * self._task_cfg.angular_velocity_weight
            # + boundary_rew * self._task_cfg.boundary_weight
            + progress_rew * self._task_cfg.progress_weight
            + collision_penalty_rew * self._task_cfg.collision_penalty_weight
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
        self.scalar_logger.log(
            "task_reward", "GoToPose6DoF/AVG/weighted_collision_penalty", collision_penalty_rew * self._task_cfg.collision_penalty_weight
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
        # Sample random goal positions in a box space

        x_range = (-0.7, 0.56)
        y_range = (-2.7, 2.7)
        z_range = (0.6, 1.7)

        self._target_positions[env_ids, 0] = self._rng.sample_uniform_torch(*x_range, 1, ids=env_ids).squeeze(-1) + self._env_origins[env_ids, 0]
        self._target_positions[env_ids, 1] = self._rng.sample_uniform_torch(*y_range, 1, ids=env_ids).squeeze(-1) + self._env_origins[env_ids, 1]
        self._target_positions[env_ids, 2] = self._rng.sample_uniform_torch(*z_range, 1, ids=env_ids).squeeze(-1)


        # padding = 0.25
        # target_xyz = torch.concat((
        #     self._rng.sample_uniform_torch(-self.wall_width / 2 + padding, self.wall_width / 2 - padding, 1, ids=env_ids),
            
        #     self._rng.sample_uniform_torch(-self.wall_length / 2 + padding, self.wall_length / 2 - padding, 1, ids=env_ids),
            
        #     self._rng.sample_uniform_torch(self.height_from_floor + padding, self.wall_height + self.height_from_floor - padding, 1, ids=env_ids)

        # ), dim=-1).unsqueeze(0).reshape(-1, 3) + self._env_origins[env_ids]

        # self._target_positions[env_ids] = (target_xyz)
        
        yaw_offset = self._rng.sample_uniform_torch(-math.pi, math.pi, 1, ids=env_ids)
        pitch_offset = self._rng.sample_uniform_torch(
            -math.pi / 2, math.pi / 2, 1, ids=env_ids
        )  # Limit pitch to avoid upside-down flips
        roll_offset = self._rng.sample_uniform_torch(-math.pi, math.pi, 1, ids=env_ids)

        # Step 2: Convert these into a quaternion using Euler angles
        self._target_orientations[env_ids] = math_utils.quat_from_euler_xyz(roll_offset, pitch_offset, yaw_offset)

        # Update the visual markers for goal pos
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
        
        # Randomize obstacles positions
        obstacles_positions, mask = self.randomize_obstacles_positions(env_ids)
        # set the hight of the obstacles in the mask that are false to storage height
        for i in range(obstacles_positions.shape[0]):
            obstacles_positions[i, ~mask[i], 2] -= self._task_cfg.obstacles_storage_height_pos
        self.obstacles.write_object_link_pose_to_sim(obstacles_positions, env_ids=self._env_ids[env_ids])

        # Remove ISS box walls from current task environments from the storage locaiton
        # Only done once at reset
        if self.first_reset:
            env_ids = torch.arange(0, self._num_envs, device=self._device)
            task_iss_box_position = self.iss_box.data.object_state_w[self._env_ids[env_ids], :, :7].clone()
            task_iss_box_position[:, :, 2] -= self._task_cfg.iss_box_storage_height_pos
            self.iss_box.write_object_link_pose_to_sim(task_iss_box_position, env_ids=self._env_ids[env_ids])
            self.first_reset = False

        # Add obstacles to the scene
        # obstacles_positions, mask = self.randomize_obstacles_positions(env_ids)
        # self._pos_obstacles_in_env = self.obstacles_generator.get_positions_with_storage(obstacles_positions, mask, env_ids)
        # self._pos_obstacles_in_env[:, :, 3:] = self.obstacles.data.object_com_quat_w[env_ids]
        # self.obstacles.write_object_link_pose_to_sim(self._pos_obstacles_in_env, env_ids=env_ids)
    
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
        z_range = (0.6, 1.7)

        x = self._rng.sample_uniform_torch(*x_range, (number_obstacles_to_generate,), ids=env_ids)
        y = self._rng.sample_uniform_torch(*y_range, (number_obstacles_to_generate,), ids=env_ids)
        z = self._rng.sample_uniform_torch(*z_range, (number_obstacles_to_generate,), ids=env_ids)

        xyz = torch.stack((x, y, z), dim=2)
        xyz[..., :2] += self._env_origins[env_ids].unsqueeze(1)[..., :2]

        # **Step 5: Compute distances from obstacles to robot and target**
        distance_obstacle_to_target = torch.norm(xyz - self._target_positions[env_ids].unsqueeze(1), dim=-1)
        distance_obstacle_to_robot = torch.norm(xyz - self._robot.root_link_pos_w[env_ids].unsqueeze(1), dim=-1)

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
        num_visible_obstacles_per_env = self._gen_actions[env_ids, 7] * (self._task_cfg.max_num_vis_obstacles - self._task_cfg.min_num_obstacles) + self._task_cfg.min_num_obstacles

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

        self.goal_pos_visualizer.visualize(self._markers_pos, self._markers_rot)

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