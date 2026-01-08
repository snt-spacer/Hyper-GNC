# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseModelCfg

from .common import SpaceType, ViewerCfg
from .ui import BaseEnvWindow

import numpy as np

def get_lookat_point_flexible(camera_position, camera_orientation, distance, rotation_order='zyx', forward_axis='-z'):
    """
    Calculates the 3D point the camera is looking at, with flexible conventions.

    Args:
        camera_position (tuple): The camera's (x, y, z) position.
        camera_orientation (tuple): The camera's (pitch, yaw, roll) orientation in degrees.
        distance (float): The distance from the camera to the look-at point.
        rotation_order (str): The order of rotation axes (e.g., 'xyz', 'zyx').
        forward_axis (str): The camera's initial forward direction ('+x', '-x', '+y', '-y', '+z', '-z').

    Returns:
        numpy.ndarray: The (x, y, z) coordinates of the look-at point.
    """
    pitch, yaw, roll = np.radians(camera_orientation)

    # 1. Define the initial forward vector based on convention
    if forward_axis == '-z':
        forward_vector = np.array([0, 0, -distance])
    elif forward_axis == '+z':
        forward_vector = np.array([0, 0, distance])
    elif forward_axis == '-y':
        forward_vector = np.array([0, -distance, 0])
    # Add other cases as needed

    # 2. Create the rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    R_y = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    R_z = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])

    # 3. Combine rotations based on the specified order
    if rotation_order == 'zyx':
        combined_rotation_matrix = R_z @ R_y @ R_x
    elif rotation_order == 'xyz':
        combined_rotation_matrix = R_x @ R_y @ R_z
    elif rotation_order == 'yxz':
        combined_rotation_matrix = R_y @ R_x @ R_z
    else:
        raise ValueError("Invalid rotation order.")

    # 4. Rotate the vector and calculate the look-at point
    rotated_forward_vector = combined_rotation_matrix @ forward_vector
    lookat_point = np.array(camera_position) + rotated_forward_vector
    
    return tuple(lookat_point.tolist())
  
'''
Go To Pose
  camera_position=(0.8, 1.45, 1.71),
  camera_orientation=(83.9, -0.0, 159.46),
Go To Pose Box
  camera_pos = (-0.39, 3.56, 1.58)
    camera_ori = (79.39, -0.0, -172.97)
'''
@configclass
class DirectRLEnvCfg:
    """Configuration for an RL environment defined with the direct workflow.

    Please refer to the :class:`isaaclab.envs.direct_rl_env.DirectRLEnv` class for more details.
    """
    camera_pos = (-0.56, -2.45, 1.59)
    camera_ori = (83.52, -0.0, -23.8)
    # simulation settings
    viewer: ViewerCfg = ViewerCfg(
      eye=camera_pos,
      lookat=get_lookat_point_flexible(
          camera_position=camera_pos,
          camera_orientation=camera_ori,
          distance=np.linalg.norm(np.array(camera_pos)),
          rotation_order='zyx',
          forward_axis='-z'
      ),
      # origin_type="asset_root",
      # asset_name="ModularFreeflyer"
    )
    """Viewer configuration. Default is ViewerCfg()."""

    sim: SimulationCfg = SimulationCfg()
    """Physics simulation configuration. Default is SimulationCfg()."""

    # ui settings
    ui_window_class_type: type | None = BaseEnvWindow
    """The class type of the UI window. Default is None.

    If None, then no UI window is created.

    Note:
        If you want to make your own UI window, you can create a class that inherits from
        from :class:`isaaclab.envs.ui.base_env_window.BaseEnvWindow`. Then, you can set
        this attribute to your class type.
    """

    # general settings
    seed: int | None = None
    """The seed for the random number generator. Defaults to None, in which case the seed is not set.

    Note:
      The seed is set at the beginning of the environment initialization. This ensures that the environment
      creation is deterministic and behaves similarly across different runs.
    """

    decimation: int = MISSING
    """Number of control action updates @ sim dt per policy dt.

    For instance, if the simulation dt is 0.01s and the policy dt is 0.1s, then the decimation is 10.
    This means that the control action is updated every 10 simulation steps.
    """

    is_finite_horizon: bool = False
    """Whether the learning task is treated as a finite or infinite horizon problem for the agent.
    Defaults to False, which means the task is treated as an infinite horizon problem.

    This flag handles the subtleties of finite and infinite horizon tasks:

    * **Finite horizon**: no penalty or bootstrapping value is required by the the agent for
      running out of time. However, the environment still needs to terminate the episode after the
      time limit is reached.
    * **Infinite horizon**: the agent needs to bootstrap the value of the state at the end of the episode.
      This is done by sending a time-limit (or truncated) done signal to the agent, which triggers this
      bootstrapping calculation.

    If True, then the environment is treated as a finite horizon problem and no time-out (or truncated) done signal
    is sent to the agent. If False, then the environment is treated as an infinite horizon problem and a time-out
    (or truncated) done signal is sent to the agent.

    Note:
        The base :class:`ManagerBasedRLEnv` class does not use this flag directly. It is used by the environment
        wrappers to determine what type of done signal to send to the corresponding learning agent.
    """

    episode_length_s: float = MISSING
    """Duration of an episode (in seconds).

    Based on the decimation rate and physics time step, the episode length is calculated as:

    .. code-block:: python

        episode_length_steps = ceil(episode_length_s / (decimation_rate * physics_time_step))

    For example, if the decimation rate is 10, the physics time step is 0.01, and the episode length is 10 seconds,
    then the episode length in steps is 100.
    """

    # environment settings
    scene: InteractiveSceneCfg = MISSING
    """Scene settings.

    Please refer to the :class:`isaaclab.scene.InteractiveSceneCfg` class for more details.
    """

    events: object | None = None
    """Event settings. Defaults to None, in which case no events are applied through the event manager.

    Please refer to the :class:`isaaclab.managers.EventManager` class for more details.
    """

    observation_space: SpaceType = MISSING
    """Observation space definition.

    The space can be defined either using Gymnasium :py:mod:`~gymnasium.spaces` (when a more detailed
    specification of the space is desired) or basic Python data types (for simplicity).

    .. list-table::
        :header-rows: 1

        * - Gymnasium space
          - Python data type
        * - :class:`~gymnasium.spaces.Box`
          - Integer or list of integers (e.g.: ``7``, ``[64, 64, 3]``)
        * - :class:`~gymnasium.spaces.Discrete`
          - Single-element set (e.g.: ``{2}``)
        * - :class:`~gymnasium.spaces.MultiDiscrete`
          - List of single-element sets (e.g.: ``[{2}, {5}]``)
        * - :class:`~gymnasium.spaces.Dict`
          - Dictionary (e.g.: ``{"joints": 7, "rgb": [64, 64, 3], "gripper": {2}}``)
        * - :class:`~gymnasium.spaces.Tuple`
          - Tuple (e.g.: ``(7, [64, 64, 3], {2})``)
    """

    num_observations: int | None = None
    """The dimension of the observation space from each environment instance.

    .. warning::

        This attribute is deprecated. Use :attr:`~isaaclab.envs.DirectRLEnvCfg.observation_space` instead.
    """

    state_space: SpaceType | None = None
    """State space definition.

    This is useful for asymmetric actor-critic and defines the observation space for the critic.

    The space can be defined either using Gymnasium :py:mod:`~gymnasium.spaces` (when a more detailed
    specification of the space is desired) or basic Python data types (for simplicity).

    .. list-table::
        :header-rows: 1

        * - Gymnasium space
          - Python data type
        * - :class:`~gymnasium.spaces.Box`
          - Integer or list of integers (e.g.: ``7``, ``[64, 64, 3]``)
        * - :class:`~gymnasium.spaces.Discrete`
          - Single-element set (e.g.: ``{2}``)
        * - :class:`~gymnasium.spaces.MultiDiscrete`
          - List of single-element sets (e.g.: ``[{2}, {5}]``)
        * - :class:`~gymnasium.spaces.Dict`
          - Dictionary (e.g.: ``{"joints": 7, "rgb": [64, 64, 3], "gripper": {2}}``)
        * - :class:`~gymnasium.spaces.Tuple`
          - Tuple (e.g.: ``(7, [64, 64, 3], {2})``)
    """

    num_states: int | None = None
    """The dimension of the state-space from each environment instance.

    .. warning::

        This attribute is deprecated. Use :attr:`~isaaclab.envs.DirectRLEnvCfg.state_space` instead.
    """

    observation_noise_model: NoiseModelCfg | None = None
    """The noise model to apply to the computed observations from the environment. Default is None, which means no noise is added.

    Please refer to the :class:`isaaclab.utils.noise.NoiseModel` class for more details.
    """

    action_space: SpaceType = MISSING
    """Action space definition.

    The space can be defined either using Gymnasium :py:mod:`~gymnasium.spaces` (when a more detailed
    specification of the space is desired) or basic Python data types (for simplicity).

    .. list-table::
        :header-rows: 1

        * - Gymnasium space
          - Python data type
        * - :class:`~gymnasium.spaces.Box`
          - Integer or list of integers (e.g.: ``7``, ``[64, 64, 3]``)
        * - :class:`~gymnasium.spaces.Discrete`
          - Single-element set (e.g.: ``{2}``)
        * - :class:`~gymnasium.spaces.MultiDiscrete`
          - List of single-element sets (e.g.: ``[{2}, {5}]``)
        * - :class:`~gymnasium.spaces.Dict`
          - Dictionary (e.g.: ``{"joints": 7, "rgb": [64, 64, 3], "gripper": {2}}``)
        * - :class:`~gymnasium.spaces.Tuple`
          - Tuple (e.g.: ``(7, [64, 64, 3], {2})``)
    """

    num_actions: int | None = None
    """The dimension of the action space for each environment.

    .. warning::

        This attribute is deprecated. Use :attr:`~isaaclab.envs.DirectRLEnvCfg.action_space` instead.
    """

    action_noise_model: NoiseModelCfg | None = None
    """The noise model applied to the actions provided to the environment. Default is None, which means no noise is added.

    Please refer to the :class:`isaaclab.utils.noise.NoiseModel` class for more details.
    """

    rerender_on_reset: bool = False
    """Whether a render step is performed again after at least one environment has been reset.
    Defaults to False, which means no render step will be performed after reset.

    * When this is False, data collected from sensors after performing reset will be stale and will not reflect the
      latest states in simulation caused by the reset.
    * When this is True, an extra render step will be performed to update the sensor data
      to reflect the latest states from the reset. This comes at a cost of performance as an additional render
      step will be performed after each time an environment is reset.

    """

    wait_for_textures: bool = True
    """True to wait for assets to be loaded completely, False otherwise. Defaults to True."""
