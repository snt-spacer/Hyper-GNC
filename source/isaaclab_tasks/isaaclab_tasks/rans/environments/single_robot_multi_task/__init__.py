# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-RANS-MultiTask-v0",
    entry_point=f"{__name__}.auto_env_gen_multitask:MultiTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.auto_env_gen_multitask:MultiTaskEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_ppo-discrete_cfg_entry_point": f"{agents.__name__}:rl_games_ppo-discrete_cfg.yaml",
        "skrl_ppo-discrete_cfg_entry_point": f"{agents.__name__}:skrl_ppo-discrete_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SingleRobotMultiTaskPPORunnerCfg",
        "rsl_rl_ppo-beta_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-beta_cfg:SingleRobotMultiTaskPPORunnerCfg",
        "rsl_rl_ppo-beta-memory_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-beta-memory_cfg:SingleRobotMultiTaskPPORunnerCfg",
        "rsl_rl_ppo-multi-critic_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-multi-critic_cfg:SingleRobotMultiTaskPPORunnerCfg",
        "rsl_rl_ppo-memory_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-memory_cfg:SingleRobotMultiTaskPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-RANS-MultiTaskTest-v0",
    entry_point=f"{__name__}.multitask_test:SingleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multitask_test:SingleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_ppo-discrete_cfg_entry_point": f"{agents.__name__}:rl_games_ppo-discrete_cfg.yaml",
        "skrl_ppo-discrete_cfg_entry_point": f"{agents.__name__}:skrl_ppo-discrete_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SingleRobotMultiTaskPPORunnerCfg",
        "rsl_rl_ppo_beta_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_beta_cfg:SingleRobotMultiTaskPPORunnerCfg",
    },
)
