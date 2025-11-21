# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticMemoryCfg, RslRlPpoAlgorithmCfg


@configclass
class SingleRobotMultiTaskPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 4000
    save_interval = 100
    experiment_name = "mtrl_control_intball_6DOF"
    logger = "wandb"
    wandb_kwargs = {
        "project": "mtrl_control_intball_6DOF",
        "entity": "spacer-rl",
        "group": "zeroG",
    }
    empirical_normalization = False
    policy = RslRlPpoActorCriticMemoryCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[256, 256],
        activation="tanh",
        clip_actions=True,
        clip_actions_range=[0, 1],
        use_embeddings=True,
        embeddings_size=4,
        generator_size=(64, 64),
        num_memory_obs=4,
        network_type="hybrid", #pure, hybrid
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-3,
        schedule="adaptive", #adaptive, fixed
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
