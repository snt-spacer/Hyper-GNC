# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticBetaMemoryCfg, RslRlPpoAlgorithmCfg


@configclass
class SingleRobotMultiTaskPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 4000
    save_interval = 1000
    experiment_name = "multitask_memory_control_beta_pcgrad"
    logger = "wandb"
    wandb_kwargs = {
        "project": "multitask_memory_control_beta_pcgrad",
        "entity": "spacer-rl",
        "group": "zeroG",
    }
    empirical_normalization = False
    policy = RslRlPpoActorCriticBetaMemoryCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[256, 256],
        activation="tanh",
        clip_actions=False,
        clip_actions_range=[-1, 1],
        use_embeddings=False,
        embeddings_size=5,
        generator_size=(64, 64),
        num_memory_obs=5, # task specific dimension
        network_type="hybrid", #pure, hybrid
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-3,
        schedule="adaptive", #adaptive, fixed
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
