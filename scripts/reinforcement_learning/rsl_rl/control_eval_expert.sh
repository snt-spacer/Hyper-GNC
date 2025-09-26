
#!/bin/bash

# Common evaluation parameters
SCRIPT_PATH="./scripts/reinforcement_learning/rsl_rl/eval_control.py"
NUM_ENVS=4096

# Common arguments that apply to all evaluations (without task, as it will be extracted from checkpoint)
COMMON_ARGS="--task=Isaac-RANS-MultiTask-v0 --headless --num_envs=${NUM_ENVS}"

# Array of checkpoint paths (relative to BASE_LOG_DIR)

# Experts
# CHECKPOINTS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_13-40-53_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_14-05-17_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_14-29-44_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_14-54-10_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_15-18-28_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_15-42-48_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-6/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_16-07-01_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-7/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_16-31-37_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-8/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_16-55-54_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-9/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_17-19-36_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-10/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_17-43-52_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_18-08-31_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_18-32-31_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_18-56-59_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_19-21-06_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_19-45-24_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-6/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_20-10-29_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-7/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_20-34-43_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-8/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_20-59-12_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-9/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_21-24-22_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-10/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_21-49-01_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1/model_2350.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_22-17-14_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2/model_2350.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_22-44-58_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3/model_2350.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_23-13-10_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4/model_2350.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_23-41-48_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5/model_2350.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_00-10-01_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-6/model_2350.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_00-38-23_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-7/model_2350.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_01-06-25_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-8/model_2350.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_01-34-14_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-9/model_2350.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_02-02-39_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-10/model_2350.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_02-31-28_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_02-57-35_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_03-24-24_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_03-50-36_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_04-17-17_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_04-44-17_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-6/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_05-11-15_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-7/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_05-38-04_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-8/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_06-04-38_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-9/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-26_06-31-49_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-10/model_3999.pt
# )

# NEW EVAL
# CHECKPOINTS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_17-01-14_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_17-27-40_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_17-54-17_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_18-20-41_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_18-47-16_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_19-13-45_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-6/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_19-40-14_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-7/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_20-07-03_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-8/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_20-33-34_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-9/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_21-00-10_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-10/model_3999.pt 
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_21-26-57_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_21-53-23_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_22-20-17_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_22-46-57_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_23-13-35_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_23-40-28_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-6/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_00-07-14_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-7/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_00-33-48_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-8/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_01-00-23_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-9/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_01-26-52_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-10/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_01-53-37_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_02-21-30_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_02-49-20_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_03-17-26_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_03-45-10_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_04-13-04_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-6/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_04-40-48_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-7/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_05-08-34_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-8/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_05-36-32_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-9/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_06-04-04_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-10/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_12-11-23_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_12-40-28_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_13-09-12_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_13-37-58_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_14-06-53_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_14-35-52_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-6/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_15-04-48_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-7/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_15-33-44_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-8/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_16-02-57_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-9/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-12_16-32-07_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-10/model_3999.pt
# )

# Expert L 
CHECKPOINTS=(
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_12-25-35_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1/model_3000.pt
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_12-54-36_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2/model_3000.pt
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_13-23-22_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3/model_3000.pt
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_13-52-31_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4/model_3000.pt
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_14-21-47_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5/model_3000.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_14-50-34_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1/model_3000.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_15-17-15_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2/model_3000.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_15-43-52_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3/model_3000.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_16-10-15_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4/model_3000.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_16-36-46_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5/model_3000.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_17-03-26_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1/model_3000.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_17-30-04_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2/model_3000.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_17-56-58_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3/model_3000.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_18-23-36_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4/model_3000.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_18-50-24_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5/model_3000.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_19-17-29_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1/model_3000.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_19-45-45_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2/model_3000.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_20-13-52_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3/model_3000.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_20-41-46_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4/model_3000.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-26_21-09-28_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5/model_3000.pt
)

########################################################################################################################
# VANILLA 2.0
########################################################################################################################
CHECKPOINTS=(
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_16-39-04_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1/model_3999.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_17-08-03_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2/model_3999.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_17-37-20_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3/model_3999.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_18-06-26_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4/model_3999.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_18-35-25_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5/model_3999.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_19-03-53_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1/model_3999.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_19-30-16_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2/model_3999.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_19-56-54_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3/model_3999.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_20-23-05_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4/model_3999.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_20-49-18_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5/model_3999.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_21-15-35_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1/model_3750.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_21-42-12_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2/model_3900.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_22-08-52_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3/model_3850.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_22-35-35_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4/model_3900.pt
    /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_23-01-53_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5/model_3999.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_23-27-59_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1/model_3999.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_23-55-49_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2/model_3999.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-09-01_00-23-46_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3/model_3999.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-09-01_00-51-20_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4/model_3999.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-09-01_01-18-58_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5/model_3999.pt
)


# Function to extract task from checkpoint path
extract_task_from_checkpoint() {
    local checkpoint_path="$1"
    local basename=$(basename "${checkpoint_path}")
    local dirname=$(dirname "${checkpoint_path}")
    local parent_dirname=$(basename "${dirname}")
    
    # Extract task from the parent directory name
    # More flexible pattern that looks for task names between underscores
    # Matches any of the known task names regardless of algorithm or robot
    if [[ "${parent_dirname}" =~ _(GoToPosition|GoToPose|TrackVelocities|Rendezvous)_ ]]; then
        local task_name="${BASH_REMATCH[1]}"
        echo "${task_name}"
    else
        echo "Orbital Not exist"
    fi
}

# Function to run evaluation for a single checkpoint
run_evaluation() {
    local checkpoint_path="$1"
    local task=$(extract_task_from_checkpoint "${checkpoint_path}")
    
    echo "Running evaluation for checkpoint: ${checkpoint_path}"
    echo "Extracted task: ${task}"
    ./isaaclab.sh -p ${SCRIPT_PATH} ${COMMON_ARGS} env.tasks_names="['${task}']" env.robot_name=ModularFreeflyer --checkpoint=${checkpoint_path}

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed successfully for: ${checkpoint_path}"
    else
        echo "✗ Evaluation failed for: ${checkpoint_path}"
    fi
    echo "----------------------------------------"
}

# Main execution: iterate through all checkpoints
echo "Starting control evaluation for ${#CHECKPOINTS[@]} checkpoints..."
echo "Number of environments: ${NUM_ENVS}"
echo "========================================"

for checkpoint in "${CHECKPOINTS[@]}"; do
    run_evaluation "${checkpoint}"
done

echo "All evaluations completed!"