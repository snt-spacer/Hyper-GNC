#!/bin/bash

# --- Configuration ---
OUTPUT_DIR_NAME="multitask_eval_control_hypernet_general_obs_task_id"
BASE_OUTPUT_DIR="/workspace/isaaclab/source/${OUTPUT_DIR_NAME}"
mkdir -p "$BASE_OUTPUT_DIR"
EVALUATION_LOG_FILE="${BASE_OUTPUT_DIR}/evaluation_run_$(date +"%Y-%m-%d_%H-%M-%S").log"

# Define your robot and tasks
robot="IntBall2"
BASE_TASKS=(GoToPose3D TrackVelocities3D GoThroughPoses3D GoToPosition3DWithObstacles)
num_envs=64 # Base number of environments, adjust if needed
algorithm="ppo-beta-memory" #ppo, ppo-memory, ppo-beta, ppo-beta-memory
runs_per_env=2

# Hypernetwork 32x32
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_15-43-21_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_1500.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-13-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_1100.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_16-44-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_1050.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-14-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_1050.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_17-45-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_1600.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-15-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_850.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_18-47-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_850.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-18-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_1500.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_19-48-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_1850.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-23_20-19-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_1900.pt
# )

# Baseline with info about thrusters
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-21-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_950.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_10-48-38_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_1600.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-15-42_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_1700.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_11-42-24_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_1550.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-17-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_1650.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_07-44-01_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_1550.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-11-02_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_1800.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_08-38-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_1950.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-05-05_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_1500.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-25_09-32-09_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_1000.pt
# )

# Hypernetwork 32x32 with general obs task id
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-28_18-11-11_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-42/model_2700.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_10-40-36_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_2700.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_11-41-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3500.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_12-42-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_500.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_13-44-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3700.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_14-45-35_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3000.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_15-46-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_3600.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_16-48-38_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_3650.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_17-49-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_250.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_18-50-32_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-29_19-52-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_1150.pt
# )

# Hypernet 64x64 critic obs has task id
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_09-04-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_10-05-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_11-07-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_12-09-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_13-10-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_14-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_15-12-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_16-15-55_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_3700.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_17-17-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_3600.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-07-31_18-19-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_3999.pt
# )

# Hypernet 64x64x64 critic obs has task id
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_14-58-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_2550.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_16-00-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3050.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_17-01-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_18-03-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3000.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_19-04-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3700.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_20-06-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_3850.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_21-07-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_1400.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_22-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_2600.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-01_23-10-59_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_3250.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_00-12-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_3000.pt
# )

# Hypernet w Gobs Tid 512x128x64x32 
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_11-14-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_300.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_12-17-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_13-20-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3000.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_14-22-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_550.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_15-24-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_300.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_16-28-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_500.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_17-32-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_700.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_18-35-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_3800.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_20-42-11_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_400.pt
# )

# Hypernet w Gobs Tid 32x32x32x32x32x32
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-02_23-44-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_3700.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_00-46-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_550.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_01-48-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3200.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_02-51-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3650.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_03-54-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3400.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_04-56-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_3700.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_07-01-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_08-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_09-05-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_1950.pt
# )

# Hypernet w Gobs Tid 64, 64, 64, 128, 64, 64, 64
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_13-50-19_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_14-52-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_750.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_14-52-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_1400.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_16-59-17_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_600.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_18-01-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_400.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_19-04-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_800.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_20-07-35_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_300.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-03_22-15-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_3999.pt
# )

# Hypernet Critic w Gobs Tid 64x64x64
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_10-48-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_450.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_13-07-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_14-16-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_1300.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_16-35-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_200.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-04_18-55-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_650.pt
# )

# Hypernet Embeddings w Gobs Tid 64x64x64
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_16-34-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_1100.pt #NG
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_17-36-32_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_2250.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_18-38-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_1750.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_19-41-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3650.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_20-42-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3850.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_21-44-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_3999.pt #NG
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_22-46-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_3400.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-05_23-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_300.pt #NG
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_00-49-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_300.pt #NG
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_01-51-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_1400.pt #NG
# )

# New rendezvous reward Hypernet w Gobs Tid 64x64x64
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-06_11-49-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_3999.pt
# )


# PROPER EVAL 
# Hypernet Init 64x64x64 w Gobs Tinf
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_18-34-24_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_1750.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_19-36-10_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_700.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_20-38-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_2350.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_21-39-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_1950.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-09_23-43-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_500.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_00-45-42_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_1100.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_01-47-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_1200.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_02-49-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_700.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_03-52-03_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_3999.pt
# )

# Hypernet Critic Init 64x64x64 Tinfo in Gobs 
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_01-33-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_1800.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_06-36-22_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_2550.pt
# )

# Hypernet Critic Emb 32x32 w Gobs Tinf
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_10-01-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_1950.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_11-09-48_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_2650.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_12-18-36_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3000.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_13-27-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3500.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_14-37-12_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3300.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_15-46-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_2300.pt #MNG 700
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_16-56-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_2050.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_18-05-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_2650.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_19-14-49_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_2850.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-11_20-23-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_1150.pt
# )

# Hypernet Critic Emb 32x32 w/o Gobs Tinf
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_09-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_2950.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_10-11-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3600.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_11-23-25_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3700.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_12-32-30_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_1100.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_13-40-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_1550.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_14-49-26_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_1250.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_15-57-56_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_900.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_17-06-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_1300.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_18-13-50_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_2150.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_19-21-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_900.pt 
# )

# Hypernet Critic Emb 64x64x64 w/o Gobs Tinf
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_21-50-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_200.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-13_23-00-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_200.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_00-09-18_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_200.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_01-18-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_1500.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_02-26-08_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_350.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_03-34-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_150.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_05-52-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_250.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_07-03-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_200.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_08-12-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_200.pt   
# )

# Hypernet Emb 32x32 w/o Gobs Tinf
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_09-08-45_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_1300.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_10-12-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_2000.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_11-16-09_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_1000.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_13-21-15_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3000.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_15-25-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_1400.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_16-28-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_1000.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-15_17-29-41_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_3000.pt
# )

# Reward with action rate at target Hypernet last layer
# Hypernet Critic Emb 64, 64, 64 w/o Gobs Tinf
MODEL_PATHS=(
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_08-26-39_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3999.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_09-25-57_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3999.pt #NG
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_10-25-14_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3999.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_11-24-07_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3999.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_12-22-53_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_3750.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_13-21-54_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_3300.pt #NG
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_14-20-58_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_2800.pt 
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_15-19-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_3200.pt
    # /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-21_16-17-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_1300.pt #NG
)

# Hypernet Critic L Emb 32x32 w/o Tinf in Gobs 
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_07-23-51_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_3000.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_08-22-47_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3000.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_09-20-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3500.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_10-18-16_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3000.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_11-17-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3500.pt
# )

# Hypernet Critic L Emb 64x64 w Tinf in Gobs (tanh, 256C)
# MODEL_PATHS=(
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_06-53-13_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_3999.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_07-51-23_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3999.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_08-50-04_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3999.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_09-48-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3000.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-25_10-47-20_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3000.pt
# )

# Hypernet Critic L Emb 64x64 w/o Tinf in Gobs (tanh, 256C) 
# MODEL_PATHS=(
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-28_09-32-40_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_3950.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-28_10-30-37_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3999.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-28_11-28-52_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3999.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-28_12-27-02_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3950.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-28_13-25-30_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3999.pt
# )

# Hypernet Critic L Emb 32x32 w/o Tinf in Gobs (tanh, 256C) 
# MODEL_PATHS=(
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-28_17-49-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_3400.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-28_18-48-33_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3200.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-28_19-46-44_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3999.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-28_20-44-32_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3999.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-28_21-43-01_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3999.pt
# )

########################################################################################################################
# VANILLA 2.0
########################################################################################################################
# Hypernet Critic L Emb 64x64 w Tinf in Gobs (tanh, 256C) 
# MODEL_PATHS=(
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_18-31-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_3999.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_19-30-27_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3999.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_20-28-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3999.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_21-27-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3400.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_22-25-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3900.pt
# )

########################################################################################################################
# IntBall2
########################################################################################################################
# Hypernet Beta
MODEL_PATHS=(
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-11-28_10-07-51_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-42/model_0.pt
)

# Check if model paths are provided
if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    echo "Error: No model paths provided. Please edit the script and add your .pth file paths to the MODEL_PATHS array."
    exit 1
fi

total_start_time=$(date +%s)
echo "Starting Evaluation Script..." | tee -a "$EVALUATION_LOG_FILE"
echo "Log file: $EVALUATION_LOG_FILE" | tee -a "$EVALUATION_LOG_FILE"
echo "Found ${#MODEL_PATHS[@]} models to evaluate." | tee -a "$EVALUATION_LOG_FILE"

# Determine the number of tasks for calculating EVAL_NUM_ENVS
NUM_TASKS=${#BASE_TASKS[@]}
EVAL_NUM_ENVS=$((num_envs * NUM_TASKS))

# Convert BASE_TASKS array to a comma-separated string
IFS=, EVAL_TASKS_NAMES="${BASE_TASKS[*]}"

echo "Evaluating all tasks simultaneously: [${EVAL_TASKS_NAMES}]" | tee -a "$EVALUATION_LOG_FILE"
echo "Total environments: ${EVAL_NUM_ENVS} (${num_envs} per task × ${NUM_TASKS} tasks)" | tee -a "$EVALUATION_LOG_FILE"

# --- Main Loop: Evaluation for each model path ---
for model_path in "${MODEL_PATHS[@]}"
do
    run_start_time=$(date +%s)

    # Check if the model file exists
    if [ ! -f "$model_path" ]; then
        echo "--------------------------------------------------------" | tee -a "$EVALUATION_LOG_FILE"
        echo "Error: Model file not found: $model_path" | tee -a "$EVALUATION_LOG_FILE"
        echo "Skipping evaluation for this model." | tee -a "$EVALUATION_LOG_FILE"
        echo "--------------------------------------------------------" | tee -a "$EVALUATION_LOG_FILE"
        continue # Skip to the next model
    fi

    echo "--------------------------------------------------------" | tee -a "$EVALUATION_LOG_FILE"
    echo "Starting evaluation for Model: $model_path" | tee -a "$EVALUATION_LOG_FILE"
    echo "Evaluating all tasks simultaneously: [${EVAL_TASKS_NAMES}]" | tee -a "$EVALUATION_LOG_FILE"

    # Remove all csv files from previous runs from the metrics folder
    # experiment_dir_name=$(basename "$(dirname "${model_path}")")
    experiment_path=$(dirname "${model_path}")
    echo "Removing previous CSV files from: ${experiment_path}/metrics"
    rm -rf "${experiment_path}/metrics"

    # Execute the evaluation script for all tasks at once
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_control.py \
        --task=Isaac-RANS-MultiTask-v0 \
        --headless \
        --num_envs="${EVAL_NUM_ENVS}" \
        --checkpoint="${model_path}" \
        --algorithm="${algorithm}" \
        --runs_per_env="${runs_per_env}" \
        env.robot_name="${robot}" \
        env.tasks_names="[${EVAL_TASKS_NAMES}]" #>> "$EVALUATION_LOG_FILE" 2>&1

    # Check if the evaluation command executed successfully
    if [ $? -eq 0 ]; then
        echo "Evaluation completed successfully for all tasks." | tee -a "$EVALUATION_LOG_FILE"
    else
        echo "Evaluation failed. Check logs in $EVALUATION_LOG_FILE for details." | tee -a "$EVALUATION_LOG_FILE"
    fi

    run_end_time=$(date +%s)
    run_duration=$((run_end_time - run_start_time))
    echo "Evaluation for model $model_path finished in $(($run_duration / 60)) minutes and $(($run_duration % 60)) seconds." | tee -a "$EVALUATION_LOG_FILE"
    echo "--------------------------------------------------------" | tee -a "$EVALUATION_LOG_FILE"
    echo "" | tee -a "$EVALUATION_LOG_FILE"

done

total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))
total_hours=$(($total_duration / 3600))
total_minutes=$((($total_duration % 3600) / 60))
total_seconds=$(($total_duration % 60))

echo "All evaluation runs completed." | tee -a "$EVALUATION_LOG_FILE"
echo "Total execution time: $total_hours hours, $total_minutes minutes, and $total_seconds seconds." | tee -a "$EVALUATION_LOG_FILE"