#!/bin/bash

# Configuration
ALGORITHM="ppo-memory"
ROBOT="FloatingPlatform"
OUTPUT_LOGS_PATH="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/train_logs/"
OUTPUT_MODELS_PATH="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/train_eval_models_logs/"
TASKS_NAMES="GoToPose,TrackVelocities,Rendezvous,GoToPositionWithObstacles"

# Evaluation parameters
EVAL_NUM_ENVS=128
RUNS_PER_ENV=1
MODEL_NUM="3999"
# ZERO_SHOT_TASKS=(GoToPose3DBox GoToPosition3D)

# List of model directories to evaluate
MODEL_PATHS=(
"/workspace/isaaclab/logs/rsl_rl/MASS_ONLY_D0.05/2026-01-21_20-12-20_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"
"/workspace/isaaclab/logs/rsl_rl/MASS_ONLY_D0.1/2026-01-21_21-34-00_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"
"/workspace/isaaclab/logs/rsl_rl/MASS_ONLY_D0.15/2026-01-21_22-55-28_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"
"/workspace/isaaclab/logs/rsl_rl/MASS_ONLY_D0.2/2026-01-22_00-17-19_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"
"/workspace/isaaclab/logs/rsl_rl/COM_ONLY_D0.05/2026-01-22_01-38-41_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"
"/workspace/isaaclab/logs/rsl_rl/COM_ONLY_D0.1/2026-01-22_03-02-50_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"
"/workspace/isaaclab/logs/rsl_rl/COM_ONLY_D0.15/2026-01-22_04-26-43_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"
"/workspace/isaaclab/logs/rsl_rl/COM_ONLY_D0.2/2026-01-22_05-50-28_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"
"/workspace/isaaclab/logs/rsl_rl/WRENCH_ONLY_W0.1/2026-01-22_07-16-25_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"
"/workspace/isaaclab/logs/rsl_rl/WRENCH_ONLY_W0.2/2026-01-22_09-18-44_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"
"/workspace/isaaclab/logs/rsl_rl/WRENCH_ONLY_W0.3/2026-01-22_11-22-34_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"
"/workspace/isaaclab/logs/rsl_rl/WRENCH_ONLY_W0.4/2026-01-22_13-25-33_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"
"/workspace/isaaclab/logs/rsl_rl/ALL_ACTIVE_D0.05_W0.1/2026-01-22_15-30-44_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"
"/workspace/isaaclab/logs/rsl_rl/ALL_ACTIVE_D0.1_W0.2/2026-01-22_17-37-03_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"
"/workspace/isaaclab/logs/rsl_rl/ALL_ACTIVE_D0.15_W0.3/2026-01-22_19-45-49_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"
"/workspace/isaaclab/logs/rsl_rl/ALL_ACTIVE_D0.2_W0.4/2026-01-22_21-54-52_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"
"/workspace/isaaclab/logs/rsl_rl/BASELINE_NO_RANDOMIZATION/2026-01-23_09-03-43_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"

)

reset_all() {
    echo "Resetting all randomization flags to False..."
    sed -i "/mass_rand_cfg:/ s/enable=True/enable=False/" "$ROBOT_CFG_PATH"
    sed -i "/com_rand_cfg:/ s/enable=True/enable=False/" "$ROBOT_CFG_PATH"
    sed -i "/wrench_rand_cfg/,/)/ s/enable=True/enable=False/" "$ROBOT_CFG_PATH"
}

reset_all

# mkdir -p "$OUTPUT_LOGS_PATH"
# mkdir -p "$OUTPUT_MODELS_PATH"

for FULL_PATH in "${MODEL_PATHS[@]}"
do
    # Extract details from the path
    # EXP_NAME is the folder before the timestamp folder
    EXP_NAME=$(basename $(dirname "$FULL_PATH"))
    # Extract seed from the end of the folder name (e.g., "seed-1" -> "1")
    SEED=$(echo "$FULL_PATH" | grep -oP 'seed-\K\d+')
    CHECKPOINT_PATH="${FULL_PATH}/model_${MODEL_NUM}.pt"
    
    # OUTPUT_FILE="${OUTPUT_LOGS_PATH}eval_only_${EXP_NAME}_seed-${SEED}.txt"
    # rm -f "$OUTPUT_FILE"

    echo "----------------------------------------"
    echo "Evaluating Path: $FULL_PATH"
    echo "Seed: $SEED | Exp: $EXP_NAME"

    # Multi-task Evaluation
    echo "Running Multi-task evaluation..."
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_control.py \
        --task=Isaac-RANS-MultiTask-v0 \
        --headless \
        --num_envs="${EVAL_NUM_ENVS}" \
        --checkpoint="${CHECKPOINT_PATH}" \
        --algorithm="${ALGORITHM}" \
        --runs_per_env="${RUNS_PER_ENV}" \
        env.robot_name="${ROBOT}" \
        env.tasks_names="[${TASKS_NAMES}]" #>> "$OUTPUT_FILE" 2>&1         # 

    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed successfully."
    else
        echo "✗ Evaluation failed. Check logs in $OUTPUT_FILE"
    fi

done