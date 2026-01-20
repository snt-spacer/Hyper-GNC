#!/bin/bash

# Configuration
ALGORITHM="ppo-beta-memory"
ROBOT="IntBall2"
OUTPUT_LOGS_PATH="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/train_logs/"
OUTPUT_MODELS_PATH="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/train_eval_models_logs/"
TASKS_NAMES="GoToPose3D,TrackVelocities3D,GoThroughPoses3D,GoToPosition3DWithObstacles"

# Evaluation parameters
EVAL_NUM_ENVS=64
RUNS_PER_ENV=2
MODEL_NUM="3999"
ZERO_SHOT_TASKS=(GoToPose3DBox GoToPosition3D)

# List of model directories to evaluate
MODEL_PATHS=(
    "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_S_mtcr_pcgrad_new_obstacles_TaskID/2026-01-19_18-49-41_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5"
)

mkdir -p "$OUTPUT_LOGS_PATH"
mkdir -p "$OUTPUT_MODELS_PATH"

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
        env.tasks_names="[${TASKS_NAMES}]" #>> "$OUTPUT_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed successfully."
    else
        echo "✗ Evaluation failed. Check logs in $OUTPUT_FILE"
    fi

    # Zero-shot evaluation
    for TASK in "${ZERO_SHOT_TASKS[@]}"
    do
        echo "Running Zero-shot: $TASK"
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_control.py \
            --task=Isaac-RANS-Single-v0 \
            --headless \
            --num_envs=${EVAL_NUM_ENVS} \
            --runs_per_env=${RUNS_PER_ENV} \
            --algorithm=${ALGORITHM} \
            env.robot_name=${ROBOT} \
            env.task_name=${TASK} \
            --checkpoint=${CHECKPOINT_PATH} #>> "$OUTPUT_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "  ✓ $TASK zero-shot done."
        else
            echo "  ✗ $TASK zero-shot failed."
        fi
    done

    # Log the evaluated path
    echo "$FULL_PATH" >> "${OUTPUT_MODELS_PATH}evaluated_${EXP_NAME}.txt"

done