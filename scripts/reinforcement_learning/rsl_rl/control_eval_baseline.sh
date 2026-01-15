#!/bin/bash

# --- Configuration ---
OUTPUT_DIR_NAME="multitask_eval_control_hypernet_general_obs_task_id"
BASE_OUTPUT_DIR="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/${OUTPUT_DIR_NAME}"
mkdir -p "$BASE_OUTPUT_DIR"
EVALUATION_LOG_FILE="${BASE_OUTPUT_DIR}/evaluation_run_$(date +"%Y-%m-%d_%H-%M-%S").log"

# Define your robot and tasks
robot="ModularFreeflyer"
BASE_TASKS=(GoToPosition GoToPose TrackVelocities Rendezvous)
num_envs=1024 # Base number of environments, adjust if needed
algorithm="ppo"
runs_per_env=1

# PROPER EVAL 
# Baseline w Gobs Tinf
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_11-30-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_12-24-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_13-18-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_14-11-59_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_15-06-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-01-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_16-55-10_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_17-50-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_18-44-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-10_19-39-06_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_3999.pt
# )

#Baseline w/o gobs tinf
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_13-10-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_14-04-11_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-00-56_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_15-55-17_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_16-49-14_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_17-42-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-6/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_18-36-07_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-7/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_19-31-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-8/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_20-25-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-9/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-14_21-18-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-10/model_3999.pt
# )

# Baseline L 32x32 w Tinf in Gobs
# MODEL_PATHS=(
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_13-56-13_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_14-50-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_15-46-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_16-40-36_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3999.pt
#     /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-22_17-35-32_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3999.pt
# )

# Baseline L 32x32 w Tinf in Gobs (tanh C 256) 
# MODEL_PATHS=(
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-00-16_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_3000.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_09-55-55_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3999.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_10-51-46_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3999.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_11-48-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3999.pt
# /workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-27_12-43-08_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3999.pt
# )

########################################################################################################################
# VANILLA 2.0
########################################################################################################################
# Baseline L 32x32 w Tinf in Gobs (tanh C 256) 
MODEL_PATHS=(
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_09-27-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/model_3999.pt
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_10-20-58_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/model_3999.pt
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_11-15-40_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/model_3999.pt
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_12-10-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/model_3999.pt
/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_13-05-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/model_3999.pt
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
    echo "Removing previous CSV files from: ${experiment_path}/metrics/"
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
        env.tasks_names="[${EVAL_TASKS_NAMES}]" >> "$EVALUATION_LOG_FILE" 2>&1

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