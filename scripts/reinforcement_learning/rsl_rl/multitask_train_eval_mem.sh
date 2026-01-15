#!/bin/bash

# --- Configuration ---
OUTPUT_DIR_NAME="multitask_memory_normW"
BASE_OUTPUT_DIR="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/${OUTPUT_DIR_NAME}"
mkdir -p "$BASE_OUTPUT_DIR"
MODEL_TRACKER_FILE="${BASE_OUTPUT_DIR}/trained_models_tracker.log"

# Define your robots and tasks
robots=(FloatingPlatform)
BASE_TASKS=(GoToPosition GoToPose TrackVelocities GoThroughPoses)
num_envs=4096 # Base number of environments
algorithm="ppo-memory" #ppo, ppo-memory, ppo-beta
runs_per_env=1

total_start_time=$(date +%s)

# Convert the tasks array into a comma-separated string for Python arguments
IFS=, EVAL_TASKS_NAMES="${BASE_TASKS[*]}"
NUM_TASKS=${#BASE_TASKS[@]}

# --- Main Loop: Training and Evaluation for each robot and seed ---
for robot in "${robots[@]}"
do
    for seed in {1..10}
    do
        run_start_time=$(date +%s)

        # Generate a unique timestamp for the current training run's log file
        TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
        CLEAN_TASKS_NAMES=$(echo "$EVAL_TASKS_NAMES" | tr ',' '-')
        TRAINING_LOG_FILE="${BASE_OUTPUT_DIR}/training_${robot}_${CLEAN_TASKS_NAMES}_seed-${seed}_${TIMESTAMP}.log"

        echo "--------------------------------------------------------"
        echo "Starting run for Robot: $robot, Seed: $seed, Tasks: $EVAL_TASKS_NAMES"
        echo "Training logs will be written to: $TRAINING_LOG_FILE"
        echo "Training with num_envs: ${num_envs}"

        # --- Training Phase Execution ---
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
            --task=Isaac-RANS-MultiTask-v0 \
            --seed="${seed}" \
            --num_envs="${num_envs}" \
            --headless \
            --algorithm="${algorithm}" \
            env.robot_name="${robot}" \
            env.tasks_names="[${EVAL_TASKS_NAMES}]" > "$TRAINING_LOG_FILE" 2>&1

        # Check if the training command executed successfully
        if [ $? -eq 0 ]; then
            echo "Training completed successfully for Robot: $robot, Seed: $seed, Tasks: $CLEAN_TASKS_NAMES."
            run_end_time=$(date +%s)
            run_duration=$((run_end_time - run_start_time))
            echo "Run duration: $(($run_duration / 60)) minutes and $(($run_duration % 60)) seconds."
        else
            echo "Training failed for Robot: $robot, Seed: $seed, Tasks: $CLEAN_TASKS_NAMES. Check logs in $TRAINING_LOG_FILE for details."
            echo "--------------------------------------------------------"
            continue
        fi

        # --- Post-Training: Extract Paths and Prepare for Evaluation ---
        LOG_BASE_DIR=$(grep -oP 'Logging experiment in directory: \K[^ ]+' "$TRAINING_LOG_FILE" | tail -1)
        MODEL_RELATIVE_PATH=$(grep -oP 'wandb: Syncing run \K[^ ]+' "$TRAINING_LOG_FILE" | tail -1)
        TOTAL_STEPS_STR=$(grep -oP 'Learning iteration [0-9]+/\K[0-9]+' "$TRAINING_LOG_FILE" | tail -1)

        # --- Error Handling for Path Extractions ---
        if [ -z "$LOG_BASE_DIR" ] || [ -z "$MODEL_RELATIVE_PATH" ] || [ -z "$TOTAL_STEPS_STR" ]; then
            echo "Error: One or more critical pieces of information (LOG_BASE_DIR, MODEL_RELATIVE_PATH, TOTAL_STEPS_STR) could not be extracted from $TRAINING_LOG_FILE."
            echo "    LOG_BASE_DIR: '$LOG_BASE_DIR'"
            echo "    MODEL_RELATIVE_PATH: '$MODEL_RELATIVE_PATH'"
            echo "    TOTAL_STEPS_STR: '$TOTAL_STEPS_STR'"
            echo "Skipping evaluation for Robot: $robot, Seed: $seed, Tasks: $CLEAN_TASKS_NAMES. and continuing to the next run."
            echo "--------------------------------------------------------"
            continue
        fi

        MODEL_NUMBER=$((TOTAL_STEPS_STR - 1))
        EXPERIMENT_DIR="${LOG_BASE_DIR}/${MODEL_RELATIVE_PATH}"
        FINAL_MODEL_CHECKPOINT_PATH="${EXPERIMENT_DIR}/model_${MODEL_NUMBER}.pt"

        echo "Derived Model Checkpoint Path: $FINAL_MODEL_CHECKPOINT_PATH"

        # Save the extracted model path to the overall tracker file
        echo "${FINAL_MODEL_CHECKPOINT_PATH}" >> "$MODEL_TRACKER_FILE"
        echo "Model path added to: $MODEL_TRACKER_FILE"

        # --- Evaluation Phase Execution ---
        EVAL_NUM_ENVS=$((num_envs * NUM_TASKS)) # Multiply num_envs by NUM_TASKS for multi-task evaluation
        echo "Evaluation with num_envs: ${EVAL_NUM_ENVS}"
        echo "Evaluating all tasks simultaneously: [${EVAL_TASKS_NAMES}]"

        # Execute evaluation for all tasks at once
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval.py \
            --task=Isaac-RANS-MultiTask-v0 \
            --headless \
            --num_envs="${EVAL_NUM_ENVS}" \
            --checkpoint="${FINAL_MODEL_CHECKPOINT_PATH}" \
            --algorithm="${algorithm}" \
            --runs_per_env="${runs_per_env}" \
            env.robot_name="${robot}" \
            env.tasks_names="[${EVAL_TASKS_NAMES}]" >> "$TRAINING_LOG_FILE" 2>&1 # '>>' to append to the log

        if [ $? -eq 0 ]; then
            echo "Evaluation completed successfully for all tasks."
        else
            echo "Evaluation failed for Robot: $robot, Seed: $seed, Tasks: $CLEAN_TASKS_NAMES. Check logs in $TRAINING_LOG_FILE for details."
            echo "--------------------------------------------------------"
            continue
        fi
        echo "Evaluation finished for all tasks: [${EVAL_TASKS_NAMES}]."
        echo "--------------------------------------------------------"
        echo ""
    done
done

total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))
total_hours=$(($total_duration / 3600))
total_minutes=$((($total_duration % 3600) / 60))
total_seconds=$(($total_duration % 60))

echo "All training and evaluation runs completed."
echo "Total execution time: $total_hours hours, $total_minutes minutes, and $total_seconds seconds."