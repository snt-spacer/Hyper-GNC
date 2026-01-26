#!/bin/bash

# --- 1. CONFIGURATION PATHS ---
ALGORITHM="ppo-memory"
ROBOT="FloatingPlatform"
ROBOT_CFG_PATH="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/robots_cfg/floating_platform_cfg.py"
AGENT_CFG_PATH="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/environments/single_robot_multi_task/agents/rsl_rl_ppo-memory_cfg.py"
OUTPUT_LOGS_PATH="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/train_logs/"
OUTPUT_MODELS_PATH="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/train_eval_models_logs/"

# --- 2. EXPERIMENT PARAMETERS ---
TASKS_NAMES="GoToPose, TrackVelocities, Rendezvous, GoToPositionWithObstacles"
ZERO_SHOT_TASKS=(GoToPosition)
SEED=1
MODEL_NUM="3999"
EVAL_NUM_ENVS=64
RUNS_PER_ENV=2

# Increments
# DELTAS=(0.05 0.1 0.15 0.2)           # For Mass/CoM
# WRENCH_VALS=(0.1 0.2 0.3 0.4)        # For Wrench Force/Torque

DELTAS=(0.15)           # For Mass/CoM
WRENCH_VALS=(0.2)        # For Wrench Force/Torque

# --- 3. HELPER FUNCTIONS ---

reset_all() {
    echo "Resetting all randomization flags to False..."
    sed -i "/mass_rand_cfg:/ s/enable=True/enable=False/" "$ROBOT_CFG_PATH"
    sed -i "/com_rand_cfg:/ s/enable=True/enable=False/" "$ROBOT_CFG_PATH"
    sed -i "/wrench_rand_cfg/,/)/ s/enable=True/enable=False/" "$ROBOT_CFG_PATH"
}

# Master Function: Training + Multi-task Eval + Zero-shot Eval
run_full_cycle() {
    local LABEL=$1
    echo "========================================================="
    echo " PROCESS START: $LABEL"
    echo "========================================================="
    
    OUTPUT_FILE="${OUTPUT_LOGS_PATH}sim2real_log_${LABEL}_seed-${SEED}.txt"
    rm -f "$OUTPUT_FILE"

    # A. Update Experiment Name in Agent Config to prevent overwriting/resuming
    sed -i "s/experiment_name = \".*\"/experiment_name = \"${LABEL}\"/" "$AGENT_CFG_PATH"

    # B. Training
    echo "-> Training..."
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_control.py \
        --headless --num_envs=4096 --task=Isaac-RANS-MultiTask-v0 \
        env.robot_name=$ROBOT env.tasks_names="[${TASKS_NAMES}]" \
        --algorithm=$ALGORITHM --seed=$SEED >> "$OUTPUT_FILE" 2>&1

    # C. Extract Model Name for Evaluation
    MODEL_NAME=$(grep -oP 'wandb: Syncing run \K.+' "$OUTPUT_FILE" | tail -1)
    EXP_NAME=$LABEL
    CHECKPOINT_PATH="/workspace/isaaclab/logs/rsl_rl/${EXP_NAME}/${MODEL_NAME}/model_${MODEL_NUM}.pt"

    # D. Multi-Task Evaluation
    echo "-> Evaluating Multi-Task..."
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_control.py \
        --task=Isaac-RANS-MultiTask-v0 --headless --num_envs="${EVAL_NUM_ENVS}" \
        --checkpoint="${CHECKPOINT_PATH}" --algorithm="${ALGORITHM}" \
        --runs_per_env="${RUNS_PER_ENV}" env.robot_name="${ROBOT}" \
        env.tasks_names="[${TASKS_NAMES}]" >> "$OUTPUT_FILE" 2>&1

    # # E. Zero-Shot Evaluation
    # for TASK in "${ZERO_SHOT_TASKS[@]}"; do
    #     echo "-> Evaluating Zero-shot: $TASK"
    #     ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_control.py \
    #         --task=Isaac-RANS-Single-v0 --headless --num_envs=${EVAL_NUM_ENVS} \
    #         --runs_per_env=${RUNS_PER_ENV} --algorithm=${ALGORITHM} \
    #         env.robot_name=${ROBOT} env.task_name=${TASK} \
    #         --checkpoint=${CHECKPOINT_PATH} >> "$OUTPUT_FILE" 2>&1
    # done

    # F. Logging Path
    echo "/workspace/isaaclab/logs/rsl_rl/${EXP_NAME}/${MODEL_NAME}" >> "${OUTPUT_MODELS_PATH}sim2realFP_summary.txt"
}

# --- 4. EXECUTION PHASES ---

mkdir -p "$OUTPUT_LOGS_PATH" "$OUTPUT_MODELS_PATH"
reset_all

#  ---------------------------------------------------------
# PHASE 0: Baseline (No Randomization)
# ---------------------------------------------------------
run_full_cycle "BASELINE_NO_RANDOMIZATION_Sim2RealFP"

# ---------------------------------------------------------
# PHASE 1: Mass Only
# ---------------------------------------------------------
# for D in "${DELTAS[@]}"; do
#     reset_all
#     # Matches line with mass_rand_cfg and changes enable and max_delta
#     sed -i "/mass_rand_cfg:/ s/enable=False/enable=True/" "$ROBOT_CFG_PATH"
#     sed -i "/mass_rand_cfg:/ s/max_delta=[0-9.]*/max_delta=$D/" "$ROBOT_CFG_PATH"
#     run_full_cycle "MASS_ONLY_D${D}"
# done

# ---------------------------------------------------------
# PHASE 2: CoM Only
# ---------------------------------------------------------
# for D in "${DELTAS[@]}"; do
#     reset_all
#     # Matches line with com_rand_cfg and changes enable and max_delta
#     sed -i "/com_rand_cfg:/ s/enable=False/enable=True/" "$ROBOT_CFG_PATH"
#     sed -i "/com_rand_cfg:/ s/max_delta=[0-9.]*/max_delta=$D/" "$ROBOT_CFG_PATH"
#     run_full_cycle "COM_ONLY_D${D}"
# done

# ---------------------------------------------------------
# PHASE 3: Wrench Only
# ---------------------------------------------------------
# for W in "${WRENCH_VALS[@]}"; do
#     reset_all
#     # Matches the block from wrench_rand_cfg until the closing ')'
#     sed -i "/wrench_rand_cfg/,/)/ s/enable=False/enable=True/" "$ROBOT_CFG_PATH"
#     sed -i "/uniform_force=/ s/(0, [0-9.]*)/(0, $W)/" "$ROBOT_CFG_PATH"
#     sed -i "/uniform_torque=/ s/(0, [0-9.]*)/(0, $W)/" "$ROBOT_CFG_PATH"
#     run_full_cycle "WRENCH_ONLY_W${W}"
# done

# ---------------------------------------------------------
# PHASE 4: ALL ACTIVATED
# ---------------------------------------------------------
for i in "${!DELTAS[@]}"; do
    D=${DELTAS[$i]}
    W=${WRENCH_VALS[$i]}
    reset_all
    
    # Enable all
    sed -i "/mass_rand_cfg:/ s/enable=False/enable=True/" "$ROBOT_CFG_PATH"
    sed -i "/com_rand_cfg:/ s/enable=False/enable=True/" "$ROBOT_CFG_PATH"
    sed -i "/wrench_rand_cfg/,/)/ s/enable=False/enable=True/" "$ROBOT_CFG_PATH"
    
    # Update values
    sed -i "/mass_rand_cfg:/ s/max_delta=[0-9.]*/max_delta=$D/" "$ROBOT_CFG_PATH"
    sed -i "/com_rand_cfg:/ s/max_delta=[0-9.]*/max_delta=$D/" "$ROBOT_CFG_PATH"
    sed -i "/uniform_force=/ s/(0, [0-9.]*)/(0, $W)/" "$ROBOT_CFG_PATH"
    sed -i "/uniform_torque=/ s/(0, [0-9.]*)/(0, $W)/" "$ROBOT_CFG_PATH"
    
    run_full_cycle "ALL_ACTIVE_D${D}_W${W}_Sim2RealFP"
done

echo "Suite complete. Check $OUTPUT_MODELS_PATH/experiment_summary.txt for model locations."