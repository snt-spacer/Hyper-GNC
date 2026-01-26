#!/bin/bash
ALGORITHM="ppo"
ROBOT="FloatingPlatform"
OUTPUT_LOGS_PATH="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/train_logs/"
OUTPUT_MODELS_PATH="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/train_eval_models_logs/"
CONFIG_PATH="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/environments/single_robot_multi_task/agents/rsl_rl_ppo_cfg.py"
TASKS_NAMES="GoToPose,TrackVelocities,Rendezvous,GoToPositionWithObstacles"

# Evaluation parameters
EVAL_NUM_ENVS=64 # Base number of environments, adjust if needed
RUNS_PER_ENV=2
MODEL_NUM="3999"

mkdir -p "$OUTPUT_LOGS_PATH"
mkdir -p "$OUTPUT_MODELS_PATH"

TASKS=(
    "GoToPose"
    "TrackVelocities"
    "Rendezvous"
    "GoToPositionWithObstacles"
    "GoToPosition"
)

ROBOT_NAME="FloatingPlatform"

for _task in "${TASKS[@]}"; 
do
    for SEED in {1..1}
    do
        echo "Training task: ${_task} with seed: ${SEED}"
        OUTPUT_FILE="${OUTPUT_LOGS_PATH}expert-floatingplatform-${_task}-${SEED}.txt"
        rm -f $OUTPUT_FILE
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_control.py --task=Isaac-RANS-Single-v0 --headless env.robot_name=${ROBOT_NAME} env.task_name=${_task} --algorithm=$ALGORITHM --seed=$SEED --num_envs=1024 >> $OUTPUT_FILE 2>&1

        if [ $? -eq 0 ]; then
            echo "✓ Training completed successfully."
        else
            echo "✗ Training failed. Check logs in $OUTPUT_FILE for details."
        fi

        sleep 1

        # Evaluation
        echo "Now evaluating the trained model. Did you touch some grass?"
        MODEL_NAME=$(grep -oP 'wandb: Syncing run \K.+' "$OUTPUT_FILE" | tail -1)
        EXP_NAME=$(grep "experiment_name =" $CONFIG_PATH | cut -d '"' -f 2)
        CHECKPOINT_PATH="/workspace/isaaclab/logs/rsl_rl/${EXP_NAME}/${MODEL_NAME}/model_${MODEL_NUM}.pt"
        echo "Evaluating model $CHECKPOINT_PATH"
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_control.py \
            --task=Isaac-RANS-Single-v0 \
            --headless \
            --num_envs="${EVAL_NUM_ENVS}" \
            --checkpoint="${CHECKPOINT_PATH}" \
            --algorithm="${ALGORITHM}" \
            --runs_per_env="${RUNS_PER_ENV}" \
            env.robot_name="${ROBOT}" \
            env.task_name="${_task}" >> $OUTPUT_FILE 2>&1

        if [ $? -eq 0 ]; then
            echo "✓ Evaluation completed successfully."
        else
            echo "✗ Evaluation failed. Check logs in $OUTPUT_FILE for details."
        fi
    done

    EVAL_PATH="/workspace/isaaclab/logs/rsl_rl/${EXP_NAME}/${MODEL_NAME}"
    echo $EVAL_PATH >> "${OUTPUT_MODELS_PATH}${EXP_NAME}.txt"
    echo "----------------------------------------"
done