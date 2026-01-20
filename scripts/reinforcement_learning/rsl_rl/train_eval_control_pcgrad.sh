#!/bin/bash
ALGORITHM="ppo"
ROBOT="IntBall2"
OUTPUT_LOGS_PATH="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/train_logs/"
OUTPUT_MODELS_PATH="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/train_eval_models_logs/"
CONFIG_PATH="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/environments/single_robot_multi_task/agents/rsl_rl_ppo_cfg.py"
TASKS_NAMES="GoToPose3D,TrackVelocities3D,GoThroughPoses3D,GoToPosition3DWithObstacles"

# Evaluation parameters
EVAL_NUM_ENVS=64 # Base number of environments, adjust if needed
RUNS_PER_ENV=2
MODEL_NUM="3999"
ZERO_SHOT_TASKS=(GoToPose3DBox GoToPosition3D)

mkdir -p "$OUTPUT_LOGS_PATH"
mkdir -p "$OUTPUT_MODELS_PATH"

start_time=$(date +%s.%N)
for SEED in {1..5}
do
    # Training
    echo "No worries is training, go touch some grass."
    echo "Seed: $SEED"
    OUTPUT_FILE="${OUTPUT_LOGS_PATH}pcgrad-semEmb_seed-${SEED}.txt"
    rm -f $OUTPUT_FILE
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_control.py \
        --headless \
        --task=Isaac-RANS-MultiTask-v0 \
        env.robot_name=$ROBOT \
        env.tasks_names="[${TASKS_NAMES}]" \
        --algorithm=$ALGORITHM \
        --seed=$SEED >> $OUTPUT_FILE 2>&1

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
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_control.py \
        --task=Isaac-RANS-MultiTask-v0 \
        --headless \
        --num_envs="${EVAL_NUM_ENVS}" \
        --checkpoint="${CHECKPOINT_PATH}" \
        --algorithm="${ALGORITHM}" \
        --runs_per_env="${RUNS_PER_ENV}" \
        env.robot_name="${ROBOT}" \
        env.tasks_names="[${TASKS_NAMES}]" >> $OUTPUT_FILE 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed successfully."
    else
        echo "✗ Evaluation failed. Check logs in $OUTPUT_FILE for details."
    fi

    # Zero-shot evaluation
    for TASK in "${ZERO_SHOT_TASKS[@]}"
    do
        echo "Zero-shot: $TASK"
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_control.py \
            --task=Isaac-RANS-Single-v0 \
            --headless \
            --num_envs=${EVAL_NUM_ENVS} \
            --runs_per_env=${RUNS_PER_ENV} \
            --algorithm=${ALGORITHM} \
            env.robot_name=${ROBOT} \
            env.task_name=${TASK} \
            --checkpoint=${CHECKPOINT_PATH} >> $OUTPUT_FILE 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✓"
        else
            echo "✗ Evaluation failed. Check logs in $OUTPUT_FILE for details."
        fi

    done

    # Save eval path to single file
    EVAL_PATH="/workspace/isaaclab/logs/rsl_rl/${EXP_NAME}/${MODEL_NAME}"
    echo $EVAL_PATH >> "${OUTPUT_MODELS_PATH}${EXP_NAME}.txt"
    echo "----------------------------------------"

done