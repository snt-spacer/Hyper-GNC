#!/bin/bash
TASKS=(
    # "GoToPose3D"
    # "TrackVelocities3D"
    # "GoThroughPoses3D"
    "GoToPosition3DWithObstacles"
    # "GoToPosition3D"
    # "GoToPose3DBox"
)

ROBOT_NAME="IntBall2"

for _task in "${TASKS[@]}"; 
do
    for SEED in {1..1}
    do
        echo "Training task: ${_task} with seed: ${SEED}"
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_control.py --task=Isaac-RANS-Single-v0 --headless env.robot_name=${ROBOT_NAME} env.task_name=${_task} --algorithm=ppo --seed=$SEED --num_envs=1024
    done
done