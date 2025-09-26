#!/bin/bash
TASKS=(
    "Rendezvous"
    "GoToPosition"
    "GoToPose"
    "TrackVelocities"
)

for _task in "${TASKS[@]}"; 
do
    for SEED in {1..1}
    do
        echo "Training task: ${_task} with seed: ${SEED}"
        ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_control.py --task=Isaac-RANS-MultiTask-v0 --headless env.robot_name=FloatingPlatform env.tasks_names="['${_task}']" --algorithm=ppo --seed=$SEED --num_envs=1024
    done
done