#!/bin/bash
ALGORITHM="ppo-memory"
ROBOT="FloatingPlatform"
for SEED in {1..5}
do
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_control.py --headless --task=Isaac-RANS-MultiTask-v0 env.robot_name=$ROBOT env.tasks_names='[GoToPose, TrackVelocities, GoThroughPoses]' --algorithm=$ALGORITHM --seed=$SEED --num_envs=4095
    sleep 5
done
