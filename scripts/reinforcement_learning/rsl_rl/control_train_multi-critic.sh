#!/bin/bash
ALGORITHM="ppo-multi-critic"
ROBOT="IntBall2"
TASKS='[GoToPose3D, TrackVelocities3D, GoThroughPoses3D, GoToPosition3DWithObstacles]'
for SEED in {1..5}
do
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_control.py --task=Isaac-RANS-MultiTask-v0 --headless env.robot_name=$ROBOT env.tasks_names=$TASKS --algorithm=$ALGORITHM --seed=$SEED
    sleep 5
done