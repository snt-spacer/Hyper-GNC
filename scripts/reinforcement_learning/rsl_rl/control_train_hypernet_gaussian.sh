#!/bin/bash
ALGORITHM="ppo-memory"
ROBOT="IntBall2"
for SEED in {1..5}
do
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_control.py --headless --task=Isaac-RANS-MultiTask-v0 env.robot_name=$ROBOT env.tasks_names='[GoToPose3D, TrackVelocities3D, GoThroughPoses3D, GoToPosition3DWithObstacles]' --algorithm=$ALGORITHM --seed=$SEED
    sleep 5
done
