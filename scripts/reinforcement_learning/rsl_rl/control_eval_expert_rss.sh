#!/bin/bash

echo "Starting evaluation for GoToPose3D experts..."
./scripts/reinforcement_learning/rsl_rl/control_eval_expert_gotopose3d.sh

echo "Starting evaluation for TrackVelocities3D experts..."
./scripts/reinforcement_learning/rsl_rl/control_eval_expert_trackvelocities3d.sh

echo "Starting evaluation for GoThroughPoses3D experts..."
./scripts/reinforcement_learning/rsl_rl/control_eval_expert_gothroughposes3d.sh

echo "Starting evaluation for GoToPosition3DWithObstacles experts..."
./scripts/reinforcement_learning/rsl_rl/control_eval_expert_gotopositionwithobstacles.sh

echo "Starting evaluation for GoToPosition3D experts..."
./scripts/reinforcement_learning/rsl_rl/control_eval_expert_gotoposition3d.sh

echo "Starting evaluation for GoToPose3DBox experts..."
./scripts/reinforcement_learning/rsl_rl/control_eval_expert_gotoposebox.sh

echo "All expert evaluations completed!"
