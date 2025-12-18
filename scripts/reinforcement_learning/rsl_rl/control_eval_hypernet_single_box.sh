
#!/bin/bash

# Common evaluation parameters
SCRIPT_PATH="./scripts/reinforcement_learning/rsl_rl/eval_control.py"
TASK="Isaac-RANS-Single-v0"
NUM_ENVS=32
runs_per_env=4
BASE_LOG_DIR="logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad"
robot="IntBall2"
task_name="GoToPose3DBox"
algorithm="ppo-beta-memory" #ppo, ppo-memory, ppo-beta, ppo-beta-memory

# Common arguments that apply to all evaluations
COMMON_ARGS="--task=${TASK} --headless --num_envs=${NUM_ENVS} --runs_per_env=${runs_per_env} --algorithm=${algorithm} env.robot_name=${robot} env.task_name=${task_name}"

# Array of checkpoint paths (relative to BASE_LOG_DIR)
# CHECKPOINTS=(
# "2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/model_2000.pt"
# "2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/model_2000.pt"
# "2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/model_2000.pt"
# "2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/model_2000.pt"
# "2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/model_2000.pt"
# )

# # Hypernet Beta PCGrad RSS
# CHECKPOINTS=(
# "2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/model_3999.pt"
# "2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/model_3999.pt"
# "2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/model_3999.pt"
# "2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/model_3999.pt"
# "2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/model_3999.pt"
# )


# # Hypernet Beta RSS
# CHECKPOINTS=(
# 2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/model_3999.pt
# 2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/model_3999.pt
# 2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/model_3999.pt
# 2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/model_3999.pt
# 2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/model_3999.pt
# )

# Hypernet curriculum 14,2 RSS
# CHECKPOINTS=(
# 2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/model_3999.pt
# )

# Hypernet curriculum 7,0 RSS
# CHECKPOINTS=(
# 2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/model_3999.pt
# )

# Hypernet beta_multicrit_pcgrad
CHECKPOINTS=(
2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/model_3999.pt
2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/model_3999.pt
2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/model_3999.pt
2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/model_3999.pt
2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/model_3999.pt
)


# Function to run evaluation for a single checkpoint
run_evaluation() {
    local checkpoint_path="$1"
    local full_checkpoint_path="${BASE_LOG_DIR}/${checkpoint_path}"
    
    echo "Running evaluation for checkpoint: ${checkpoint_path}"
    ./isaaclab.sh -p ${SCRIPT_PATH} ${COMMON_ARGS} --checkpoint=${full_checkpoint_path}
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed successfully for: ${checkpoint_path}"
    else
        echo "✗ Evaluation failed for: ${checkpoint_path}"
    fi
    echo "----------------------------------------"
}

# Main execution: iterate through all checkpoints
echo "Starting control evaluation for ${#CHECKPOINTS[@]} checkpoints..."
echo "Task: ${TASK}"
echo "Number of environments: ${NUM_ENVS}"
echo "========================================"

for checkpoint in "${CHECKPOINTS[@]}"; do
    run_evaluation "${checkpoint}"
done

echo "All evaluations completed!"