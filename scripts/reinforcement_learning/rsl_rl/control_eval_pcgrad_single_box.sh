
#!/bin/bash

# Common evaluation parameters
SCRIPT_PATH="./scripts/reinforcement_learning/rsl_rl/eval_control.py"
TASK="Isaac-RANS-Single-v0"
NUM_ENVS=32
runs_per_env=4
BASE_LOG_DIR="logs/rsl_rl/mtrl_intball2_pcgrad"
robot="IntBall2"
task_name="GoToPose3DBox"
algorithm="ppo" #ppo, ppo-memory, ppo-beta, ppo-beta-memory ppo-multi-critic

# Common arguments that apply to all evaluations
COMMON_ARGS="--task=${TASK} --headless --num_envs=${NUM_ENVS} --runs_per_env=${runs_per_env} --algorithm=${algorithm} env.robot_name=${robot} env.task_name=${task_name}"

# PCGRAD RSS 14-3
# CHECKPOINTS=(
# 2025-12-17_09-19-33_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/model_3999.pt
# 2025-12-17_12-00-58_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/model_3999.pt
# 2025-12-17_14-42-24_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/model_3999.pt
# 2025-12-17_17-22-25_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/model_3999.pt
# 2025-12-17_20-03-17_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/model_3999.pt
# )

# # PCGRAD RSS 7-0
# CHECKPOINTS=(
# 2025-12-19_14-49-48_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/model_3999.pt
# 2025-12-19_17-53-34_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/model_3999.pt
# 2025-12-19_20-33-29_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/model_3999.pt
# 2025-12-19_23-13-38_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/model_3999.pt
# 2025-12-20_01-53-26_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/model_3999.pt
# )

# PCGrad TaskID 7-0
CHECKPOINTS=(
2026-01-06_12-29-20_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/model_3999.pt
2026-01-06_15-08-09_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/model_3999.pt
2026-01-06_17-48-08_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/model_3999.pt
2026-01-06_20-27-30_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/model_3999.pt
2026-01-06_23-06-49_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/model_3999.pt
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