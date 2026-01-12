
#!/bin/bash

# Common evaluation parameters
SCRIPT_PATH="./scripts/reinforcement_learning/rsl_rl/eval_control.py"
TASK="Isaac-RANS-Single-v0"
NUM_ENVS=126
runs_per_env=4
BASE_LOG_DIR="logs/rsl_rl/mtrl_intball2_pcgrad_simple_semEmb"
robot="IntBall2"
task_name="GoToPosition3D"
algorithm="ppo" #ppo, ppo-memory, ppo-beta, ppo-beta-memory

# Common arguments that apply to all evaluations
COMMON_ARGS="--task=${TASK} --headless --num_envs=${NUM_ENVS} --runs_per_env=${runs_per_env} --algorithm=${algorithm} env.robot_name=${robot} env.task_name=${task_name}"

# Array of checkpoint paths (relative to BASE_LOG_DIR)
# CHECKPOINTS=(
# 2026-01-08_17-30-08_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D_IntBall2_r-0_seed-1/model_3999.pt
# 2026-01-08_19-40-59_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D_IntBall2_r-0_seed-2/model_3999.pt
# 2026-01-08_21-51-48_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D_IntBall2_r-0_seed-3/model_3999.pt
# 2026-01-09_00-02-37_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D_IntBall2_r-0_seed-4/model_3999.pt
# 2026-01-09_02-13-32_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D_IntBall2_r-0_seed-5/model_3999.pt

# )

# TaskID
# CHECKPOINTS=(
# 2026-01-09_10-18-16_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D_IntBall2_r-0_seed-1/model_3999.pt
# 2026-01-09_12-28-42_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D_IntBall2_r-0_seed-2/model_3999.pt
# 2026-01-09_14-39-51_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D_IntBall2_r-0_seed-3/model_3999.pt
# 2026-01-09_16-50-18_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D_IntBall2_r-0_seed-4/model_3999.pt
# 2026-01-09_19-00-35_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D_IntBall2_r-0_seed-5/model_3999.pt

# )


#SemEmb
CHECKPOINTS=(
2026-01-11_21-24-51_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D_IntBall2_r-0_seed-1/model_3999.pt
2026-01-11_23-35-36_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D_IntBall2_r-0_seed-2/model_3999.pt
2026-01-12_01-46-25_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D_IntBall2_r-0_seed-3/model_3999.pt
2026-01-12_03-57-27_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D_IntBall2_r-0_seed-4/model_3999.pt
2026-01-12_06-08-29_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D_IntBall2_r-0_seed-5/model_3999.pt
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