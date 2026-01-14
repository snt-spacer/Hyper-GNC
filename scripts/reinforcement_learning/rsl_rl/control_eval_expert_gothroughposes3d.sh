
#!/bin/bash

# Common evaluation parameters
SCRIPT_PATH="./scripts/reinforcement_learning/rsl_rl/eval_control.py"
TASK="Isaac-RANS-Single-v0"
NUM_ENVS=32
runs_per_env=4
BASE_LOG_DIR="logs/rsl_rl/intball_experts"
robot="IntBall2"
task_name="GoThroughPoses3D"
algorithm="ppo" #ppo, ppo-memory, ppo-beta, ppo-beta-memory

# Common arguments that apply to all evaluations
COMMON_ARGS="--task=${TASK} --headless --num_envs=${NUM_ENVS} --runs_per_env=${runs_per_env} --algorithm=${algorithm} env.robot_name=${robot} env.task_name=${task_name}"

# Array of checkpoint paths (relative to BASE_LOG_DIR)

#Experts
CHECKPOINTS=(
    2026-01-07_12-27-29_rsl-rl_ppo_GoThroughPoses3D_IntBall2_r-0_seed-1/model_3999.pt
    2026-01-07_13-04-45_rsl-rl_ppo_GoThroughPoses3D_IntBall2_r-0_seed-2/model_3999.pt
    2026-01-07_13-42-15_rsl-rl_ppo_GoThroughPoses3D_IntBall2_r-0_seed-3/model_3999.pt
    2026-01-07_14-19-20_rsl-rl_ppo_GoThroughPoses3D_IntBall2_r-0_seed-4/model_3999.pt
    2026-01-07_14-55-28_rsl-rl_ppo_GoThroughPoses3D_IntBall2_r-0_seed-5/model_3999.pt
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