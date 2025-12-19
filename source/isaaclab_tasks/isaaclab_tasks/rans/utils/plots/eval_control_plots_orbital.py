import torch
from tasks import TaskPlotsFactory
from robots import RobotPlotsFactory

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import yaml

def main():
    # Define your CSV file groups directly, now each group can contain multiple sets of files (for different seeds)
    list_of_grouped_csv_data = [
        # Multi-critic
        # {
        #     "group_name": "MultiCritic  GoToPose3D",
        #     "task_name": "GoToPose3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-01_20-51-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-01_20-51-17_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-01_20-51-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-01_20-51-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_08-27-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-02_08-27-36_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_08-27-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_08-27-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_13-56-40_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-02_13-56-40_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_13-56-40_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_13-56-40_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_16-07-00_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-02_16-07-00_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_16-07-00_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_16-07-00_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_18-16-56_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-02_18-16-56_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_18-16-56_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_18-16-56_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "MultiCritic  GoThroughPoses3D",
        #     "task_name": "GoThroughPoses3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-01_20-51-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-01_20-51-17_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-01_20-51-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-01_20-51-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_08-27-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-02_08-27-36_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_08-27-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_08-27-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_13-56-40_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-02_13-56-40_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_13-56-40_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_13-56-40_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_16-07-00_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-02_16-07-00_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_16-07-00_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_16-07-00_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_18-16-56_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-02_18-16-56_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_18-16-56_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_18-16-56_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "MultiCritic  TrackVelocities3D",
        #     "task_name": "TrackVelocities3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-01_20-51-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-01_20-51-17_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-01_20-51-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-01_20-51-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_08-27-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-02_08-27-36_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_08-27-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_08-27-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_13-56-40_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-02_13-56-40_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_13-56-40_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_13-56-40_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_16-07-00_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-02_16-07-00_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_16-07-00_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_16-07-00_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_18-16-56_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-02_18-16-56_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_18-16-56_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_18-16-56_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "MultiCritic  GoToPosition3DWithObstacles",
        #     "task_name": "GoToPosition3DWithObstacles",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-01_20-51-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-01_20-51-17_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-01_20-51-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-01_20-51-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_08-27-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-02_08-27-36_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_08-27-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_08-27-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_13-56-40_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-02_13-56-40_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_13-56-40_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_13-56-40_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_16-07-00_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-02_16-07-00_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_16-07-00_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_16-07-00_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_18-16-56_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-02_18-16-56_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_18-16-56_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-02_18-16-56_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },

        #PCGrad
        # {
        #     "group_name": "PCGrad GoToPose3D",
        #     "task_name": "GoToPose3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_11-12-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-03_11-12-01_rsl-rl_ppo_GoToPose3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_11-12-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_11-12-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_15-32-34_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-03_15-32-34_rsl-rl_ppo_GoToPose3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_15-32-34_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_15-32-34_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_18-26-42_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-03_18-26-42_rsl-rl_ppo_GoToPose3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_18-26-42_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_18-26-42_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_21-21-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-03_21-21-01_rsl-rl_ppo_GoToPose3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_21-21-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_21-21-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-04_00-15-37_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-04_00-15-37_rsl-rl_ppo_GoToPose3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-04_00-15-37_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-04_00-15-37_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "PCGrad GoThroughPoses3D",
        #     "task_name": "GoThroughPoses3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_11-12-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-03_11-12-01_rsl-rl_ppo_GoThroughPoses3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_11-12-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_11-12-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_15-32-34_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-03_15-32-34_rsl-rl_ppo_GoThroughPoses3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_15-32-34_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_15-32-34_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_18-26-42_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-03_18-26-42_rsl-rl_ppo_GoThroughPoses3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_18-26-42_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_18-26-42_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_21-21-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-03_21-21-01_rsl-rl_ppo_GoThroughPoses3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_21-21-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_21-21-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-04_00-15-37_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-04_00-15-37_rsl-rl_ppo_GoThroughPoses3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-04_00-15-37_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-04_00-15-37_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "PCGrad TrackVelocities3D",
        #     "task_name": "TrackVelocities3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_11-12-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-03_11-12-01_rsl-rl_ppo_TrackVelocities3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_11-12-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_11-12-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_15-32-34_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-03_15-32-34_rsl-rl_ppo_TrackVelocities3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_15-32-34_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_15-32-34_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_18-26-42_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-03_18-26-42_rsl-rl_ppo_TrackVelocities3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_18-26-42_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_18-26-42_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_21-21-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-03_21-21-01_rsl-rl_ppo_TrackVelocities3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_21-21-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_21-21-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-04_00-15-37_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-04_00-15-37_rsl-rl_ppo_TrackVelocities3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-04_00-15-37_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-04_00-15-37_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "PCGrad GoToPosition3DWithObstacles",
        #     "task_name": "GoToPosition3DWithObstacles",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_11-12-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-03_11-12-01_rsl-rl_ppo_GoToPosition3DWithObstacles_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_11-12-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_11-12-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_15-32-34_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-03_15-32-34_rsl-rl_ppo_GoToPosition3DWithObstacles_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_15-32-34_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_15-32-34_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_18-26-42_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-03_18-26-42_rsl-rl_ppo_GoToPosition3DWithObstacles_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_18-26-42_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_18-26-42_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_21-21-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-03_21-21-01_rsl-rl_ppo_GoToPosition3DWithObstacles_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_21-21-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-03_21-21-01_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-04_00-15-37_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-04_00-15-37_rsl-rl_ppo_GoToPosition3DWithObstacles_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-04_00-15-37_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_pcgrad/2025-12-04_00-15-37_rsl-rl_ppo_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },

        # Hypernet beta
        # {
        #     "group_name": "Hypernet Beta GoToPose3D",
        #     "task_name": "GoToPose3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-01_22-04-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-01_22-04-46_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-01_22-04-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-01_22-04-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_00-16-14_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-02_00-16-14_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_00-16-14_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_00-16-14_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_02-25-20_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-02_02-25-20_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_02-25-20_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_02-25-20_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_04-35-47_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-02_04-35-47_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_04-35-47_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_04-35-47_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_06-46-37_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-02_06-46-37_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_06-46-37_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_06-46-37_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta GoThroughPoses3D",
        #     "task_name": "GoThroughPoses3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-01_22-04-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-01_22-04-46_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-01_22-04-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-01_22-04-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_00-16-14_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-02_00-16-14_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_00-16-14_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_00-16-14_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_02-25-20_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-02_02-25-20_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_02-25-20_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_02-25-20_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_04-35-47_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-02_04-35-47_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_04-35-47_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_04-35-47_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_06-46-37_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-02_06-46-37_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_06-46-37_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_06-46-37_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta TrackVelocities3D",
        #     "task_name": "TrackVelocities3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-01_22-04-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-01_22-04-46_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-01_22-04-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-01_22-04-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_00-16-14_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-02_00-16-14_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_00-16-14_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_00-16-14_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_02-25-20_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-02_02-25-20_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_02-25-20_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_02-25-20_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_04-35-47_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-02_04-35-47_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_04-35-47_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_04-35-47_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_06-46-37_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-02_06-46-37_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_06-46-37_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_06-46-37_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta GoToPosition3DWithObstacles",
        #     "task_name": "GoToPosition3DWithObstacles",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-01_22-04-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-01_22-04-46_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-01_22-04-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-01_22-04-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_00-16-14_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-02_00-16-14_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_00-16-14_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_00-16-14_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_02-25-20_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-02_02-25-20_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_02-25-20_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_02-25-20_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_04-35-47_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-02_04-35-47_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_04-35-47_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_04-35-47_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_06-46-37_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-02_06-46-37_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_06-46-37_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-02_06-46-37_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # Hypernet beta 2
        # {
        #     "group_name": "Hypernet Beta GoToPose3D",
        #     "task_name": "GoToPose3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_08-45-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-05_08-45-34_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_08-45-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_08-45-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_12-29-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-05_12-29-34_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_12-29-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_12-29-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_16-04-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-05_16-04-34_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_16-04-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_16-04-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_19-39-19_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-05_19-39-19_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_19-39-19_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_19-39-19_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_23-12-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-05_23-12-53_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_23-12-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_23-12-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta GoThroughPoses3D",
        #     "task_name": "GoThroughPoses3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_08-45-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-05_08-45-34_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_08-45-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_08-45-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_12-29-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-05_12-29-34_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_12-29-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_12-29-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_16-04-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-05_16-04-34_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_16-04-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_16-04-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_19-39-19_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-05_19-39-19_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_19-39-19_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_19-39-19_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_23-12-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-05_23-12-53_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_23-12-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_23-12-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta TrackVelocities3D",
        #     "task_name": "TrackVelocities3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_08-45-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-05_08-45-34_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_08-45-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_08-45-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_12-29-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-05_12-29-34_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_12-29-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_12-29-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_16-04-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-05_16-04-34_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_16-04-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_16-04-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_19-39-19_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-05_19-39-19_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_19-39-19_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_19-39-19_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_23-12-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-05_23-12-53_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_23-12-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_23-12-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta GoToPosition3DWithObstacles",
        #     "task_name": "GoToPosition3DWithObstacles",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_08-45-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-05_08-45-34_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_08-45-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_08-45-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_12-29-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-05_12-29-34_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_12-29-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_12-29-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_16-04-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-05_16-04-34_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_16-04-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_16-04-34_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_19-39-19_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-05_19-39-19_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_19-39-19_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_19-39-19_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_23-12-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-05_23-12-53_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_23-12-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-05_23-12-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # }
        
        # Hypernet Beta Obstacles
        # {
        #     "group_name": "Hypernet Beta GoToPose3D",
        #     "task_name": "GoToPose3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta GoThroughPoses3D",
        #     "task_name": "GoThroughPoses3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta TrackVelocities3D",
        #     "task_name": "TrackVelocities3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta GoToPosition3DWithObstacles",
        #     "task_name": "GoToPosition3DWithObstacles",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta GoToPose3D",
        #     "task_name": "GoToPose3DBox",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_19-02-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_21-24-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-09_23-25-27_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_01-10-28_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-10_02-49-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        
        # Multicritic Obstacles
        # {
        #     "group_name": "Multicritic GoToPose3D",
        #     "task_name": "GoToPose3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Multicritic TrackVelocities3D",
        #     "task_name": "TrackVelocities3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Multicritic GoThroughPoses3D",
        #     "task_name": "GoThroughPoses3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Multicritic GoToPosition3DWithObstacles",
        #     "task_name": "GoToPosition3DWithObstacles",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Multicritic GoToPose3DBox",
        #     "task_name": "GoToPose3DBox",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3DBox_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3DBox_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3DBox_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3DBox_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3DBox_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Multicritic VeloStabilization3D",
        #     "task_name": "VeloStabilization3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_VeloStabilization3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_19-12-58_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_VeloStabilization3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_21-25-55_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_VeloStabilization3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-09_22-43-03_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_VeloStabilization3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_00-00-36_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_VeloStabilization3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-10_01-18-07_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        
        # Multicritic - rss
        # {
        #     "group_name": "Multicritic GoToPose3D",
        #     "task_name": "GoToPose3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Multicritic TrackVelocities3D",
        #     "task_name": "TrackVelocities3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Multicritic GoThroughPoses3D",
        #     "task_name": "GoThroughPoses3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Multicritic GoToPosition3DWithObstacles",
        #     "task_name": "GoToPosition3DWithObstacles",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Multicritic GoToPose3DBox",
        #     "task_name": "GoToPose3DBox",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3DBox_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3DBox_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3DBox_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3DBox_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3DBox_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Multicritic VeloStabilization3D",
        #     "task_name": "VeloStabilization3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_VeloStabilization3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_15-58-01_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_VeloStabilization3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_17-11-54_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_VeloStabilization3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_18-24-51_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_VeloStabilization3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_19-38-08_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_VeloStabilization3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-15_20-51-27_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        
        # Hypernet beta pcgrad rss
        # {
        #     "group_name": "Hypernet Beta PCGrad GoToPose3D",
        #     "task_name": "GoToPose3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta PCGrad TrackVelocities3D",
        #     "task_name": "TrackVelocities3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta PCGrad GoThroughPoses3D",
        #     "task_name": "GoThroughPoses3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta PCGrad GoToPosition3DWithObstacles",
        #     "task_name": "GoToPosition3DWithObstacles",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta PCGrad GoToPose3DBox",
        #     "task_name": "GoToPose3DBox",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta PCGrad VeloStabilization3D",
        #     "task_name": "VeloStabilization3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_VeloStabilization3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_21-29-48_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_VeloStabilization3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_23-54-26_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_VeloStabilization3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_02-18-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_VeloStabilization3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_04-42-25_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_VeloStabilization3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_07-06-13_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        
        # Hypernet beta rss
        # {
        #     "group_name": "Hypernet Beta GoToPose3D",
        #     "task_name": "GoToPose3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta TrackVelocities3D",
        #     "task_name": "TrackVelocities3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta GoThroughPoses3D",
        #     "task_name": "GoThroughPoses3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta GoToPosition3DWithObstacles",
        #     "task_name": "GoToPosition3DWithObstacles",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet Beta GoToPose3DBox",
        #     "task_name": "GoToPose3DBox",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_12-49-21_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-2_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_14-18-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-3_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_15-47-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-4_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_17-16-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #         },
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-5_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-15_18-45-22_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #         }
        #     ]
        # }
        
        # Hypernet curriculum 14-2 RSS
        # {
        #     "group_name": "Hypernet curriculum 14-2 GoToPose3D",
        #     "task_name": "GoToPose3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet curriculum 14-2 TrackVelocities3D",
        #     "task_name": "TrackVelocities3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet curriculum 14-2 GoThroughPoses3D",
        #     "task_name": "GoThroughPoses3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet curriculum 14-2 GoToPosition3DWithObstacles",
        #     "task_name": "GoToPosition3DWithObstacles",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet curriculum 14-2 GoToPose3DBox",
        #     "task_name": "GoToPose3DBox",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_16-34-46_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        
        # Hypernet curriculum 7-0 RSS
        # {
        #     "group_name": "Hypernet curriculum 7-0 GoToPose3D",
        #     "task_name": "GoToPose3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet curriculum 7-0 TrackVelocities3D",
        #     "task_name": "TrackVelocities3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet curriculum 7-0 GoThroughPoses3D",
        #     "task_name": "GoThroughPoses3D",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet curriculum 7-0 GoToPosition3DWithObstacles",
        #     "task_name": "GoToPosition3DWithObstacles",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         }
        #     ]
        # },
        # {
        #     "group_name": "Hypernet curriculum 7-0 GoToPose3DBox",
        #     "task_name": "GoToPose3DBox",
        #     "robot_name": "IntBall2",
        #     "runs": [
        #         {
        #             "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-1_metrics.csv",
        #             "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #             "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta/2025-12-16_21-48-51_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #         }
        #     ]
        # }
        
        # Hypernet beta_multicrit_pcgrad 14-3
        # {
        # "group_name": "Hypernet beta MCPCG GoToPose3D",
        # "task_name": "GoToPose3D",
        # "robot_name": "IntBall2",
        # "runs": [
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-1_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-2_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-3_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-4_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-5_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #     }
        # ]
        # },
        # {
        # "group_name": "Hypernet beta MCPCG TrackVelocities3D",
        # "task_name": "TrackVelocities3D",
        # "robot_name": "IntBall2",
        # "runs": [
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-1_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-2_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-3_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-4_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-5_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #     }
        # ]
        # },
        # {
        # "group_name": "Hypernet beta MCPCG GoThroughPoses3D",
        # "task_name": "GoThroughPoses3D",
        # "robot_name": "IntBall2",
        # "runs": [
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-1_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-2_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-3_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-4_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-5_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #     }
        # ]
        # },
        # {
        # "group_name": "Hypernet beta MCPCG GoToPosition3DWithObstacles",
        # "task_name": "GoToPosition3DWithObstacles",
        # "robot_name": "IntBall2",
        # "runs": [
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-1_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-2_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-3_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-4_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-5_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #     }
        # ]
        # },
        # {
        # "group_name": "Hypernet beta MCPCG GoToPose3DBox",
        # "task_name": "GoToPose3DBox",
        # "robot_name": "IntBall2",
        # "runs": [
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-1_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-2_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-3_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-4_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-5_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #     }
        # ]
        # },
        # {
        # "group_name": "Hypernet beta MCPCG VeloStabilization3D",
        # "task_name": "VeloStabilization3D",
        # "robot_name": "IntBall2",
        # "runs": [
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_VeloStabilization3D_IntBall2_r-0_seed-1_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_VeloStabilization3D_IntBall2_r-0_seed-2_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_VeloStabilization3D_IntBall2_r-0_seed-3_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_VeloStabilization3D_IntBall2_r-0_seed-4_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_VeloStabilization3D_IntBall2_r-0_seed-5_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #     }
        # ]
        # },
        # {
        # "group_name": "Hypernet beta MCPCG GoToPosition3D",
        # "task_name": "GoToPosition3D",
        # "robot_name": "IntBall2",
        # "runs": [
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPosition3D_IntBall2_r-0_seed-1_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPosition3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_09-18-23_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPosition3D_IntBall2_r-0_seed-2_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPosition3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_14-48-17_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPosition3D_IntBall2_r-0_seed-3_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPosition3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_17-12-40_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPosition3D_IntBall2_r-0_seed-4_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPosition3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_19-36-39_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPosition3D_IntBall2_r-0_seed-5_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPosition3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-17_21-59-18_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #     }
        # ]
        # },     
        
        # Multi-critic 14-3
        # {
        # "group_name": "Multi-Critic GoToPose3D",
        # "task_name": "GoToPose3D",
        # "robot_name": "IntBall2",
        # "runs": [
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-1_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-2_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-3_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-4_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D_IntBall2_r-0_seed-5_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #     }
        # ]
        # },
        # {
        # "group_name": "Multi-Critic TrackVelocities3D",
        # "task_name": "TrackVelocities3D",
        # "robot_name": "IntBall2",
        # "runs": [
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-1_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-2_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-3_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-4_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_TrackVelocities3D_IntBall2_r-0_seed-5_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #     }
        # ]
        # },
        # {
        # "group_name": "Multi-Critic GoThroughPoses3D",
        # "task_name": "GoThroughPoses3D",
        # "robot_name": "IntBall2",
        # "runs": [
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-1_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-2_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-3_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-4_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoThroughPoses3D_IntBall2_r-0_seed-5_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoThroughPoses3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #     }
        # ]
        # },
        # {
        # "group_name": "Multi-Critic GoToPosition3DWithObstacles",
        # "task_name": "GoToPosition3DWithObstacles",
        # "robot_name": "IntBall2",
        # "runs": [
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-1_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-2_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-3_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-4_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPosition3DWithObstacles_IntBall2_r-0_seed-5_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #     }
        # ]
        # },
        # {
        # "group_name": "Multi-Critic GoToPose3DBox",
        # "task_name": "GoToPose3DBox",
        # "robot_name": "IntBall2",
        # "runs": [
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3DBox_IntBall2_r-0_seed-1_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3DBox_IntBall2_r-0_seed-2_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3DBox_IntBall2_r-0_seed-3_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3DBox_IntBall2_r-0_seed-4_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3DBox_IntBall2_r-0_seed-5_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3DBox.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #     }
        # ]
        # },
        # {
        # "group_name": "Multi-Critic VeloStabilization3D",
        # "task_name": "VeloStabilization3D",
        # "robot_name": "IntBall2",
        # "runs": [
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_VeloStabilization3D_IntBall2_r-0_seed-1_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_VeloStabilization3D_IntBall2_r-0_seed-2_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_VeloStabilization3D_IntBall2_r-0_seed-3_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_VeloStabilization3D_IntBall2_r-0_seed-4_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_VeloStabilization3D_IntBall2_r-0_seed-5_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_VeloStabilization3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #     }
        # ]
        # },
        # {
        # "group_name": "Multi-Critic GoToPosition3D",
        # "task_name": "GoToPosition3D",
        # "robot_name": "IntBall2",
        # "runs": [
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPosition3D_IntBall2_r-0_seed-1_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPosition3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_09-21-17_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPosition3D_IntBall2_r-0_seed-2_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPosition3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_10-33-44_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPosition3D_IntBall2_r-0_seed-3_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPosition3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_11-46-33_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPosition3D_IntBall2_r-0_seed-4_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPosition3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_12-59-02_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
        #     },
        #     {
        #     "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPosition3D_IntBall2_r-0_seed-5_metrics.csv",
        #     "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPosition3D.csv",
        #     "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/mtrl_intball2_multi_critic/2025-12-17_14-11-24_rsl-rl_ppo-multi-critic_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
        #     }
        # ]
        # },
        
        # Hypernet MTCR PCG 7-0
        
        {
            "group_name": "Hypernet beta MCPCG 7:0 GoToPose3D",
            "task_name": "GoToPose3D",
            "robot_name": "IntBall2",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D_IntBall2_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet beta MCPCG 7:0 TrackVelocities3D",
            "task_name": "TrackVelocities3D",
            "robot_name": "IntBall2",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_TrackVelocities3D_IntBall2_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet beta MCPCG 7:0 GoThroughPoses3D",
            "task_name": "GoThroughPoses3D",
            "robot_name": "IntBall2",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoThroughPoses3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoThroughPoses3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoThroughPoses3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoThroughPoses3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoThroughPoses3D_IntBall2_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoThroughPoses3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet beta MCPCG 7:0 GoToPosition3DWithObstacles",
            "task_name": "GoToPosition3DWithObstacles",
            "robot_name": "IntBall2",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPosition3DWithObstacles_IntBall2_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPosition3DWithObstacles.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet beta MCPCG 7:0 GoToPose3DBox",
            "task_name": "GoToPose3DBox",
            "robot_name": "IntBall2",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPose3DBox.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPose3DBox.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPose3DBox.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPose3DBox.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3DBox_IntBall2_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPose3DBox.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet beta MCPCG 7:0 VeloStabilization3D",
            "task_name": "VeloStabilization3D",
            "robot_name": "IntBall2",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_VeloStabilization3D_IntBall2_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_VeloStabilization3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_VeloStabilization3D_IntBall2_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_VeloStabilization3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_VeloStabilization3D_IntBall2_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_VeloStabilization3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_VeloStabilization3D_IntBall2_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_VeloStabilization3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_VeloStabilization3D_IntBall2_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_VeloStabilization3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet beta MCPCG 7:0 GoToPosition3D",
            "task_name": "GoToPosition3D",
            "robot_name": "IntBall2",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPosition3D_IntBall2_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/extracted_trajectories_GoToPosition3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_11-13-53_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPosition3D_IntBall2_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/extracted_trajectories_GoToPosition3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_13-40-15_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPosition3D_IntBall2_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/extracted_trajectories_GoToPosition3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_16-20-38_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPosition3D_IntBall2_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/extracted_trajectories_GoToPosition3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_18-43-04_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPosition3D_IntBall2_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/extracted_trajectories_GoToPosition3D.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control_beta_multicrit_pcgrad/2025-12-18_21-06-45_rsl-rl_ppo-beta-memory_GoToPose3D-TrackVelocities3D-GoThroughPoses3D-GoToPosition3DWithObstacles_IntBall2_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        
    ]
    

    # rm -rf source/isaaclab_tasks/isaaclab_tasks/rans/utils/multiTask_scripts_plus_summaries/plots_vani2/metrics_summary.txt

    plot_cfg = {
        "title": "",
        "box_colors": [
            "#FF3D50",
            "#FFA034",
            "#2FA1FF",
            "#A734FF",
            "#FFFF3D",
            "#4DFF3D",
            "#FF3DBB",
            "#623652",
            "#00CED1",  # Dark Turquoise
            "#1E90FF",  # Dodger Blue
            "#4682B4",  # Steel Blue
            "#32CD32",  # Lime Green
            "#008080",  # Teal
            "#20B2AA",  # Light Sea Green
            "#8A2BE2",  # Blue Violet
            "#9932CC",  # Dark Orchid
            "#BA55D3",  # Medium Orchid
            "#FF8C00",  # Dark Orange
            "#D2691E",  # Chocolate
            "#B8860B",  # Dark Goldenrod
            "#FF69B4",  # Hot Pink
            "#DB7093",  # Pale Violet Red
            "#C71585",  # Medium Violet Red
            "#DC143C",  # Crimson
            "#B22222",  # Firebrick
            "#808080",  # Grey
            "#D3D3D3",  # Light Grey
        ],
        "runs_names": [], # This will be filled with group names
        "zoom_in": False,
    }

    save_plots_folder_path = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/multiTask_scripts_plus_summaries/hypernet_MTCR_PCG_rss_7_0" # Specify the folder path where you want to save the plots
    if not os.path.exists(save_plots_folder_path):
        os.makedirs(save_plots_folder_path)

    dfs = {}
    trajectories_dfs = {}
    labels = {} # This will now store labels for individual seeds within a group
    env_infos = {} # Store env_info for each group - collect all runs' env_info
    list_of_tasks = []
    robot_names_per_task = {} # To store robot name for each task

    for group_data in list_of_grouped_csv_data:
        group_name = group_data["group_name"]
        task_name = group_data["task_name"]
        robot_name = group_data["robot_name"]
        plot_cfg["runs_names"].append(group_name)

        # Add task and robot to their respective lists/dicts
        if task_name not in list_of_tasks:
            list_of_tasks.append(task_name)
        robot_names_per_task[task_name] = robot_name # Assuming robot_name is consistent per task

        dfs[group_name] = []
        trajectories_dfs[group_name] = []
        labels[group_name] = []
        env_infos[group_name] = []  # Store env_info for each run in this group

        # Loop through individual runs (seeds) within this group
        for run_info in group_data["runs"]:
            try:
                # Load metrics CSV
                metrics_file_path = run_info["metrics_csv"]
                df = pd.read_csv(metrics_file_path)
                dfs[group_name].append(df)
                labels[group_name].append(f"{group_name} - {os.path.basename(os.path.dirname(os.path.dirname(metrics_file_path)))}") # Label with group name and folder name
                
                # Load trajectories CSV
                trajectories_file_path = run_info["trajectories_csv"]
                if os.path.exists(trajectories_file_path):
                    trajectories_df = pd.read_csv(trajectories_file_path)
                    trajectories_dfs[group_name].append(trajectories_df)
                    # pass # iNGORE TRAJECTORIES FOR NOW
                else:
                    print(f"Warning: No trajectories file found at {trajectories_file_path} for group {group_name}")
                    trajectories_dfs[group_name].append(pd.DataFrame()) # Empty DataFrame as fallback

                # Load env info for each run
                env_info_file_path = run_info["env_info_yaml"]
                if os.path.exists(env_info_file_path):
                    with open(env_info_file_path, 'r') as f:
                        env_info_data = yaml.safe_load(f)
                        env_infos[group_name].append(env_info_data)
                else:
                    print(f"Warning: No env_info file found at {env_info_file_path} for group {group_name}")
                    env_infos[group_name].append({})  # Empty dict as fallback

            except Exception as e:
                print(f"Error reading file for run in group {group_name}: {e}")
                exit(0)

    # Save env_infos as YAML file - structure it properly for readability
    env_infos_yaml_path = os.path.join(save_plots_folder_path, "env_infos.yaml")
    
    # Create a more structured format for saving
    structured_env_infos = {}
    for group_name, env_info_list in env_infos.items():
        structured_env_infos[group_name] = {}
        for i, env_info in enumerate(env_info_list):
            structured_env_infos[group_name][f"run_{i+1}"] = env_info
    
    with open(env_infos_yaml_path, 'w') as f:
        yaml.dump(structured_env_infos, f, default_flow_style=False, allow_unicode=True)
    print(f"Environment information saved to: {env_infos_yaml_path}")

    # Group by task for plotting
    task_dfs = {}
    task_trajectories_dfs = {}
    task_labels = {}
    task_env_infos = {}

    for task in set(list_of_tasks): # Use set to get unique task names
        task_dfs[task] = {}
        task_trajectories_dfs[task] = {}
        task_labels[task] = {}
        task_env_infos[task] = {}

        for group_name, group_dfs in dfs.items():
            # Check if this group belongs to the current task
            # This assumes that group_name itself somehow implies the task,
            # or you might need a more explicit mapping if not.
            # For this example, we'll use the task_name stored in `list_of_grouped_csv_data`
            # and access it via the `group_data` structure.
            found_group_task_name = None
            for g_data in list_of_grouped_csv_data:
                if g_data["group_name"] == group_name:
                    found_group_task_name = g_data["task_name"]
                    break

            if found_group_task_name == task:
                task_dfs[task][group_name] = group_dfs
                task_trajectories_dfs[task][group_name] = trajectories_dfs[group_name]
                task_labels[task][group_name] = labels[group_name]
                task_env_infos[task][group_name] = env_infos.get(group_name, []) # Get list of env_info for this group

    # Create plots for each task
    for task_name in task_dfs:
        # Create a combined plot_cfg for TaskPlotsFactory if needed
        # For now, we pass the general plot_cfg. The TaskPlotsFactory should handle grouping internally.

        task_plots_factory = TaskPlotsFactory.create(
            task_name,
            dfs=task_dfs[task_name],
            trajectories_dfs=task_trajectories_dfs[task_name],
            labels=task_labels[task_name], # Pass the labels for each group
            env_info=task_env_infos[task_name], # Pass environment info per group (if multiple exist)
            folder_path=save_plots_folder_path,
            plot_cfg=plot_cfg,
        )
        task_plots_factory.plot()

        current_robot_name = robot_names_per_task.get(task_name)
        if current_robot_name:
            robot_plots_factory = RobotPlotsFactory.create(
                current_robot_name,
                dfs=task_dfs[task_name],
                trajectories_dfs=task_trajectories_dfs[task_name],
                labels=task_labels[task_name],
                env_info=task_env_infos[task_name],
                folder_path=save_plots_folder_path,
                plot_cfg=plot_cfg,
            )
            robot_plots_factory.plot()
        else:
            print(f"Warning: Could not determine robot name for task {task_name}. Skipping robot plots.")


if __name__ == "__main__":
    main()