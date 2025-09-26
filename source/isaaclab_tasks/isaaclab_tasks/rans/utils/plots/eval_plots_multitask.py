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
        # Expert L 32x32 w Tinf in Gobs (tanh C 256)
        {
            "group_name": "Expert L 32x32 w Tinf in Gobs (tanh C 256) GoToPosition",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_19-03-53_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1/metrics/2025-08-31_19-03-53_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_19-03-53_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_19-03-53_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_19-30-16_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2/metrics/2025-08-31_19-30-16_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_19-30-16_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_19-30-16_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_19-56-54_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3/metrics/2025-08-31_19-56-54_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_19-56-54_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_19-56-54_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_20-23-05_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4/metrics/2025-08-31_20-23-05_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_20-23-05_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_20-23-05_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_20-49-18_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5/metrics/2025-08-31_20-49-18_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_20-49-18_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_20-49-18_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Expert L 32x32 w Tinf in Gobs (tanh C 256) GoToPose",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_21-15-35_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1/metrics/2025-08-31_21-15-35_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_21-15-35_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_21-15-35_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                }, #5
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_21-42-12_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2/metrics/2025-08-31_21-42-12_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_21-42-12_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_21-42-12_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },#22
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_22-08-52_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3/metrics/2025-08-31_22-08-52_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_22-08-52_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_22-08-52_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },#0.5
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_22-35-35_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4/metrics/2025-08-31_22-35-35_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_22-35-35_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_22-35-35_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },# 5.7
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_23-01-53_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5/metrics/2025-08-31_23-01-53_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_23-01-53_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_23-01-53_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }#6
            ]
        },
        {
            "group_name": "Expert L 32x32 w Tinf in Gobs (tanh C 256) TrackVelocities",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_23-27-59_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1/metrics/2025-08-31_23-27-59_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_23-27-59_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_23-27-59_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_23-55-49_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2/metrics/2025-08-31_23-55-49_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_23-55-49_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_23-55-49_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-09-01_00-23-46_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3/metrics/2025-09-01_00-23-46_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-09-01_00-23-46_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-09-01_00-23-46_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-09-01_00-51-20_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4/metrics/2025-09-01_00-51-20_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-09-01_00-51-20_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-09-01_00-51-20_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-09-01_01-18-58_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5/metrics/2025-09-01_01-18-58_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-09-01_01-18-58_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-09-01_01-18-58_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Expert L 32x32 w Tinf in Gobs (tanh C 256) Rendezvous",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_16-39-04_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-31_16-39-04_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_16-39-04_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_16-39-04_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_17-08-03_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-31_17-08-03_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_17-08-03_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_17-08-03_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_17-37-20_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-31_17-37-20_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_17-37-20_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_17-37-20_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_18-06-26_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-31_18-06-26_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_18-06-26_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_18-06-26_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_18-35-25_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-31_18-35-25_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_18-35-25_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_18-35-25_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },

        # Baseline L 32x32 w Tinf in Gobs (tanh C 256)
        {
            "group_name": "Baseline L 32x32 w Tinf in Gobs (tanh C 256) GoToPosition",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_09-27-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-31_09-27-03_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_09-27-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_09-27-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_10-20-58_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-31_10-20-58_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_10-20-58_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_10-20-58_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_11-15-40_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-31_11-15-40_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_11-15-40_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_11-15-40_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_12-10-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-31_12-10-00_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_12-10-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_12-10-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_13-05-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-31_13-05-04_rsl-rl_ppo_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_13-05-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_13-05-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Baseline L 32x32 w Tinf in Gobs (tanh C 256) GoToPose",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_09-27-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-31_09-27-03_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_09-27-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_09-27-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_10-20-58_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-31_10-20-58_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_10-20-58_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_10-20-58_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_11-15-40_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-31_11-15-40_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_11-15-40_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_11-15-40_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_12-10-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-31_12-10-00_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_12-10-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_12-10-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_13-05-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-31_13-05-04_rsl-rl_ppo_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_13-05-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_13-05-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Baseline L 32x32 w Tinf in Gobs (tanh C 256) TrackVelocities",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_09-27-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-31_09-27-03_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_09-27-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_09-27-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_10-20-58_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-31_10-20-58_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_10-20-58_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_10-20-58_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_11-15-40_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-31_11-15-40_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_11-15-40_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_11-15-40_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_12-10-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-31_12-10-00_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_12-10-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_12-10-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_13-05-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-31_13-05-04_rsl-rl_ppo_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_13-05-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_13-05-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Baseline L 32x32 w Tinf in Gobs (tanh C 256) Rendezvous",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_09-27-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-31_09-27-03_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_09-27-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_09-27-03_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_10-20-58_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-31_10-20-58_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_10-20-58_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_10-20-58_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_11-15-40_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-31_11-15-40_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_11-15-40_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_11-15-40_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_12-10-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-31_12-10-00_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_12-10-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_12-10-00_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_13-05-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-31_13-05-04_rsl-rl_ppo_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_13-05-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-31_13-05-04_rsl-rl_ppo_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },

        # Hypernet Critic L Emb 64x64 w Tinf in Gobs (tanh, 256C) 
        {
            "group_name": "Hypernet Critic L Emb 64x64 w Tinf in Gobs (tanh, 256C) GoToPosition",
            "task_name": "GoToPosition",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_18-31-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-30_18-31-29_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_18-31-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_18-31-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_19-30-27_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-30_19-30-27_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_19-30-27_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_19-30-27_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_20-28-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-30_20-28-46_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_20-28-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_20-28-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_21-27-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-30_21-27-06_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_21-27-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_21-27-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_22-25-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-30_22-25-46_rsl-rl_ppo-memory_GoToPosition_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_22-25-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPosition.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_22-25-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Critic L Emb 64x64 w Tinf in Gobs (tanh, 256C) GoToPose",
            "task_name": "GoToPose",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_18-31-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-30_18-31-29_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_18-31-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_18-31-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_19-30-27_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-30_19-30-27_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_19-30-27_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_19-30-27_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_20-28-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-30_20-28-46_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_20-28-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_20-28-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_21-27-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-30_21-27-06_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_21-27-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_21-27-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_22-25-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-30_22-25-46_rsl-rl_ppo-memory_GoToPose_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_22-25-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_GoToPose.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_22-25-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Critic L Emb 64x64 w Tinf in Gobs (tanh, 256C) TrackVelocities",
            "task_name": "TrackVelocities",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_18-31-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-30_18-31-29_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_18-31-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_18-31-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_19-30-27_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-30_19-30-27_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_19-30-27_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_19-30-27_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_20-28-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-30_20-28-46_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_20-28-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_20-28-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_21-27-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-30_21-27-06_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_21-27-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_21-27-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_22-25-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-30_22-25-46_rsl-rl_ppo-memory_TrackVelocities_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_22-25-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_TrackVelocities.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_22-25-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
                }
            ]
        },
        {
            "group_name": "Hypernet Critic L Emb 64x64 w Tinf in Gobs (tanh, 256C) Rendezvous",
            "task_name": "Rendezvous",
            "robot_name": "ModularFreeflyer",
            "runs": [
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_18-31-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/2025-08-30_18-31-29_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-1_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_18-31-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_18-31-29_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-1/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_19-30-27_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/2025-08-30_19-30-27_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-2_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_19-30-27_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_19-30-27_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-2/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_20-28-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/2025-08-30_20-28-46_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-3_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_20-28-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_20-28-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-3/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_21-27-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/2025-08-30_21-27-06_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-4_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_21-27-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_21-27-06_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-4/metrics/env_info.yaml"
                },
                {
                    "metrics_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_22-25-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/2025-08-30_22-25-46_rsl-rl_ppo-memory_Rendezvous_ModularFreeflyer_r-0_seed-5_metrics.csv",
                    "trajectories_csv": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_22-25-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/extracted_trajectories_Rendezvous.csv",
                    "env_info_yaml": "/workspace/isaaclab/logs/rsl_rl/multitask_memory_control/2025-08-30_22-25-46_rsl-rl_ppo-memory_GoToPosition-GoToPose-TrackVelocities-Rendezvous_ModularFreeflyer_r-0_seed-5/metrics/env_info.yaml"
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

    save_plots_folder_path = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/rans/utils/multiTask_scripts_plus_summaries/plots_vani6" # Specify the folder path where you want to save the plots
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