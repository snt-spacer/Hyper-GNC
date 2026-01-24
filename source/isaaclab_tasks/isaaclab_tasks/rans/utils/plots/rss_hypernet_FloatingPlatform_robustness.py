import torch
from tasks import TaskPlotsFactory
from robots import RobotPlotsFactory

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import yaml

def main():
    base = "/workspace/isaaclab/logs/rsl_rl/BASELINE_NO_RANDOMIZATION/2026-01-23_09-03-43_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1"
    com_1 = '/workspace/isaaclab/logs/rsl_rl/MASS_ONLY_D0.05/2026-01-21_20-12-20_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1'
    com_2 = '/workspace/isaaclab/logs/rsl_rl/MASS_ONLY_D0.1/2026-01-21_21-34-00_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1'
    com_3 = '/workspace/isaaclab/logs/rsl_rl/MASS_ONLY_D0.15/2026-01-21_22-55-28_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1'
    com_4 = '/workspace/isaaclab/logs/rsl_rl/MASS_ONLY_D0.2/2026-01-22_00-17-19_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1'
    
    mass_1 = '/workspace/isaaclab/logs/rsl_rl/COM_ONLY_D0.05/2026-01-22_01-38-41_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1'
    mass_2 = '/workspace/isaaclab/logs/rsl_rl/COM_ONLY_D0.1/2026-01-22_03-02-50_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1'
    mass_3 = '/workspace/isaaclab/logs/rsl_rl/COM_ONLY_D0.15/2026-01-22_04-26-43_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1'
    mass_4 = '/workspace/isaaclab/logs/rsl_rl/COM_ONLY_D0.2/2026-01-22_05-50-28_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1'
    
    wrench_1 = '/workspace/isaaclab/logs/rsl_rl/WRENCH_ONLY_W0.1/2026-01-22_07-16-25_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1'
    wrench_2 = '/workspace/isaaclab/logs/rsl_rl/WRENCH_ONLY_W0.2/2026-01-22_09-18-44_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1'
    wrench_3 = '/workspace/isaaclab/logs/rsl_rl/WRENCH_ONLY_W0.3/2026-01-22_11-22-34_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1'
    wrench_4 = '/workspace/isaaclab/logs/rsl_rl/WRENCH_ONLY_W0.4/2026-01-22_13-25-33_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1'
    
    all_1 = '/workspace/isaaclab/logs/rsl_rl/ALL_ACTIVE_D0.05_W0.1/2026-01-22_15-30-44_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1'
    all_2 = '/workspace/isaaclab/logs/rsl_rl/ALL_ACTIVE_D0.1_W0.2/2026-01-22_17-37-03_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1'
    all_3 = '/workspace/isaaclab/logs/rsl_rl/ALL_ACTIVE_D0.15_W0.3/2026-01-22_19-45-49_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1'
    all_4 = '/workspace/isaaclab/logs/rsl_rl/ALL_ACTIVE_D0.2_W0.4/2026-01-22_21-54-52_rsl-rl_ppo-memory_GoToPose-TrackVelocities-Rendezvous-GoToPositionWithObstacles_FloatingPlatform_r-0_seed-1'

    tasks = ['GoToPose', 'TrackVelocities', 'Rendezvous', 'GoToPositionWithObstacles']
    randomizations = [com_1, com_2, com_3, com_4,
                      mass_1, mass_2, mass_3, mass_4,
                      wrench_1, wrench_2, wrench_3, wrench_4,
                      all_1, all_2, all_3, all_4
                    ]
    
    
    dict_results = {}
    
        
    for randomization in randomizations:
        rand_score = 0
        for task in tasks:
            base_date = base.split('/')[6].split("_")[0]
            base_time = base.split('/')[6].split("_")[1]
            metrics_file_name = f"{base_date}_{base_time}_rsl-rl_ppo-memory_{task}_FloatingPlatform_r-0_seed-1_metrics.csv"
            base_path_to_metrics = os.path.join(base, "metrics", metrics_file_name)
            base_df = pd.read_csv(base_path_to_metrics).dropna()
            
            rand_date = randomization.split('/')[6].split("_")[0]
            rand_time = randomization.split('/')[6].split("_")[1]
            metrics_file_name = f"{rand_date}_{rand_time}_rsl-rl_ppo-memory_{task}_FloatingPlatform_r-0_seed-1_metrics.csv"
            path_to_metrics = os.path.join(randomization, "metrics", metrics_file_name)
            
            if os.path.exists(base_path_to_metrics) and os.path.exists(path_to_metrics):
                rand_df = pd.read_csv(path_to_metrics).dropna()
                
                if task == 'GoToPose':
                    base_pose_metric_1 = base_df["final_position_distance.m"].mean()
                    base_pose_metric_2 = base_df["final_orientation_error.rad"].mean()
                    pose_metric_1 = rand_df["final_position_distance.m"].mean()
                    pose_metric_2 = rand_df["final_orientation_error.rad"].mean()
                    pose_score = ((pose_metric_1 / base_pose_metric_1) + (pose_metric_2 / base_pose_metric_2)) / 2
                    rand_score += pose_score
                    
                    print(f"Randomization: {randomization.split('/')[5]}, Task: {task}, Pose Score: {pose_score}, Rand Score: {rand_score}")

                elif task == 'TrackVelocities':
                    base_track_metric_1 = base_df["linear_velocity_error.m/s"].mean()
                    base_track_metric_2 = base_df["lateral_velocity_error.m/s"].mean()
                    base_track_metric_3 = base_df["angular_velocity_error.m/s"].mean()
                    track_metric_1 = rand_df["linear_velocity_error.m/s"].mean()
                    track_metric_2 = rand_df["lateral_velocity_error.m/s"].mean()
                    track_metric_3 = rand_df["angular_velocity_error.m/s"].mean()
                    track_score = ((track_metric_1 / base_track_metric_1) + (track_metric_2 / base_track_metric_2) + (track_metric_3 / base_track_metric_3)) / 3
                    rand_score += track_score
                    print(f"Randomization: {randomization.split('/')[5]}, Task: {task}, Track Score: {track_score}, Rand Score: {rand_score}")
                    
                    
                elif task == 'Rendezvous':
                    base_rend_metric_1 = base_df["orientation_error.rad"].mean()
                    rend_metric_1 = rand_df["orientation_error.rad"].mean()
                    base_rend_metric_2 = base_df["success_rate.u"].mean()
                    rend_metric_2 = rand_df["success_rate.u"].mean()
                    rend_score = (rend_metric_1 / base_rend_metric_1) + (rend_metric_2 / base_rend_metric_2) / 2
                    rand_score += rend_score
                    print(f"Randomization: {randomization.split('/')[5]}, Task: {task}, Rend Score: {rend_score}, Rand Score: {rand_score}")
                    
                elif task == 'GoToPositionWithObstacles':
                    base_posi_metric_1 = base_df["final_position_distance.m"].mean()
                    posi_metric_1 = rand_df["final_position_distance.m"].mean()
                    posi_score = (posi_metric_1 / base_posi_metric_1)
                    rand_score += posi_score
                    print(f"Randomization: {randomization.split('/')[5]}, Task: {task}, Pos Score: {posi_score}, Rand Score: {rand_score}")
                    print(base_posi_metric_1, posi_metric_1, posi_score)
                    # breakpoint()

            else:
                print(f"Metrics file not found for {task} in either base or randomized path.")
                    
        dict_results[randomization.split('/')[5]] = rand_score / 4  # Average over 4 tasks
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)

    
    print(dict_results)

    # Define groupings and their labels
    groups = [
        ('COM_ONLY', 'CoM\noffset'),
        ('MASS_ONLY', 'Mass\noffset'),
        ('WRENCH_ONLY', 'Wrench and\nTorque force'),
        ('ALL_ACTIVE', 'All')
    ]

    for i, (prefix, label) in enumerate(groups):
        # Filter dictionary for relevant keys and extract X/Y values
        subset = {k: v for k, v in dict_results.items() if k.startswith(prefix)}
        
        # Extract numbers from keys for X-axis (e.g., '0.05' from 'COM_ONLY_D0.05')
        # For 'All', we'll just use a simple 1, 2, 3, 4 index to match your image style
        if prefix == 'ALL_ACTIVE':
            x_vals = [1, 2, 3, 4]
        else:
            x_vals = [float(k.split('_')[-1][1:]) for k in subset.keys()]
        
        y_vals = list(subset.values())

        # Plot on the specific subplot
        axes[i].plot(x_vals, y_vals, marker='o', color='gray', markerfacecolor='white', markersize=10, linewidth=1)
        
        # Formatting
        axes[i].set_xlabel(label, fontsize=14)
        # axes[i].set_ylim(0, 2)
        axes[i].set_facecolor('#f0f0f0') # Light gray background like the image
        axes[i].tick_params(axis='both', which='major', labelsize=10)

    # Label the first Y axis
    axes[0].set_ylabel('Floating P.', fontsize=16)

    plt.tight_layout()
    plt.savefig("/workspace/isaaclab/source/rss_hypernet_FloatingPlatform_robustness.png", dpi=300)
        
        
        

if __name__ == "__main__":
    main()