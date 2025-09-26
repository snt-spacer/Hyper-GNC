#!/usr/bin/env python3
"""
Simple example script to plot velocity profiles for trajectory data.
This script demonstrates basic usage of the plotting functionality.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def simple_velocity_plot():
    """
    Simple function to plot linear velocities and angular velocity for all trajectories (first 80 steps only).
    """
    
    # Load the data
    csv_file = 'source/isaaclab_tasks/isaaclab_tasks/rans/utils/multiTask_scripts_plus_summaries/plots_vani5/extracted_trajectories_GoToPosition.csv'
    print(f"Loading data from {csv_file}...")
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found!")
        print("Please make sure the CSV file is in the current directory.")
        return
    
    # Get all available trajectories
    available_trajectories = sorted(df['trajectory'].unique())
    print(f"Found {len(available_trajectories)} trajectories: {available_trajectories}")
    
    # Create the plot with subplots for each velocity component
    plt.figure(figsize=(15, 12))
    
    # Define colors for different trajectories (cycling through a colormap)
    colors = plt.cm.tab20(np.linspace(0, 1, len(available_trajectories)))
    
    # Subplot 1: Linear Velocity X
    plt.subplot(3, 1, 1)
    for i, trajectory_id in enumerate(available_trajectories):
        traj_data = df[df['trajectory'] == trajectory_id]
        if not traj_data.empty:
            # Limit to first 80 steps
            traj_data_limited = traj_data[traj_data['step'] < 80]
            if not traj_data_limited.empty:
                time_steps = traj_data_limited['step'].values
                linear_vel_x = traj_data_limited['linear_velocity_x'].values
                plt.plot(time_steps, linear_vel_x, color=colors[i], linewidth=1.5, alpha=0.7, label=f'Traj {trajectory_id}')
    
    plt.title('Linear Velocity X - All Trajectories (First 80 Steps)', fontsize=14)
    plt.ylabel('Velocity X (m/s)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
    
    # Subplot 2: Linear Velocity Y
    plt.subplot(3, 1, 2)
    for i, trajectory_id in enumerate(available_trajectories):
        traj_data = df[df['trajectory'] == trajectory_id]
        if not traj_data.empty:
            # Limit to first 80 steps
            traj_data_limited = traj_data[traj_data['step'] < 80]
            if not traj_data_limited.empty:
                time_steps = traj_data_limited['step'].values
                linear_vel_y = traj_data_limited['linear_velocity_y'].values
                plt.plot(time_steps, linear_vel_y, color=colors[i], linewidth=1.5, alpha=0.7, label=f'Traj {trajectory_id}')
    
    plt.title('Linear Velocity Y - All Trajectories (First 80 Steps)', fontsize=14)
    plt.ylabel('Velocity Y (m/s)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
    
    # Subplot 3: Angular Velocity Z
    plt.subplot(3, 1, 3)
    for i, trajectory_id in enumerate(available_trajectories):
        traj_data = df[df['trajectory'] == trajectory_id]
        if not traj_data.empty:
            # Limit to first 80 steps
            traj_data_limited = traj_data[traj_data['step'] < 80]
            if not traj_data_limited.empty:
                time_steps = traj_data_limited['step'].values
                angular_vel_z = traj_data_limited['angular_velocity_z'].values
                plt.plot(time_steps, angular_vel_z, color=colors[i], linewidth=1.5, alpha=0.7, label=f'Traj {trajectory_id}')
    
    plt.title('Angular Velocity Z - All Trajectories (First 80 Steps)', fontsize=14)
    plt.ylabel('Angular Velocity Z (rad/s)')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
    
    plt.tight_layout()
    plt.suptitle('Velocity Profiles for All Trajectories (First 80 Steps)', fontsize=16, y=0.98)
    
    # Print some basic statistics for all trajectories combined (first 80 steps only)
    df_limited = df[df['step'] < 80]
    print(f"\nBasic Statistics for All Trajectories (First 80 Steps):")
    print(f"Total data points: {len(df_limited)}")
    print(f"Number of trajectories: {len(available_trajectories)}")
    
    all_linear_vel_x = df_limited['linear_velocity_x'].values
    all_linear_vel_y = df_limited['linear_velocity_y'].values
    all_angular_vel_z = df_limited['angular_velocity_z'].values
    
    print(f"Linear Velocity X: mean={np.mean(all_linear_vel_x):.3f}, max={np.max(all_linear_vel_x):.3f}, min={np.min(all_linear_vel_x):.3f}")
    print(f"Linear Velocity Y: mean={np.mean(all_linear_vel_y):.3f}, max={np.max(all_linear_vel_y):.3f}, min={np.min(all_linear_vel_y):.3f}")
    print(f"Angular Velocity Z: mean={np.mean(all_angular_vel_z):.3f}, max={np.max(all_angular_vel_z):.3f}, min={np.min(all_angular_vel_z):.3f}")
    
    # Show the plot
    # plt.show()
    filename = 'source/all_trajectories_velocities_first80steps.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")


if __name__ == "__main__":
    simple_velocity_plot()
