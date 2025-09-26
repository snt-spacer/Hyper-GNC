#!/usr/bin/env python3
"""
Script to plot linear velocities (x, y) and angular velocity (z) over time for a single trajectory
from the extracted_trajectories_GoToPosition.csv file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot_trajectory_velocities(csv_file, trajectory_id=1, save_plot=False, output_file=None):
    """
    Plot linear velocities and angular velocity over time for a single trajectory.
    
    Args:
        csv_file (str): Path to the CSV file
        trajectory_id (int): ID of the trajectory to plot (default: 1)
        save_plot (bool): Whether to save the plot to file
        output_file (str): Output filename for the plot (optional)
    """
    
    # Read the CSV file
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Filter data for the specified trajectory
    traj_data = df[df['trajectory'] == trajectory_id]
    
    if traj_data.empty:
        print(f"No data found for trajectory {trajectory_id}")
        available_trajectories = df['trajectory'].unique()
        print(f"Available trajectories: {sorted(available_trajectories)}")
        return
    
    print(f"Found {len(traj_data)} time steps for trajectory {trajectory_id}")
    
    # Extract time steps and velocities
    time_steps = traj_data['step'][:80].values
    linear_vel_x = traj_data['linear_velocity_x'].values
    linear_vel_y = traj_data['linear_velocity_y'].values
    angular_vel_z = traj_data['angular_velocity_z'].values
    
    # Create the plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Velocity Profiles for Trajectory {trajectory_id}', fontsize=16, fontweight='bold')
    
    # Plot linear velocity x
    axes[0].plot(time_steps, linear_vel_x, 'b-', linewidth=2, label='Linear Velocity X')
    axes[0].set_ylabel('Linear Velocity X (m/s)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title('Linear Velocity in X Direction')
    
    # Plot linear velocity y
    axes[1].plot(time_steps, linear_vel_y, 'g-', linewidth=2, label='Linear Velocity Y')
    axes[1].set_ylabel('Linear Velocity Y (m/s)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_title('Linear Velocity in Y Direction')
    
    # Plot angular velocity z
    axes[2].plot(time_steps, angular_vel_z, 'r-', linewidth=2, label='Angular Velocity Z')
    axes[2].set_ylabel('Angular Velocity Z (rad/s)', fontsize=12)
    axes[2].set_xlabel('Time Step', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_title('Angular Velocity about Z Axis')
    
    # Adjust layout
    plt.tight_layout()
    
    # Print some statistics
    print(f"\nVelocity Statistics for Trajectory {trajectory_id}:")
    print(f"Linear Velocity X - Mean: {np.mean(linear_vel_x):.4f}, Std: {np.std(linear_vel_x):.4f}, Range: [{np.min(linear_vel_x):.4f}, {np.max(linear_vel_x):.4f}]")
    print(f"Linear Velocity Y - Mean: {np.mean(linear_vel_y):.4f}, Std: {np.std(linear_vel_y):.4f}, Range: [{np.min(linear_vel_y):.4f}, {np.max(linear_vel_y):.4f}]")
    print(f"Angular Velocity Z - Mean: {np.mean(angular_vel_z):.4f}, Std: {np.std(angular_vel_z):.4f}, Range: [{np.min(angular_vel_z):.4f}, {np.max(angular_vel_z):.4f}]")
    
    # Save or show plot
    if save_plot:
        if output_file is None:
            output_file = f'trajectory_{trajectory_id}_velocities.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {output_file}")
    else:
        plt.show()


def plot_multiple_trajectories_comparison(csv_file, trajectory_ids=[1, 2, 3], save_plot=False):
    """
    Plot velocity comparisons for multiple trajectories.
    
    Args:
        csv_file (str): Path to the CSV file
        trajectory_ids (list): List of trajectory IDs to compare
        save_plot (bool): Whether to save the plot to file
    """
    
    # Read the CSV file
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Create the plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Velocity Profiles Comparison Across Trajectories', fontsize=16, fontweight='bold')
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for i, traj_id in enumerate(trajectory_ids):
        traj_data = df[df['trajectory'] == traj_id]
        
        if traj_data.empty:
            print(f"No data found for trajectory {traj_id}")
            continue
            
        time_steps = traj_data['step'].values
        linear_vel_x = traj_data['linear_velocity_x'].values
        linear_vel_y = traj_data['linear_velocity_y'].values
        angular_vel_z = traj_data['angular_velocity_z'].values
        
        color = colors[i % len(colors)]
        
        # Plot linear velocity x
        axes[0].plot(time_steps, linear_vel_x, color=color, linewidth=2, 
                    label=f'Trajectory {traj_id}', alpha=0.8)
        
        # Plot linear velocity y
        axes[1].plot(time_steps, linear_vel_y, color=color, linewidth=2, 
                    label=f'Trajectory {traj_id}', alpha=0.8)
        
        # Plot angular velocity z
        axes[2].plot(time_steps, angular_vel_z, color=color, linewidth=2, 
                    label=f'Trajectory {traj_id}', alpha=0.8)
    
    # Configure axes
    axes[0].set_ylabel('Linear Velocity X (m/s)', fontsize=12)
    axes[0].set_title('Linear Velocity in X Direction')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_ylabel('Linear Velocity Y (m/s)', fontsize=12)
    axes[1].set_title('Linear Velocity in Y Direction')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].set_ylabel('Angular Velocity Z (rad/s)', fontsize=12)
    axes[2].set_xlabel('Time Step', fontsize=12)
    axes[2].set_title('Angular Velocity about Z Axis')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_plot:
        output_file = f'trajectories_comparison_velocities.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved as {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot velocity profiles from trajectory data')
    parser.add_argument('--csv_file', default='extracted_trajectories_GoToPosition.csv',
                       help='Path to the CSV file (default: extracted_trajectories_GoToPosition.csv)')
    parser.add_argument('--trajectory_id', type=int, default=1,
                       help='Trajectory ID to plot (default: 1)')
    parser.add_argument('--save', action='store_true',
                       help='Save the plot instead of displaying it')
    parser.add_argument('--output', type=str,
                       help='Output filename for the plot')
    parser.add_argument('--compare', nargs='+', type=int,
                       help='Compare multiple trajectories (provide list of trajectory IDs)')
    
    args = parser.parse_args()
    
    if args.compare:
        plot_multiple_trajectories_comparison(args.csv_file, args.compare, args.save)
    else:
        plot_trajectory_velocities(args.csv_file, args.trajectory_id, args.save, args.output)


if __name__ == "__main__":
    main()
