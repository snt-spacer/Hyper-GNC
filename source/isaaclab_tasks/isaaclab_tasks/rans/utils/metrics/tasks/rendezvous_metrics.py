from . import BaseTaskMetrics, Registerable
import torch

class RendezvousMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name: str, task_index: int = 0) -> None:
        super().__init__(env=env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, task_name=task_name, task_index=task_index)

    @BaseTaskMetrics.register
    def orientation_error(self):
        print("[INFO][METRICS][TASK][Rendezvous] Orientation error")
        
        masked_error = self.trajectories['orientation_error'] * self.trajectories_masks
        sum_masks = torch.sum(self.trajectories_masks, dim=1)
        # Prevent division by zero
        sum_masks[sum_masks == 0] = 1.0
        
        avg_error = torch.sum(masked_error, dim=1) / sum_masks
        
        self.metrics["orientation_error.rad"] = avg_error

    @BaseTaskMetrics.register
    def success_rate(self):
        print("[INFO][METRICS][TASK][Rendezvous] Success rate")
        
        masked_completed = self.trajectories['trajectory_completed'].float() * self.trajectories_masks
        is_completed = torch.max(masked_completed, dim=1).values
        
        self.metrics["success_rate.u"] = is_completed
        
        
    # @BaseTaskMetrics.register
    # def avg_orientation_error(self):
    #     """
    #     Calculate the average orientation error for each trajectory.
        
    #     This function computes the mean orientation error between the robot's actual heading
    #     and the target heading for each segment of the trajectory (between waypoints/goals).
    #     """
    #     print("[INFO][METRICS][TASK] Average orientation error")
    #     mean_avg_angle_errors = []  # Store mean orientation error for each trajectory
    #     failed_trajectories = 0     # Count trajectories that didn't complete properly
        
    #     # Process each trajectory individually
    #     for traj_idx, trajectory_target_index in enumerate(self.trajectories["target_index"]):
    #         # Find the index where the robot exceeded the maximum number of goals
    #         # This determines where to cut the trajectory (end of valid data)
    #         cut_idx = torch.where(trajectory_target_index >= self.trajectories["num_goals"][traj_idx, 0])[0]

    #         # Handle multiple cut indices by taking the first one
    #         if cut_idx.numel() > 1:
    #             cut_idx = cut_idx[0].unsqueeze(-1)  # Use the first index if there are multiple 
            
    #         # If no cut index found, trajectory failed to complete
    #         if cut_idx.numel() == 0:
    #             failed_trajectories += 1
    #             continue  # Skip this trajectory and mark as failed

    #         # Find where the target index increments (robot reaches new waypoints)
    #         # These increments indicate transitions between trajectory segments
    #         diffs = torch.diff(trajectory_target_index[:cut_idx.item()])
    #         increment_indices = torch.where(diffs > 0)[0] + 1  # Add 1 to get the actual increment position
    #         increment_indices = torch.cat([increment_indices, cut_idx])  # Include the final cut index

    #         # Split the heading data into segments based on waypoint transitions
    #         # Each segment represents the robot's path between two consecutive waypoints
    #         headings = torch.tensor_split(self.trajectories["heading"][traj_idx].cpu(), increment_indices.cpu())
    #         target_headings = self.trajectories["target_headings"][traj_idx]

    #         # Calculate orientation error for each segment
    #         avg_angle_errors = []
    #         for segment_idx, (heading, target_heading) in enumerate(zip(headings, target_headings)):
    #             # Calculate heading error: difference between target and actual heading
    #             # Repeat target heading to match the length of the segment
    #             heading_error = target_heading[segment_idx].repeat(heading.shape[0]).cpu() - heading
                
    #             # Normalize angle error to [-π, π] range and take absolute value
    #             # arctan2(sin(θ), cos(θ)) ensures proper angle wrapping
    #             angle_error = torch.abs(torch.arctan2(torch.sin(heading_error), torch.cos(heading_error)))
                
    #             # Calculate mean error for this segment
    #             avg_angle_errors.append(torch.mean(angle_error.float()))
            
    #         # Calculate overall mean orientation error for this trajectory
    #         mean_avg_angle_errors.append(torch.mean(torch.stack(avg_angle_errors)))

    #     # Add NaN values for failed trajectories to maintain consistent array size
    #     for i in range(failed_trajectories):
    #         mean_avg_angle_errors.append(torch.tensor(float('nan'), device=self.trajectories["target_index"].device))

    #     # Store results in metrics dictionary
    #     self.metrics["mean_orientation_error.rad"] = torch.tensor(mean_avg_angle_errors, device=self.trajectories["target_index"].device)
    #     self.env_info["failed_trajectories"] = failed_trajectories

        

    # @BaseTaskMetrics.register
    # def sr_num_gates(self):
    #     num_successful_trajectories = torch.sum(self.trajectories['total_goals_reached'][:,-1] >= self.trajectories['num_goals'][:, 0])
    #     self.env_info["sr_num_gates"] = 100 * (num_successful_trajectories / self.trajectories['num_goals'].shape[0]).item()

    # @BaseTaskMetrics.register
    # def avg_orientation_error_path(self):
    #     """Calculate the average orientation error along the path after the robot arrived to the first marker."""
    #     print("[INFO][METRICS][TASK] Average orientation error path")
    #     mean_avg_angle_errors = []
    #     failed_trajectories = 0
    #     for traj_idx, trajectory_target_index in enumerate(self.trajectories["target_index"]):
    #         cut_idx = torch.where(trajectory_target_index >= self.trajectories["num_goals"][traj_idx, 0])[0]

    #         if cut_idx.numel() > 1:
    #             cut_idx = cut_idx[0].unsqueeze(-1)  # Use the first index if there are multiple 
    #         if cut_idx.numel() == 0:
    #             failed_trajectories += 1
    #             continue  # Skip if no index found

    #         diffs = torch.diff(trajectory_target_index[:cut_idx.item()])
    #         increment_indices = torch.where(diffs > 0)[0] + 1
    #         first_increment = increment_indices[0]
    #         increment_indices = increment_indices[1:] - first_increment
    #         increment_indices = torch.cat([increment_indices, cut_idx - first_increment])

    #         headings = torch.tensor_split(self.trajectories["heading"][traj_idx][first_increment:].cpu(), increment_indices.cpu())
    #         target_headings = self.trajectories["target_headings"][traj_idx][increment_indices[0]:]

    #         avg_angle_errors = []
    #         for segment_idx, (heading, target_heading) in enumerate(zip(headings, target_headings)):
    #             heading_error = target_heading[segment_idx].repeat(heading.shape[0]).cpu() - heading
    #             angle_error = torch.abs(torch.arctan2(torch.sin(heading_error), torch.cos(heading_error)))  # Normalize the angle error to [-pi, pi] and take the absolute value
    #             avg_angle_errors.append(torch.mean(angle_error.float()))

    #         if torch.any(torch.isnan(torch.stack(avg_angle_errors))):
    #             breakpoint()
    #         mean_avg_angle_errors.append(torch.mean(torch.stack(avg_angle_errors)))

    #     for i in range(failed_trajectories):
    #         mean_avg_angle_errors.append(torch.tensor(float('nan'), device=self.trajectories["target_index"].device))

    #     self.metrics["mean_orientation_error_path.rad"] = torch.tensor(mean_avg_angle_errors, device=self.trajectories["target_index"].device)