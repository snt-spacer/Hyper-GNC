from . import BaseTaskMetrics, Registerable
import torch

class GoToPose3DMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name: str, task_index: int = 0) -> None:
        super().__init__(env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, task_name=task_name, task_index=task_index)

    @BaseTaskMetrics.register
    def final_position_distance(self):
        """ Difference between the target position and the final position of the robot. """
        print("[INFO][METRICS][TASK][GoToPose6DoF] Final position delta")

        masked_distances = self.trajectories['position_dist'] * self.trajectories_masks
        final_position_delta = masked_distances[torch.arange(0, masked_distances.shape[0], device=masked_distances.device), self.last_true_index]
        self.metrics["final_position_distance.m"] = final_position_delta

    @BaseTaskMetrics.register
    def final_orientation_error(self):
        """ Difference between the target heading and the final heading of the robot. """
        print("[INFO][METRICS][TASK][GoToPose6DoF] Final orientation error")

        # masked_target_heading = self.trajectories['target_heading'] * self.trajectories_masks
        # masked_headings = self.trajectories['heading'] * self.trajectories_masks

        # final_target_heading = masked_target_heading[torch.arange(0, masked_target_heading.shape[0], device=masked_target_heading.device), self.last_true_index]
        # final_heading = masked_headings[torch.arange(0, masked_headings.shape[0], device=masked_headings.device), self.last_true_index]

        # heading_error = final_target_heading - final_heading
        # angle_error = torch.abs(torch.arctan2(torch.sin(heading_error), torch.cos(heading_error)))  # Normalize the angle error to [-pi, pi] and take the absolute value

        # self.metrics["final_orientation_error.rad"] = angle_error

        masked_orientation_error = self.trajectories['orientation_error'] * self.trajectories_masks
        final_orientation_error = masked_orientation_error[torch.arange(0, masked_orientation_error.shape[0], device=masked_orientation_error.device), self.last_true_index]
        self.metrics["final_orientation_error.rad"] = final_orientation_error