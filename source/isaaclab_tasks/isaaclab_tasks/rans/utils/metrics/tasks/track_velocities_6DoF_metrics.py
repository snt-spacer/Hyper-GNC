from . import BaseTaskMetrics, Registerable
import torch

class TrackVelocities3DMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name: str, task_index: int = 0) -> None:
        super().__init__(env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, task_name=task_name, task_index=task_index)

    @BaseTaskMetrics.register
    def track_velocity_error(self):
        print("[INFO][METRICS][TASK][TrackVelocities6DoF] Track velocity error")
        masked_lin_vel_error = self.trajectories['error_linear_velocity'] * self.trajectories_masks
        masked_lat_vel_error = self.trajectories['error_lateral_velocity'] * self.trajectories_masks
        masked_vert_vel_error = self.trajectories['error_vertical_velocity'] * self.trajectories_masks
        masked_yaw_vel_error = self.trajectories['error_yaw_velocity'] * self.trajectories_masks
        masked_pitch_vel_error = self.trajectories['error_pitch_velocity'] * self.trajectories_masks
        masked_roll_vel_error = self.trajectories['error_roll_velocity'] * self.trajectories_masks
        

        len_trajec = torch.sum(self.trajectories_masks, dim=1)
        avg_lin_vel_error = torch.sum(torch.abs(masked_lin_vel_error), dim=1) / len_trajec
        avg_lat_vel_error = torch.sum(torch.abs(masked_lat_vel_error), dim=1) / len_trajec
        avg_vert_vel_error = torch.sum(torch.abs(masked_vert_vel_error), dim=1) / len_trajec
        avg_yaw_vel_error = torch.sum(torch.abs(masked_yaw_vel_error), dim=1) / len_trajec
        avg_pitch_vel_error = torch.sum(torch.abs(masked_pitch_vel_error), dim=1) / len_trajec
        avg_roll_vel_error = torch.sum(torch.abs(masked_roll_vel_error), dim=1) / len_trajec

        self.metrics["avg_linear_velocity_error.m/s"] = avg_lin_vel_error
        self.metrics["avg_lateral_velocity_error.m/s"] = avg_lat_vel_error
        self.metrics["avg_vertical_velocity_error.m/s"] = avg_vert_vel_error
        self.metrics["avg_yaw_velocity_error.rad/s"] = avg_yaw_vel_error
        self.metrics["avg_pitch_velocity_error.rad/s"] = avg_pitch_vel_error
        self.metrics["avg_roll_velocity_error.rad/s"] = avg_roll_vel_error