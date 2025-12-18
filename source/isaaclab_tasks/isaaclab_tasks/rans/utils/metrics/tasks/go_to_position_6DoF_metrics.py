from . import BaseTaskMetrics, Registerable
import torch

class GoToPosition3DMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name: str, task_index: int = 0) -> None:
        super().__init__(env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, task_name=task_name, task_index=task_index)

    @BaseTaskMetrics.register
    def time_to_reach_position_threshold(self):
        print("[INFO][METRICS][TASK] Time to reach position threshold")
        if "MultiTask" in self.env.unwrapped.__class__.__name__:
            threshold = self.env.unwrapped.tasks_apis[self.task_index]._task_cfg.position_tolerance
        else:
            threshold = self.env.unwrapped.task_api._task_cfg.position_tolerance
        masked_distances = self.trajectories['position_distance'] * self.trajectories_masks
        reached_threshold = masked_distances <= threshold
        len_trajec = self.trajectories['position_distance'].shape[1]
        reached_idx = torch.argmax(reached_threshold.int(), dim=1)
        # argmax returns 0 if all values are false
        all_false = ~torch.any(reached_threshold, dim=1)
        reached_idx[all_false] = len_trajec

        episode_length_in_s = reached_idx * self.step_dt

        self.metrics["time_to_reach_position_threshold.s"] = episode_length_in_s

    @BaseTaskMetrics.register
    def final_position_distance(self):
        """ Difference between the target position and the final position of the robot. """
        print("[INFO][METRICS][TASK] Final position delta")
        masked_distances = self.trajectories['position_distance'] * self.trajectories_masks
        final_position_delta = masked_distances[torch.arange(0, masked_distances.shape[0], device=masked_distances.device), self.last_true_index]
        self.metrics["final_position_distance.m"] = final_position_delta
        
    @BaseTaskMetrics.register
    def final_velocity_magnitude(self):
        """ Final linear velocity magnitude of the robot. """
        print("[INFO][METRICS][TASK] Final linear velocity magnitude")
        masked_linear_velocities = self.trajectories['linear_velocity'] * self.trajectories_masks.unsqueeze(-1)
        
        final_linear_velocity = masked_linear_velocities[torch.arange(0, masked_linear_velocities.shape[0], device=masked_linear_velocities.device), self.last_true_index][:, 0]
        final_lateral_velocity = masked_linear_velocities[torch.arange(0, masked_linear_velocities.shape[0], device=masked_linear_velocities.device), self.last_true_index][:, 1]
        final_vertical_velocity = masked_linear_velocities[torch.arange(0, masked_linear_velocities.shape[0], device=masked_linear_velocities.device), self.last_true_index][:, 2]

        masked_angular_velocities = self.trajectories['angular_velocity'] * self.trajectories_masks.unsqueeze(-1)
        final_roll_velocity = masked_angular_velocities[torch.arange(0, masked_angular_velocities.shape[0], device=masked_angular_velocities.device), self.last_true_index][:, 0]
        final_pitch_velocity = masked_angular_velocities[torch.arange(0, masked_angular_velocities.shape[0], device=masked_angular_velocities.device), self.last_true_index][:, 1]
        final_yaw_velocity = masked_angular_velocities[torch.arange(0, masked_angular_velocities.shape[0], device=masked_angular_velocities.device), self.last_true_index][:, 2]
        
        
        self.metrics["final_linear_velocity.m/s"] = final_linear_velocity
        self.metrics["final_lateral_velocity.m/s"] = final_lateral_velocity
        self.metrics["final_vertical_velocity.m/s"] = final_vertical_velocity
        self.metrics["final_roll_velocity.rad/s"] = final_roll_velocity
        self.metrics["final_pitch_velocity.rad/s"] = final_pitch_velocity
        self.metrics["final_yaw_velocity.rad/s"] = final_yaw_velocity
        
        lin_vel_magnitude = torch.sqrt(final_linear_velocity**2 + final_lateral_velocity**2 + final_vertical_velocity**2)
        ang_vel_magnitude = torch.sqrt(final_roll_velocity**2 + final_pitch_velocity**2 + final_yaw_velocity**2)
        self.metrics["final_magnitude_velocity.m/s"] = lin_vel_magnitude
        self.metrics["final_magnitude_angular_velocity.rad/s"] = ang_vel_magnitude