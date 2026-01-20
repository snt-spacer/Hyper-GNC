from . import BaseTaskMetrics, Registerable
import torch

class GoToPositionWithObstaclesMetrics(BaseTaskMetrics, Registerable):
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

    # @BaseTaskMetrics.register
    # def convergence_time(self):
    #     #TODO: how diff from time_to_reach_threshold?
    #     pass

    # @BaseTaskMetrics.register
    # def position_overshoot(self):
    #     print("[INFO][METRICS][TASK] Position overshoot")
    #     if "MultiTask" in self.env.unwrapped.__class__.__name__:
    #         threshold = self.env.unwrapped.tasks_apis[self.task_index]._task_cfg.position_tolerance
    #     else:
    #         threshold = self.env.unwrapped.task_api._task_cfg.position_tolerance
    #     masked_distances = self.trajectories['position_distance'] * self.trajectories_masks
    #     reached_threshold = masked_distances <= threshold
        
    #     len_trajec = self.trajectories['position_distance'].shape[1] # argmax returns 0 if all values are false
    #     reached_idx = torch.argmax(reached_threshold.int(), dim=1)
    #     all_false = ~torch.any(reached_threshold, dim=1)
    #     reached_idx[all_false] = len_trajec - 1

    #     overshoot_idx = self.get_indx_of_n_true_values(num_consecutive=5, bool_tensor=reached_threshold, init_indx=reached_idx) #TODO: Hardcoded 5
    #     overshoot_idx = torch.where(overshoot_idx == -1, 0, overshoot_idx)
    #     overshoot_num_steps = overshoot_idx - reached_idx 

    #     self.metrics["num_steps_position_overshoot.u"] = overshoot_num_steps
    #     # Calculate the overshoot distance
    #     indx = torch.arange(masked_distances.shape[0], device=masked_distances.device)
    #     self.metrics["overshoot_distance_error.m"] = masked_distances[indx, overshoot_idx] - masked_distances[indx, reached_idx]
        
    # @BaseTaskMetrics.register
    # def trajectory_efficiency(self):
    #     print("[INFO][METRICS][TASK] Trajectory efficiency")
    #     # Calculate path
    #     masked_positions = self.trajectories['position'] * self.trajectories_masks.unsqueeze(-1)
    #     diff_positions = masked_positions[:, 1:] - masked_positions[:, :-1]
    #     segment_lengths = torch.linalg.vector_norm(diff_positions, dim=-1)
    #     total_distance = torch.sum(segment_lengths, dim=1)

    #     # Calculate shortest path
    #     masked_targets = self.trajectories['target_position'] * self.trajectories_masks.unsqueeze(-1)
    #     start_positions = masked_positions[:, 0]
    #     final_targets = masked_targets[:, 0]
    #     shortest_path = torch.linalg.vector_norm(final_targets - start_positions[:, :2], dim=-1)

    #     efficiency = shortest_path / total_distance
    #     self.metrics["trajectory_efficiency.u"] = efficiency

    # @BaseTaskMetrics.register
    # def jerckiness(self):
    #     print("[INFO][METRICS][TASK] Jerkiness")
    #     last_n_steps = 20
    #     masked_pos_dist = self.trajectories['position_distance'] * self.trajectories_masks
    #     last_pos = torch.stack([row[start_idx - last_n_steps: start_idx] for row, start_idx in zip(masked_pos_dist, self.last_true_index)])

    #     vel = last_pos[:, 1:] - last_pos[:, :-1]
    #     acc = vel[:, 1:] - vel[:, :-1]
    #     jerk = torch.mean(acc[:, 1:] - acc[:, :-1], dim=1)

    #     self.metrics[f"mean_last_{last_n_steps}_steps_jerk.u"] = jerk


    # @BaseTaskMetrics.register
    # def time_to_half_init_velocity(self):
    #     print("[INFO][METRICS][TASK] Time to half initial velocity")
    #     if "MultiTask" in self.env.unwrapped.__class__.__name__:
    #         init_vel = self.env.unwrapped.tasks_apis[self.task_index]._task_cfg.spawn_max_lin_vel
    #     else:
    #         init_vel = self.env.unwrapped.task_api._task_cfg.spawn_max_lin_vel

    #     len_trajec = self.trajectories['half_init_lin_vel_x'].shape[1]

    #     masked_half_init_vel_x = self.trajectories['half_init_lin_vel_x'] * self.trajectories_masks
    #     half_init_vel_x_idx = torch.argmax(masked_half_init_vel_x.int(), dim=1)
    #     all_false = ~torch.any(masked_half_init_vel_x.int(), dim=1) # argmax returns 0 if all values are false
    #     half_init_vel_x_idx[all_false] = len_trajec

    #     masked_half_init_vel_y = self.trajectories['half_init_lin_vel_y'] * self.trajectories_masks
    #     half_init_vel_y_idx = torch.argmax(masked_half_init_vel_y.int(), dim=1)
    #     all_false = ~torch.any(masked_half_init_vel_y.int(), dim=1)
    #     half_init_vel_y_idx[all_false] = len_trajec

    #     masked_half_init_ang_vel = self.trajectories['half_init_ang_vel'] * self.trajectories_masks
    #     half_init_ang_vel_idx = torch.argmax(masked_half_init_ang_vel.int(), dim=1)
    #     all_false = ~torch.any(masked_half_init_ang_vel.int(), dim=1)
    #     half_init_ang_vel_idx[all_false] = len_trajec

    #     time_s_to_reach_half_init_vel_x = half_init_vel_x_idx * self.step_dt
    #     time_s_to_reach_half_init_vel_y = half_init_vel_y_idx * self.step_dt
    #     time_s_to_reach_half_init_ang_vel = half_init_ang_vel_idx * self.step_dt

    #     self.metrics["time_to_half_initial_linear_velocity_x.s"] = time_s_to_reach_half_init_vel_x
    #     self.metrics["time_to_half_initial_linear_velocity_y.s"] = time_s_to_reach_half_init_vel_y
    #     self.metrics["time_to_half_initial_angular_velocity.s"] = time_s_to_reach_half_init_ang_vel
        
    
    # @BaseTaskMetrics.register
    # def final_velocity_error(self):
    #     print("[INFO][METRICS][TASK] Final velocity error")

    #     masked_distances = self.trajectories['position_distance'] * self.trajectories_masks
    #     final_position_delta = masked_distances[torch.arange(0, masked_distances.shape[0], device=masked_distances.device), self.last_true_index]
    #     self.metrics["final_position_distance.m"] = final_position_delta

    #     masked_lin_vel_x = self.trajectories['linear_velocity'][..., 0] * self.trajectories_masks
    #     masked_lin_vel_y = self.trajectories['linear_velocity'][..., 1] * self.trajectories_masks
    #     masked_ang_vel = self.trajectories['angular_velocity'][..., -1] * self.trajectories_masks

    #     final_lin_vel_x = masked_lin_vel_x[torch.arange(0, masked_lin_vel_x.shape[0], device=masked_lin_vel_x.device), self.last_true_index]
    #     final_lin_vel_y = masked_lin_vel_y[torch.arange(0, masked_lin_vel_y.shape[0], device=masked_lin_vel_y.device), self.last_true_index]
    #     final_ang_vel = masked_ang_vel[torch.arange(0, masked_ang_vel.shape[0], device=masked_ang_vel.device), self.last_true_index]

    #     # The final velocities are also the error in this case, as we want the robot to stop
    #     self.metrics["final_linear_velocity_error_y.m/s"] = torch.abs(final_lin_vel_y)
    #     self.metrics["final_linear_velocity_error_x.m/s"] = torch.abs(final_lin_vel_x)
    #     self.metrics["final_angular_velocity_error.rad/s"] = torch.abs(final_ang_vel)