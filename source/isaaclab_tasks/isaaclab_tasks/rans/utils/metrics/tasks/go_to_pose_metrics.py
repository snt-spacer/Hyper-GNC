from . import BaseTaskMetrics, Registerable
import torch

class GoToPoseMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name: str, task_index: int = 0) -> None:
        super().__init__(env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, task_name=task_name, task_index=task_index)

        self.pos_threshold = 0
        self.heading_threshold = 0
    
    # @BaseTaskMetrics.register
    # def time_to_reach_position_threshold(self):
    #     print("[INFO][METRICS][TASK] Time to reach position threshold")
    #     reached_idx = self.get_reached_idx()
    #     episode_length_in_s = reached_idx * self.step_dt
    #     self.metrics["time_to_reach_position_threshold.s"] = episode_length_in_s


    @BaseTaskMetrics.register
    def final_position_distance(self):
        """ Difference between the target position and the final position of the robot. """
        print("[INFO][METRICS][TASK] Final position delta")
        masked_distances = self.trajectories['position_distance'] * self.trajectories_masks
        final_position_delta = masked_distances[torch.arange(0, masked_distances.shape[0], device=masked_distances.device), self.last_true_index]
        self.metrics["final_position_distance.m"] = final_position_delta

    # @BaseTaskMetrics.register
    # def final_position_heading_error(self):
    #     """ Difference between the target position and the final heading of the robot. """
    #     print("[INFO][METRICS][TASK] Final position heading error")

    #     masked_sin = self.trajectories['sin_heading_to_target_error'] * self.trajectories_masks
    #     final_sin = masked_sin[torch.arange(0, masked_sin.shape[0], device=masked_sin.device), self.last_true_index]

    #     masked_cos = self.trajectories['cos_heading_to_target_error'] * self.trajectories_masks
    #     final_cos = masked_cos[torch.arange(0, masked_cos.shape[0], device=masked_cos.device), self.last_true_index]

    #     angle_error = torch.arctan2(final_sin, final_cos)

    #     self.metrics["final_position_heading_error.rad"] = angle_error

    @BaseTaskMetrics.register
    def final_orientation_error(self):
        """ Difference between the target heading and the final heading of the robot. """
        print("[INFO][METRICS][TASK] Final orientation error")

        masked_target_heading = self.trajectories['target_heading'] * self.trajectories_masks
        masked_headings = self.trajectories['heading'] * self.trajectories_masks

        final_target_heading = masked_target_heading[torch.arange(0, masked_target_heading.shape[0], device=masked_target_heading.device), self.last_true_index]
        final_heading = masked_headings[torch.arange(0, masked_headings.shape[0], device=masked_headings.device), self.last_true_index]

        heading_error = final_target_heading - final_heading
        angle_error = torch.abs(torch.arctan2(torch.sin(heading_error), torch.cos(heading_error)))  # Normalize the angle error to [-pi, pi] and take the absolute value

        self.metrics["final_orientation_error.rad"] = angle_error

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
    #     print("[INFO][METRICS][ROBOT] Jerkiness")
    #     last_n_steps = 20
    #     masked_pos_dist = self.trajectories['position_distance'] * self.trajectories_masks
    #     last_pos = torch.stack([row[start_idx - last_n_steps: start_idx] for row, start_idx in zip(masked_pos_dist, self.last_true_index)])

    #     vel = last_pos[:, 1:] - last_pos[:, :-1]
    #     acc = vel[:, 1:] - vel[:, :-1]
    #     jerk = torch.mean(acc[:, 1:] - acc[:, :-1], dim=1)

    #     self.metrics[f"mean_last_{last_n_steps}_steps_jerk.u"] = jerk

    # @BaseTaskMetrics.register
    # def success_rate(self):
    #     """ Success rate of the task, defined as the percentage of episodes where the robot reached the target position and heading. """
    #     print("[INFO][METRICS][TASK] Success rate")
    #     _, under_threshold = self.get_reached_idx_and_stay_under_threshold()
    #     self.metrics["success_rate.u"] = under_threshold.sum() / under_threshold.shape[0]

