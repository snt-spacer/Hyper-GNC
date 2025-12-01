from . import BaseTaskMetrics, Registerable
import torch

class GoToPosition3DWithObstaclesMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name: str, task_index: int = 0) -> None:
        super().__init__(env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, task_name=task_name, task_index=task_index)

    BaseTaskMetrics.register
    def time_to_reach_position_threshold(self):
        print("[INFO][METRICS][TASK][GoToPosition6DoFWithObstacles] Time to reach position threshold")
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
        print("[INFO][METRICS][TASK][GoToPosition6DoFWithObstacles] Final position delta")
        masked_distances = self.trajectories['position_distance'] * self.trajectories_masks
        final_position_delta = masked_distances[torch.arange(0, masked_distances.shape[0], device=masked_distances.device), self.last_true_index]
        self.metrics["final_position_distance.m"] = final_position_delta