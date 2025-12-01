from . import BaseTaskMetrics, Registerable
import torch

class GoThroughPoses3DMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name: str, task_index: int = 0) -> None:
        super().__init__(env=env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, task_name=task_name, task_index=task_index)
        
    @BaseTaskMetrics.register
    def orientation_error(self):
        print("[INFO][METRICS][TASK][GoThroughPoses6DoF] Orientation error")
        
        masked_error = self.trajectories['orientation_error'] * self.trajectories_masks
        sum_masks = torch.sum(self.trajectories_masks, dim=1)
        # Prevent division by zero
        sum_masks[sum_masks == 0] = 1.0
        
        avg_error = torch.sum(masked_error, dim=1) / sum_masks
        
        self.metrics["orientation_error.rad"] = avg_error

    @BaseTaskMetrics.register
    def success_rate(self):
        print("[INFO][METRICS][TASK][GoThroughPoses6DoF] Success rate")
        
        masked_completed = self.trajectories['trajectory_completed'].float() * self.trajectories_masks
        is_completed = torch.max(masked_completed, dim=1).values
        
        self.metrics["success_rate.u"] = is_completed
