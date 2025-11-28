from . import BaseTaskMetrics, Registerable
import torch

class GoThroughPoses6DoFMetrics(BaseTaskMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name: str, task_index: int = 0) -> None:
        super().__init__(env=env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, task_name=task_name, task_index=task_index)
        
        @BaseTaskMetrics.register
        def orientation_error_following_path(self):
            print("[INFO][METRICS][TASK][GoThroughPoses6DoF] Orientation error following path")