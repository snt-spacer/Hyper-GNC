from . import BaseRobotMetrics, Registerable
import torch

class IntBall2Metrics(BaseRobotMetrics, Registerable):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, robot_name: str) -> None:
        super().__init__(env, folder_path=folder_path, physics_dt=physics_dt, step_dt=step_dt, robot_name=robot_name)
    