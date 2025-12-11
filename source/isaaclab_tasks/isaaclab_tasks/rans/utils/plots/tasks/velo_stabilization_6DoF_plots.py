from . import BaseTaskPlots, Registerable
import torch

class VeloStabilization3DPlots(BaseTaskPlots, Registerable):
    def __init__(self, dfs: dict, trajectories_dfs: dict, labels: dict, env_info:dict, folder_path:list, plot_cfg:dict) -> None:
        super().__init__(dfs=dfs, trajectories_dfs=trajectories_dfs, labels=labels, env_info=env_info, folder_path=folder_path, plot_cfg=plot_cfg)

        self.task_name = "velo_stabilization_3d"
        keys_set = set()
        for group_dfs in dfs.values():
            for df in group_dfs:
                keys_set.update(
                    key for key in df.columns if key.startswith("avg_linear_velocity_error")
                )
                keys_set.update(
                    key for key in df.columns if key.startswith("avg_lateral_velocity_error")
                )
                keys_set.update(
                    key for key in df.columns if key.startswith("avg_vertical_velocity_error")
                )
                keys_set.update(
                    key for key in df.columns if key.startswith("avg_yaw_velocity_error")
                )
                keys_set.update(
                    key for key in df.columns if key.startswith("avg_pitch_velocity_error")
                )
                keys_set.update(
                    key for key in df.columns if key.startswith("avg_roll_velocity_error")
                )
                

        self.labels_to_plot = list(keys_set)

    def plot(self):
        for label_to_plot in self.labels_to_plot:
            self.boxplot(label_to_plot)

        # if len(self._trajectories_dfs) > 0:
        #     self.plot_linear_velocity_over_time()
        #     self.plot_angular_velocity_over_time()
        #     self.plot_linear_velocity_error_over_time()
        #     self.plot_angular_velocity_error_over_time()
        #     self.plot_actions_over_time()