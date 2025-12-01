from . import BaseTaskPlots, Registerable
import torch

class GoToPose3DPlots(BaseTaskPlots, Registerable):
    def __init__(self, dfs: dict, trajectories_dfs: dict, labels: dict, env_info:dict, folder_path:list, plot_cfg:dict) -> None:
        super().__init__(dfs=dfs, trajectories_dfs=trajectories_dfs, labels=labels, env_info=env_info, folder_path=folder_path, plot_cfg=plot_cfg)

        self.task_name = "go_to_pose_3d"

        # Box plots
        keys_set = set()
        for group_dfs in dfs.values():
            for df in group_dfs:
                keys_set.update(
                    key for key in df.columns if key.startswith("final_position_distance")
                )
                keys_set.update(
                    key for key in df.columns if key.startswith("final_orientation_error")
                )
        self.labels_box_plot = list(keys_set)


    def plot(self):
        for label_to_plot in self.labels_box_plot:
            self.boxplot(label_to_plot)