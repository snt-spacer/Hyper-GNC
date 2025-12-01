from . import BaseRobotPlots, Registerable

class IntBall2Plots(BaseRobotPlots, Registerable):
    def __init__(self, dfs: dict, trajectories_dfs: dict, labels: dict, env_info:dict, folder_path:list, plot_cfg:dict) -> None:
        super().__init__(dfs=dfs, trajectories_dfs=trajectories_dfs, labels=labels, env_info=env_info, folder_path=folder_path, plot_cfg=plot_cfg)

        self.robot_name = "IntBall2"
        
        keys_set = set()
        for group_dfs in dfs.values():
            for df in group_dfs:
                # keys_set.update(
                #     key for key in df.columns if key.startswith("mean_trajectory_action_rate")
                # )
                # keys_set.update(
                #     key for key in df.columns if key.startswith("mean_trajectory_energy")
                # )
                # keys_set.update(
                #     key for key in df.columns if key.startswith("avg_mass_per_trajectory")
                # )
                pass
                
        self.labels_to_plot = list(keys_set)

    def plot(self):
        for label_to_plot in self.labels_to_plot:
            self.boxplot(label_to_plot)