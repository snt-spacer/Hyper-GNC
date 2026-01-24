import torch

class AutoRegister:
    def __init_subclass__(cls, **kwargs):
        """Ensure each subclass gets its own independent registry."""
        super().__init_subclass__(**kwargs)

        # Unique for each subclass + inherit from parent class
        cls._registry = getattr(super(cls, cls), '_registry', {}).copy()

        for name, value in cls.__dict__.items():
            # If an attribute is a function and has our marker, register it.
            if callable(value) and getattr(value, '_auto_register', False):
                cls._registry[name] = value

    @staticmethod
    def register(func: callable) -> callable:
        """Decorator that simply marks a function so that __init_subclass__
        knows it should be placed in the registry.
        """
        func._auto_register = True
        return func

    @classmethod
    def get_registered_methods(cls) -> dict[str, callable]:
        """Retrieve registered methods."""
        return cls._registry


class BaseRobotMetrics(AutoRegister):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, robot_name: str) -> None:
        self.env = env
        self.folder_path = folder_path
        self.physics_dt = physics_dt
        self.step_dt = step_dt

        self.robot_name = robot_name

        self.metrics = {}
        self.env_info = {}

    def populate_env_info(self)-> None:
        """Populate environment information. Subclass should implement this method."""
        pass

    def generate_metrics(
            self, 
            trajectories: dict, 
            cutoff_indices: torch.Tensor,
            trajectories_masks: torch.Tensor, 
        ) -> None:

        self.trajectories = trajectories
        self.cutoff_indices = cutoff_indices - 1
        self.trajectories_masks = trajectories_masks

        for metric_fnc in self.get_registered_methods().values():
            metric_fnc(self)

        for key, value in self.metrics.items():
            self.metrics

    @property
    def last_true_index(self) -> torch.Tensor:
        """Returns the last true index of a masked tensor along the first dimension.
            Args:
                masked_tensor (torch.Tensor): The tensor to find the last true index for.
            Returns:
                torch.Tensor: A tensor containing the last true indices for each row.
        """
        # traj_len = self.trajectories_masks.shape[1]
        # last_true_idx = torch.argmax(~self.trajectories_masks.int(), dim=1)
        # all_true = torch.all(self.trajectories_masks, dim=1)
        # last_true_idx[all_true] = traj_len
        # return last_true_idx
        return self.trajectories_masks.sum(dim=1) - 1
    
    @AutoRegister.register
    def action_rate(self):
        # print("[INFO][METRICS][ROBOT] Action rate")
        # masked_unaltered_actions = self.trajectories['actions'] * self.trajectories_masks.unsqueeze(-1)
        # action_rate = torch.mean(
        #     torch.sum(
        #         torch.square(masked_unaltered_actions[:, 1:] - masked_unaltered_actions[:, :-1]), dim=-1
        #     )[:, 1:], dim=1)
        # self.metrics["mean_trajectory_action_rate.u"] = action_rate
        pass

    @AutoRegister.register
    def energy(self):
        # print("[INFO][METRICS][ROBOT] Energy")
        # masked_actions = self.trajectories['actions'] * self.trajectories_masks.unsqueeze(-1)

        # energy = torch.stack([torch.mean(row[:end_idx]) for row, end_idx in zip(masked_actions, self.last_true_index)])
        # energy = torch.mean(torch.sum(masked_actions ** 2, dim=-1), dim=1)
        # self.metrics["mean_trajectory_energy.u"] = energy
        pass