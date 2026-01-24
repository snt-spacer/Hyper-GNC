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


class BaseTaskMetrics(AutoRegister):
    def __init__(self, env, folder_path: str, physics_dt: float, step_dt: float, task_name:str, task_index: int = 0) -> None:
        self.env = env
        self.folder_path = folder_path
        self.physics_dt = physics_dt
        self.step_dt = step_dt
        self.task_name = task_name
        self.task_index = task_index

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

    def get_indx_of_n_true_values(self, num_consecutive: int, bool_tensor: torch.Tensor, init_indx: torch.Tensor) -> torch.Tensor:
        """ Check that there are N consecutive True values and return the index of the first one
            Args:
                num_consecutive (int): Number of consecutive True values to check for.
                bool_tensor (torch.Tensor): The tensor to check for consecutive True values.
                init_indx (torch.Tensor): The initial indexs to start checking from.
            Returns:
                torch.Tensor: A tensor containing the index of the first occurrence of N consecutive True values.
        """
        len_trajec = bool_tensor.shape[1]
        results = torch.zeros(bool_tensor.shape[0], dtype=torch.int64, device=bool_tensor.device)
        for row_idx, row in enumerate(bool_tensor):
            start_idx = init_indx[row_idx].item()
            if start_idx == len_trajec:
                results[row_idx] = len_trajec - 1
                continue

            found = False
            for i in range(start_idx, len(row) - num_consecutive + 1):
                if all(row[i + j] for j in range(num_consecutive)):
                    results[row_idx] = i
                    found = True
                    break
            if not found:
                results[row_idx] = -1

        return results

    def get_reached_idx(self) -> torch.Tensor:
        # Position threshold
        if "MultiTask" in self.env.unwrapped.__class__.__name__:
            self.pos_threshold = self.env.unwrapped.tasks_apis[self.task_index]._task_cfg.position_tolerance
        else:
            self.pos_threshold = self.env.unwrapped.task_api._task_cfg.position_tolerance
        masked_distances = self.trajectories['position_distance'] * self.trajectories_masks
        pos_reached_threshold_mask = masked_distances <= self.pos_threshold

        # Heading threshold
        if "MultiTask" in self.env.unwrapped.__class__.__name__:
            heading_threshold = self.env.unwrapped.tasks_apis[self.task_index]._task_cfg.heading_tolerance
        else:
            heading_threshold = self.env.unwrapped.task_api._task_cfg.heading_tolerance
        heading_error = torch.arctan2(
            torch.sin(self.trajectories['target_heading'] - self.trajectories['heading']),
            torch.cos(self.trajectories['target_heading'] - self.trajectories['heading']),
        ) * self.trajectories_masks
        heading_reached_threshold = torch.abs(heading_error) <= heading_threshold

        # Combine position and heading thresholds
        self.reached_threshold_mask = pos_reached_threshold_mask & heading_reached_threshold

        # First index where the threshold is reached
        len_trajec = self.trajectories_masks.sum(dim=-1) #self.trajectories['position_distance'].shape[1] 
        reached_idx = torch.argmax(self.reached_threshold_mask.int(), dim=1)
        all_false = ~torch.any(self.reached_threshold_mask, dim=1) # argmax returns 0 if all values are false
        reached_idx[all_false] = len_trajec[all_false]

        return reached_idx

    def get_reached_idx_and_stay_under_threshold(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the index where the robot reaches the target position and stays under the threshold for a given number of consecutive steps.
        """
        reached_idx = self.get_reached_idx()
        under_threshold = []
        for i, trajectory in enumerate(self.reached_threshold_mask):
            if torch.any(trajectory[reached_idx[i]:]==False) or len(trajectory[reached_idx[i]:]) == 0:
                under_threshold.append(False)
            else:
                under_threshold.append(True)
        
        under_threshold = torch.tensor(under_threshold, device=self.reached_threshold_mask.device)
        return reached_idx, under_threshold
    
    def trajectory_shortest_distance(self, target_positions) -> torch.Tensor:
        """Calculate the shortest (euclidian) path length of the trajectory based on target positions."""

        # Handle different tensor shapes
        if len(target_positions.shape) < 4:
            target_positions = target_positions.unsqueeze(2)

        total_distance = torch.zeros(target_positions.shape[0], device=target_positions.device)
        
        # Check if we have enough goals to calculate distances
        if target_positions.shape[2] <= 1:
            # If only one goal, return the initial distance
            if 'position_distance' in self.trajectories:
                return self.trajectories['position_distance'][:, 0]
            else:
                return total_distance
        
        for current_goal in range(target_positions.shape[2] - 1):
            if current_goal == 0:
                # For the first goal, use the initial position distance if available
                if 'position_distance' in self.trajectories and self.trajectories['position_distance'].shape[1] > current_goal:
                    total_distance += self.trajectories['position_distance'][:, current_goal]
                else:
                    # Calculate distance from start to first goal
                    x_diff = target_positions[:, 0, current_goal + 1, 0] - target_positions[:, 0, current_goal, 0]
                    y_diff = target_positions[:, 0, current_goal + 1, 1] - target_positions[:, 0, current_goal, 1]
                    total_distance += torch.sqrt(x_diff**2 + y_diff**2)
            else:
                # Calculate distance between consecutive goals
                x_diff = target_positions[:, 0, current_goal + 1, 0] - target_positions[:, 0, current_goal, 0]
                y_diff = target_positions[:, 0, current_goal + 1, 1] - target_positions[:, 0, current_goal, 1]
                total_distance += torch.sqrt(x_diff**2 + y_diff**2)

        return total_distance

    def robot_distance_traveled(self, robot_positions) -> torch.Tensor:
        """Calculate the path length of the robot trajectory based on robot positions."""

        total_distance = torch.zeros(robot_positions.shape[0], device=robot_positions.device)
        x_dif = robot_positions[:, 1:, 0] - robot_positions[:, :-1, 0]
        y_dif = robot_positions[:, 1:, 1] - robot_positions[:, :-1, 1]
        total_distance += torch.sqrt(x_dif**2 + y_dif**2).sum(dim=1)

        return total_distance