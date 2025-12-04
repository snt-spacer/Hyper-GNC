import math
import torch
from .rng_utils import PerEnvSeededRNG

class ObjectStorage:
    def __init__(
        self,
        num_envs: int,
        max_num_vis_objects_in_env: int,
        store_height: float,
        rng: PerEnvSeededRNG,
        device: str = "cuda",
    ) -> None:
        """
        Args:
            num_envs (int): The number of environments.
            env_origin (torch.tensor): The origin of the environment.
            max_num_vis_objects_in_env (int): The maximum number of objects visible in the environment.
            store_height (float): The height position to storage the objects.
            device (str): The device to use.
        """

        assert num_envs > 0, "The number of environments must be greater than 0."

        self.num_envs = num_envs
        self._max_num_vis_objects_in_env = max_num_vis_objects_in_env
        self._store_height = store_height
        self._rng = rng
        self._device = device
        self.storage_buff = None

    def create_storage_buffer(self, env_origin: torch.Tensor) -> torch.Tensor:
        """
        Generates a tensor representing hidden objects' positions for each environment in 3D space.

        Args:
            env_origin (torch.Tensor): The origin positions for each environment (shape: [num_envs, 3]).

        Returns:
            torch.Tensor: A tensor containing the coordinates and quaternions of hidden objects (shape: [num_envs, num_objects, 7]).
        """
        # Define grid size in each axis
        grid_size = int(math.ceil(self._max_num_vis_objects_in_env ** (1/3)))  # Cubic root for 3D grid

        # Create 3D grid of positions
        num = torch.arange(grid_size, device=self._device)
        x, y, z = torch.meshgrid(num, num, num, indexing="ij")  # 3D grid
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

        # Scale positions & offset by environment origin
        spacing = self._store_height / grid_size
        xyz = torch.stack((x, y, z), dim=1).float() * spacing
        xyz = xyz.unsqueeze(0).repeat(self.num_envs, 1, 1)
        xyz += env_origin.unsqueeze(1)

        # Generate default quaternions (identity rot)
        xyzw = torch.zeros(self.num_envs, xyz.shape[1], 4, device=self._device)
        xyzw[:, :, 0] = 1.0 

        # Concatenate positions & quaternions
        self.storage_buff = torch.cat((xyz, xyzw), dim=2)  # Shape: [num_envs, num_objects, 7]

        return self.storage_buff


    def get_positions_with_storage(
        self, objects_pos: torch.tensor, mask: torch.tensor, env_ids: torch.tensor
    ) -> torch.tensor:
        """
        Returns the position of the objects for each environment.

        Args:
            objects_pos (torch.tensor): The position of the objects.
            mask (torch.tensor): The mask to apply to the objects.
            env_ids (torch.tensor): The ids of the environments.

        Returns:
            torch.tensor: The position of the objects for each environment.
        """

        objects_pos[~mask] = self.storage_buff[env_ids, :self._max_num_vis_objects_in_env][~mask]

        return objects_pos