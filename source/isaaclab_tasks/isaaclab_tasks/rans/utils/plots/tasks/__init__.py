from .base_task_plots import BaseTaskPlots

class Registerable:
    """Registerable class.

    All classes that inherit from this class are automatically registered in the PlotsFactory.
    """
    def __init_subclass__(cls: BaseTaskPlots) -> None:
        """Register the class in the factory.
        
        When a class inherits from this class, it is automatically registered in the factory.

        Args:
            cls (BaseTaskPlots): The task Plots class to register
        """
        cls_name = cls.__name__[:-5]  # Remove "Plots" from the class name
        TaskPlotsFactory.register(cls_name, cls)

class TaskPlotsFactory:
    """Task Plots factory class.

    The factory is used to create task plots objects. Task plots calculates the plots for that class.
    """ 
    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable) -> None:
        """Register a task plots class in the factory.

        Task plots classes are used to calculate plots from a dictionary of torch.Tensors.

        Args:
            name (str): The name of the task plots class.
            sub_class (Registerable): The task plots class to register.
        """
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name:str, *args, **kwargs) -> BaseTaskPlots:
        """Create a task plots object.

        Args:
            cls_name (str): The name of the task plots class.
        """
        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered. Available modules: {cls.registry.keys()}")

        return cls.registry[cls_name](*args, **kwargs)
    

from .go_through_positions_plots import GoThroughPositionsPlots
from .go_to_position_plots import GoToPositionPlots
from .go_to_pose_plots import GoToPosePlots
from .go_through_poses_plots import GoThroughPosesPlots
from .track_velocities_plots import TrackVelocitiesPlots
from .race_gates_plots import RaceGatesPlots
from .rendezvous_plots import RendezvousPlots
from .go_to_position_6DoF_with_obstacles_plots import GoToPosition3DWithObstaclesPlots
from .go_to_pose_6DoF_plots import GoToPose3DPlots
from .go_through_poses_6DoF_plots import GoThroughPoses3DPlots
from .track_velocities_6DoF_plots import TrackVelocities3DPlots