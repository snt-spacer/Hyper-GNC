from .base_task_metrics import BaseTaskMetrics

class Registerable:
    """Registerable class.

    All classes that inherit from this class are automatically registered in the MetricsFactory.
    """
    def __init_subclass__(cls: BaseTaskMetrics) -> None:
        """Register the class in the factory.
        
        When a class inherits from this class, it is automatically registered in the factory.

        Args:
            cls (BaseTaskMetrics): The task metrics class to register
        """
        cls_name = cls.__name__[:-7]  # Remove "Metrics" from the class name
        TaskMetricsFactory.register(cls_name, cls)

class TaskMetricsFactory:
    """Task Metrics factory class.

    The factory is used to create task metrics objects. Task metrics calculates the metrics for that class.
    """ 
    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable) -> None:
        """Register a task metrics class in the factory.

        Task metrics classes are used to calculate metrics from a dictionary of torch.Tensors.

        Args:
            name (str): The name of the task metrics class.
            sub_class (Registerable): The task metrics class to register.
        """
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name:str, *args, **kwargs) -> BaseTaskMetrics:
        """Create a task metrics object.

        Args:
            cls_name (str): The name of the task metrics class.
        """
        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered. Available modules: {cls.registry.keys()}")

        return cls.registry[cls_name](*args, **kwargs)
    

from .go_through_positions_metrics import GoThroughPositionsMetrics
from .go_to_position_metrics import GoToPositionMetrics
from .go_to_pose_metrics import GoToPoseMetrics
from .go_through_poses_metrics import GoThroughPosesMetrics
from .track_velocities_metrics import TrackVelocitiesMetrics
from .rendezvous_metrics import RendezvousMetrics
from .race_gates_metrics import RaceGatesMetrics
from .go_to_position_6DoF_with_obstacles import GoToPosition3DWithObstaclesMetrics
from .go_to_pose_6DoF_metrics import GoToPose3DMetrics
from .go_through_poses_6DoF_metrics import GoThroughPoses3DMetrics
from .track_velocities_6DoF_metrics import TrackVelocities3DMetrics