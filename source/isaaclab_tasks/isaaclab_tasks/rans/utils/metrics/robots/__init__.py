from .base_robot_metrics import BaseRobotMetrics

class Registerable:
    """Registerable class.

    All classes that inherit from this class are automatically registered in the RobotMetricsFactory.
    """
    def __init_subclass__(cls: BaseRobotMetrics) -> None:
        """Register the class in the factory.
        
        When a class inherits from this class, it is automatically registered in the factory.

        Args:
            cls (BaseRobotMetrics): The robot metrics class to register
        """
        cls_name = cls.__name__[:-7]  # Remove "Metrics" from the class name
        RobotMetricsFactory.register(cls_name, cls)

class RobotMetricsFactory:
    """Robot Metrics factory class.

    The factory is used to create robot metrics objects. Robot metrics calculates the metrics for that class.
    """ 
    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable) -> None:
        """Register a robot metrics class in the factory.

        Robot metrics classes are used to calculate metrics from a dictionary of torch.Tensors.

        Args:
            name (str): The name of the robot metrics class.
            sub_class (Registerable): The robot metrics class to register.
        """
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name:str, *args, **kwargs) -> BaseRobotMetrics:
        """Create a robot metrics object.

        Args:
            cls_name (str): The name of the robot metrics class.
        """
        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered. Available modules: {cls.registry.keys()}")

        return cls.registry[cls_name](*args, **kwargs)
    

from .floating_platform_metrics import FloatingPlatformMetrics
from .jetbot_metrics import JetbotMetrics
from .turtlebot2_metrics import Turtlebot2Metrics
from .leatherback_metrics import LeatherbackMetrics
from .modular_freeflyer_metrics import ModularFreeflyerMetrics
from .intBall2 import IntBall2Metrics