from .base_robot_plots import BaseRobotPlots

class Registerable:
    """Registerable class.

    All classes that inherit from this class are automatically registered in the RobotPlotsFactory.
    """
    def __init_subclass__(cls: BaseRobotPlots) -> None:
        """Register the class in the factory.
        
        When a class inherits from this class, it is automatically registered in the factory.

        Args:
            cls (BaseRobotPlots): The robot plots class to register
        """
        cls_name = cls.__name__[:-5]  # Remove "Plots" from the class name
        RobotPlotsFactory.register(cls_name, cls)

class RobotPlotsFactory:
    """Robot Plots factory class.

    The factory is used to create robot plots objects. Robot plots calculates the plots for that class.
    """ 
    registry = {}

    @classmethod
    def register(cls, name: str, sub_class: Registerable) -> None:
        """Register a robot plots class in the factory.

        Robot plots classes are used to calculate plots from a dictionary of torch.Tensors.

        Args:
            name (str): The name of the robot plots class.
            sub_class (Registerable): The robot plots class to register.
        """
        if name in cls.registry:
            raise ValueError(f"Module {name} already registered.")
        cls.registry[name] = sub_class

    @classmethod
    def create(cls, cls_name:str, *args, **kwargs) -> BaseRobotPlots:
        """Create a robot plots object.

        Args:
            cls_name (str): The name of the robot plots class.
        """
        if cls_name not in cls.registry:
            raise ValueError(f"Module {cls_name} not registered. Available modules: {cls.registry.keys()}")

        return cls.registry[cls_name](*args, **kwargs)
    

from .floating_platform_plots import FloatingPlatformPlots
from .jetbot_plots import JetbotPlots
from .turtlebot2_plots import Turtlebot2Plots
from .leatherback_plots import LeatherbackPlots
from .modular_freeflyer_plots import ModularFreeflyerPlots
from .intball2_plots import IntBall2Plots