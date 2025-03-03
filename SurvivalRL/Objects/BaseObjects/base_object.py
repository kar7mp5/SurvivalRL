import matplotlib
from BaseObjects import Position
from SurvivalRL import GameObject


class BaseObject:
    """ 
    A base class for all game objects that move and interact within the simulation.

    This class serves as the foundation for different graphical objects (e.g., circles, rectangles)
    by defining common properties such as position, movement speed, and color. 
    Subclasses are expected to implement specific behaviors related to rendering, collision detection, and updates.
    """

    def __init__(
        self, 
        game: GameObject, 
        ax: matplotlib.axes.Axes,
        x: float, y: float, 
        target_speed: float, 
        colour: str,
        name: str = None):
        """
        Initializes a BaseObject instance with essential properties.

        Args:
            game (GameObject): The game manager that oversees all objects.
            ax (matplotlib.axes.Axes): The axis on which the object will be drawn.
            x (float): Initial x-coordinate of the object.
            y (float): Initial y-coordinate of the object.
            target_speed (float): Movement speed of the object.
            colour (str): Object's display color.
            name (str, optional): Name label for the object (default: None).
        """
        self.game = game
        self.ax = ax
        self.pos = Position(x, y)  # Represents the object's position in a 2D space
        self.target_speed = target_speed  # Speed at which the object moves
        self.colour = colour  # Color used to render the object
        self.shape = None  # To be defined in subclasses
        if name is not None:
            self.name = name  # Assign a name to the object if provided

    def update(self):
        """
        Updates the object's state.

        This method should be implemented in subclasses to define how the object moves
        and interacts with other elements in the environment.

        Raises:
            NotImplementedError: This method must be overridden in subclasses.
        """
        raise NotImplementedError

    def draw(self):
        """
        Draws the object on the designated axis.

        Subclasses must implement this method to specify how the object should be rendered.

        Raises:
            NotImplementedError: This method must be overridden in subclasses.
        """
        raise NotImplementedError

    def get_grid_cell(self):
        """ 
        Determines the grid cell coordinates based on the object's position.

        This is useful for spatial partitioning, which improves performance by reducing the number 
        of collision checks required.

        Returns:
            tuple: A tuple representing the grid cell (x, y) in which the object is located.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    def is_colliding(self, other):
        """ 
        Checks whether the object is colliding with another game entity.

        This method must be implemented in subclasses to define how collisions are detected
        for different object types (e.g., rectangle vs. rectangle, circle vs. rectangle).

        Args:
            other (BaseObject): Another game object to check for collisions.

        Returns:
            bool: True if the objects are colliding, False otherwise.

        Raises:
            NotImplementedError: This method must be overridden in subclasses.
        """
        raise NotImplementedError

    def resolve_collision(self, other):
        """ 
        Resolves collisions by applying appropriate physics responses.

        This function should implement a reaction mechanism when the object collides with 
        another entity, such as bouncing off or stopping movement.

        Args:
            other (BaseObject): The object with which this one has collided.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError