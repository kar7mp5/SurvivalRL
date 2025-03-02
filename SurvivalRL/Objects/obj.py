import numpy as np


class Position:    
    """ 
    A simple class to store and manage 2D positions. 
    This class allows retrieving and updating position values.
    """

    def __init__(self, x: float, y: float):
        """
        Initializes a Position object with x and y coordinates.

        Args:
            x (float): Initial x-coordinate.
            y (float): Initial y-coordinate.
        """
        self.x = x
        self.y = y

    def __call__(self):
        """
        Returns the current position as a tuple.

        Returns:
            tuple: A tuple containing (x, y) coordinates.
        """
        return (self.x, self.y)

    def move(self, dx: float, dy: float):
        """
        Moves the position by a given displacement in the x and y directions.

        Args:
            dx (float): Change in x-coordinate.
            dy (float): Change in y-coordinate.
        """
        self.x += dx
        self.y += dy

    def move_towards(self, target_x: float, target_y: float, max_speed: float):
        """
        Moves the position toward a target point using an ease-in-out function.

        Args:
            target_x (float): Target x-coordinate.
            target_y (float): Target y-coordinate.
            max_speed (float): Maximum movement speed.
        """
        direction_x = target_x - self.x
        direction_y = target_y - self.y
        distance = np.hypot(direction_x, direction_y)  # Euclidean distance

        if distance < 0.1:
            return True
        
        # Normalize direction
        direction_x /= distance
        direction_y /= distance

        # Ease In-Out interpolation:
        # - Slow start, fast middle, slow end
        ease_factor = np.clip((distance / 5), 0.1, 1.0)  # Scale distance into (0.1, 1)
        speed = max_speed * ease_factor

        # Move toward the target
        self.x += direction_x * speed
        self.y += direction_y * speed
        return False


class Obj:
    """ 
    A parent class for moving objects (shapes).
    This class provides a base for different graphical objects that can be drawn and updated.
    """

    def __init__(self, x: float, y: float, target_speed: float, colour: str, name: str=None):
        """
        Initializes an Obj with a position and color.

        Args:
            x (float): Initial x-coordinate of the object.
            y (float): Initial y-coordinate of the object.
            colour (str): Color of the object.
        """
        self.pos = Position(x, y)
        self.target_speed = target_speed
        self.colour = colour
        self.shape = None # Shape will be defined in the subclasses
        if name is not None:
            self.name = name

    def draw(self, ax):
        """
        Abstract method to draw the object on a given axis.

        This method must be implemented in subclasses to define how the object is drawn.

        Args:
            ax (matplotlib.axes.Axes): The axis on which the object will be drawn.

        Raises:
            NotImplementedError: If called directly from the Obj class.
        """
        raise NotImplementedError

    def update(self, fps, objects, grid):
        """
        Abstract method to update the object's position based on the current frame.

        Args:
            frame (int): Current animation frame number.

        Raises:
            NotImplementedError: If called directly from the Obj class.
        """
        raise NotImplementedError