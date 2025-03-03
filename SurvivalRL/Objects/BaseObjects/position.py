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