from obj import Obj
from SurvivalRL import Config

import matplotlib.patches as patches
import matplotlib
import numpy as np


class Circle(Obj):
    """ 
    A Circle class that inherits from Obj.
    This class represents a moving circular object in a 2D space.
    """

    def __init__(
        self,
        ax: matplotlib.axes.Axes, 
        x: float, y: float, 
        radius: float, 
        target_speed: float, 
        colour: str, 
        name: str = None):
        """
        Initializes a Circle object.

        Args:
            ax (matplotlib.axes.Axes): The axis where the circle will be drawn.
            x (float): Initial x-coordinate of the circle.
            y (float): Initial y-coordinate of the circle.
            radius (float): Radius of the circle.
            colour (str): Color of the circle.
            name (str, optional): Name label displayed above the circle. Defaults to None.
        """
        super().__init__(x, y, target_speed, colour, name)
        self.radius = radius
        self.label = ax.text(x, y + radius + 0.5, self.name, ha="center", va="bottom", fontsize=10, color="black")

        # Direction arrow for movement visualization
        self.direction_arrow, = ax.plot([x, x], [y, y], color="red", linewidth=2, marker="o", markersize=6)

        self.set_new_target()

    def set_new_target(self):
        """ 
        Sets a new random target position within a reasonable distance.
        
        Ensures that the new target is not too close to the current position.
        """
        while True:
            new_x = np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2)
            new_y = np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2)
            distance = np.hypot(new_x - self.pos.x, new_y - self.pos.y)

            if distance > Config.MIN_TARGET_DISTANCE:
                self.target_x = new_x
                self.target_y = new_y
                break

    def draw(self, ax):
        """
        Draws the circle on the given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The axis where the circle will be drawn.
        """
        self.shape = patches.Circle(self.pos(), self.radius, color=self.colour)
        ax.add_patch(self.shape)

    def update(self, fps, objects, grid):
        """
        Updates the circle's position by moving it towards its target.

        The circle moves in both x and y directions based on its target.
        If a collision is detected, the movement is adjusted accordingly.

        Args:
            fps (int): The frames per second for movement calculations.
            objects (list): A list of all objects in the scene.
            grid (dict): The spatial partitioning grid for optimized collision detection.
        """
        prev_x, prev_y = self.pos.x, self.pos.y
        max_speed = self.target_speed * (60 / fps)
        reached_target = self.pos.move_towards(self.target_x, self.target_y, max_speed)

        if reached_target:
            self.set_new_target()

        cell_x, cell_y = self.get_grid_cell()
        possible_collisions = grid.get((cell_x, cell_y), [])

        for other in possible_collisions:
            if other is not self and self.is_colliding(other):
                self.resolve_collision(other)
                self.shape.set_color("red")
            else:
                self.shape.set_color(self.colour)

        dx = self.pos.x - prev_x
        dy = self.pos.y - prev_y
        direction_length = np.hypot(dx, dy)

        if direction_length > 0.01:
            dx /= direction_length
            dy /= direction_length
            arrow_length = max(1, direction_length * 5)

            # Updates the direction arrow to indicate movement direction
            self.direction_arrow.set_data([self.pos.x, self.pos.x + dx * arrow_length], 
                                          [self.pos.y, self.pos.y + dy * arrow_length])

        self.shape.set_center(self.pos())
        self.label.set_position((self.pos.x, self.pos.y + self.radius + 0.5))

    def get_grid_cell(self):
        """ 
        Gets the grid cell coordinates based on the circle's position.

        Returns:
            tuple: A tuple containing the grid cell coordinates (x, y).
        """
        return int(self.pos.x // Config.GRID_SIZE), int(self.pos.y // Config.GRID_SIZE)

    def is_colliding(self, other):
        """ 
        Checks if this circle is colliding with another object.

        Supports collision detection with both `Circle` and `Rectangle` objects.

        Args:
            other (Obj): Another object in the scene.

        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        from Objects import Rectangle

        if isinstance(other, Circle):
            # Circle-to-Circle collision detection
            distance = np.hypot(self.pos.x - other.pos.x, self.pos.y - other.pos.y)
            return distance < (self.radius + other.radius)

        elif isinstance(other, Rectangle):
            # Circle-to-Rectangle collision detection (SAT method)
            circle_dist_x = abs(self.pos.x - (other.pos.x + other.width / 2))
            circle_dist_y = abs(self.pos.y - (other.pos.y + other.height / 2))

            if circle_dist_x > (other.width / 2 + self.radius) or circle_dist_y > (other.height / 2 + self.radius):
                return False

            if circle_dist_x <= (other.width / 2) or circle_dist_y <= (other.height / 2):
                return True

            corner_dist_sq = (circle_dist_x - other.width / 2) ** 2 + (circle_dist_y - other.height / 2) ** 2
            return corner_dist_sq <= (self.radius ** 2)

        return False  # No collision detected

    def resolve_collision(self, other):
        """ 
        Resolves collision by applying a bounce effect and setting a new target.

        The object moves away from the collision direction and finds a new target.

        Args:
            other (Obj): The object that this circle has collided with.
        """
        from Objects import Rectangle

        direction_x = self.pos.x - other.pos.x
        direction_y = self.pos.y - other.pos.y
        distance = np.hypot(direction_x, direction_y)

        # Cannot resolve collision if at the same position
        if distance == 0:
            return

        # Normalize direction vector
        direction_x /= distance
        direction_y /= distance

        if isinstance(other, Circle):
            # Circle-to-Circle collision resolution
            overlap = (self.radius + other.radius) - distance

        elif isinstance(other, Rectangle):
            # Circle-to-Rectangle collision resolution
            overlap_x = (self.radius + other.width / 2) - abs(self.pos.x - (other.pos.x + other.width / 2))
            overlap_y = (self.radius + other.height / 2) - abs(self.pos.y - (other.pos.y + other.height / 2))
            overlap = min(overlap_x, overlap_y)  # Uses the smallest overlap
        else:
            return

        bounce_x = direction_x * overlap * 0.5
        bounce_y = direction_y * overlap * 0.5

        self.pos.move(bounce_x, bounce_y)
        other.pos.move(-bounce_x, -bounce_y)

        self.set_new_target()
        other.set_new_target()
