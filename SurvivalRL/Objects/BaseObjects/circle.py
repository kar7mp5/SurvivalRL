from BaseObjects import BaseObject
from SurvivalRL import Config, GameObject

import matplotlib.patches as patches
import matplotlib
import numpy as np


class Circle(BaseObject):
    """ 
    A Circle class that inherits from Obj.
    This class represents a moving circular object in a 2D space.
    """

    def __init__(
        self,
        game: GameObject,
        ax: matplotlib.axes.Axes, 
        x: float, y: float, 
        energy: float, 
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
        super().__init__(game, ax, x, y, energy, target_speed, colour, name)
        
        self.radius = radius
        
        self.label = self.ax.text(x, y + radius + 0.5, 
                                  f'{self.name} {self.energy}', 
                                  ha="center", va="bottom", 
                                  fontsize=10, 
                                  color="black")

        # Always show a red rectangle around the hitbox in debugging mode
        if Config.DEBUG_MODE:
            self.hitbox = patches.Rectangle(
                (self.pos.x - self.radius, self.pos.y - self.radius),
                2 * self.radius, 2 * self.radius,
                linewidth=1, edgecolor='red', facecolor='none'
            )
            self.ax.add_patch(self.hitbox)

        # Direction arrow for movement visualization
        self.direction_arrow, = self.ax.plot([x, x], [y, y], 
                                             color="red", 
                                             linewidth=2, 
                                             marker="o", markersize=3)

    def update(self):
        # Update hitbox position
        if Config.DEBUG_MODE:
            self.hitbox.set_xy((self.pos.x - self.radius, self.pos.y - self.radius))

    def draw(self):
        """Draws the circle on the given matplotlib axis."""
        self.shape = patches.Circle(self.pos(), self.radius, color=self.colour)
        self.ax.add_patch(self.shape)

    """
    Collision System
    """
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
        from BaseObjects import Rectangle

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

        bounce_x = direction_x * overlap * 0.7
        bounce_y = direction_y * overlap * 0.7

        self.pos.move(bounce_x, bounce_y)
        other.pos.move(-bounce_x, -bounce_y)

        self.set_new_target()
        other.set_new_target()