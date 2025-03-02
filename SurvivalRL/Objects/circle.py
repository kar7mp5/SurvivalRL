from obj import Obj
from SurvivalRL import Config

import matplotlib.patches as patches
import numpy as np


class Circle(Obj):
    """ 
    A Circle class that inherits from Obj.
    This class represents a moving circular object in a 2D space.
    """

    def __init__(self, x: float, y: float, radius: float, colour: str):
        """
        Initializes a Circle object.

        Args:
            x (float): Initial x-coordinate of the circle.
            y (float): Initial y-coordinate of the circle.
            radius (float): Radius of the circle.
            colour (str): Color of the circle.
        """
        super().__init__(x, y, colour)
        self.radius = radius
        self.set_new_target()

    def set_new_target(self):
        """ Sets a new random target position at a reasonable distance. """
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
        Updates the circle's position by moving it randomly within a defined range.

        The circle moves in both x and y directions with a random displacement.
        The new position is applied to the shape.
        """
        max_speed = 1/4 * (60 / fps)
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

        self.shape.set_center(self.pos())

    def get_grid_cell(self):
        """ Gets the grid cell coordinates based on the circle's position. """
        return int(self.pos.x // Config.GRID_SIZE), int(self.pos.y // Config.GRID_SIZE)

    def is_colliding(self, other):
        """ Checks if this circle is colliding with another circle. """
        distance = np.hypot(self.pos.x - other.pos.x, self.pos.y - other.pos.y)
        return distance < (self.radius + other.radius)

    def resolve_collision(self, other):
        """ Resolves collision by applying a random bounce direction. """
        direction_x = self.pos.x - other.pos.x
        direction_y = self.pos.y - other.pos.y
        distance = np.hypot(direction_x, direction_y)

        if distance == 0:
            return

        direction_x /= distance
        direction_y /= distance

        overlap = (self.radius + other.radius) - distance

        random_angle = np.random.uniform(0, 2 * np.pi)
        bounce_x = np.cos(random_angle) * overlap * 0.3
        bounce_y = np.sin(random_angle) * overlap * 0.3

        self.pos.move(bounce_x, bounce_y)
        other.pos.move(-bounce_x, -bounce_y)

        self.set_new_target()