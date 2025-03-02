from obj import Obj

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

    def draw(self, ax):
        """
        Draws the circle on the given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The axis where the circle will be drawn.
        """
        self.shape = patches.Circle(self.pos(), self.radius, color=self.colour)
        ax.add_patch(self.shape)

    def update(self, fps):
        """
        Updates the circle's position by moving it randomly within a defined range.

        The circle moves in both x and y directions with a random displacement.
        The new position is applied to the shape.
        """
        speed_multiplier = 60 / fps
        dx, dy = np.random.uniform(-2, 1) * speed_multiplier, np.random.uniform(-1, 1) * speed_multiplier
        self.pos.move(dx, dy)
        self.shape.set_center(self.pos())