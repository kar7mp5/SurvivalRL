from obj import Obj

import matplotlib.patches as patches
import numpy as np


class Rectangle(Obj):
    """ 
    A Rectangle object that moves randomly in a 2D space. 
    This class inherits from Obj and represents a moving rectangular shape.
    """

    def __init__(self, x: float, y: float, width: float, height: float, colour: str):
        """
        Initializes a Rectangle object.

        Args:
            x (float): Initial x-coordinate of the rectangle.
            y (float): Initial y-coordinate of the rectangle.
            width (float): Width of the rectangle.
            height (float): Height of the rectangle.
            colour (str): Color of the rectangle.
        """
        super().__init__(x, y, colour)
        self.width = width
        self.height = height

    def draw(self, ax):
        """
        Creates and adds the rectangle shape to the given matplotlib Axes.

        Args:
            ax (matplotlib.axes.Axes): The axis where the rectangle will be drawn.
        """
        self.shape = patches.Rectangle(self.pos(), self.width, self.height, color=self.colour)
        ax.add_patch(self.shape)

    def update(self, fps):
        """
        Updates the rectangle's position by moving it randomly within a defined range.

        The rectangle moves in both x and y directions with a random displacement.
        The new position is applied to the shape.
        """
        speed_multiplier = 60 / fps
        dx, dy = np.random.uniform(-2, 1) * speed_multiplier, np.random.uniform(-1, 1) * speed_multiplier
        self.pos.move(dx, dy)
        self.shape.set_xy(self.pos())
