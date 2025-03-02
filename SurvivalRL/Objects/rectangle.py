from obj import Obj

import matplotlib.patches as patches
import numpy as np


class Rectangle(Obj):
    """ Rectangle object that moves randomly """
    def __init__(self, x: float, y: float, width: float, height: float, colour: str):
        super().__init__(x, y, colour)
        self.width = width
        self.height = height

    def draw(self, ax):
        """ Create and add rectangle shape to Axes """
        self.shape = patches.Rectangle(self.pos(), self.width, self.height, color=self.colour)
        ax.add_patch(self.shape)

    def update(self):
        """ Moves the rectangle and updates its position """
        dx, dy = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
        self.pos.move(dx, dy)
        self.shape.set_xy(self.pos())  # Move shape (for rectangles)