from obj import Obj

import matplotlib.patches as patches


class Circle(Obj):
    """ Circle object that moves randomly """
    def __init__(self, x: float, y: float, radius: float, colour: str):
        super().__init__(x, y, colour)
        self.radius = radius

    def draw(self, ax):
        """ Create and add circle shape to Axes """
        self.shape = patches.Circle(self.pos(), self.radius, color=self.colour)
        ax.add_patch(self.shape)
