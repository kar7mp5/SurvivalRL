import matplotlib.pyplot as plt
import numpy as np


class Position:    
    """ A simple class to store 2D position """
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __call__(self):
        return (self.x, self.y)

    def move(self, dx: float, dy: float):
        """ Move the position by dx and dy """
        self.x += dx
        self.y += dy


class Obj:
    """ Parent class for moving objects (shapes) """
    def __init__(self, x: float, y: float, colour: str):
        self.pos = Position(x, y)  # Position object
        self.colour = colour
        self.shape = None  # To be defined in subclasses

    def draw(self, ax):
        """ Adds shape to the axis (to be implemented in child classes) """
        raise NotImplementedError

    def update(self):
        """ Moves the object to a new random position within bounds """
        dx, dy = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
        self.pos.move(dx, dy)
        self.shape.set_center(self.pos())  # Move shape (for circles)