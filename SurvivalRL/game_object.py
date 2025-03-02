import matplotlib.pyplot as plt
import numpy as np


WINDOW_SIZE = 20

class GameObject:
    """ Manages all objects in the game """
    def __init__(self, ax):
        self.ax = ax
        self.objects = []

    def add_object(self, obj):
        """ Add an object to the game and draw it """
        obj.draw(self.ax)
        self.objects.append(obj)

    def update(self):
        """ Updates all objects in each frame """
        for obj in self.objects:
            obj.update()
        return [obj.shape for obj in self.objects]