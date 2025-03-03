import matplotlib.axes
from BaseObjects import Circle
from SurvivalRL import Config, GameObject

import matplotlib.patches as patches
import matplotlib
import numpy as np


class Plant(Circle):
    
    def __init__(
        self, 
        game: GameObject, 
        ax: matplotlib.axes.Axes, 
        x: float, y: float, 
        energy: float,
        radius: float, 
        colour: str, 
        name: str = None):
        super().__init__(game, ax, x, y, energy, radius, 0, colour, name)
        
    def update(self, fps, grid):        
        # Update debug label with movement tracking information
        super().update()
        if Config.DEBUG_MODE is True:
            self.label.set_text(f'{self.name}\nPos: ({self.pos.x:.2f}, {self.pos.y:.2f})\n'
                                f'Energy: {self.energy:.2f}')

            self.label.set_fontsize(6)

        self.shape.set_center(self.pos())
        self.label.set_position((self.pos.x, self.pos.y + self.radius + 0.5))

    def division(self):
        """
        Divide Cells
        """
        return None

    def set_new_target(self):
        return None
    
    def resolve_collision(self, other):
        # super().resolve_collision(other)
        other.set_new_target()

    def remove(self):
        """
        Removes the Predator from the game and also from the matplotlib figure.
        """
        if self in self.game.objects:
            self.game.objects.remove(self)  # Remove from the game list
            
            # Remove from the matplotlib figure
            if self.shape is not None:
                self.shape.remove()

            # Remove movement arrow if exists
            if hasattr(self, "direction_arrow"):
                self.direction_arrow.remove()

            # Remove the name label if exists
            if hasattr(self, "label"):
                self.label.remove()
            
            # Remove the hitbox if exists
            if hasattr(self, "hitbox"):
                self.hitbox.remove()

            del self  # Delete the object