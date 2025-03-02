from Objects import Circle
from SurvivalRL import Config, GameObject

import matplotlib.patches as patches
import matplotlib
import numpy as np


class Plant(Circle):
    
    def __init__(self, game, ax, x, y, radius, colour, name = None):
        super().__init__(game, ax, x, y, radius, 0, colour, name)
    
    def update(self, fps, grid):
        self.shape.set_center(self.pos())
        self.label.set_position((self.pos.x, self.pos.y + self.radius + 0.5))

    def division(self):
        """
        Divide Cells
        """
        self.game.add_object(Circle(
            game=self.game,
            ax=self.ax,
            x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            radius=1,
            target_speed=np.random.uniform(0.1, 0.3),
            colour=np.random.choice(["blue", "green", "purple", "orange"]),
            name=f"Clone Cell"
        ))