import matplotlib.axes
from BaseObjects import Circle
from SurvivalRL import Config, GameObject

import matplotlib.patches as patches
import matplotlib
import numpy as np


class Herbivore(Circle):
    
    def __init__(
        self, 
        game: GameObject,
        ax: matplotlib.axes.Axes, 
        x: float,
        y: float,
        energy: float,
        radius: float, 
        target_speed: float, 
        colour: str, 
        name: str = None):
        super().__init__(game, ax, x, y, energy, radius, target_speed, colour, name)

        self.test = True

        self.set_new_target()

    def update(self, fps, grid):
        """
        Updates the circle's position by moving it towards its target.

        The circle moves in both x and y directions based on its target.
        If a collision is detected, the movement is adjusted accordingly.

        Args:
            fps (int): The frames per second for movement calculations.
            objects (list): A list of all objects in the scene.
            grid (dict): The spatial partitioning grid for optimized collision detection.
        """
        super().update()
        prev_x, prev_y = self.pos.x, self.pos.y
        max_speed = self.target_speed * (60 / fps)
        reached_target = self.pos.move_towards(self.target_x, self.target_y, max_speed)

        # Update debug label with movement tracking information
        if Config.DEBUG_MODE is True:
            self.label.set_text(f'{self.name}\nPos: ({self.pos.x:.2f}, {self.pos.y:.2f})\n'
                                f'Target: ({self.target_x:.2f}, {self.target_y:.2f})\n'
                                f'Speed: {max_speed:.2f}\nEnergy: {self.energy:.2f}')
            self.label.set_fontsize(6)

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

        dx = self.pos.x - prev_x
        dy = self.pos.y - prev_y
        direction_length = np.hypot(dx, dy)

        if direction_length > 0.01:
            dx /= direction_length
            dy /= direction_length
            arrow_length = max(1, direction_length * 5)

            # Updates the direction arrow to indicate movement direction
            self.direction_arrow.set_data([self.pos.x, self.pos.x + dx * arrow_length], 
                                          [self.pos.y, self.pos.y + dy * arrow_length])

        self.shape.set_center(self.pos())
        self.label.set_position((self.pos.x, self.pos.y + self.radius + 0.5))

    def set_new_target(self):
        """ 
        Sets a new random target position within a reasonable distance.
        
        Ensures that the new target is not too close to the current position.
        """
        while True:
            new_x = np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2)
            new_y = np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2)
            distance = np.hypot(new_x - self.pos.x, new_y - self.pos.y)

            if distance > Config.MIN_TARGET_DISTANCE:
                self.target_x = new_x
                self.target_y = new_y
                break
        
    def resolve_collision(self, other):
        from Objects import Plant
        super().resolve_collision(other)
        # if self.test:
        #     self.division()
        #     self.test = False
        if isinstance(other, Plant):
            self.division()
            other.remove()
        self.set_new_target()
        
    def division(self):
        """
        Creates a new Predator instance (cell division).
        
        A new predator with similar properties is added to the game at a random position.
        """
        self.game.add_object(Herbivore(
            game=self.game,
            ax=self.ax,
            x=self.pos.x + np.random.uniform(-1, 1),
            y=self.pos.x + np.random.uniform(-1, 1),
            energy=100,
            radius=self.radius,
            target_speed=np.random.uniform(0.1, 0.3),
            colour=np.random.choice(["purple", "orange"]),
            name=f"Herbivore Clone",
        ))

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