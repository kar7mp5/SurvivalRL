from BaseObjects import Circle
from SurvivalRL import Config, GameObject

import matplotlib
import numpy as np


class Plant(Circle):

    SHADES = [
        "#008000",  # Green
        "#006400",  # DarkGreen
        "#228B22",  # ForestGreen
        "#32CD32",  # LimeGreen
        "#00FF00",  # Lime
        "#7CFC00",  # LawnGreen
        "#ADFF2F",  # GreenYellow
        "#9ACD32",  # YellowGreen
        "#6B8E23",  # OliveDrab
        "#20B2AA"   # LightSeaGreen (blue-green)
    ]

    isDebug: bool = Config.DEBUG_MODE and Config.PLANT

    def __init__(
        self, 
        game: GameObject, 
        ax: matplotlib.axes.Axes, 
        x: float, y: float, 
        energy: float,
        radius: float, 
        colour: str, 
        name: str = None):
        super().__init__(game, ax, x, y, energy, radius, 0.01, colour, name)

        # self.set_new_target()

    def update(self, fps, grid):        
        # Update debug label with movement tracking information
        super().update()
        prev_x, prev_y = self.pos.x, self.pos.y

        # Update debug label with movement tracking information
        if self.isDebug is True:
            if self.name is not None:
                self.label.set_text(f'{self.name}\nPos: ({self.pos.x:.2f}, {self.pos.y:.2f})\n'
                                    f'Energy: {self.energy:.2f}')
            else:
                self.label.set_text(f'Pos: ({self.pos.x:.2f}, {self.pos.y:.2f})\n'
                                    f'Energy: {self.energy:.2f}')
            self.label.set_fontsize(6)


        self.energy += 1
        if self.energy >= 100:
            self.energy -= 80
            self.division()

        # Check collision
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

    def division(self):
        """
        Creates a new Predator instance (cell division).
        
        A new predator with similar properties is added to the game at a random position.
        """
        choice = np.random.choice([-1, 1])
        random_x = self.radius * choice * (1 + np.random.uniform(0, 1))
        choice = np.random.choice([-1, 1])
        random_y = self.radius * choice * (1 + np.random.uniform(0, 1))
        self.game.add_object(Plant(
            game=self.game,
            ax=self.ax,
            x=self.pos.x + random_x,
            y=self.pos.x + random_y,
            energy=self.energy // 4,
            radius=self.radius,
            colour=np.random.choice(self.SHADES),
        ))

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
        super().resolve_collision(other)

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