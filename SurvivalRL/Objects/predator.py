import matplotlib.axes
from BaseObjects import Rectangle
from SurvivalRL import Config, GameObject

import matplotlib
import numpy as np


class Predator(Rectangle):

    def __init__(
        self, 
        game: GameObject, 
        ax: matplotlib.axes.Axes, 
        x: float, 
        y: float,
        energy: float,
        width: float, height: float,
        target_speed: float, 
        colour: str, 
        name: str = None):
        """
        Initializes a Rectangle object.

        Args:
            game (GameObject): The game instance managing all objects.
            ax (matplotlib.axes.Axes): The axis where the rectangle will be drawn.
            x (float): Initial x-coordinate of the rectangle.
            y (float): Initial y-coordinate of the rectangle.
            width (float): Width of the rectangle.
            height (float): Height of the rectangle.
            target_speed (float): Speed of movement.
            colour (str): Color of the rectangle.
            name (str, optional): Name label displayed above the rectangle.
        """
        super().__init__(game, ax, x, y, energy, width, height, target_speed, colour, name)
        
        self.set_new_target()

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

    def update(self, fps, grid):
        """Updates the rectangle's position and handles collisions."""
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
            if other is not self and self.aabb_collision(other):
                self.resolve_collision(other)

        dx = self.pos.x - prev_x
        dy = self.pos.y - prev_y
        direction_length = np.hypot(dx, dy)

        if direction_length > 0.01:
            dx /= direction_length
            dy /= direction_length
            arrow_length = max(1, direction_length * 5)

            self.direction_arrow.set_data(
                [self.pos.x + self.width / 2, self.pos.x + self.width / 2 + dx * arrow_length], 
                [self.pos.y + self.height / 2, self.pos.y + self.height / 2 + dy * arrow_length]
            )

            self.rotation_angle = np.degrees(np.arctan2(dy, dx))
            self.apply_rotation()

        self.shape.set_xy(self.pos())
        self.label.set_position((self.pos.x + self.width / 2, self.pos.y + self.height + 0.5))

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
        from BaseObjects import Circle
        super().resolve_collision(other)
        if isinstance(other, Circle):
            other.remove()
        self.set_new_target()
        
    def division(self):
        """
        Creates a new Predator instance (cell division).
        
        A new predator with similar properties is added to the game at a random position.
        """
        self.game.add_object(Predator(
            game=self.game,
            ax=self.ax,
            x=self.pos.x + np.random.uniform(-5, 5),
            y=self.pos.x + np.random.uniform(-5, 5),
            energy=100,
            width=self.width,
            height=self.height,
            target_speed=np.random.uniform(0.1, 0.3),
            colour=np.random.choice(["purple", "orange"]),
            name=f"Predator Clone"
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