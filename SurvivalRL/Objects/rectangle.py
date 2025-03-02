from obj import Obj
from SurvivalRL import Config, GameObject

from matplotlib.transforms import Affine2D
import matplotlib.patches as patches
import matplotlib
import numpy as np


class Rectangle(Obj):
    """ 
    A Rectangle object that moves and rotates based on its movement direction.
    """

    def __init__(
        self,
        game: GameObject,
        ax: matplotlib.axes.Axes, 
        x: float, y: float, 
        width: float, height: float, 
        target_speed: float, 
        colour: str, 
        name: str = None):
        """
        Initializes a Rectangle object.

        Args:
            ax (matplotlib.axes.Axes): The axis where the rectangle will be drawn.
            x (float): Initial x-coordinate of the rectangle.
            y (float): Initial y-coordinate of the rectangle.
            width (float): Width of the rectangle.
            height (float): Height of the rectangle.
            colour (str): Color of the rectangle.
            name (str, optional): Name label displayed above the rectangle.
        """
        super().__init__(game, ax, x, y, target_speed, colour, name)

        self.width = width
        self.height = height
        self.rotation_angle = 0  

        # Direction arrow for movement visualization
        self.direction_arrow, = self.ax.plot([x, x], [y, y], color="red", marker="o", linewidth=2)

        # Label displaying the name of the rectangle
        self.label = self.ax.text(x + width / 2, y + height + 0.5, self.name, ha="center", va="bottom", fontsize=10, color="black")

        self.set_new_target()

    def set_new_target(self):
        """ 
        Sets a new random target position at a reasonable distance.
        
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

    def draw(self):
        """
        Draws the rectangle on the given matplotlib axis.

        Args:
            ax (matplotlib.axes.Axes): The axis where the rectangle will be drawn.
        """
        self.shape = patches.Rectangle(self.pos(), self.width, self.height, color=self.colour, angle=0)
        self.ax.add_patch(self.shape)

    def update(self, fps, grid):
        """
        Updates the rectangle's position by moving it towards its target.

        Also updates the direction arrow and rotates the rectangle to align with its movement direction.

        Args:
            fps (int): The frames per second for movement calculations.
            objects (list): A list of all objects in the scene.
            grid (dict): The spatial partitioning grid for optimized collision detection.
        """
        prev_x, prev_y = self.pos.x, self.pos.y
        max_speed = self.target_speed * (60 / fps)
        reached_target = self.pos.move_towards(self.target_x, self.target_y, max_speed)

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

            self.direction_arrow.set_data([self.pos.x + self.width / 2, self.pos.x + self.width / 2 + dx * arrow_length], 
                                          [self.pos.y + self.height / 2, self.pos.y + self.height / 2 + dy * arrow_length])

            self.rotation_angle = np.degrees(np.arctan2(dy, dx))
            self.apply_rotation()

        self.shape.set_xy(self.pos())
        self.label.set_position((self.pos.x + self.width / 2, self.pos.y + self.height + 0.5))

    def division(self, game: GameObject):
        """
        Divide Cells
        """
        game.add_object(Rectangle(
            ax=self.ax,
            x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            width=2,
            height=2,
            target_speed=np.random.uniform(0.1, 0.3),
            colour=np.random.choice(["blue", "green", "purple", "orange"]),
            name=f"Clone Rect"
        ))

    """
    Collision System
    """
    def apply_rotation(self):
        """ 
        Applies rotation transform to the rectangle.
        
        Uses `Affine2D` to rotate the rectangle around its center based on movement direction.
        """
        transform = Affine2D().rotate_deg_around(self.pos.x + self.width / 2, self.pos.y + self.height / 2, self.rotation_angle)
        self.shape.set_transform(transform + self.ax.transData)

    def get_grid_cell(self):
        """ 
        Gets the grid cell coordinates based on the rectangle's position.

        Returns:
            tuple: A tuple containing the grid cell coordinates (x, y).
        """
        return int(self.pos.x // Config.GRID_SIZE), int(self.pos.y // Config.GRID_SIZE)

    def is_colliding(self, other):
        """ 
        Checks if this rectangle is colliding with another object.

        Supports collision detection with both `Circle` and `Rectangle` objects.

        Args:
            other (Obj): Another object in the scene.

        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        from Objects import Circle  

        if isinstance(other, Rectangle):
            return (self.pos.x < other.pos.x + other.width and
                    self.pos.x + self.width > other.pos.x and
                    self.pos.y < other.pos.y + other.height and
                    self.pos.y + self.height > other.pos.y)

        elif isinstance(other, Circle):  
            circle_dist_x = abs(other.pos.x - (self.pos.x + self.width / 2))
            circle_dist_y = abs(other.pos.y - (self.pos.y + self.height / 2))

            if circle_dist_x > (self.width / 2 + other.radius) or circle_dist_y > (self.height / 2 + other.radius):
                return False

            if circle_dist_x <= (self.width / 2) or circle_dist_y <= (self.height / 2):
                return True

            corner_dist_sq = (circle_dist_x - self.width / 2) ** 2 + (circle_dist_y - self.height / 2) ** 2
            return corner_dist_sq <= (other.radius ** 2)

        return False  

    def resolve_collision(self, other):
        """ 
        Resolves collision by applying a bounce effect and setting a new target.

        The object moves away from the collision direction and finds a new target.

        Args:
            other (Obj): The object that this rectangle has collided with.
        """
        from SurvivalRL.Objects.circle import Circle  

        direction_x = self.pos.x - other.pos.x
        direction_y = self.pos.y - other.pos.y
        distance = np.hypot(direction_x, direction_y)

        if distance == 0:
            return  

        direction_x /= distance
        direction_y /= distance

        if isinstance(other, Rectangle):
            overlap_x = (self.width / 2 + other.width / 2) - abs(self.pos.x - other.pos.x)
            overlap_y = (self.height / 2 + other.height / 2) - abs(self.pos.y - other.pos.y)
            overlap = min(overlap_x, overlap_y)  

        elif isinstance(other, Circle):
            overlap_x = (other.radius + self.width / 2) - abs(self.pos.x - (other.pos.x))
            overlap_y = (other.radius + self.height / 2) - abs(self.pos.y - (other.pos.y))
            overlap = min(overlap_x, overlap_y)  

        else:
            return  

        bounce_x = direction_x * overlap * 0.3
        bounce_y = direction_y * overlap * 0.3

        self.pos.move(bounce_x, bounce_y)
        other.pos.move(-bounce_x, -bounce_y)

        self.set_new_target()
        other.set_new_target()