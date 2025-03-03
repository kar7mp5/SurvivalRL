from BaseObjects import BaseObject

from SurvivalRL import Config, GameObject
from matplotlib.transforms import Affine2D
import matplotlib.patches as patches
import matplotlib
import numpy as np


def _simplex_contains_origin(simplex):
    """
    Checks if the simplex contains the origin using the winding number test.

    Args:
        simplex (list): A list of points forming a simplex.

    Returns:
        bool: True if the simplex contains the origin, False otherwise.
    """
    if len(simplex) == 2:
        a, b = np.array(simplex[0]), np.array(simplex[1])
        ab = b - a
        ao = -a

        # Ensure 2D vectors
        if ab.shape != (2,) or ao.shape != (2,):
            return False  # Invalid data

        return np.dot(np.cross(ab, ao), np.cross(ab, ab)) >= 0

    elif len(simplex) == 3:
        a, b, c = np.array(simplex[0]), np.array(simplex[1]), np.array(simplex[2])
        ab = b - a
        ac = c - a
        ao = -a

        # Ensure 2D vectors
        if ab.shape != (2,) or ac.shape != (2,) or ao.shape != (2,):
            return False  # Invalid data

        # Compute perpendicular edges
        ab_perp = np.cross(ab, ao)
        ac_perp = np.cross(ac, ao)

        # Ensure result is scalar
        if np.isscalar(ab_perp) and np.isscalar(ac_perp):
            if ab_perp > 0:
                return False
            elif ac_perp > 0:
                return False

            return True  # Origin is inside the simplex

    return False  # Invalid simplex case


class Rectangle(BaseObject):
    """ 
    A Rectangle object that moves and rotates based on its movement direction.
    """

    def __init__(
        self,
        game: GameObject,
        ax: matplotlib.axes.Axes, 
        x: float, y: float, 
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
        super().__init__(game, ax, x, y, energy, target_speed, colour, name)

        self.width = width
        self.height = height
        self.rotation_angle = 0  

        self.direction_arrow, = self.ax.plot([x, x], [y, y], 
                                             color="brown", 
                                             linewidth=2, 
                                             marker="o", markersize=3)

        self.label = self.ax.text(x + width / 2, y + height + 0.5, 
                                  f'{self.name} {self.energy}', 
                                  ha="center", va="bottom", 
                                  fontsize=10, 
                                  color="black")

        # Always show a red rectangle around the hitbox in debugging mode
        if Config.DEBUG_MODE:
            self.hitbox = patches.Rectangle(
                (self.pos.x, self.pos.y),
                self.width, self.height,
                linewidth=1, edgecolor='red', facecolor='none'
            )
            self.ax.add_patch(self.hitbox)

        self.set_new_target()

    def update(self):
        # Update hitbox position
        if Config.DEBUG_MODE:
            self.hitbox.set_xy((self.pos.x, self.pos.y))

    def draw(self):
        """Draws the circle on the given matplotlib axis."""
        if not hasattr(self, "shape") or self.shape is None:
            self.shape = patches.Rectangle(self.pos(), self.width, self.height, color=self.colour, angle=0)
            self.ax.add_patch(self.shape)

    """
    Collision System
    """
    def get_grid_cell(self):
        """ 
        Gets the grid cell coordinates based on the circle's position.

        Returns:
            tuple: A tuple containing the grid cell coordinates (x, y).
        """
        return int(self.pos.x // Config.GRID_SIZE), int(self.pos.y // Config.GRID_SIZE)

    def aabb_collision(self, other):
        """Checks AABB for Rectangle vs Rectangle and applies Circle collision detection."""
        from Objects import Circle

        if isinstance(other, Rectangle):
            return (
                self.pos.x < other.pos.x + other.width and
                self.pos.x + self.width > other.pos.x and
                self.pos.y < other.pos.y + other.height and
                self.pos.y + self.height > other.pos.y
            )

        elif isinstance(other, Circle):
            nearest_x = max(self.pos.x, min(other.pos.x, self.pos.x + self.width))
            nearest_y = max(self.pos.y, min(other.pos.y, self.pos.y + self.height))
            
            dx = other.pos.x - nearest_x
            dy = other.pos.y - nearest_y
            distance_squared = dx**2 + dy**2

            return distance_squared < other.radius ** 2

    def apply_rotation(self):
        """ 
        Applies rotation transform to the rectangle.
        
        Uses `Affine2D` to rotate the rectangle around its center based on movement direction.
        """
        if self.shape is None:
            return

        transform = Affine2D().rotate_deg_around(self.pos.x + self.width / 2, self.pos.y + self.height / 2, self.rotation_angle)
        self.shape.set_transform(transform + self.ax.transData)

    def is_colliding(self, other):
        """ 
        Uses the GJK (Gilbert-Johnson-Keerthi) algorithm to check if this rotated rectangle collides with another object.

        Args:
            other (Obj): Another object in the scene.

        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        from BaseObjects import Circle  

        if isinstance(other, Rectangle):
            return self._gjk_collision_rectangle(other)
        
        elif isinstance(other, Circle):
            return self._gjk_collision_circle(other)

        return False  # No collision

    def _gjk_collision_rectangle(self, other):
        """
        Uses the GJK (Gilbert-Johnson-Keerthi) algorithm to check collision between two rotated rectangles.

        Args:
            other (Rectangle): Another rectangle object.

        Returns:
            bool: True if the rectangles collide, False otherwise.
        """
        def support(shape1, shape2, direction):
            """ Finds the farthest point in the Minkowski Difference along a given direction. """
            farthest1 = max(shape1, key=lambda p: np.dot(p, direction))
            farthest2 = min(shape2, key=lambda p: np.dot(p, direction))
            return np.array(farthest1) - np.array(farthest2)

        shape1 = self._get_rotated_corners()
        shape2 = other._get_rotated_corners()

        direction = np.array([1, 0])  # Initial direction
        simplex = [support(shape1, shape2, direction)]

        while True:
            direction = -simplex[-1]  # Move towards the origin
            new_point = support(shape1, shape2, direction)

            if np.dot(new_point, direction) < 0:
                return False  # No collision

            simplex.append(new_point)

            if len(simplex) == 3 and _simplex_contains_origin(simplex):
                return True  # Collision detected

    def _gjk_collision_circle(self, circle):
        """
        Uses the GJK (Gilbert-Johnson-Keerthi) algorithm to check collision between a rotated rectangle and a circle.

        Args:
            circle (Circle): A circle object.

        Returns:
            bool: True if the rectangle and circle collide, False otherwise.
        """
        def support(shape1, shape2, direction):
            """ Finds the farthest point in the Minkowski Difference along a given direction. """
            farthest1 = max(shape1, key=lambda p: np.dot(p, direction))
            farthest2 = min(shape2, key=lambda p: np.dot(p, direction))
            return np.array(farthest1) - np.array(farthest2)

        shape1 = self._get_rotated_corners()
        shape2 = [circle.pos()]  # Treat circle as a single point

        direction = np.array([1, 0])  # Initial direction
        simplex = [support(shape1, shape2, direction)]

        while True:
            direction = -simplex[-1]
            new_point = support(shape1, shape2, direction)

            if np.dot(new_point, direction) < 0:
                return False  # No collision

            simplex.append(new_point)

            if len(simplex) == 3 and _simplex_contains_origin(simplex):
                return True  # Collision detected

    def _get_rotated_corners(self):
        """
        Computes the four corners of the rotated rectangle.

        Returns:
            list: A list of (x, y) tuples representing the corners of the rectangle.
        """
        cx, cy = self.pos.x + self.width / 2, self.pos.y + self.height / 2  # Rectangle center
        hw, hh = self.width / 2, self.height / 2  # Half-width and half-height

        # Rotation matrix
        angle_rad = np.radians(self.rotation_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Compute corners
        corners = [
            (cx + cos_a * hw - sin_a * hh, cy + sin_a * hw + cos_a * hh),
            (cx - cos_a * hw - sin_a * hh, cy - sin_a * hw + cos_a * hh),
            (cx - cos_a * hw + sin_a * hh, cy - sin_a * hw - cos_a * hh),
            (cx + cos_a * hw + sin_a * hh, cy + sin_a * hw - cos_a * hh)
        ]

        return corners

    def resolve_collision(self, other):
        """Handles collision response by applying bounce effect."""
        from Objects import Circle  

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

        bounce_x = direction_x * overlap * 0.7
        bounce_y = direction_y * overlap * 0.7

        self.pos.move(bounce_x, bounce_y)
        other.pos.move(-bounce_x, -bounce_y)

        self.set_new_target()
        other.set_new_target()