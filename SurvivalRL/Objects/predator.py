from BaseObjects import Rectangle
from SurvivalRL import Config, GameObject

import matplotlib
import numpy as np
import matplotlib.patches as patches


class Predator(Rectangle):

    SHADES = [
        "#FF0000",  # Red
        "#CD0000",  # DarkRed
        "#DC143C",  # Crimson
        "#B22222",  # FireBrick
        "#8B0000",  # DarkRed (Deeper)
        "#FF4500",  # OrangeRed
        "#FA8072",  # Salmon
        "#E9967A",  # DarkSalmon
        "#FF6347",  # Tomato
        "#FF7F50"   # Coral
    ]

    isDebug: bool = Config.DEBUG_MODE and Config.PREDATOR

    FOV_ANGLE = 30
    FOV_RADIUS = 6
    DIVISION_UNIT = 500
    ENERGY_UNIT = 1000
    GRID_UPDATE_THRESHOLD = 0.2
 
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
        
        # Add FOV fan shape
        self.fov_patch = patches.Wedge(
            center=(self.pos.x + self.init_width/2, self.pos.y + self.init_height/2),
            r=self.FOV_RADIUS,
            theta1=0,
            theta2=0,
            color='cyan',
            alpha=0.3
        )
        self.ax.add_patch(self.fov_patch)

        self.current_target = None
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
        """
        Updates the predator's position, energy, FOV detection, reproduction, and collisions.

        Args:
            fps (int): The frames per second used for time-based movement.
            grid (dict): Spatial hash grid for efficient neighbor detection.
        """
        super().update()

        # Decrease energy over time; remove if energy is depleted
        self.energy -= 0.1
        if self.energy <= 0:
            self.remove()
            return

        # Adjust size based on current energy level
        self.width = max(2, self.init_width * self.energy / self.ENERGY_UNIT)
        self.height = max(2, self.init_height * self.energy / self.ENERGY_UNIT)
        if (abs(self.width - self.last_grid_width) >= self.GRID_UPDATE_THRESHOLD or
            abs(self.height - self.last_grid_height) >= self.GRID_UPDATE_THRESHOLD):
            self.last_grid_width = self.width
            self.last_grid_height = self.height

        # from Objects import Herbivore
        # # Detect herbivores within FOV and set target to chase
        # herbivore = self.detect_in_fov_type(grid, target_type=Herbivore)
        # if herbivore:
        #     dx = self.pos.x - herbivore.pos.x
        #     dy = self.pos.y - herbivore.pos.y
        #     norm = max(np.hypot(dx, dy), 1e-5)
        #     self.target_x = self.pos.x + (dx / norm) * 5
        #     self.target_y = self.pos.y + (dy / norm) * 5

        # Move toward target
        prev_x, prev_y = self.pos.x, self.pos.y
        max_speed = self.target_speed * (60 / fps)
        reached_target = self.pos.move_towards(self.target_x, self.target_y, max_speed)

        # Update FOV visualization based on current direction
        herbivore = None
        self.draw_fov(herbivore is not None)

        # If reached target location, choose a new random destination
        if reached_target:
            self.set_new_target()

        # Check for collisions with nearby objects
        possible_collisions = grid.retrieve_nearby(self)
        for other in possible_collisions:
            if other is self:
                continue
            if self.aabb_collision(other):
                self.resolve_collision(other)
                self.shape.set_color("gray")
            else:
                self.shape.set_color(self.colour)

        # Update movement direction arrow
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

        # Update debug label (if enabled)
        if self.isDebug:
            self.label.set_text(f'{self.name}\nPos: ({self.pos.x:.2f}, {self.pos.y:.2f})\n'
                                f'Target: ({self.target_x:.2f}, {self.target_y:.2f})\n'
                                f'Speed: {max_speed:.2f}\nEnergy: {self.energy:.2f}')
            self.label.set_fontsize(Config.DEBUG_FONT_SIZE)

        self.shape.set_xy(self.pos())
        self.label.set_position((self.pos.x + self.width / 2, self.pos.y + self.height + 0.5))

        # Attempt reproduction with nearby predator
        # self.try_reproduce_in_fov(grid)

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

        from Objects import Herbivore, Plant
        # Consume energy from entities
        if isinstance(other, Herbivore):
            self.energy += other.energy * 0.9
            other.remove()
        elif isinstance(other, Plant):
            self.energy += other.energy * 0.5
            other.remove()
        # Duplicate self
        elif isinstance(other, Predator):
            energy_sum = self.energy + other.energy
            if energy_sum >= self.DIVISION_UNIT:
                self.energy -= (self.energy / energy_sum) * self.DIVISION_UNIT 
                other.energy -= (other.energy / energy_sum) * self.DIVISION_UNIT
                self.division()

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
            energy=self.DIVISION_UNIT,
            width=self.width,
            height=self.height,
            target_speed=np.random.uniform(0.1, 0.3),
            colour=np.random.choice(self.SHADES),
        ))

    def remove(self):
        """
        Removes the Predator from the game and also from the matplotlib figure.
        """
        if self in self.game.objects:
            self.game.objects.remove(self)  # Remove from the game list
            
            if self.shape is not None:
                self.shape.remove()
            if hasattr(self, "direction_arrow"):
                self.direction_arrow.remove()
            if hasattr(self, "label"):
                self.label.remove()
            if hasattr(self, "hitbox"):
                self.hitbox.remove()
            if hasattr(self, "fov_patch"):
                    self.fov_patch.remove()
            
            del self  # Delete the object

    """
    FOV
    """
    def is_in_fov(self, obj):
        """
        Checks if the given object is still inside the Predator's FOV.
        """
        if obj is None:
            return False

        dx = obj.pos.x - self.pos.x
        dy = obj.pos.y - self.pos.y
        distance_sq = dx * dx + dy * dy  # Euclidean distance squared

        if distance_sq > self.FOV_RADIUS ** 2:
            return False  # Out of range

        # Calculate FOV degree
        angle = np.degrees(np.arctan2(dy, dx))
        direction_angle = np.degrees(np.arctan2(self.target_y - self.pos.y, self.target_x - self.pos.x))
        angle_diff = (angle - direction_angle + 180) % 360 - 180  

        return abs(angle_diff) <= self.FOV_ANGLE / 2

    def detect_in_fov(self, grid):
        """
        Detects the nearest object within the Field of View (FOV) using Spatial Hash Grid.
        """
        possible_objects = grid.retrieve_in_fov_range(self.pos.x, self.pos.y, self.FOV_RADIUS)
        best_target = None
        min_distance_sq = self.FOV_RADIUS ** 2  # Calculate distance

        for obj in possible_objects:
            if obj is self:
                continue

            dx = obj.pos.x - self.pos.x
            dy = obj.pos.y - self.pos.y
            distance_sq = dx * dx + dy * dy  # Rather than np.hypot

            if distance_sq > min_distance_sq:
                continue  # Out of FOV range

            # Calculate FOV degree
            angle = np.degrees(np.arctan2(dy, dx))
            direction_angle = np.degrees(np.arctan2(self.target_y - self.pos.y, self.target_x - self.pos.x))
            angle_diff = (angle - direction_angle + 180) % 360 - 180  

            if abs(angle_diff) <= self.FOV_ANGLE / 2 and distance_sq < min_distance_sq:
                best_target = obj
                min_distance_sq = distance_sq

        return best_target

    def detect_in_fov_type(self, grid, target_type):
        """
        Detects the nearest object of the given type within the predator's FOV.
        """
        candidates = grid.retrieve_in_fov_range(self.pos.x, self.pos.y, self.FOV_RADIUS)
        best_target = None
        min_distance_sq = self.FOV_RADIUS ** 2

        for obj in candidates:
            if obj is self or not isinstance(obj, target_type):
                continue
            dx = obj.pos.x - self.pos.x
            dy = obj.pos.y - self.pos.y
            dist_sq = dx * dx + dy * dy
            angle = np.degrees(np.arctan2(dy, dx))
            dir_angle = np.degrees(np.arctan2(self.target_y - self.pos.y, self.target_x - self.pos.x))
            diff = (angle - dir_angle + 180) % 360 - 180

            if dist_sq < min_distance_sq and abs(diff) <= self.FOV_ANGLE / 2:
                best_target = obj
                min_distance_sq = dist_sq

        return best_target

    def try_reproduce_in_fov(self, grid):
        """
        Attempts reproduction with another Predator in FOV if both have sufficient energy.
        """
        from Objects import Predator
        mate = self.detect_in_fov_type(grid, Predator)
        if mate and mate is not self:
            dx = mate.pos.x - self.pos.x
            dy = mate.pos.y - self.pos.y
            distance_sq = dx * dx + dy * dy
            if distance_sq < 4.0:  # Close enough to reproduce
                if self.energy >= self.DIVISION_UNIT and mate.energy >= self.DIVISION_UNIT:
                    self.energy -= self.DIVISION_UNIT // 2
                    mate.energy -= self.DIVISION_UNIT // 2
                    self.division()

    def draw_fov(self, target_detected):
        """
        Visualizes the Field of View (FOV) as a sector (wedge).
        """
        direction_angle = np.degrees(np.arctan2(self.target_y - self.pos.y, self.target_x - self.pos.x))

        self.fov_patch.set_center((self.pos.x + self.width/2, self.pos.y + self.height/2))
        self.fov_patch.set_theta1(direction_angle - self.FOV_ANGLE / 2)
        self.fov_patch.set_theta2(direction_angle + self.FOV_ANGLE / 2)

        # Adjust FOV radius based on energy level
        adjusted_fov_radius = max(5, self.FOV_RADIUS * (self.energy / self.ENERGY_UNIT))
        self.fov_patch.set_radius(adjusted_fov_radius)
        
        # Change color when a target is detected
        self.fov_patch.set_color("red" if target_detected else "cyan")
        self.fov_patch.set_alpha(0.3) # Set transparency fro visibility