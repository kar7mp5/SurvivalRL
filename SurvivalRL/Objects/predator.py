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
    FOV_RADIUS = 10
    DIVISION_UNIT = 250
    ENERGY_UNIT = 500
    GRID_UPDATE_THRESHOLD = 10
 
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
        """Updates the rectangle's position and handles collisions."""
        super().update()

        # Energy logic
        self.energy -= 0.1
        if self.energy <= 0:
            self.remove()

        # Adjust size based on energy level
        self.width = max(1, self.init_width * self.energy / self.ENERGY_UNIT)
        self.height = max(1, self.init_height * self.energy / self.ENERGY_UNIT)
        
        # Update grid position only if the size change is significant
        if (abs(self.width - self.last_grid_width) >= self.GRID_UPDATE_THRESHOLD or
            abs(self.height - self.last_grid_height) >= self.GRID_UPDATE_THRESHOLD):
            self.last_grid_width = self.width
            self.last_grid_height = self.height

        from Objects import Herbivore
        prev_x, prev_y = self.pos.x, self.pos.y
        max_speed = self.target_speed * (60 / fps)
        reached_target = self.pos.move_towards(self.target_x, self.target_y, max_speed)

        # Check object in FOV
        if self.current_target and self.is_in_fov(self.current_target):
            detected_target = self.current_target
        else:
            detected_target = self.detect_in_fov(grid)
            self.current_target = detected_target
    
        # Detect objects in FOV
        if isinstance(detected_target, Predator):
            self.set_new_target()  
        elif isinstance(detected_target, Herbivore):
            self.target_x, self.target_y = detected_target.pos.x, detected_target.pos.y
            reached_target = self.pos.move_towards(self.target_x, self.target_y, max_speed)
        
        # Draw FOV fan shape
        self.draw_fov(detected_target is not None)

        if reached_target:
            self.set_new_target()

        # Check collision
        possible_collisions = grid.retrieve_nearby(self)
        for other in possible_collisions:
            if other is not self and self.aabb_collision(other):
                self.resolve_collision(other)
                self.shape.set_color("gray")
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
            self.direction_arrow.set_data(
                [self.pos.x + self.width / 2, self.pos.x + self.width / 2 + dx * arrow_length], 
                [self.pos.y + self.height / 2, self.pos.y + self.height / 2 + dy * arrow_length]
            )

            # Apply rectangle rotation
            self.rotation_angle = np.degrees(np.arctan2(dy, dx))
            self.apply_rotation()

        # Update debug label with movement tracking information
        if self.isDebug is True:
            self.label.set_text(f'{self.name}\nPos: ({self.pos.x:.2f}, {self.pos.y:.2f})\n'
                                f'Target: ({self.target_x:.2f}, {self.target_y:.2f})\n'
                                f'Speed: {max_speed:.2f}\nEnergy: {self.energy:.2f}')
            self.label.set_fontsize(Config.DEBUG_FONT_SIZE)

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
            energy=100,
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