from collections import defaultdict
from SurvivalRL import Config
import numpy as np
import matplotlib.patches as patches


class SpatialHashGrid:
    """
    Efficient spatial partitioning using a hash grid for fast collision detection.
    """
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = {}

    def _get_cell_key(self, x, y):
        return (int(x // self.cell_size), int(y // self.cell_size))

    def insert(self, obj):
        """Insert object into the correct grid cell."""
        cell_key = self._get_cell_key(obj.pos.x, obj.pos.y)
        if cell_key not in self.grid:
            self.grid[cell_key] = []
        self.grid[cell_key].append(obj)

    def retrieve_nearby(self, obj):
        """Retrieve nearby objects for collision checking."""
        cell_x, cell_y = self._get_cell_key(obj.pos.x, obj.pos.y)
        possible_objects = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cell_key = (cell_x + dx, cell_y + dy)
                if cell_key in self.grid:
                    possible_objects.extend(self.grid[cell_key])
        return possible_objects

    def retrieve_in_fov_range(self, x, y, fov_radius):
        """Retrieve all objects within a given FOV radius using an adaptive cell range."""
        cell_x, cell_y = self._get_cell_key(x, y)
        search_radius = int(np.ceil(fov_radius / self.cell_size))  # 검색할 셀 개수

        possible_objects = []
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                cell_key = (cell_x + dx, cell_y + dy)
                if cell_key in self.grid:
                    possible_objects.extend(self.grid[cell_key])

        return possible_objects

    def clear(self):
        """Clear the spatial hash grid (for each frame update)."""
        self.grid = {}


class GameObject:
    """ 
    Manages all objects in the game using a Spatial Hash Grid.
    """

    def __init__(self, ax, cell_size=Config.GRID_SIZE):
        """
        Initializes the GameObject manager.

        Args:
            ax (matplotlib.axes.Axes): The axis where objects will be drawn and managed.
            cell_size (int): Size of each cell in the spatial hash grid.
        """
        self.ax = ax
        self.objects = []
        self.spatial_grid = SpatialHashGrid(cell_size)
        self.grid_patches = []  # Stores grid patches for visualization.

    def add_object(self, obj):
        """
        Adds an object to the game and draws it on the axis.

        Args:
            obj (Obj): An instance of a game object (e.g., Circle, Rectangle).
        """
        obj.draw()
        self.objects.append(obj)

    def update(self, fps):
        """
        Updates all objects in the game by calling their respective update methods.

        Args:
            fps (float): Frames per second value to ensure smooth updates.

        Returns:
            list: A list of updated shapes for animation rendering.
        """
        # Clear the Spatial Hash Grid before inserting objects
        self.spatial_grid.clear()

        # Insert all objects into the Spatial Hash Grid
        for obj in self.objects:
            self.spatial_grid.insert(obj)

        # Update all objects
        for obj in self.objects:
            obj.update(fps, self.spatial_grid)

        self.draw_grid()  # Redraw the spatial grid for visualization

        return [obj.shape for obj in self.objects]

    def draw_grid(self):
        """ 
        Draws the spatial hash grid on the figure, highlighting active cells and FOV-affected cells.
        """
        # Remove existing grid patches before redrawing
        for patch in self.grid_patches:
            patch.remove()
        self.grid_patches.clear()

        # Stores active grid cells and FOV-affected cells
        fov_cells = set()
        object_cells = set()

        # Store cells that contain objects
        for obj in self.objects:
            cell_x, cell_y = self.spatial_grid._get_cell_key(obj.pos.x, obj.pos.y)
            object_cells.add((cell_x, cell_y))

            # Add all cells within the FOV radius if the object has an FOV property
            if getattr(obj, "FOV_RADIUS", None):  # Ensure FOV_RADIUS exists
                fov_radius = obj.FOV_RADIUS
                search_radius = int(np.ceil(fov_radius / self.spatial_grid.cell_size))

                for dx in range(-search_radius, search_radius + 1):
                    for dy in range(-search_radius, search_radius + 1):
                        fov_cell_x, fov_cell_y = cell_x + dx, cell_y + dy

                        # Ensure the cell is within the actual circular FOV range
                        cell_center_x = (fov_cell_x + 0.5) * self.spatial_grid.cell_size
                        cell_center_y = (fov_cell_y + 0.5) * self.spatial_grid.cell_size
                        distance_sq = (cell_center_x - obj.pos.x) ** 2 + (cell_center_y - obj.pos.y) ** 2

                        if distance_sq <= fov_radius ** 2:  # Confirm within circular FOV range
                            fov_cells.add((fov_cell_x, fov_cell_y))

        # Highlight active grid cells, which contain objects
        for (cell_x, cell_y) in self.spatial_grid.grid.keys():
            color = "gray"  # Default color for active grid cells
            if (cell_x, cell_y) in fov_cells:
                color = "blue"  # Highlight FOV-affected cells in blue

            rect = patches.Rectangle(
                (cell_x * self.spatial_grid.cell_size, cell_y * self.spatial_grid.cell_size),
                self.spatial_grid.cell_size, self.spatial_grid.cell_size,
                linewidth=1, edgecolor=color, facecolor='none', alpha=0.5
            )
            self.ax.add_patch(rect)
            self.grid_patches.append(rect)

    def reset_objects(self):
        """
        Fully resets all game objects and removes their visual shapes.
        """
        for obj in self.objects:
            if hasattr(obj, 'shape') and obj.shape:
                try:
                    obj.shape.remove()
                except Exception:
                    pass
            if hasattr(obj, 'label') and obj.label:
                try:
                    obj.label.remove()
                except Exception:
                    pass
            if hasattr(obj, 'direction_arrow') and obj.direction_arrow:
                try:
                    obj.direction_arrow.remove()
                except Exception:
                    pass
            if hasattr(obj, 'hitbox') and obj.hitbox:
                try:
                    obj.hitbox.remove()
                except Exception:
                    pass

        self.objects.clear()
        self.spatial_grid.clear()

        for patch in self.grid_patches:
            try:
                patch.remove()
            except Exception:
                pass
        self.grid_patches.clear()

        for patch in list(self.ax.patches):
            try:
                patch.remove()
            except Exception:
                pass

        for line in list(self.ax.lines):
            try:
                line.remove()
            except Exception:
                pass