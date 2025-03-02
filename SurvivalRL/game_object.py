from collections import defaultdict
from SurvivalRL import Config


class GameObject:
    """ 
    Manages all objects in the game.
    This class is responsible for handling multiple objects, updating their states, and rendering them on a given matplotlib axis.
    """

    def __init__(self, ax):
        """
        Initializes the GameObject manager.

        Args:
            ax (matplotlib.axes.Axes): The axis where objects will be drawn and managed.
        """
        self.ax = ax
        self.objects = []
        self.grid = defaultdict(list)
        self.grid_lines = []
        self.draw_grid()

    def add_object(self, obj):
        """
        Adds an object to the game and draws it on the axis.

        Args:
            obj (Obj): An instance of a game object (e.g., Circle, Rectangle).
        """
        obj.draw()
        self.objects.append(obj)

    def draw_grid(self):
        """ Draws the spatial grid on the figure. """
        for x in range(-Config.WINDOW_SIZE // 2, Config.WINDOW_SIZE // 2 + 1, Config.GRID_SIZE):
            self.ax.axvline(x, color="gray", linestyle="--", linewidth=0.5)
        for y in range(-Config.WINDOW_SIZE // 2, Config.WINDOW_SIZE // 2 + 1, Config.GRID_SIZE):
            self.ax.axhline(y, color="gray", linestyle="--", linewidth=0.5)

    def update(self, fps):
        """
        Updates all objects in the game by calling their respective update methods.

        Returns:
            list: A list of updated shapes for animation rendering.
        """
        self.grid.clear()

        for obj in self.objects:
            cell_x, cell_y = obj.get_grid_cell()
            self.grid[(cell_x, cell_y)].append(obj)

        for obj in self.objects:
            obj.update(fps, self.grid)

        return [obj.shape for obj in self.objects]