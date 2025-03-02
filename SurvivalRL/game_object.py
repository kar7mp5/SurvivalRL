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

    def add_object(self, obj):
        """
        Adds an object to the game and draws it on the axis.

        Args:
            obj (Obj): An instance of a game object (e.g., Circle, Rectangle).
        """
        obj.draw(self.ax)
        self.objects.append(obj)

    def update(self):
        """
        Updates all objects in the game by calling their respective update methods.

        Returns:
            list: A list of updated shapes for animation rendering.
        """
        for obj in self.objects:
            obj.update()
        return [obj.shape for obj in self.objects]