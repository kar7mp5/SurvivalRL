class Position:    
    """ 
    A simple class to store and manage 2D positions. 
    This class allows retrieving and updating position values.
    """

    def __init__(self, x: float, y: float):
        """
        Initializes a Position object with x and y coordinates.

        Args:
            x (float): Initial x-coordinate.
            y (float): Initial y-coordinate.
        """
        self.x = x
        self.y = y

    def __call__(self):
        """
        Returns the current position as a tuple.

        Returns:
            tuple: A tuple containing (x, y) coordinates.
        """
        return (self.x, self.y)

    def move(self, dx: float, dy: float):
        """
        Moves the position by a given displacement in the x and y directions.

        Args:
            dx (float): Change in x-coordinate.
            dy (float): Change in y-coordinate.
        """
        self.x += dx
        self.y += dy


class Obj:
    """ 
    A parent class for moving objects (shapes).
    This class provides a base for different graphical objects that can be drawn and updated.
    """

    def __init__(self, x: float, y: float, colour: str):
        """
        Initializes an Obj with a position and color.

        Args:
            x (float): Initial x-coordinate of the object.
            y (float): Initial y-coordinate of the object.
            colour (str): Color of the object.
        """
        self.pos = Position(x, y)
        self.colour = colour
        self.shape = None  # Shape will be defined in the subclasses

    def draw(self, ax):
        """
        Abstract method to draw the object on a given axis.

        This method must be implemented in subclasses to define how the object is drawn.

        Args:
            ax (matplotlib.axes.Axes): The axis on which the object will be drawn.

        Raises:
            NotImplementedError: If called directly from the Obj class.
        """
        raise NotImplementedError