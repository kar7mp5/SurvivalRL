import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use("TkAgg") # WSL matplotlib animation renderer

# Load game object
from SurvivalRL import GameObject, Circle, Rectangle

WINDOW_SIZE: int = 20

target_fps = 60
interval = 1000 / target_fps

if __name__=='__main__':
    fig, ax = plt.subplots()
    ax.set_xlim(-WINDOW_SIZE//2, WINDOW_SIZE//2)
    ax.set_ylim(-WINDOW_SIZE//2, WINDOW_SIZE//2)

    game = GameObject(ax)

    # Add objects
    game.add_object(Circle(x=0, y=0, radius=1, colour="blue"))
    # game.add_object(Rectangle(x=-5, y=-5, width=2, height=2, colour="red"))

    def animate(frame):
        """ Updates all objects in each frame """
        fps = 1000 / interval
        return game.update(fps)

    ani = animation.FuncAnimation(fig, animate, frames=200, interval=interval, blit=True)

    ani.save("result.gif", writer="pillow", fps=20)
    # plt.show()