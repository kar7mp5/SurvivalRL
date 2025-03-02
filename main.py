import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use("TkAgg") # WSL matplotlib animation renderer

# Load game object
from SurvivalRL import GameObject, Circle, Rectangle


if __name__=='__main__':
    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    game = GameObject(ax)

    # Add objects
    game.add_object(Circle(x=0, y=0, radius=1, colour="blue"))
    game.add_object(Rectangle(x=-5, y=-5, width=2, height=2, colour="red"))

    def animate(frame):
        """ Updates all objects in each frame """
        return game.update()

    ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True)

    ani.save("result.gif", writer="pillow", fps=20)
    plt.show()