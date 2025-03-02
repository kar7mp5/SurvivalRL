import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use("TkAgg") # WSL matplotlib animation renderer

import numpy as np

# Load game object
from SurvivalRL import Config, GameObject, Circle, Rectangle

target_fps = 30
interval = 1000 / target_fps
duration = 10
frames = target_fps * duration


if __name__=='__main__':
    fig, ax = plt.subplots()
    ax.set_xlim(-Config.WINDOW_SIZE//2, Config.WINDOW_SIZE//2)
    ax.set_ylim(-Config.WINDOW_SIZE//2, Config.WINDOW_SIZE//2)

    game = GameObject(ax)

    # Add objects
    for _ in range(5):
        game.add_object(Circle(
            x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            radius=1,
            colour=np.random.choice(["blue", "green", "purple", "orange"])
        ))

    # game.add_object(Rectangle(x=-5, y=-5, width=2, height=2, colour="red"))

    def animate(frame):
        """ Updates all objects in each frame """
        fps = 1000 / interval
        return game.update(fps)

    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True)

    ani.save("result.gif", writer="pillow", fps=target_fps)
    # plt.show()