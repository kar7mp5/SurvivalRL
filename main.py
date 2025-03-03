import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use("TkAgg") # WSL matplotlib animation renderer

import numpy as np

# Load game object
from SurvivalRL import Config, GameObject, Predator, Herbivore, Plant

target_fps = 30
interval = 1000 / target_fps
duration = 20
frames = target_fps * duration


if __name__=='__main__':
    fig, ax = plt.subplots()
    ax.set_xlim(-Config.WINDOW_SIZE//2, Config.WINDOW_SIZE//2)
    ax.set_ylim(-Config.WINDOW_SIZE//2, Config.WINDOW_SIZE//2)

    game = GameObject(ax)

    # Add objects
    for i in range(3):
        game.add_object(Herbivore(
            game=game,
            ax=ax,
            x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            radius=1,
            target_speed=np.random.uniform(0.1, 0.2),
            colour=np.random.choice(["blue", "green", "purple", "orange"]),
            name=f"Herbivore {i+1}"
        ))

    for i in range(10):
        game.add_object(Plant(
            game=game,
            ax=ax,
            x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            radius=1,
            colour="green",
            name=f"Plant {i+1}"
        ))

    for i in range(5):
        game.add_object(Predator(
            game=game,
            ax=ax,
            x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            width=2,
            height=2,
            target_speed=np.random.uniform(0.1, 0.2),
            colour=np.random.choice(["blue", "green", "purple", "orange"]),
            name=f"Predator {i+1}"
        ))

    def animate(frame):
        """Updates all objects in each frame"""
        return game.update(target_fps)

    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
    ani.save("result.gif", writer="pillow", fps=target_fps)
    # plt.show()