import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for fast rendering

import numpy as np
from collections import deque
from tqdm import tqdm
from SurvivalRL import Config, GameObject, Predator, Herbivore, Plant


if __name__ == '__main__':
    np.random.seed(41)

    # Set up the figure layout (Simulation + Population Plot)
    fig, (ax_sim, ax_plot) = plt.subplots(1, 2, figsize=(12, 6))

    # Configure simulation subplot
    ax_sim.set_xlim(-Config.WINDOW_SIZE // 2, Config.WINDOW_SIZE // 2)
    ax_sim.set_ylim(-Config.WINDOW_SIZE // 2, Config.WINDOW_SIZE // 2)
    ax_sim.set_title("Simulation")

    game = GameObject(ax_sim)

    # Add objects to the simulation
    for _ in range(Config.HERBI_NUM):
        game.add_object(Herbivore(
            game=game, ax=ax_sim,
            x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            energy=100,
            radius=np.random.uniform(1, 2),
            target_speed=np.random.uniform(0.1, 0.5),
            colour="blue",
        ))

    for _ in range(Config.PRED_NUM):
        game.add_object(Predator(
            game=game, ax=ax_sim,
            x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            energy=500,
            width=np.random.uniform(2, 4),
            height=np.random.uniform(2, 4),
            target_speed=np.random.uniform(0.1, 0.5),
            colour="red",
        ))

    for _ in range(Config.PLANT_NUM):
        game.add_object(Plant(
            game=game, ax=ax_sim,
            x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            energy=100,
            radius=np.random.uniform(1, 2),
            colour="green"
        ))

    # Add real-time count label inside the simulation
    label = ax_sim.text(
        Config.WINDOW_SIZE // 2 - 25,
        Config.WINDOW_SIZE // 2 - 15,
        "",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    # Configure population tracking plot
    ax_plot.set_xlim(0, Config.FRAMES)
    ax_plot.set_ylim(0, 20)  # Set an initial y-limit
    ax_plot.set_title("Population Over Time")
    ax_plot.set_xlabel("Time (frames)")
    ax_plot.set_ylabel("Population")

    # Initialize deque for tracking population history (Limited to Recent Data)
    population_data = {
        "Predator": deque(maxlen=500),
        "Herbivore": deque(maxlen=500),
        "Plant": deque(maxlen=500)
    }

    # Create empty lines for real-time updating
    line_herb, = ax_plot.plot([], [], color="blue", label="Herbivores")
    line_pred, = ax_plot.plot([], [], color="red", label="Predators")
    line_plant, = ax_plot.plot([], [], color="green", label="Plants")
    ax_plot.legend()

    def animate(frame):
        """Optimized animation update function with blitting."""
        game.update(Config.TARGET_FPS)  # Update game state

        # Count different object types
        herbivore_count = sum(isinstance(obj, Herbivore) for obj in game.objects)
        predator_count = sum(isinstance(obj, Predator) for obj in game.objects)
        plant_count = sum(isinstance(obj, Plant) for obj in game.objects)

        # Update the label with live counts
        label.set_text(
            f"Herbivores: {herbivore_count}\n"
            f"Predators: {predator_count}\n"
            f"Plants: {plant_count}"
        )

        # Store new population data (Limited to last 500 frames)
        population_data["Herbivore"].append(herbivore_count)
        population_data["Predator"].append(predator_count)
        population_data["Plant"].append(plant_count)

        # Convert population data to NumPy for speed
        x_data = np.arange(len(population_data["Herbivore"]))
        line_herb.set_data(x_data, np.array(population_data["Herbivore"]))
        line_pred.set_data(x_data, np.array(population_data["Predator"]))
        line_plant.set_data(x_data, np.array(population_data["Plant"]))

        # Adjust y-limits dynamically
        max_population = max(
            max(population_data["Herbivore"], default=10),
            max(population_data["Predator"], default=10),
            max(population_data["Plant"], default=10)
        )
        ax_plot.set_ylim(0, max_population + 5)

        return line_herb, line_pred, line_plant, label

    # Faster Animation with Blitting
    ani = animation.FuncAnimation(
        fig=fig,
        func=animate,
        frames=tqdm(range(Config.FRAMES), desc="Rendering Frames"),
        interval=Config.INTERVAL,
        blit=True,  # Enables blitting (only updates changes)
        cache_frame_data=False  # Prevents memory buildup
    )

    # Save the animation with optimized FFmpeg
    print("Saving animation as MP4...")
    ani.save("result.mp4", writer="ffmpeg", fps=Config.TARGET_FPS)
    print("Animation saved successfully!")

    # Show the plots
    # plt.show()
