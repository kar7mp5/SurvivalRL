import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use("TkAgg")  # WSL matplotlib animation renderer

import numpy as np
from collections import deque
from SurvivalRL import Config, GameObject, Predator, Herbivore, Plant


if __name__=='__main__':
    # Set up the figure layout (one simulation + one population plot)
    fig, (ax_sim, ax_plot) = plt.subplots(1, 2, figsize=(12, 6))

    # Configure simulation subplot
    ax_sim.set_xlim(-Config.WINDOW_SIZE // 2, Config.WINDOW_SIZE // 2)
    ax_sim.set_ylim(-Config.WINDOW_SIZE // 2, Config.WINDOW_SIZE // 2)
    ax_sim.set_title("Simulation")

    game = GameObject(ax_sim)

    # Add objects to the simulation
    for i in range(5):
        game.add_object(Herbivore(
            game=game,
            ax=ax_sim,
            x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            energy=100,
            radius=np.random.uniform(1, 2),
            target_speed=np.random.uniform(0.1, 0.5),
            colour="blue",
            name=f"Herbivore {i+1}",
        ))

    for i in range(3):
        game.add_object(Predator(
            game=game,
            ax=ax_sim,
            x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            energy=100,
            width=np.random.uniform(2, 4),
            height=np.random.uniform(2, 4),
            target_speed=np.random.uniform(0.1, 0.5),
            colour="red",
            name=f"Predator {i+1}",
        ))

    for i in range(5):
        game.add_object(Plant(
            game=game,
            ax=ax_sim,
            x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            energy=100,
            radius=np.random.uniform(1, 2),
            colour="green",
            name=f"Plant {i+1}",
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
    ax_plot.set_xlim(0, Config.FRAMES)  # Time range
    ax_plot.set_ylim(0, max(15, len(game.objects)))  # Dynamic y-limit
    ax_plot.set_title("Population Over Time")
    ax_plot.set_xlabel("Time (frames)")
    ax_plot.set_ylabel("Population")

    # Initialize deque for tracking population history
    population_data = {
        "Predator": deque(maxlen=Config.FRAMES),
        "Herbivore": deque(maxlen=Config.FRAMES),
        "Plant": deque(maxlen=Config.FRAMES)
    }

    # Create empty lines for real-time updating
    line_herb, = ax_plot.plot([], [], color="blue", label="Herbivores")
    line_pred, = ax_plot.plot([], [], color="red", label="Predators")
    line_plant, = ax_plot.plot([], [], color="green", label="Plants")

    ax_plot.legend()


    def animate(frame):
        """Updates the simulation, object count label, and population plot."""
        print(frame)
        # Test for plant duplication at frame 80
        if frame%100 == 0:
            for k in range(5):
                game.add_object(Plant(
                    game=game,
                    ax=ax_sim,
                    x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
                    y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
                    energy=100,
                    radius=np.random.uniform(1, 3),
                    colour="green",
                    name=f"Plant {k+1}",
                ))

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

        # Append new population data
        population_data["Herbivore"].append(herbivore_count)
        population_data["Predator"].append(predator_count)
        population_data["Plant"].append(plant_count)

        # Update line plots
        x_data = list(range(len(population_data["Herbivore"])))
        line_herb.set_data(x_data, list(population_data["Herbivore"]))
        line_pred.set_data(x_data, list(population_data["Predator"]))
        line_plant.set_data(x_data, list(population_data["Plant"]))

        # Adjust y-limits dynamically based on max population
        max_population = max(
            max(population_data["Herbivore"]),
            max(population_data["Predator"]),
            max(population_data["Plant"]),
            10
        )
        ax_plot.set_ylim(0, max_population + 5)

        return label, line_herb, line_pred, line_plant


    ani = animation.FuncAnimation(
        fig=fig, 
        func=animate, 
        frames=Config.FRAMES, 
        interval=Config.INTERVAL, 
        blit=False)

    # Save the animation
    ani.save("single_simulation_with_population_graph.gif", writer="pillow", fps=Config.TARGET_FPS)
    # Save the animation as an MP4 file
    # ani.save("single_simulation_with_population_graph.mp4", writer="ffmpeg", fps=Config.TARGET_FPS)

    # Show the plots
    # plt.show()