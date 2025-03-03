import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use("TkAgg")  # WSL matplotlib animation renderer

import numpy as np
from collections import deque
from SurvivalRL import Config, GameObject, Predator, Herbivore, Plant


# Number of simulations
NUM_SIMULATIONS = 2  # You can adjust this (e.g., 4 for 2x2 layout)

# Set up the figure layout (each simulation has a corresponding time-series plot)
fig, axes = plt.subplots(NUM_SIMULATIONS, 2, figsize=(14, 6 * NUM_SIMULATIONS))

# Ensure we can iterate over axes properly if only one simulation is used
if NUM_SIMULATIONS == 1:
    axes = [axes]

# Store game instances and tracking data
games = []
labels = []
population_data = []  # Stores population history for plotting
lines = []  # Stores matplotlib line objects for real-time updates

# Define colors for plotting
colors = {"Herbivore": "blue", "Predator": "red", "Plant": "green"}

for i in range(NUM_SIMULATIONS):
    ax_sim = axes[i][0]  # Left: Simulation
    ax_plot = axes[i][1]  # Right: Time-series plot

    # Configure simulation subplot
    ax_sim.set_xlim(-Config.WINDOW_SIZE // 2, Config.WINDOW_SIZE // 2)
    ax_sim.set_ylim(-Config.WINDOW_SIZE // 2, Config.WINDOW_SIZE // 2)
    ax_sim.set_title(f"Simulation {i+1}")

    game = GameObject(ax_sim)

    # Add objects to the simulation
    for j in range(3):
        game.add_object(Herbivore(
            game=game,
            ax=ax_sim,
            x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            radius=2,
            target_speed=np.random.uniform(0.1, 0.5),
            colour="blue",
            name=f"Herbivore {j+1}"
        ))

    for j in range(5):
        game.add_object(Predator(
            game=game,
            ax=ax_sim,
            x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            width=4,
            height=4,
            target_speed=np.random.uniform(0.1, 0.5),
            colour="red",
            name=f"Predator {j+1}"
        ))

    for j in range(10):
        game.add_object(Plant(
            game=game,
            ax=ax_sim,
            x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
            radius=1,
            colour="green",
            name=f"Plant {j+1}"
        ))

    games.append(game)

    # Add real-time count label inside the simulation
    label = ax_sim.text(
        Config.WINDOW_SIZE // 2 - 22,
        Config.WINDOW_SIZE // 2 - 15,
        "",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8)
    )
    labels.append(label)

    # Configure population tracking plot
    ax_plot.set_xlim(0, Config.FRAMES)  # Time range
    ax_plot.set_ylim(0, max(15, len(game.objects)))  # Dynamic y-limit
    ax_plot.set_title(f"Population Over Time - Simulation {i+1}")
    ax_plot.set_xlabel("Time (frames)")
    ax_plot.set_ylabel("Population")
    
    # Initialize empty deque for tracking population history
    pop_data = {
        "Predator": deque(maxlen=Config.FRAMES),
        "Herbivore": deque(maxlen=Config.FRAMES),
        "Plant": deque(maxlen=Config.FRAMES)
    }
    population_data.append(pop_data)

    # Create empty lines for real-time updating
    line_herb, = ax_plot.plot([], [], color="blue", label="Herbivores")
    line_pred, = ax_plot.plot([], [], color="red", label="Predators")
    line_plant, = ax_plot.plot([], [], color="green", label="Plants")
    
    ax_plot.legend()
    lines.append((line_herb, line_pred, line_plant))


def animate(frame):
    """Updates all simulations, object count labels, and population plots."""
    updates = []
    
    for i, game in enumerate(games):
        
        # Test for duplication
        if frame == 80:
            for k in range(10):
                game.add_object(Plant(
                    game=game,
                    ax=axes[i][0],
                    x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
                    y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
                    radius=2,
                    colour="green",
                    name=f"Plant {k+1}"
                ))

        updates.extend(game.update(Config.TARGET_FPS))  # Update game state
        
        # Count different object types
        herbivore_count = sum(isinstance(obj, Herbivore) for obj in game.objects)
        predator_count = sum(isinstance(obj, Predator) for obj in game.objects)
        plant_count = sum(isinstance(obj, Plant) for obj in game.objects)

        # Update the label with live counts
        labels[i].set_text(
            f"Herbivores: {herbivore_count}\n"
            f"Predators: {predator_count}\n"
            f"Plants: {plant_count}"
        )

        # Append new population data
        population_data[i]["Herbivore"].append(herbivore_count)
        population_data[i]["Predator"].append(predator_count)
        population_data[i]["Plant"].append(plant_count)

        # Update line plots
        line_herb, line_pred, line_plant = lines[i]

        x_data = list(range(len(population_data[i]["Herbivore"])))
        line_herb.set_data(x_data, list(population_data[i]["Herbivore"]))
        line_pred.set_data(x_data, list(population_data[i]["Predator"]))
        line_plant.set_data(x_data, list(population_data[i]["Plant"]))

        # Adjust y-limits dynamically based on max population
        max_population = max(
            max(population_data[i]["Herbivore"]),
            max(population_data[i]["Predator"]),
            max(population_data[i]["Plant"]),
            10
        )
        axes[i][1].set_ylim(0, max_population + 5)

    return updates + labels + [line for sublist in lines for line in sublist]  # Return updated objects, labels, and plots


ani = animation.FuncAnimation(
    fig=fig, 
    func=animate, 
    frames=Config.FRAMES, 
    interval=Config.INTERVAL, 
    blit=False)

# Save the animation
ani.save("multi_simulation_with_population_graphs.gif", writer="pillow", fps=Config.TARGET_FPS)

# Show the plots
# plt.show()
