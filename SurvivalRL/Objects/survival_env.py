import os
import subprocess
from collections import deque
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use("Agg")  # Switch to non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from tqdm import tqdm

import gym
from gym import spaces

from SurvivalRL import Config, GameObject  # Custom module imports

import concurrent.futures

def save_frame(args):
    """ Saves a single frame as a PNG image. Used in multiprocessing. """
    frame_data, filename = args
    image = Image.fromarray(frame_data)
    image.save(filename)


class SurvivalEnv(gym.Env):
    """
    Reinforcement Learning Environment for Predator-Prey Simulation.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(SurvivalEnv, self).__init__()
        from Objects import Predator, Herbivore, Plant
        # State: [x, y, energy, speed]
        self.observation_space = spaces.Box(
            low=np.array([-Config.WINDOW_SIZE / 2, -Config.WINDOW_SIZE / 2, 0, 0]),
            high=np.array([Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2, 500, 1]),
            dtype=np.float32
        )

        # Action: [dx, dy] movement
        self.action_space = spaces.Box(low=np.array([-1, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

        # Initialize Matplotlib figure for rendering
        self.fig, (self.ax, self.ax_plot) = plt.subplots(1, 2, figsize=(12, 6))

        self.ax.set_xlim(-Config.WINDOW_SIZE // 2, Config.WINDOW_SIZE // 2)
        self.ax.set_ylim(-Config.WINDOW_SIZE // 2, Config.WINDOW_SIZE // 2)
        self.ax.set_title("Survival Simulation")

        # Game Object Manager
        self.game = GameObject(self.ax)
        self.grid = self.game.spatial_grid

        self.reset()

    def reset(self):
        from Objects import Predator, Herbivore, Plant
        self.game.spatial_grid.clear()

        for _ in range(Config.PRED_NUM):
            self.predator = Predator(
                game=self.game,
                ax=self.ax,
                x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
                y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
                energy=500,
                width=3,
                height=3,
                target_speed=0.3,
                colour="red"
            )
            self.game.add_object(self.predator)

        for _ in range(Config.HERBI_NUM):
            self.game.add_object(Herbivore(
                game=self.game,
                ax=self.ax,
                x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
                y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
                energy=100,
                radius=1.5,
                target_speed=0.4,
                colour="blue"
            ))

        for _ in range(Config.PLANT_NUM):
            self.game.add_object(Plant(
                game=self.game,
                ax=self.ax,
                x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
                y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
                energy=100,
                radius=1.5,
                colour="green"
            ))

        return self._get_observation()

    def _get_observation(self):
        """ Returns the current state as an observation. """
        return np.array([self.predator.pos.x, self.predator.pos.y, self.predator.energy, self.predator.target_speed])

    def step(self, action):
        """
        Executes a single step in the environment based on the given action.
        
        Args:
            action (tuple): A tuple containing movement values (dx, dy) and optionally a detection flag.
        
        Returns:
            tuple: (observation, reward, done, info)
                - observation: The updated state of the environment.
                - reward: The reward for this step.
                - done: A boolean indicating if the episode has ended.
                - info: Additional debug information (empty dictionary).
        """
        from Objects import Plant, Predator  # Import object classes

        # Extract movement values and detection flag
        if len(action) == 2:
            dx, dy = action
            detect_target = 0  # Default: No detection
        else:
            dx, dy, detect_target = action

        detected_target = None
        # If detection is activated, find the nearest object within the FOV
        if detect_target > 0.5:
            detected_target = self.detect_in_fov(self.predator.pos.x, self.predator.pos.y, self.predator.FOV_RADIUS)

        # If an object is detected, determine whether to chase or evade
        if detected_target:
            if isinstance(detected_target, Plant):
                # If a plant is detected, move towards it
                self.target_x, self.target_y = detected_target.pos.x, detected_target.pos.y
            elif isinstance(detected_target, Predator):
                # If another predator is detected, move in the opposite direction (escape)
                self.target_x, self.target_y = -detected_target.pos.x, -detected_target.pos.y  

        # Apply movement updates
        self.predator.pos.x += dx * 5  # Move predator in the x-direction
        self.predator.pos.y += dy * 5  # Move predator in the y-direction
        self.predator.energy -= 0.1  # Reduce energy with each movement

        # Check termination condition (if energy is depleted)
        done = self.predator.energy <= 0

        # Assign reward: penalty if dead (-10), otherwise +1 for each step
        reward = -10 if done else 1

        return self._get_observation(), reward, done, {}  # Return updated state, reward, and episode status

    def detect_in_fov(self, x, y, fov_radius):
        """
        Detects the nearest object within the Field of View (FOV) using a Spatial Hash Grid.
        
        Args:
            x (float): X-coordinate of the observer.
            y (float): Y-coordinate of the observer.
            fov_radius (float): The radius defining the FOV range.

        Returns:
            Object or None: The nearest detected object within the FOV, or None if no object is found.
        """
        # Retrieve all possible objects within the FOV range using a spatial hash grid
        possible_objects = self.grid.retrieve_in_fov_range(x, y, fov_radius)
        
        best_target = None  # Variable to store the nearest object
        min_distance_sq = fov_radius ** 2  # Maximum search distance (squared for efficiency)

        for obj in possible_objects:
            # Skip self-detection (e.g., predator should not detect itself)
            if obj is self.predator:
                continue  

            # Compute squared Euclidean distance between the object and observer
            dx = obj.pos.x - x
            dy = obj.pos.y - y
            distance_sq = dx * dx + dy * dy

            # Ignore objects outside the FOV radius
            if distance_sq > min_distance_sq:
                continue  

            # Update the closest object if this one is nearer
            if distance_sq < min_distance_sq:
                best_target = obj
                min_distance_sq = distance_sq  # Update the minimum distance

        return best_target  # Return the nearest object within FOV

    def render(self, mode="human", save_as="train.mp4"):
        """
        Ultra-fast rendering using optimized Matplotlib updates and parallel frame saving.
        Now includes blitting for faster rendering.
        """
        from Objects import Herbivore, Predator, Plant

        # Create a directory to store frames
        frame_dir = "frames"
        os.makedirs(frame_dir, exist_ok=True)

        # Prepare frames for parallel saving
        frame_data_list = []

        # Initialize population tracking with limited deque size
        population_data = {
            "Predator": deque(maxlen=Config.FRAMES),
            "Herbivore": deque(maxlen=Config.FRAMES),
            "Plant": deque(maxlen=Config.FRAMES)
        }

        # **Blitting Optimization (Redraw Only When Needed)**
        self.ax_plot.set_xlim(0, Config.FRAMES)
        self.ax_plot.set_ylim(0, 20)  
        self.ax_plot.set_title("Population Over Time")
        self.ax_plot.set_xlabel("Time (frames)")
        self.ax_plot.set_ylabel("Population")

        line_herb, = self.ax_plot.plot([], [], color="blue", label="Herbivores")
        line_pred, = self.ax_plot.plot([], [], color="red", label="Predators")
        line_plant, = self.ax_plot.plot([], [], color="green", label="Plants")
        self.ax_plot.legend()

        plt.tight_layout()

        for i in tqdm(range(Config.FRAMES), desc="Capturing Frames", unit="frame"):
            self.game.update(Config.TARGET_FPS)  # Update game state

            # Count population
            herbivore_count = sum(isinstance(obj, Herbivore) for obj in self.game.objects)
            predator_count = sum(isinstance(obj, Predator) for obj in self.game.objects)
            plant_count = sum(isinstance(obj, Plant) for obj in self.game.objects)

            # Store recent population data
            population_data["Herbivore"].append(herbivore_count)
            population_data["Predator"].append(predator_count)
            population_data["Plant"].append(plant_count)

            # **Fast Population Plot Update**
            x_data = np.arange(len(population_data["Herbivore"]))  # Precompute range
            line_herb.set_data(x_data, np.array(population_data["Herbivore"]))
            line_pred.set_data(x_data, np.array(population_data["Predator"]))
            line_plant.set_data(x_data, np.array(population_data["Plant"]))

            max_population = max(
                max(population_data["Herbivore"], default=10),
                max(population_data["Predator"], default=10),
                max(population_data["Plant"], default=10)
            )
            self.ax_plot.set_ylim(0, max_population + 5)

            # **Blitting Optimization: Draw Only Changes**
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()

            # Convert figure to NumPy array (Faster Processing)
            frame = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
            frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
            frame = frame[:, :, [1, 2, 3]]

            # Store frame data
            filename = os.path.join(frame_dir, f"frame_{i:05d}.png")
            frame_data_list.append((frame, filename))

        # **Use ThreadPoolExecutor for Parallel Frame Saving**
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            list(tqdm(executor.map(save_frame, frame_data_list), total=len(frame_data_list), desc="Saving Frames", unit="frame"))

        # **GPU-Accelerated FFmpeg Encoding**
        output_video = save_as
        fps = Config.TARGET_FPS
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frame_dir, "frame_%05d.png"),
            "-c:v", "h264_nvenc",  # Use NVIDIA NVENC for GPU acceleration
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            output_video
        ]

        print("Converting images to MP4 using GPU-accelerated ffmpeg...")
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("MP4 Video Created Successfully!")

        # Cleanup frames
        for file in os.listdir(frame_dir):
            os.remove(os.path.join(frame_dir, file))
        os.rmdir(frame_dir)

    def close(self):
        plt.close(self.fig)
