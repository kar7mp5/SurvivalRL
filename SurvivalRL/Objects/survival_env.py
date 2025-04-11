import os
import subprocess
from collections import deque
from multiprocessing import cpu_count
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import gym
from gym import spaces

from SurvivalRL import Config, GameObject
import concurrent.futures


def save_frame(args):
    """Saves a single frame as an image file.

    Args:
        args (tuple): A tuple containing (frame_data, filename).
    """
    frame_data, filename = args
    image = Image.fromarray(frame_data)
    image.save(filename)


class SurvivalEnv(gym.Env):
    """A dual-agent Gym environment for simulating survival behavior.

    Agents: Predator and Herbivore
    Reward logic is handled internally by the agent objects themselves.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        """Initializes the simulation environment, including rendering and spatial grid."""
        super(SurvivalEnv, self).__init__()
        from Objects import Predator, Herbivore, Plant

        # Observation: [pred_x, pred_y, pred_energy, pred_speed, herb_x, herb_y, herb_energy, herb_speed]
        self.observation_space = spaces.Box(
            low=np.array([-Config.WINDOW_SIZE / 2] * 2 + [0, 0] + [-Config.WINDOW_SIZE / 2] * 2 + [0, 0]),
            high=np.array([Config.WINDOW_SIZE / 2] * 2 + [500, 1] + [Config.WINDOW_SIZE / 2] * 2 + [500, 1]),
            dtype=np.float32
        )

        # Action: 3 values for each agent (dx, dy, detect)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        # Rendering setup
        self.fig, (self.ax, self.ax_plot) = plt.subplots(1, 2, figsize=(12, 6))
        self.ax.set_xlim(-Config.WINDOW_SIZE // 2, Config.WINDOW_SIZE // 2)
        self.ax.set_ylim(-Config.WINDOW_SIZE // 2, Config.WINDOW_SIZE // 2)
        self.ax.set_title("Survival Simulation")

        self.game = GameObject(self.ax)
        self.grid = self.game.spatial_grid

    def reset(self):
        """Resets the environment with new agents and plant entities.

        Returns:
            dict: A dictionary containing initial observations for both predator and herbivore.
        """
        from Objects import Predator, Herbivore, Plant
        self.game.reset_objects()
        self.grid = self.game.spatial_grid

        print('---RESET ENVIRONMENT---')
        print(f'PRED: {Config.PRED_NUM}')
        print(f'HERBI: {Config.HERBI_NUM}')
        print(f'PLANT: {Config.PLANT_NUM}')

        # Spawn predator
        for _ in range(Config.PRED_NUM):
            self.predator = Predator(
                game=self.game,
                ax=self.ax,
                x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
                y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
                energy=150,
                width=3,
                height=3,
                target_speed=0.3,
                colour="red"
            )
            self.game.add_object(self.predator)

        # Spawn herbivore
        for _ in range(Config.HERBI_NUM):
            self.herbivore = Herbivore(
                game=self.game,
                ax=self.ax,
                x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
                y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
                energy=100,
                radius=1.5,
                target_speed=0.4,
                colour="blue"
            )
            self.game.add_object(self.herbivore)

        # Spawn plants
        for _ in range(Config.PLANT_NUM):
            self.game.add_object(Plant(
                game=self.game,
                ax=self.ax,
                x=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
                y=np.random.uniform(-Config.WINDOW_SIZE / 2, Config.WINDOW_SIZE / 2),
                energy=10,
                radius=1.5,
                colour="green"
            ))

        return self._get_obs()

    def _get_obs(self):
        """Builds observation dictionary from current agent states.

        Returns:
            dict: Observations for 'predator' and 'herbivore'.
        """
        return {
            "predator": np.array([
                self.predator.pos.x,
                self.predator.pos.y,
                self.predator.energy,
                self.predator.target_speed,
            ], dtype=np.float32),
            "herbivore": np.array([
                self.herbivore.pos.x,
                self.herbivore.pos.y,
                self.herbivore.energy,
                self.herbivore.target_speed,
            ], dtype=np.float32),
        }

    def step(self, action):
        """Steps the environment forward using each agent's action.

        Args:
            action (dict): Dictionary containing 'predator' and 'herbivore' actions.

        Returns:
            Tuple: observation (dict), reward (dict), done (dict), info (dict)
        """
        predator_reward, predator_contrib, predator_done = self.predator.compute_reward(
            action["predator"], self.grid, self.game.objects
        )
        herbivore_reward, herbivore_contrib, herbivore_done = self.herbivore.compute_reward(
            action["herbivore"], self.grid, self.game.objects
        )

        obs = self._get_obs()
        reward = {"predator": predator_reward, "herbivore": herbivore_reward}
        done = {
            "predator": predator_done,
            "herbivore": herbivore_done,
            "__all__": predator_done and herbivore_done
        }
        info = {
            "predator_breakdown": predator_contrib,
            "herbivore_breakdown": herbivore_contrib
        }

        return obs, reward, done, info

    def render(self, mode="human", save_as="train.mp4"):
        """Renders the full episode and saves it as a video file.

        Args:
            mode (str): Rendering mode. Defaults to "human".
            save_as (str): Output file name for the video.
        """
        from Objects import Herbivore, Predator, Plant

        frame_dir = "frames"
        os.makedirs(frame_dir, exist_ok=True)
        frame_data_list = []

        # Track population over time
        population_data = {
            "Predator": deque(maxlen=Config.FRAMES),
            "Herbivore": deque(maxlen=Config.FRAMES),
            "Plant": deque(maxlen=Config.FRAMES)
        }

        # Set up live plot
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
            self.game.update(Config.TARGET_FPS)

            # Count current population
            herbivore_count = sum(isinstance(obj, Herbivore) for obj in self.game.objects)
            predator_count = sum(isinstance(obj, Predator) for obj in self.game.objects)
            plant_count = sum(isinstance(obj, Plant) for obj in self.game.objects)

            population_data["Herbivore"].append(herbivore_count)
            population_data["Predator"].append(predator_count)
            population_data["Plant"].append(plant_count)

            # Update plot data
            x_data = np.arange(len(population_data["Herbivore"]))
            line_herb.set_data(x_data, np.array(population_data["Herbivore"]))
            line_pred.set_data(x_data, np.array(population_data["Predator"]))
            line_plant.set_data(x_data, np.array(population_data["Plant"]))

            max_population = max(
                max(population_data["Herbivore"], default=10),
                max(population_data["Predator"], default=10),
                max(population_data["Plant"], default=10)
            )
            self.ax_plot.set_ylim(0, max_population + 5)

            self.fig.canvas.flush_events()
            self.fig.canvas.draw()

            # Capture current frame
            frame = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
            frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
            frame = frame[:, :, [1, 2, 3]]

            filename = os.path.join(frame_dir, f"frame_{i:05d}.png")
            frame_data_list.append((frame, filename))

        # Save frames as video
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            list(tqdm(executor.map(save_frame, frame_data_list), total=len(frame_data_list), desc="Saving Frames", unit="frame"))

        output_video = save_as
        fps = Config.TARGET_FPS
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frame_dir, "frame_%05d.png"),
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            output_video
        ]

        print("Converting images to MP4 using CPU-based ffmpeg...")
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("MP4 Video Created Successfully!")

        # Cleanup
        for file in os.listdir(frame_dir):
            os.remove(os.path.join(frame_dir, file))
        os.rmdir(frame_dir)

    def close(self):
        """Closes the matplotlib figure and cleans up the rendering session."""
        plt.close(self.fig)
