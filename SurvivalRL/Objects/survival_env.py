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
    frame_data, filename = args
    image = Image.fromarray(frame_data)
    image.save(filename)


class SurvivalEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(SurvivalEnv, self).__init__()
        from Objects import Predator, Herbivore, Plant

        self.observation_space = spaces.Box(
            low=np.array([-Config.WINDOW_SIZE / 2] * 2 + [0, 0] + [-Config.WINDOW_SIZE / 2] * 2 + [0, 0]),
            high=np.array([Config.WINDOW_SIZE / 2] * 2 + [500, 1] + [Config.WINDOW_SIZE / 2] * 2 + [500, 1]),
            dtype=np.float32
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        self.fig, (self.ax, self.ax_plot) = plt.subplots(1, 2, figsize=(12, 6))
        self.ax.set_xlim(-Config.WINDOW_SIZE // 2, Config.WINDOW_SIZE // 2)
        self.ax.set_ylim(-Config.WINDOW_SIZE // 2, Config.WINDOW_SIZE // 2)
        self.ax.set_title("Survival Simulation")

        self.game = GameObject(self.ax)
        self.grid = self.game.spatial_grid

        self.reset()

    def reset(self):
        from Objects import Predator, Herbivore, Plant
        self.game.spatial_grid.clear()
        # self.game.objects.clear()

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

        return self._get_obs()

    def _get_obs(self):
        return np.array([
            self.predator.pos.x, self.predator.pos.y, self.predator.energy, self.predator.target_speed,
            self.herbivore.pos.x, self.herbivore.pos.y, self.herbivore.energy, self.herbivore.target_speed
        ], dtype=np.float32)

    def step(self, action):
        from Objects import Plant

        pdx, pdy, pdetect = action[0:3]
        hdx, hdy, hdetect = action[3:6]

        predator_reward = 1
        herbivore_reward = 1
        predator_done = False
        herbivore_done = False

        predator_target = None
        if pdetect > 0.5:
            predator_target = self.predator.detect_in_fov_type(self.grid, Plant)
            if predator_target:
                self.predator.target_x = predator_target.pos.x
                self.predator.target_y = predator_target.pos.y

        self.predator.update(Config.TARGET_FPS, self.grid)
        self.predator.pos.x += pdx * 5
        self.predator.pos.y += pdy * 5
        self.predator.energy -= 0.1

        if self.predator.energy <= 0:
            predator_done = True
            predator_reward = -10

        if predator_target and self._approached(self.predator, predator_target):
            predator_reward += 2

        if self.predator.try_reproduce_in_fov(self.grid):
            predator_reward += 5

        herbivore_target = None
        if hdetect > 0.5:
            herbivore_target = self.herbivore.detect_in_fov_for_type(self.grid, Plant)
            if herbivore_target:
                self.herbivore.target_x = herbivore_target.pos.x
                self.herbivore.target_y = herbivore_target.pos.y

        self.herbivore.update(Config.TARGET_FPS, self.grid)
        self.herbivore.pos.x += hdx * 5
        self.herbivore.pos.y += hdy * 5
        self.herbivore.energy -= 0.1

        if self.herbivore.energy <= 0:
            herbivore_done = True
            herbivore_reward = -10

        if herbivore_target and self._approached(self.herbivore, herbivore_target):
            herbivore_reward += 2

        if self.herbivore.try_reproduce_in_fov(self.grid):
            herbivore_reward += 5

        obs = self._get_obs()
        reward = predator_reward + herbivore_reward
        done = predator_done and herbivore_done

        return obs, reward, done, {}

    def _approached(self, agent, target, threshold=5.0):
        dx = agent.pos.x - target.pos.x
        dy = agent.pos.y - target.pos.y
        return (dx * dx + dy * dy) < threshold ** 2

    def render(self, mode="human", save_as="train.mp4"):
        from Objects import Herbivore, Predator, Plant

        frame_dir = "frames"
        os.makedirs(frame_dir, exist_ok=True)
        frame_data_list = []

        population_data = {
            "Predator": deque(maxlen=Config.FRAMES),
            "Herbivore": deque(maxlen=Config.FRAMES),
            "Plant": deque(maxlen=Config.FRAMES)
        }

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

            herbivore_count = sum(isinstance(obj, Herbivore) for obj in self.game.objects)
            predator_count = sum(isinstance(obj, Predator) for obj in self.game.objects)
            plant_count = sum(isinstance(obj, Plant) for obj in self.game.objects)

            population_data["Herbivore"].append(herbivore_count)
            population_data["Predator"].append(predator_count)
            population_data["Plant"].append(plant_count)

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

            frame = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
            frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
            frame = frame[:, :, [1, 2, 3]]

            filename = os.path.join(frame_dir, f"frame_{i:05d}.png")
            frame_data_list.append((frame, filename))

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

        for file in os.listdir(frame_dir):
            os.remove(os.path.join(frame_dir, file))
        os.rmdir(frame_dir)

    def close(self):
        plt.close(self.fig)
