import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from gym import spaces
from collections import deque
from SurvivalRL import Config, GameObject


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
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(-Config.WINDOW_SIZE // 2, Config.WINDOW_SIZE // 2)
        self.ax.set_ylim(-Config.WINDOW_SIZE // 2, Config.WINDOW_SIZE // 2)
        self.ax.set_title("Survival Simulation")

        # Game Object Manager
        self.game = GameObject(self.ax)
        self.grid = self.game.spatial_grid

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

    def reset(self):
        """환경 초기화"""
        from Objects import Predator, Herbivore, Plant
        self.game.objects.clear()  # 기존 개체 제거

        # ✅ Predator 강제 생성
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

        # Herbivores & Plants 추가
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
        from Objects import Plant, Predator
        if len(action) == 2:
            dx, dy = action
            detect_target = 0  # 기본값 (탐지 X)
        else:
            dx, dy, detect_target = action

        detected_target = None
        if detect_target > 0.5:
            detected_target = self.detect_in_fov(self.predator.pos.x, self.predator.pos.y, self.predator.FOV_RADIUS)

        # 탐지된 개체가 있으면 추적/회피 결정
        if detected_target:
            if isinstance(detected_target, Plant):
                self.target_x, self.target_y = detected_target.pos.x, detected_target.pos.y
            elif isinstance(detected_target, Predator):
                self.target_x, self.target_y = -detected_target.pos.x, -detected_target.pos.y  # 반대 방향 도망

        # 이동 적용
        self.predator.pos.x += dx * 5
        self.predator.pos.y += dy * 5
        self.predator.energy -= 0.1

        # 종료 조건
        done = self.predator.energy <= 0
        reward = -10 if done else 1

        return self._get_observation(), reward, done, {}

    def detect_in_fov(self, x, y, fov_radius):
        """
        Detects the nearest object within the Field of View (FOV) using Spatial Hash Grid.
        """
        possible_objects = self.grid.retrieve_in_fov_range(x, y, fov_radius)
        best_target = None
        min_distance_sq = fov_radius ** 2  # 최대 탐색 거리 설정

        for obj in possible_objects:
            if obj is self.predator:
                continue  # 자기 자신 제외

            dx = obj.pos.x - x
            dy = obj.pos.y - y
            distance_sq = dx * dx + dy * dy  # 유클리드 거리 계산

            if distance_sq > min_distance_sq:
                continue  # FOV 범위 초과

            # 가장 가까운 객체 선택
            if distance_sq < min_distance_sq:
                best_target = obj
                min_distance_sq = distance_sq

        return best_target

    def render(self, mode="human", save_as="train.mp4"):
        """ Renders the simulation and optionally saves it as a file. """
        def animate(frame):
            self.game.update(Config.TARGET_FPS)
            return []

        ani = animation.FuncAnimation(
            fig=self.fig,
            func=animate,
            frames=Config.FRAMES,
            interval=Config.INTERVAL,
            blit=False
        )

        if save_as.endswith(".mp4"):
            ani.save(save_as, writer="ffmpeg", fps=Config.TARGET_FPS)
        elif save_as.endswith(".gif"):
            ani.save(save_as, writer="pillow", fps=Config.TARGET_FPS)

        # plt.show()

    def close(self):
        plt.close(self.fig)
