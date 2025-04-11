# Dual Agent Training Script with Reward Breakdown Logging (Fixed)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import gym
import numpy as np
from stable_baselines3 import PPO
from SurvivalRL import Config, SurvivalEnv
from tqdm import tqdm
from collections import defaultdict


class PredatorEnvWrapper(gym.Env):
    def __init__(self):
        self.env = SurvivalEnv()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.reward_breakdown = defaultdict(float)

    def reset(self):
        obs = self.env.reset()
        self.reward_breakdown.clear()
        return obs["predator"]

    def step(self, action):
        actions = {"predator": action, "herbivore": np.array([0, 0, 0], dtype=np.float32)}
        obs, reward, done, info = self.env.step(actions)
        if "predator_breakdown" in info:
            for k, v in info["predator_breakdown"].items():
                self.reward_breakdown[k] += v

        return obs["predator"], reward["predator"], done["predator"], {}

    def render(self):
        self.env.render(save_as="predator_train.mp4")

    def close(self):
        self.env.close()


class HerbivoreEnvWrapper(gym.Env):
    def __init__(self):
        self.env = SurvivalEnv()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.reward_breakdown = defaultdict(float)

    def reset(self):
        obs = self.env.reset()
        self.reward_breakdown.clear()
        return obs["herbivore"]

    def step(self, action):
        actions = {"predator": np.array([0, 0, 0], dtype=np.float32), "herbivore": action}
        obs, reward, done, info = self.env.step(actions)

        if "herbivore_breakdown" in info:
            for k, v in info["herbivore_breakdown"].items():
                self.reward_breakdown[k] += v

        return obs["herbivore"], reward["herbivore"], done["herbivore"], {}

    def render(self):
        self.env.render(save_as="herbivore_train.mp4")

    def close(self):
        self.env.close()


def train_agent(name, env_class):
    np.random.seed(41)
    env = env_class()
    model = PPO("MlpPolicy", env, verbose=0)

    epochs = 10
    steps_per_epoch = Config.FRAMES // epochs
    batch_size = max(1024, steps_per_epoch // 4)

    for epoch in range(1, epochs + 1):
        total_reward = 0
        obs = env.reset()
        progress = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch} - {name}", unit="step")

        for _ in range(0, steps_per_epoch, batch_size):
            chunk = min(batch_size, steps_per_epoch - progress.n)
            for _ in range(chunk):
                action, _ = model.predict(obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward

                if isinstance(done, dict):
                    done = done[name]
                if done:
                    obs = env.reset()

            model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            progress.update(chunk)

        progress.close()
        avg = total_reward / steps_per_epoch
        tqdm.write(f"[{name}] Epoch {epoch} | avg reward per step: {avg:.4f}")

        breakdown = env.reward_breakdown if hasattr(env, "reward_breakdown") else {}
        if breakdown:
            tqdm.write("  Reward Breakdown:")
            for k, v in sorted(breakdown.items(), key=lambda x: -abs(x[1])):
                tqdm.write(f"   - {k}: {v:.2f}")
        env.reward_breakdown.clear()

    model.save(f"{name}_ppo_model")
    env.render()
    env.close()


if __name__ == "__main__":
    train_agent("predator", PredatorEnvWrapper)
    train_agent("herbivore", HerbivoreEnvWrapper)
