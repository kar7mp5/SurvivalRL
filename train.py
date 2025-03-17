from stable_baselines3 import PPO
from SurvivalRL import SurvivalEnv
from tqdm import tqdm


class TqdmCallback:
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar

    def __call__(self, locals_, globals_):
        self.progress_bar.update(1)


if __name__ == '__main__':
    env = SurvivalEnv()
    model = PPO("MlpPolicy", env, verbose=0)  # Remove defaul log verbose=0

    total_timesteps = 100000
    progress_bar = tqdm(total=total_timesteps, desc="Training Progress", unit="step")

    model.learn(total_timesteps=total_timesteps, callback=TqdmCallback(progress_bar))
    model.save("ppo_survival")

    progress_bar.close()

    env.render(save_as="train.mp4")
    env.close()
