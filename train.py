import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from stable_baselines3 import PPO
from SurvivalRL import Config, SurvivalEnv
from tqdm import tqdm

if __name__ == '__main__':
    env = SurvivalEnv()
    model = PPO("MlpPolicy", env, verbose=0)  # Keep verbose=0 to avoid log spam

    total_timesteps = Config.FRAMES
    batch_size = max(1024, total_timesteps // 100)  # Adjust batch size dynamically
    progress_bar = tqdm(total=total_timesteps, desc="Training Progress", unit="step")

    # Train in larger batches instead of 1-step increments
    for _ in range(0, total_timesteps, batch_size):
        steps = min(batch_size, total_timesteps - progress_bar.n)  # Handle last batch
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        progress_bar.update(steps)

    model.save("ppo_survival")

    progress_bar.close()

    # env.render(save_as="train.mp4")
    env.close()
