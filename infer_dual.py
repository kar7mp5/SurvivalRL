# infer_dual.py (dict-based observation/action inference)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from stable_baselines3 import PPO
from SurvivalRL import Config, SurvivalEnv
from tqdm import tqdm
import numpy as np

# Load trained models
pred_model = PPO.load("predator_ppo_model")
herb_model = PPO.load("herbivore_ppo_model")

# Initialize shared environment
env = SurvivalEnv()
obs = env.reset()
done = False

progress = tqdm(total=Config.FRAMES, desc="Inference", unit="frame")

for _ in range(Config.FRAMES):
    if done:
        break

    # Extract observations for each agent from the observation dict
    pred_obs = obs["predator"]
    herb_obs = obs["herbivore"]

    # Predict actions with detect flag always enabled
    pred_action, _ = pred_model.predict(pred_obs)
    herb_action, _ = herb_model.predict(herb_obs)

    pred_action = np.append(pred_action[:2], 1)  # [dx, dy, detect]
    herb_action = np.append(herb_action[:2], 1)

    actions = {
        "predator": pred_action,
        "herbivore": herb_action
    }

    # Step through the environment
    obs, reward, done, _ = env.step(actions)

    progress.update(1)

progress.close()

print("Rendering inference results...")
env.render(save_as="infer_dual.mp4")
env.close()
print("Rendering complete!")
