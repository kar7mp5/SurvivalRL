import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from stable_baselines3 import PPO
from SurvivalRL import Config, SurvivalEnv
from tqdm import tqdm

# Initialize the environment
env = SurvivalEnv()

# Load the pre-trained PPO model
model = PPO.load("ppo_survival")

# Reset the environment to get the initial observation
obs = env.reset()
done = False

# Set total frames for visualization (adjust if needed)
total_frames = Config.FRAMES
progress_bar = tqdm(total=total_frames, desc="Inference Progress", unit="frame")

# Run inference loop
frame_count = 0
while not done and frame_count < total_frames:
    action, _ = model.predict(obs)  # Get predicted action from the model
    obs, reward, done, _ = env.step(action)  # Apply action to the environment
    progress_bar.update(1)  # Update tqdm progress bar
    frame_count += 1

progress_bar.close()  # Close tqdm bar after completion

# Render the final inference as a video file
env.render(save_as="infer.mp4")
env.close()
