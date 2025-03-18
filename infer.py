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

# Set total frames for inference
total_frames = Config.FRAMES
batch_size = max(1024, total_frames // 100)  # Adjust batch size dynamically
progress_bar = tqdm(total=total_frames, desc="Inference Progress", unit="frame")

# Run inference in larger batches for efficiency
for _ in range(0, total_frames, batch_size):
    steps = min(batch_size, total_frames - progress_bar.n)  # Handle last batch
    for _ in range(steps):
        if done:
            break  # Stop if episode is finished
        action, _ = model.predict(obs)  # Get predicted action from the model
        obs, reward, done, _ = env.step(action)  # Apply action to the environment
        progress_bar.update(1)  # Update tqdm progress bar

progress_bar.close()  # Close tqdm bar after completion

# Render the final inference as a video file
print("Rendering inference results...")
env.render(save_as="infer.mp4")
print("Rendering complete!")

env.close()
