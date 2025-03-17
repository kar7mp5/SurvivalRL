from stable_baselines3 import PPO
from SurvivalRL import SurvivalEnv

env = SurvivalEnv()

model = PPO.load("ppo_survival")

obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

env.render(save_as="infer.mp4")
env.close()
