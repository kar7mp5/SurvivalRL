from stable_baselines3 import PPO
from SurvivalRL import SurvivalEnv


if __name__=='__main__':
    env = SurvivalEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000000)
    model.save("ppo_survival")

    env.render(save_as="train.mp4")
    env.close()
