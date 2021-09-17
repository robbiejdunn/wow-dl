import gym

from stable_baselines3 import DQN


def train():
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./logs/dqn_cartpole_tensorboard/")
    model.learn(total_timesteps=10000)
    model.learn(total_timesteps=50000)
