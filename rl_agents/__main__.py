import argparse
import gym
from stable_baselines3 import DQN

from rl_agents.config import parse_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training framework WoW RL agents")
    parser.add_argument("-c", "--config", required=True, help="path of the experiment configuration file")
    args = parser.parse_args()
    config = parse_config(args.config)
    env = gym.make(config["env"])
    eval_env = gym.make(config["env"])
    model = DQN(
        policy=config["policy"],
        env=env,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        train_freq=config["train_freq"],
        gradient_steps=config["gradient_steps"],
        target_update_interval=config["target_update_interval"],
        exploration_fraction=config["exploration_fraction"],
        exploration_final_eps=config["exploration_final_eps"],
        tensorboard_log="logs/dqn-cartpole",
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
    )
    model.learn(
        total_timesteps=config["n_timesteps"],
        eval_env=eval_env,
        eval_freq=10000,
        eval_log_path="logs/dqn-cartpole"
    )
