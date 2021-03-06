import argparse

from rl_agents.train import TrainingManager


if __name__ == "__main__":
    t = TrainingManager(
        env_name='WoW', 
        # env_name='CartPole-v0',
        channels=1,
        num_iterations=5000,
        collect_steps_per_iteration=1,
        learning_rate=1e-3,
        replay_buffer_max_len=100000,
        batch_size=64,
        num_eval_episodes=3,
        initial_collect_steps=100,
        log_interval=100,
        eval_interval=20,
        episode_max_steps=1000,
    )
    t.train()
