import argparse

# from rl_agents.agent import main
from rl_agents.train import TrainingManager

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="DL Agents in WoW 1.12")
#     parser.add_argument("--checkpoint", help="Checkpoint to start training from")
#     args = parser.parse_args()
#     print(args.checkpoint)
#     main()

if __name__ == "__main__":
    t = TrainingManager(
        gym_env='CartPole-v0', 
        channels=1,
        unwrapped=True,
    )
    t.train()
    # train('CartPole-v0')