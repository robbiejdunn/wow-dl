import argparse

from rl_agents.agent import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DL Agents in WoW 1.12")
    parser.add_argument("--checkpoint", help="Checkpoint to start training from")
    args = parser.parse_args()
    print(args.checkpoint)
    main()