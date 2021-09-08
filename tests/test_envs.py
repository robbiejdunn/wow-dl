from stable_baselines.common.env_checker import check_env

from rl_agents.envs import MoltenDropEnv


def test_molten_drop_valid():
    env = MoltenDropEnv()
    check_env(env)