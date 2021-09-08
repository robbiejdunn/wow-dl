import gym
import subprocess
import time
from gym import spaces
import numpy as np

from rl_agents.utils import WINDOW_FOCUS_COMMAND, get_screen


RESET_CMD = f"{WINDOW_FOCUS_COMMAND} xdotool key '2'"
EXIT_CMD = f"{WINDOW_FOCUS_COMMAND} xdotool key '1'"


class MoltenDropEnv(gym.Env):

    def __init__(self):
        super(MoltenDropEnv, self).__init__()
        self.recent_screen = None
        self.step_num = 0
        # previous pixel colour (used for reward calc)
        self.prev_pixel = 1
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(60, 60, 3), dtype=np.uint8)
    
    def _take_action(self, action):
        if action == 0:
            key = '3'
            sleep = 3
        elif action == 1:
            key = '4'
            sleep = 1.5
        elif action == 2:
            key = '5'
            sleep = 1.5
        elif action == 3:
            key = '6'
            sleep = 1.5
        cmd = f"{WINDOW_FOCUS_COMMAND} xdotool key '{key}'"
        subprocess.run(cmd, shell=True)
        time.sleep(sleep)

    def _next_observation(self):
        observation, self.recent_screen = get_screen()
        return observation
    
    def _get_reward(self):
        if self.recent_screen is None:
            return 0
        curr_pixel = self.recent_screen.getpixel((0, 0))[1]
        reward = curr_pixel + 20 * self.step_num
        self.prev_pixel = curr_pixel
        done = curr_pixel == 250 or curr_pixel == 0
        return reward, done

    def step(self, action):
        self.step_num += 1
        reward, done = self._get_reward()
        self._take_action(action)
        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        subprocess.run(RESET_CMD, shell=True)
        time.sleep(1)
        return self._next_observation()
    
    def close(self):
        subprocess.run(EXIT_CMD, shell=True)

    def render(self, mode="human", close=False):
        pass