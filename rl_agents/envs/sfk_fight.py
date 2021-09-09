import datetime
import gym
import subprocess
import time
from gym import spaces
import numpy as np

from rl_agents.utils import WINDOW_FOCUS_COMMAND
from rl_agents.state.state import get_state


RESET_CMD = [
    f"{WINDOW_FOCUS_COMMAND} xdotool key Return && xdotool mousemove 300 300 click 3",
    f"{WINDOW_FOCUS_COMMAND} xdotool key Return && xdotool type '/run ClearTarget()' && xdotool key Return",
    f"{WINDOW_FOCUS_COMMAND} xdotool key Return && xdotool type '/g .revive' && xdotool key Return",
    f"{WINDOW_FOCUS_COMMAND} xdotool key Return && xdotool type '.gm on' && xdotool key Return",
    f"{WINDOW_FOCUS_COMMAND} xdotool key Return && xdotool type '.tele sfk' && xdotool key Return",
    f"{WINDOW_FOCUS_COMMAND} xdotool key Return && xdotool type '/tar Scarlet Cavalier' && xdotool key Return",
    f"{WINDOW_FOCUS_COMMAND} xdotool key Return && xdotool type '.die' && xdotool key Return",
    f"{WINDOW_FOCUS_COMMAND} xdotool key Return && xdotool type '/tar Splintertree Guard' && xdotool key Return",
    f"{WINDOW_FOCUS_COMMAND} xdotool key Return && xdotool type '.die' && xdotool key Return",
    f"{WINDOW_FOCUS_COMMAND} xdotool key Return && xdotool type '/run ClearTarget()' && xdotool key Return",
    f"{WINDOW_FOCUS_COMMAND} xdotool key Return && xdotool type '.modify mana 6000' && xdotool key Return",
    f"{WINDOW_FOCUS_COMMAND} xdotool key Return && xdotool type '.respawn' && xdotool key Return",
    f"{WINDOW_FOCUS_COMMAND} xdotool key Return && xdotool type '/tar Splintertree Guard' && xdotool key Return",
    f"{WINDOW_FOCUS_COMMAND} xdotool key Return && xdotool type '.npc yell Save me Dizia!' && xdotool key Return",
    f"{WINDOW_FOCUS_COMMAND} xdotool key Return && xdotool type '/em started training episode at {datetime.datetime.now()}' && xdotool key Return",
]
EXIT_CMD = f"{WINDOW_FOCUS_COMMAND} xdotool key '1'"


class SFKFightEnv(gym.Env):

    def __init__(self):
        super(SFKFightEnv, self).__init__()
        self.recent_screen = None
        self.step_num = 0
        # previous pixel colour (used for reward calc)
        self.prev_pixel = 1
        self.prev_mana_pixel = 1
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(50, 100, 3), dtype=np.uint8)
        self.start_time = None
        self.last_action_time = 1
    
    def _take_action(self, action):
        if action == 0:
            key = '3'
            sleep = 1.5
        elif action == 1:
            key = '4'
            sleep = 1.5
        elif action == 2:
            key = '5'
            sleep = 1.5
        elif action == 3:
            key = '6'
            sleep = 2.5
        # print(f"Action = {key}")
        self.last_action_time = sleep
        cmd = f"{WINDOW_FOCUS_COMMAND} xdotool key '{key}'"
        subprocess.run(cmd, shell=True)
        time.sleep(sleep + 0.1)

    def _next_observation(self):
        observation, self.recent_screen = get_state()
        # self.recent_screen.save(f"aioverlay_{self.step_num}.png")
        return observation
    
    def _get_reward(self):
        if self.recent_screen is None:
            return 0
        # self.recent_screen.save(f"file_test{self.step_num}.png")

        # curr_pixel = self.recent_screen.getpixel((0, 0))[0]
        # health_pct_diff = (self.prev_pixel - curr_pixel) / 255
        # curr_mana_pixel = self.recent_screen.getpixel((50, 0))[0]
        # mana_pct_diff = (self.prev_mana_pixel - curr_mana_pixel) / 255
        
        # healing per mana per second
        # hpmps = (health_pct_diff / mana_pct_diff) / self.last_action_time
        reward = time.time() - self.start_time
        # print(f"H%diff = {health_pct_diff} M%diff = {mana_pct_diff}")
        # print(f"HPerManaPerSecond = {reward}")
        # time_elapsed = time.time() - self.start_time
        # reward = (0.01 * time_elapsed) + (10 * health_pct_diff)
        # reward = (0.1 * self.step_num) + (3 * health_pct_diff)

        # self.prev_pixel = curr_pixel
        # self.prev_mana_pixel = curr_mana_pixel
        # print ("curr", curr_pixel, "pix", pix_loaded[1, 1], "rgb", r, g, b)
        # done = curr_pixel == 255
        done = False
        if done:
            reward = -1000
        # print(f"Action reward = {reward}")
        return reward, done

    def step(self, action):
        self._take_action(action)
        self.step_num += 1
        obs = self._next_observation()
        reward, done = self._get_reward()
        # if not done:
        return obs, reward, done, {}

    def reset(self):
        if self.start_time:
            print(f"Episode took {time.time() - self.start_time} seconds")
        print("Resetting in 5 sec")
        time.sleep(5)
        for c in RESET_CMD:
            subprocess.call(c, shell=True)
        time.sleep(8)
        self.start_time = time.time()
        return self._next_observation()
    
    def close(self):
        subprocess.run(EXIT_CMD, shell=True)

    def render(self, mode="human", close=False):
        pass