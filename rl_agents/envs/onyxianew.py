import gym
import os
import time
from gym import spaces
from stable_baselines3.common.env_checker import check_env
import numpy as np

from rl_agents.state.state import get_state
from rl_agents.utils import get_window_focus_command, run_command_shell


class OnyxiaEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(OnyxiaEnv, self).__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=100, shape=(2, ), dtype=np.int32)
        self._setup_environment()

    def _setup_environment(self):
        try:
            run_command_shell(get_window_focus_command(), 7.0)
        except:
            raise ValueError("WoW window not found, is client running?")
        run_command_shell("xdotool mousemove 300 300 click 3", 2.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type --delay 20 '/run ClearTarget()' && sleep 0.2 && xdotool key Return", 3.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/g .gm on' && sleep 0.2 && xdotool key Return", 3.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/g .revive' && sleep 0.2 && xdotool key Return", 2.0)
        # Go to tank spot
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/g .tele OnyHealSpot' && sleep 0.2 && xdotool key Return", 10.0)
        # Kill Onyxia so she can be respawned on reset
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/tar Onyxia' && sleep 0.2 && xdotool key Return", 2.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/g .die' && sleep 0.2 && xdotool key Return", 3.0)
        # Remove and add bot twice due to DC bug
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/g .bot remove Tank' && sleep 0.2 && xdotool key Return", 4.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/g .bot add Tank' && sleep 0.2 && xdotool key Return", 4.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/g .bot remove Tank' && sleep 0.2 && xdotool key Return", 4.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/g .bot add Tank' && sleep 0.2 && xdotool key Return", 4.0)
        # Invite tank & set tank combat orders & no loot & stay
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/inv Tank' && sleep 0.2 && xdotool key Return", 4.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/w Tank orders combat tank' && sleep 0.2 && xdotool key Return", 3.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/w Tank collect none' && sleep 0.2 && xdotool key Return", 3.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/w Tank stay' && sleep 0.2 && xdotool key Return", 3.0)
        # Convert to raid & tele tank
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/script ConvertToRaid()' && sleep 0.2 && xdotool key Return", 6.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/tar Tank' && sleep 0.2 && xdotool key Return", 6.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/g .tele name OnyTankSpot' && sleep 0.2 && xdotool key Return", 4.0)

    def _get_state(self):
        _, game_state = get_state()
        return np.array([game_state['health'].get_health(), game_state['mana'].get_mana()], dtype=np.int32)

    def _take_action(self, action):
        key = None
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
        elif action == 4:
            sleep = 0.5
        else:
            raise ValueError("Expected action in range (0, 4)")
        if key is not None:
            run_command_shell(f"xdotool key '{key}'", 0.3)
        time.sleep(sleep)
        game_state = self._get_state()
        return game_state
    
    def reset(self):
        run_command_shell("xdotool mousemove 300 300 click 3", 4.0)
        # Kill Onyxia and Tank
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/tar Onyxia' && sleep 0.2 && xdotool key Return", 2.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/g .die' && sleep 0.2 && xdotool key Return", 2.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/tar Tank' && sleep 0.2 && xdotool key Return", 2.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/g .die' && sleep 0.2 && xdotool key Return", 2.0)
        # Reset mana
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/run ClearTarget()' && sleep 0.2 && xdotool key Return", 3.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/g .modify mana 6000' && sleep 0.2 && xdotool key Return", 3.0)
        # Revive, repair & tele tank
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/tar Tank' && sleep 0.2 && xdotool key Return", 3.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/g .revive' && sleep 0.2 && xdotool key Return", 2.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/g .repair' && sleep 0.2 && xdotool key Return", 2.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/g .tele name OnyTankSpot' && sleep 0.2 && xdotool key Return", 2.0)
        # Respawn onyxia and send tank to attack
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/tar Onyxia' && sleep 0.2 && xdotool key Return", 2.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/g .respawn' && sleep 0.2 && xdotool key Return", 2.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/tar Onyxia' && sleep 0.2 && xdotool key Return", 4.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/w Tank attack' && sleep 0.2 && xdotool key Return", 2.0)
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/tar Tank' && sleep 0.2 && xdotool key Return", 1.0)
        for filename in os.listdir('data'):
            filepath = os.path.join('data', filename)
            os.remove(filepath)
        self._state = self._get_state()
        return self._state

    def step(self, action):
        done = False
        # If health of target is 0 then end episode
        if self._state[0] <= 5:
            done = True
        else:
            self._state = self._take_action(action)
        reward = 1
        return self._state, reward, done, {}

if __name__ == "__main__":
    oe = OnyxiaEnv()
    check_env(oe)