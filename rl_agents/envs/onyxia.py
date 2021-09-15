import os
import subprocess
import time
from tf_agents.environments import py_environment, utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np

from rl_agents.state.state import get_state
from rl_agents.utils import get_window_focus_command, run_command_shell


class OnyxiaEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=4, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2, ), dtype=np.int32, minimum=0, maximum=100, name='observation'
        )
        self._state = [100, 100]
        self._episode_ended = False
        self._action_start_time = time.time()
        self._window_focus = get_window_focus_command()
        self._initial_env_setup()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    
    def _initial_env_setup(self):
        run_command_shell(self._window_focus, 7.0)
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

    def _reset(self):
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
        run_command_shell("xdotool key Return && sleep 0.2 && xdotool type '/tar Tank' && sleep 0.2 && xdotool key Return", 2.0)
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
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))
    
    def _get_state(self):
        _, game_state = get_state()
        return [game_state['health'].get_health(), game_state['mana'].get_mana()]
    
    def _take_action(self, action):
        self._action_start_time = time.time()
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

    def _step(self, action):
        # print(f"Stepped with action {action}")
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        
        # If health of target is 0 then end episode
        if self._state[0] <= 5:
            self._episode_ended = True
        else:
            self._state = self._take_action(action)
        
        reward = time.time() - self._action_start_time
        if self._episode_ended or self._state[0] == 0:
            return ts.termination(
                np.array(self._state, dtype=np.int32),
                reward,
            )
        else:
            return ts.transition(
                np.array(self._state, dtype=np.int32),
                reward=reward,
                discount=1.0,
            )

if __name__ == "__main__":
    e = SFKEnv()
    utils.validate_py_environment(e, episodes=2)
