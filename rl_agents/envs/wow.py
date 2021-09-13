import subprocess
import time
from tf_agents.environments import py_environment, utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np

from rl_agents.state.state import get_state
from rl_agents.utils import get_window_focus_command


class SFKEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=4, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2, ), dtype=np.int32, minimum=0, maximum=100, name='observation'
        )
        self._state = [100, 100]
        self._episode_ended = False
        self._episode_start_time = time.time()
        self._window_focus = get_window_focus_command()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        wf = self._window_focus
        print(f"WF command = \"{wf}\"")
        reset_commands = [
            f"{wf} xdotool mousemove 300 300 click 3",
            f"{wf} xdotool key Return && xdotool type '/run ClearTarget()' && xdotool key Return",
            f"{wf} xdotool key Return && xdotool type '/g .revive' && xdotool key Return",
            f"{wf} xdotool key Return && xdotool type '.gm on' && xdotool key Return",
            f"{wf} xdotool key Return && xdotool type '.tele sfk' && xdotool key Return",
            f"{wf} xdotool key Return && xdotool type '/tar Scarlet Cavalier' && xdotool key Return",
            f"{wf} xdotool key Return && xdotool type '.die' && xdotool key Return",
            f"{wf} xdotool key Return && xdotool type '/tar Splintertree Guard' && xdotool key Return",
            f"{wf} xdotool key Return && xdotool type '.die' && xdotool key Return",
            f"{wf} xdotool key Return && xdotool type '/run ClearTarget()' && xdotool key Return",
            f"{wf} xdotool key Return && xdotool type '.modify mana 6000' && xdotool key Return",
            f"{wf} xdotool key Return && xdotool type '.respawn' && xdotool key Return",
            f"{wf} xdotool key Return && xdotool type '/tar Splintertree Guard' && xdotool key Return",
            f"{wf} xdotool key Return && xdotool type '.npc yell Save me Dizia!' && xdotool key Return",
            f"{wf} xdotool key Return && xdotool type '/em started training episode' && xdotool key Return",
        ]
        print("Resetting in 5 seconds...")
        time.sleep(5)
        for c in reset_commands:
            subprocess.call(c, shell=True)
        time.sleep(8)
        self._state = self._get_state()
        self._episode_ended = False
        self._episode_start_time = time.time()
        return ts.restart(np.array(self._state, dtype=np.int32))
    
    def _get_state(self):
        _, game_state = get_state()
        return [game_state['health'].get_health(), game_state['mana'].get_mana()]
    
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
            cmd = f"{self._window_focus} xdotool key '{key}'"
            subprocess.run(cmd, shell=True)
        time.sleep(sleep + 0.3)
        game_state = self._get_state()
        return game_state

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        
        # If health of target is 0 then end episode
        if self._state[0] == 0:
            self._episode_ended = True
        else:
            self._state = self._take_action(action)
        
        reward = time.time() - self._episode_start_time
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
