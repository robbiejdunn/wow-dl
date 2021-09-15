import skimage.io
import subprocess
import time
import numpy as np


def run_command_shell(command: str, sleep_time: float):
    proc = subprocess.Popen(
        [command],
        shell=True,
        stdout=subprocess.PIPE
    )
    time.sleep(sleep_time)
    return proc.communicate()


def get_window_focus_command():
    window_id = int(run_command_shell("xdotool search --name 'World of Warcraft'", 1.0)[0])
    window_focus_command = f"xdotool windowactivate {window_id}"
    return window_focus_command

def read_image(path: str) -> np.ndarray:
    """ Reads an image into a numpy tensor """
    return skimage.io.imread(path, as_gray=True)
