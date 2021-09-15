import subprocess
import uuid
import skimage.io
from typing import Dict, Union
import numpy as np

from rl_agents.state.overlay import OverlayHealth, OverlayMana


def read_image(path: str) -> np.ndarray:
    """ Reads an image into a numpy tensor. """
    return skimage.io.imread(path, as_gray=True)


def parse_overlay(overlay_image: np.ndarray) -> Dict[str, Union[OverlayHealth, OverlayMana]]:
    return {
        "health": OverlayHealth(overlay_image),
        "mana": OverlayMana(overlay_image),
    }


def take_screenshot() -> str:
    """
    Uses the `scrot` command to take a screenshot.
    """
    image_path = f"data/{uuid.uuid4()}.png"
    curr_step_command = f"scrot -u '{image_path}'"
    subprocess.run(curr_step_command, shell=True)
    return image_path


def cut_overlay(image: np.ndarray) -> np.ndarray:
    """
    Cuts the overlay pixels out of the given image.

    NOTE: this should be changed to either read overlay H/W from a 
    shared location or calculate based on overlay items.
    """
    return image[:100, :100]


def get_state():
    image_path = take_screenshot()
    screen = read_image(image_path)
    screen = cut_overlay(screen)
    overlay_info = parse_overlay(screen)
    return screen, overlay_info
