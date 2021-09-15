import math
import numpy as np

from rl_agents.state.utils import OverlayItemBoundary


class OverlayMana:
    bound = OverlayItemBoundary(
        ul=(50, 0),
        ll=(50, 50),
        ur=(100, 0),
        lr=(100, 50),
    )

    def __init__(self, overlay_image: np.ndarray):
        self.pixels = self.bound.extract_from_image(overlay_image)
        self.value = self.pixels[(0, 0)]
        # print(self.value)
    
    def get_mana(self):
        return math.ceil(100 * (1 - self.value))
