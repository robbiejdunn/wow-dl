import numpy as np

from rl_agents.state.utils import OverlayItemBoundary


class OverlayHealth:
    bound = OverlayItemBoundary(
        ul=(0, 0),
        ll=(0, 50),
        ur=(50, 0),
        lr=(50, 50),
    )

    def __init__(self, overlay_image: np.ndarray):
        self.pixels = self.bound.extract_from_image(overlay_image)
        self.value = self.pixels[(0, 0)]
        # print(self.value)