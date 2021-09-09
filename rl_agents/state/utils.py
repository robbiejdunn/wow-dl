from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class OverlayItemBoundary:
    """
    Represents the bound of an item in an overlay (the 
    co-ordinates of the rectangle it covers)
    """
    # upper left, lower left, upper right, lower right
    ul: Tuple[int, int]
    ll: Tuple[int, int]
    ur: Tuple[int, int]
    lr: Tuple[int, int]

    def extract_from_image(self, image: np.ndarray) -> np.ndarray:
        """
        Returns the pixels in the image within the boundary.
        """
        return image[
            self.ul[1]: self.ll[1],
            self.ul[0]: self.ur[0],
        ]
