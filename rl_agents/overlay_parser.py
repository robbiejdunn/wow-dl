import skimage.io
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Bound:
    """
    Represents the bound of an item in an overlay (the 
    co-ordinates of the area it covers)
    """
    # upper left, lower left, upper right, lower right
    ul: Tuple[int, int]
    ll: Tuple[int, int]
    ur: Tuple[int, int]
    lr: Tuple[int, int]


class OverlayHealth:
    bound = Bound(
        ul=(0, 0),
        ll=(0, 50),
        ur=(50, 0),
        lr=(50, 50),
    )

    def __init__(self, image: skimage.io.Image):
        print("initialised")


# @dataclass
# class OverlayInformation:
#     health: float
#     mana: float


# def parse_skimage(image: skimage.io.Image):
#     """
#     Parses a scikit-image image and returns `OverlayInformation` object
#     """
#     return OverlayInformation(
#         health=0,
#         mana=0,
#     )
