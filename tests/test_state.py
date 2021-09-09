import pytest
import skimage.io
from pkg_resources import resource_filename

from rl_agents.state.overlay import OverlayHealth, OverlayMana
from rl_agents.state.utils import OverlayItemBoundary
from rl_agents.utils import read_image


def test_overlay_item_boundary():
    with pytest.raises(TypeError):
        b = OverlayItemBoundary()
    # will be flagged by mypy for types
    b = OverlayItemBoundary(
        ul="egg",
        ll="fish",
        ur="cat",
        lr="dog",
    )
    b = OverlayItemBoundary(
        ul=(0, 0),
        ll=(0, 50),
        ur=(50, 0),
        lr=(50, 50),
    )
    assert b.ul == (0, 0)
    assert b.ll == (0, 50)
    assert b.ur == (50, 0)
    assert b.lr == (50, 50)


def test_overlay_health():
    i = read_image(resource_filename("tests", "resources/overlay_image.png"))
    h = OverlayHealth(i)
    assert(h.value == 0.058823529411764705)


def test_overlay_mana():
    i = read_image(resource_filename("tests", "resources/overlay_image.png"))
    h = OverlayMana(i)
    assert(h.value == 0.44705882352941173)


if __name__ == "__main__":
    test_overlay_health()
