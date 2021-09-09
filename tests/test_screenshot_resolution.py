import pytest
import torch
from PIL import Image
from skimage import io

from rl_agents.utils import topil


FILE_PATH = "test_1280x800.png"


def test_1280_800():
    with Image.open(FILE_PATH):
        pass


def test_thealth_pixels():
    with Image.open(FILE_PATH) as im:
        ul_rgb = im.getpixel((0, 0))
        ll_rgb = im.getpixel((0, 49))
        assert im.getpixel((0, 50)) != ll_rgb
        ur_rgb = im.getpixel((49, 0))
        lr_rgb = im.getpixel((49, 49))
        assert im.getpixel((50, 0)) != ur_rgb
        print(im.getpixel((50, 0)), ur_rgb, im.getpixel((48, 0)))


def test_skimage():
    im = io.imread(FILE_PATH)
    im = im[:50, :100, :]
    im = im.transpose((2, 0, 1))
    im = torch.from_numpy(im)
    im = topil(im)
    im.show()
    ul_rgb = im.getpixel((0, 0))
    ll_rgb = im.getpixel((0, 49))
    assert im.getpixel((0, 50)) != ll_rgb
    ur_rgb = im.getpixel((49, 0))
    lr_rgb = im.getpixel((49, 49))
    assert im.getpixel((50, 0)) != ur_rgb
    print(im.getpixel((50, 0)), ur_rgb, im.getpixel((48, 0)))
    x, y = img.size()
    print(x, y)


if __name__ == "__main__":
    # test_1280_800()
    # test_thealth_pixels()
    test_skimage()
