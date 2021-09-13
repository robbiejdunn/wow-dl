import subprocess
import torch
import uuid
import numpy as np
from PIL import Image
import skimage.io

import torchvision.transforms as T


# get_window_id = subprocess.Popen(["xdotool search --name 'World of Warcraft'"], shell=True, stdout=subprocess.PIPE)
# window_id = int(get_window_id.communicate()[0])
# WINDOW_FOCUS_COMMAND = f"xdotool windowactivate {window_id} && sleep 0.1 &&"
# print(WINDOW_FOCUS_COMMAND)
# exit()

# resize = T.Compose([T.ToPILImage(), T.Resize(60, interpolation=Image.CUBIC)])
totensor = T.Compose([T.ToTensor()])

topil = T.Compose([T.ToPILImage()])


def read_image(path: str) -> np.ndarray:
    """ Reads an image into a numpy tensor """
    return skimage.io.imread(path, as_gray=True)


def get_screen():
    """ Gets screen data with screenshot """
    step_image = f"data/{uuid.uuid4()}.png"
    curr_step_command = f"scrot -u '{step_image}'"
    subprocess.run(curr_step_command, shell=True)
    screen = read_image(step_image)

    screen = screen[:50, :100, :]
    screen = screen.transpose((2, 0, 1))
    # image = image[:, :98, :49]
    image = torch.from_numpy(screen)
    image = topil(image)

    screen = np.ascontiguousarray(screen, dtype=np.float32)
    screen = torch.from_numpy(screen)
    screen = topil(screen)
    # return image too for reward extraction
    # image = resize(screen)
    screen = totensor(screen)
    screen = screen.unsqueeze(0)
    return screen, image
