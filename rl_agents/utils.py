import subprocess
import torch
import uuid
import numpy as np
from PIL import Image
from skimage import io
import torchvision.transforms as T


WINDOW_FOCUS_COMMAND = "xdotool search --name 'World of Warcraft' windowactivate && sleep 0.1 &&"

resize = T.Compose([T.ToPILImage(), T.Resize(60, interpolation=Image.CUBIC)])
totensor = T.Compose([T.ToTensor()])

topil = T.Compose([T.ToPILImage()])


def get_screen():
    """ Gets screen data with screenshot """
    step_image = f"data/{uuid.uuid4()}.png"
    # print(step_image)
    curr_step_command = f"scrot -u '{step_image}'"
    subprocess.run(curr_step_command, shell=True)
    screen = io.imread(step_image)
    # io.imshow(screen)
    # exit()
    image = screen.transpose((2, 0, 1))
    image = image[:, :60, :60]
    image = torch.from_numpy(image)
    image = topil(image)


    screen = screen.transpose((2, 0, 1))
    screen = screen[:, :60, :60]
    screen = np.ascontiguousarray(screen, dtype=np.float32)
    screen = torch.from_numpy(screen)
    screen = resize(screen)
    # return image too for reward extraction
    # image = resize(screen)
    screen = totensor(screen)
    screen = screen.unsqueeze(0)
    return screen, image
