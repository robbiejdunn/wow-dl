import torch
import torch.nn as nn
import torch.nn.functional as F

import math
device = torch.device("cpu")

class DQN(nn.Module):

    def __init__(self, h, w, outputs, channels: int):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        conv1_out_h, conv1_out_w = self.conv_output_shape((h, w), kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        conv2_out_h, conv2_out_w = self.conv_output_shape((conv1_out_h, conv1_out_w), kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        conv3_out_h, conv3_out_w = self.conv_output_shape((conv2_out_h, conv2_out_w), kernel_size=5, stride=2)
        linear_input_size = conv3_out_h * conv3_out_w * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        res = self.head(x.view(x.size(0), -1))
        return res

    """
    https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/3 DuaneNielsen
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    @staticmethod
    def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        from math import floor
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
        w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
        return h, w
