import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 50, stride=50)

        # self.conv1 = nn.Conv2d(16, 33, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(2)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=50, stride=50):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convw = conv2d_size_out(w)
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        convh = conv2d_size_out(h)
        linear_input_size = convw * convh * 2
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        res = self.head(x.view(x.size(0), -1))
        return res