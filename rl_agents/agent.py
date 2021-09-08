"""
heavily influenced by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 
"""

import gym

from rl_agents.envs import MoltenDropEnv, SFKFightEnv
import os

import math
import random
import subprocess
import time
import torch
import uuid
from PIL import Image
from skimage import io
from rl_agents.dqn import DQN
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from rl_agents.replay_memory import ReplayMemory, Transition
from rl_agents.utils import get_screen

BATCH_SIZE = 20
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 600
TARGET_UPDATE = 10

steps_done = 0
n_actions = 4
n_random = 0
n_nn = 0

device = torch.device("cpu")

policy_net = DQN(60, 60, n_actions).to(device)
target_net = DQN(60, 60, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(100)


def select_action(state):
    global steps_done, n_nn, n_random
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            n_nn += 1
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        n_random += 1
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



def main():
    # env = MoltenDropEnv()
    env = SFKFightEnv()

    num_episodes = 1500
    num_steps = 1000
    for i_episode in range(num_episodes):
        print(f"Episode {i_episode} starting")
        last_screen = env.reset()
        current_screen, _ = get_screen()
        state = current_screen - last_screen
        for t in range(num_steps):
            action = select_action(state)
            obs, reward, done, info = env.step(action.item())
            # print(f"Action = {action.item()} Reward = {reward}")
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = obs
            next_state = current_screen - last_screen

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model()

            if done:
                global n_random, n_nn
                print(f"Episiode complete after {t} steps. {n_random/(n_random + n_nn)}% random actions")
                n_random = 0
                n_nn = 0
                for filename in os.listdir('data'):
                    filepath = os.path.join('data', filename)
                    os.remove(filepath)
                break
            
            # Sleep for GCD?
            # time.sleep(2)
    env.close()
