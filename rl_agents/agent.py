"""
heavily influenced by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 
"""

import gym

from rl_agents.envs import MoltenDropEnv, SFKFightEnv
import os
import pandas as pd

import math
import random
import subprocess
import time
import torch
import uuid
import torchvision
from PIL import Image
from skimage import io
from rl_agents.dqn import DQN
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from rl_agents.replay_memory import ReplayMemory, Transition
from rl_agents.state.state import get_state

BATCH_SIZE = 200
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5000
TARGET_UPDATE = 10

steps_done = 0
n_actions = 5
n_random = 0
n_nn = 0

device = torch.device("cpu")

policy_net = DQN(100, 100, n_actions).to(device)
print(policy_net)
# exit()
target_net = DQN(100, 100, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

results_df = pd.DataFrame()


def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


def select_action(state):
    global steps_done, n_nn, n_random
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    if sample > eps_threshold:
        with torch.no_grad():
            n_nn += 1
            return policy_net(state).max(1)[1].view(1, 1), "policy"
    else:
        n_random += 1
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), "random"


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    # print("Optimizing model")
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
    global results_df
    tmp_state = None
    tmp_action = None
    tmp_nextstate = None
    env = SFKFightEnv()
    num_episodes = 1500
    num_steps = 500
    env.reset()
    for i_episode in range(num_episodes):
        print(f"Episode {i_episode} starting")
        epi_actions = {
            "Abolish Disease": 0,
            "Flash Heal": 0,
            "Divine Spirit": 0,
            "Greater Heal": 0,
            "Pass": 0,
        }
        last_screen, _ = get_state()
        # print(last_screen)
        # print(last_screen.shape)
        # exit()
        current_screen, _ = get_state()
        state = current_screen - last_screen
        for t in range(num_steps):
            action, choice = select_action(state)
            if choice == "policy":
                if action.item() == 0:
                    epi_actions["Abolish Disease"] += 1
                elif action.item() == 1:
                    epi_actions["Flash Heal"] += 1
                elif action.item() == 2:
                    epi_actions["Divine Spirit"] += 1
                elif action.item() == 3:
                    epi_actions["Greater Heal"] += 1
                elif action.item() == 4:
                    epi_actions["Pass"] += 1
            
            obs, reward, done, info = env.step(action.item())
            # print(f"Action = {action.item()} Reward = {reward}")
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = obs
            next_state = current_screen - last_screen

            # if tmp_state is not None and tmp_action is not None and tmp_next_state is not None:
            #     memory.push(tmp_state, tmp_action, tmp_next_state, reward)
            #     print(f"Pushing to memory with action {tmp_action} reward {reward}")
            memory.push(state, action, next_state, reward)
            # print(f"Pushing to memory with action {action} reward {reward}")

            tmp_state = state
            tmp_action = action
            tmp_next_state = next_state

            state = next_state

            optimize_model()

            if done:
                global n_random, n_nn
                print(f"Episiode complete after {t} steps. {n_random/(n_random + n_nn)}% random actions")
                print(epi_actions)
                _, episode_duration = env.reset()
                if episode_duration is not None:
                    results_df = results_df.append(
                        {
                            "Episode": i_episode,
                            "Duration (s)": episode_duration, 
                            "Random actions": n_random,
                            "Policy actions": n_nn,
                            "Random %": n_random / (n_random + n_nn),
                            "Abolish disease casts": epi_actions["Abolish Disease"],
                            "Flash heal casts": epi_actions["Flash Heal"],
                            "Divine spirit casts": epi_actions["Divine Spirit"],
                            "Greater heal casts": epi_actions["Greater Heal"],
                            "Passes": epi_actions["Pass"],
                        }, 
                        ignore_index=True
                    )
                    results_df.to_csv("results.csv")
                n_random = 0
                n_nn = 0
                for filename in os.listdir('data'):
                    filepath = os.path.join('data', filename)
                    os.remove(filepath)
                # pickle model
                # torch.save(policy_net.state_dict(), "test_pickle")
                break
            
            # Sleep for GCD?
            # time.sleep(2)
    env.close()
