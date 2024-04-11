import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import mse_loss
import numpy as np
import random
from collections import deque
import random
import gymnasium as gym
from datetime import timedelta
import time

from catanatron.models.player import Color
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    

class DQNAgent:
    def __init__(self, env, my_color, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        valid_actions = self.env.unwrapped.get_valid_actions()
        if np.random.rand() <= self.epsilon:
            return np.random.choice(valid_actions)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        act_values = self.model(state)
        act_values = act_values.detach().numpy().squeeze()
        # Mask invalid actions by setting their Q-values to a large negative value
        act_values[~np.isin(np.arange(self.action_size), valid_actions)] = -np.inf
        return np.argmax(act_values)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).squeeze().to(device)
        next_states = torch.FloatTensor(next_states).squeeze().to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)


        # Predict the Q-values of the current states
        q_values = self.model(states)
        # Select the Q-value for the action taken
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Predict the Q-values of the next states using the target network
        next_q_values = self.model(next_states).detach()
        # Mask invalid actions for next states as well
        valid_actions_masks = np.array([self.env.unwrapped.get_valid_actions(state_index) for state_index in range(batch_size)])
        for index, valid_actions in enumerate(valid_actions_masks):
            next_q_values[index][~np.isin(np.arange(self.action_size), valid_actions)] = -np.inf

        # Select the maximum Q-value for the next state
        next_q_value = next_q_values.max(1)[0]
        # Compute the target Q-value
        expected_q_values = rewards + (self.gamma * next_q_value * (1 - dones))

        # Compute loss
        loss = mse_loss(q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



training_logs
