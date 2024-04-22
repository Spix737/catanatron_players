import os
import csv
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import gymnasium as gym
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
from catanatron.models.enums import CITY, ROAD, SETTLEMENT, VICTORY_POINT
from catanatron.models.player import Color
from catanatron.state_functions import calculate_resource_probabilities, get_dev_cards_in_hand, get_largest_army, get_longest_road_color, get_player_buildings, player_key

def save_to_csv(file_path, game_data):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        headers = ['GameID', 'TurnCount', 'Epsilon', 'AverageLoss'] + list(game_data.keys())
        writer.writerow(headers)

class DQN(nn.Module):
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(fc3_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, env, my_color, gamma, epsilon, lr, batch_size, state_size, n_actions,
            max_mem_size=5000000, eps_end=0.01, eps_dec=26e-5):
        self.env = env
        self.state_size = state_size
        self.action_size = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_initial = epsilon
        self.epsilon_min = eps_end
        self.epsilon_decay = eps_dec
        self.learning_rate = lr
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.loss_history = []
        self.action_choices = []

        input_dims = int(np.prod(env.observation_space.shape))
        self.Q_eval = DQN(self.learning_rate, input_dims=input_dims, 
                                   fc1_dims=512, fc2_dims=256, fc3_dims=128, n_actions=n_actions)
        
        self.scheduler = ExponentialLR(self.Q_eval.optimizer, gamma=0.99)  # Set up the learning rate scheduler

        self.state_memory = np.zeros((self.mem_size, *state_size), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *state_size), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool) # end of game check

    def reset_epsilon(self):
        self.epsilon = self.epsilon_initial

    def store_transition(self, state, action, reward, state_, done):
        """
        Store the transition of the state, action, reward, new state, and done.

        Args:

        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1


    def choose_action(self, observation):
        """
        Choose the action based on the observation.

        Args:
            observation: observation of the environment

        Returns:
            action: action to take
        """
        valid_actions = self.env.unwrapped.get_valid_actions()
        if np.random.random() <= self.epsilon:
            # Exploration: Randomly choose from valid actions
            action = np.random.choice(valid_actions)
        else:
            # Exploitation: Choose the best action based on the model's prediction
            state = torch.tensor(np.array(observation), dtype=torch.float32).to(self.Q_eval.device)
            with torch.no_grad():
                actions = self.Q_eval(state)

            # Convert action values to numpy for easier manipulation
            action_values = actions.cpu().numpy().squeeze()

            # Mask invalid actions by setting their Q-values to negative infinity
            mask = np.isin(np.arange(action_values.size), valid_actions, invert=True)
            action_values[mask] = -np.inf

            # Select the action with the highest Q-value among valid actions
            action = np.argmax(action_values)

        # After choosing the action, record it
        if action != 0 and action != 289: 
            self.action_choices.append(action)
        
        return action

    def learn(self):
        """
        Learn the Q-values by sampling from the memory.
        """
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.scheduler.step()  # Step the scheduler at the end of each training episode

        # Record the loss and adjust epsilon
        self.loss_history.append(loss.item())
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
        return loss.item()

def worker(env, agent, start_event, reset_queue, result_queue):
    start_event.wait()
    while True:
        initial_state = reset_queue.get()
        if initial_state is None:
            break
        observation = initial_state
        done = False
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            agent.store_transition(observation, action, reward, observation_, done)
            loss = agent.learn()
            observation = observation_
        result_queue.put((reward, agent.epsilon, loss))

def env_initializer(num_episodes, reset_queue, env_name):
    env = gym.make(env_name)
    for _ in range(num_episodes):
        if _ % 4000 == 0:
            agent.reset_epsilon()
        state = env.reset()
        reset_queue.put(state)
    for _ in range(num_workers):
        reset_queue.put(None)

if __name__ == "__main__":
    num_workers = 4
    num_episodes = 12000
    env_name = 'catanatron_gym:catanatronReward-v1'

    reset_queue = mp.Queue()
    result_queue = mp.Queue()
    start_event = mp.Event()

    env = gym.make(env_name)
    agent = DQNAgent(env, Color.BLUE, 0.99, 1.0, 0.0005, 256, env.observation_space.shape, 290)

    initializer = mp.Process(target=env_initializer, args=(num_episodes, reset_queue, env_name))
    initializer.start()

    workers = [mp.Process(target=worker, args=(env, agent, start_event, reset_queue, result_queue)) for _ in range(num_workers)]
    for worker in workers:
        worker.start()

    start_event.set()

    initializer.join()
    for worker in workers:
        worker.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    file_path = 'model_data_r/training_outcomes.csv'
    os.makedirs('model_data_r', exist_ok=True)
    for result in results:
        reward, epsilon, loss = result
        game_data = {'reward': reward, 'epsilon': epsilon, 'loss': loss}
        save_to_csv(file_path, game_data)

    print("Training completed.")
