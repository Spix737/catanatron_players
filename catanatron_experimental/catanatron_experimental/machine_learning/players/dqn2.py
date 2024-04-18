import os
import csv
import pdb
from catanatron.models.enums import CITY, ROAD, SETTLEMENT, VICTORY_POINT, FastResource
import pandas as pd
from sklearn.model_selection import learning_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import random
from collections import deque
import random
import gymnasium as gym
from datetime import timedelta
import time
import matplotlib.pyplot as plt
from catanatron.models.player import Color
from catanatron.state_functions import calculate_resource_probabilities, get_dev_cards_in_hand, get_largest_army, get_longest_road_color, get_player_buildings, player_key

def save_to_csv(file_path, game_id, game_data, turn_count, epsilon, average_loss):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write headers if the file does not exist
        if not file_exists:
            headers = ['GameID', 'TurnCount', 'Epsilon', 'AverageLoss']  + list(game_data.keys())
            writer.writerow(headers)
        
        # Write game data
        data = [game_id, turn_count, epsilon, average_loss] + list(game_data.values())
        writer.writerow(data)

class DQN(nn.Module):
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
        super(DQN, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # Initial layer
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.relu1 = nn.ReLU()
        
        # Adding an extra hidden layer to handle complexity
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.relu2 = nn.ReLU()
        
        # Third layer
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.relu3 = nn.ReLU()
        
        # Output layer
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
            max_mem_size=100000, eps_end=0.01, eps_dec=1e-4):
        self.env = env
        self.state_size = state_size
        self.action_size = n_actions
        self.memory = deque(maxlen=5000000)
        self.gamma = gamma
        self.epsilon = epsilon
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
    
def game_end_collector(dqn_agent):

    my_color = Color.BLUE
    key = player_key(env.unwrapped.game.state, Color.BLUE)

    end_points = env.unwrapped.game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
    cities = len(get_player_buildings(env.unwrapped.game.state, my_color, CITY))
    settlements = len(get_player_buildings(env.unwrapped.game.state, my_color, SETTLEMENT))
    road = len(get_player_buildings(env.unwrapped.game.state, my_color, ROAD))
    longest = get_longest_road_color(env.unwrapped.game.state) == my_color
    largest = get_largest_army(env.unwrapped.game.state)[0] == my_color
    devvps = get_dev_cards_in_hand(env.unwrapped.game.state, my_color, VICTORY_POINT)
    probabilities = calculate_resource_probabilities(env.unwrapped.game.state)
    resource_production = { 
        'WOOD': probabilities[my_color]['WOOD'],
        'BRICK': probabilities[my_color]['BRICK'],
        'SHEEP': probabilities[my_color]['SHEEP'],
        'WHEAT': probabilities[my_color]['WHEAT'],
        'ORE': probabilities[my_color]['ORE'],
        }
    total_resource_production = sum(resource_production.values())

    total_resources_gained = env.unwrapped.my_card_counter.total_resources_gained[my_color]
    amount_of_resources_used = env.unwrapped.my_card_counter.total_resources_used[my_color]
    seven_robbers_moved = env.unwrapped.my_card_counter.total_robbers_moved[my_color] - env.unwrapped.game.state.player_state[f"{key}_PLAYED_KNIGHT"]
    knights_and_robbers_moved = env.unwrapped.my_card_counter.total_robbers_moved[my_color]
    total_robber_gain = env.unwrapped.my_card_counter.total_robber_gain[my_color]
    total_resources_lost = env.unwrapped.my_card_counter.total_resources_lost[my_color]
    total_resources_discarded = env.unwrapped.my_card_counter.total_resources_discarded[my_color]
    

    dev_cards_held = {
        'KNIGHT':env.unwrapped.game.state.player_state[f"{key}_KNIGHT_IN_HAND"],
        'VP': env.unwrapped.game.state.player_state[f"{key}_VICTORY_POINT_IN_HAND"],
        'MONOPOLY': env.unwrapped.game.state.player_state[f"{key}_MONOPOLY_IN_HAND"],
        'ROAD_BUILDING': env.unwrapped.game.state.player_state[f"{key}_ROAD_BUILDING_IN_HAND"],
        'YEAR_OF_PLENTY': env.unwrapped.game.state.player_state[f"{key}_YEAR_OF_PLENTY_IN_HAND"],
    }
    dev_cards_held_total = sum(dev_cards_held.values())
    
    dev_cards_used = {
        'KNIGHT': env.unwrapped.game.state.player_state[f"{key}_PLAYED_KNIGHT"],
        'MONOPOLY': env.unwrapped.game.state.player_state[f"{key}_PLAYED_MONOPOLY"],
        'ROAD_BUILDING': env.unwrapped.game.state.player_state[f"{key}_PLAYED_ROAD_BUILDING"],
        'YEAR_OF_PLENTY': env.unwrapped.game.state.player_state[f"{key}_PLAYED_YEAR_OF_PLENTY"],
    }
    dev_cards_used_total = sum(dev_cards_used.values())
    dev_cards_bought_total = dev_cards_held_total + dev_cards_used_total
    
    game_data = {
        'end_points': end_points,
        'cities': cities,
        'settlements': settlements,
        'road': road,
        'longest': longest,
        'largest': largest,
        'devvps': devvps,
        'resource_production': resource_production,
        'total_resource_production': total_resource_production,
        'total_resources_gained': total_resources_gained,
        'amount_of_resources_used': amount_of_resources_used,
        'seven_robbers_moved': seven_robbers_moved,
        'knights_and_robbers_moved': knights_and_robbers_moved,
        'total_robber_gain': total_robber_gain,
        'total_resources_lost': total_resources_lost,
        'total_resources_discarded': total_resources_discarded,
        'dev_cards_held': dev_cards_held,
        'dev_cards_held_total': dev_cards_held_total,
        'dev_cards_used': dev_cards_used,
        'dev_cards_used_total': dev_cards_used_total,
        'dev_cards_bought_total': dev_cards_bought_total,
    }

    return game_data


if __name__ == '__main__':
    starttime = time.perf_counter()
    file_path = 'model_data/training_outcomes.csv'
    os.makedirs('model_data', exist_ok=True)
    game_id = 0


    env = gym.make('catanatron_gym:catanatron-v1')
    agent = DQNAgent(env=env, my_color=Color.BLUE, state_size=env.observation_space.shape, gamma=0.99, epsilon=1.0, batch_size=256,
                      n_actions=290, eps_end=0.01, lr=0.0005)
    best_total_reward = 0 # flawed as max = 1
    best_end_points = 0 # max=10 
    scores, eps_history, avg_loss_per_episode = [], [], []
    n_games = 10000

    for i in range(n_games):
        score = 0
        done = False
        episode_losses = []
        observation, info = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info_ = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)
            observation = observation_
            info = info_

        # os.system('cls') # if os.name == 'nt' else 'clear')
        # os.system('clear')

        if episode_losses:
            avg_loss = np.mean(episode_losses)
            avg_loss_per_episode.append(avg_loss)
        else:
            avg_loss = 0
            avg_loss_per_episode.append(0)

        scores.append(score)
        eps_history.append(agent.epsilon)

        turn_count = env.unwrapped.game.state.num_turns
        key = player_key(env.unwrapped.game.state, Color.BLUE)
        end_points = env.unwrapped.game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
        game_stats = game_end_collector(agent)

        if (i + 1) % 1000 == 0:  # Checkpoint every 1000 episodes
            checkpoint_filename = f'model_data/dqn_model_checkpoint_{i+1}.pth'
            torch.save(agent.Q_eval.state_dict(), checkpoint_filename)
        # # Check if this episode's reward is the best so far and save the model if so
        # if score >= best_total_reward and end_points >= best_end_points:
        #     best_total_reward = score
        #     best_model_filename = f'model_data/dqn_best_model_{best_total_reward}_{best_end_points}_{i+1}.pth'
        #     torch.save(agent.Q_eval.state_dict(), best_model_filename)
        #     print(f"New best model saved with reward: {best_total_reward} at episode: {i+1}")

        
        save_to_csv(file_path, game_id, game_stats, turn_count, agent.epsilon, avg_loss)
        game_id += 1

        # avg_score = np.mean(scores[-100:])
        print('Episode: ', i, ', Points: ', end_points, ', Turns: ', turn_count ,' Score: %.2f' % score, ', Epsilon:  %.2f' % agent.epsilon)
        # if (i+1) % 100 == 0:
        #     print(f"Average loss after {i+1} games: {np.mean(agent.loss_history[-100:])}")

    try:
        epochs = range(len(scores))

        plt.figure(figsize=(12, 8))

        # Subplot 1: Scores (Rewards)
        plt.subplot(2, 2, 1)
        plt.plot(scores, label='Rewards per Episode')
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()

        # Subplot 2: Epsilon (Exploration rate)
        plt.subplot(2, 2, 2)
        plt.plot(eps_history, label='Epsilon Decay')
        plt.title('Epsilon Decay Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.legend()

        # Subplot 3: Loss History
        plt.subplot(2, 2, 3)
        plt.plot(avg_loss_per_episode, label='Average Loss per Episode')
        plt.title('Average Loss per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('model_data/training_outcomes.png')
        plt.show()


        plt.figure(figsize=(6, 4))
        plt.plot(range(len(eps_history)), eps_history, label='Epsilon')
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay Over Episodes')
        plt.legend()
        plt.show()


        plt.figure(figsize=(6, 4))
        plt.hist(agent.action_choices, bins=290, alpha=0.75, label='Action choices')
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.title('Distribution of Actions Chosen')
        plt.legend()
        plt.show()


        plt.figure(figsize=(6, 4))
        plt.hist(scores, bins=2, alpha=0.75)
        plt.xlabel('Total Reward')
        plt.ylabel('Frequency')
        plt.title('Wins vs Losses')
        plt.show()
    except Exception as err:
        print(err) 


    # try:
    #     plot_learning_curve(scores, n_games, agent.loss_history)

    # except Exception as e:
    #     print(e)

    duration = timedelta(seconds=time.perf_counter()-starttime)
    print('Job took: ', duration)
