import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from datetime import timedelta
import matplotlib.pyplot as plt
from catanatron.models.enums import CITY, ROAD, SETTLEMENT, VICTORY_POINT
from catanatron.models.player import Color
from catanatron.state_functions import calculate_resource_probabilities, get_dev_cards_in_hand, get_largest_army, get_longest_road_color, get_player_buildings, player_key

def save_to_csv(file_path, game_id, game_data, turn_count, players, epsilon, average_loss):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write headers if the file does not exist
        if not file_exists:
            headers = ['GameID', 'TurnCount', 'Players', 'Epsilon', 'AverageLoss', ]  + list(game_data.keys())
            writer.writerow(headers)
        
        # Write game data
        data = [game_id, turn_count, players, epsilon, average_loss, ] + list(game_data.values())
        writer.writerow(data)


def preprocess_observation(observation):
    # Initialize a list to gather flattened data
    data = []

    # Sort keys for consistent ordering if it's a dictionary
    if isinstance(observation, dict):
        for key in sorted(observation.keys()):
            value = observation[key]

            # Check if the value is an integer or float, wrap it into a list
            if isinstance(value, (int, float)):
                data.append([value])
            elif isinstance(value, np.ndarray):
                # Flatten the array if it's multidimensional
                data.append(value.ravel())
            elif isinstance(value, list):
                # Convert list to array, then flatten
                data.append(np.array(value).ravel())
            else:
                raise TypeError(f"Unsupported data type in observation: {type(value)}")

        # Concatenate all arrays into a single flat array
        flat_observation = np.concatenate(data)
    elif isinstance(observation, np.ndarray):
        flat_observation = observation.ravel()
    else:
        raise TypeError("Observation must be either a dictionary or a numpy array.")

    return flat_observation


class DuelingDQN(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)  
        self.fc4 = nn.Linear(fc3_dims, fc3_dims)

        # Introducing dropout for regularization
        self.dropout = nn.Dropout(p=0.2)

        # Separate streams for value and advantage
        self.value_layer = nn.Linear(fc3_dims, 1)
        self.advantage_layer = nn.Linear(fc3_dims, n_actions)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x)) 

        value = self.value_layer(x)
        advantage = self.advantage_layer(x)
        
        # Combining value and advantage to get Q-value
        q_out = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_out


class PrioritizedReplayBuffer:
    def __init__(self, max_size, input_shape, alpha=0.6):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.alpha = alpha
        
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.priority_memory = np.zeros(self.mem_size, dtype=np.float32)  # Priority of each transition
        
    def store(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        
        self.priority_memory[index] = self.priority_memory.max() if self.mem_cntr > 0 else 1.0
        self.mem_cntr += 1
    
    def sample(self, batch_size, beta=0.4):
        if self.mem_cntr < self.mem_size:
            max_mem = self.mem_cntr
        else:
            max_mem = self.mem_size
        
        priorities = self.priority_memory[:max_mem] ** self.alpha
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(max_mem, batch_size, replace=False, p=probs)
        states = self.state_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        states_ = self.new_state_memory[indices]
        dones = self.terminal_memory[indices]
        
        # Importance-sampling weights
        total = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return states, actions, rewards, states_, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        # Ensure this method can handle both single value and array of priorities
        if not isinstance(priorities, np.ndarray):
            priorities = np.array([priorities])  # Convert single priority value to an array
        for idx, prio in zip(indices, priorities):
            self.priority_memory[idx] = prio


class DQNAgent:
    def __init__(self, env, learning_rate, gamma, epsilon, state_dims, n_actions, batch_size, mem_size=2500000):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon  
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.epsilon_min = 0.01
        self.epsilon_decay = 4E-6
        self.action_choices = []

        
        self.memory = PrioritizedReplayBuffer(mem_size, state_dims)
        self.policy_net = DuelingDQN(input_dims=state_dims, fc1_dims=512, fc2_dims=256, fc3_dims=128, n_actions=n_actions)
        self.target_net = DuelingDQN(input_dims=state_dims, fc1_dims=512, fc2_dims=256, fc3_dims=128, n_actions=n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def reset_epsilon(self):
        self.epsilon = 1

    def choose_action(self, observation):
        valid_actions = self.env.unwrapped.get_valid_actions()
        if np.random.random() <= self.epsilon:
            # Exploration: Randomly choose from valid actions
            action = np.random.choice(valid_actions)
        else:
            # Exploitation: Choose the best action based on the model's prediction
            processed_observation = preprocess_observation(observation)
            state = torch.tensor([processed_observation], dtype=torch.float32).to(self.policy_net.device)
            with torch.no_grad():
                actions = self.policy_net(state)

            # Convert action values to numpy for easier manipulation
            action_values = actions.cpu().numpy().squeeze()

            # Mask invalid actions by setting their Q-values to negative infinity
            mask = np.isin(np.arange(action_values.size), valid_actions, invert=True)
            action_values[mask] = -np.inf

            # Select the action with the highest Q-value among valid actions
            action = np.argmax(action_values)

            if action != 0 and action != 289: 
                self.action_choices.append(action)
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)
    
    def sample_memory(self):
        state, action, reward, next_state, done, indices, weights = self.memory.sample(self.batch_size)
        return state, action, reward, next_state, done, indices, weights
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.optimizer.zero_grad()
        
        states, actions, rewards, next_states, dones, indices, weights = self.sample_memory()
        
        states = torch.tensor(states, dtype=torch.float32).to(self.policy_net.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.policy_net.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.policy_net.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.policy_net.device)
        dones = torch.tensor(dones, dtype=bool).to(self.policy_net.device)
        
        q_pred = self.policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q_next = self.target_net(next_states).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next
        
        loss = (q_target.detach() - q_pred) ** 2
        loss = (loss * torch.tensor(weights, dtype=torch.float32).to(self.policy_net.device)).mean()
        loss.backward()
        self.optimizer.step()
        
        # Convert loss to numpy and ensure it is in an array form
        loss_value = loss.item() + 1e-5  # Convert loss to Python scalar and add a small offset
        self.memory.update_priorities(indices, np.array([loss_value]))  # Ensure the priority is passed as an array
        
        if self.memory.mem_cntr % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
        

# def worker(env, agent, start_event, reset_queue, result_queue):
#     start_event.wait()
#     while True:
#         initial_state = reset_queue.get()
#         if initial_state is None:
#             break
#         observation = initial_state
#         done = False
#         while not done:
#             action = agent.choose_action(observation)
#             observation_, reward, done, truncated, info = env.step(action)
#             agent.store_transition(observation, action, reward, observation_, done)
#             loss = agent.learn()
#             observation = observation_
#         result_queue.put((reward, agent.epsilon, loss))

# def env_initializer(num_episodes, reset_queue, env_name):
#     env = gym.make(env_name)
#     for _ in range(num_episodes):
#         if _ % 4000 == 0:
#             agent.reset_epsilon()
#         state = env.reset()
#         reset_queue.put(state)
#     for _ in range(num_workers):
#         reset_queue.put(None)
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
    # try:
        starttime = time.perf_counter()
        file_path = 'model_data_simplerinitbuild/training_outcomes.csv'
        os.makedirs('model_data_simplerinitbuild', exist_ok=True)
        game_id = 0

        """
        AGENT COLOR = COLOR.BLUE
        """
        state_dimensons = 1336 # env.observation_space.shape

        env = gym.make('catanatron_gym:catanatronReward-v1')
        agent = DQNAgent(env=env, state_dims=state_dimensons, gamma=0.99, epsilon=1.0, batch_size=512,
                        n_actions=290, learning_rate=0.00049)
        #  eps_end=0.01, eps_dec=1.65E-6,
        best_total_reward = 0 # flawed as max = 1
        best_end_points = 0 # max=10 
        scores, eps_history, avg_loss_per_episode = [], [], []
        n_games = 6000

        for i in range(n_games):
            if i % 2000 == 0:
                agent.reset_epsilon()
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
            players = env.unwrapped.game.state.players
            game_stats = game_end_collector(agent)

            if (i) % 600 == 0:  # Checkpoint every 1000 episodes
                checkpoint_filename = f'model_data_simplerinitbuild/dqn_model_checkpoint_{i}.pth'
                torch.save(agent.policy_net.state_dict(), checkpoint_filename)
                torch.save(agent.optimizer.state_dict(), f'model_data_simplerinitbuild/dqn_optimizer_checkpoint_{i}.pth')
            # # Check if this episode's reward is the best so far and save the model if so
            # if score >= best_total_reward and end_points >= best_end_points:
            #     best_total_reward = score
            #     best_model_filename = f'model_data/dqn_best_model_{best_total_reward}_{best_end_points}_{i+1}.pth'
            #     torch.save(agent.policy_net.state_dict(), best_model_filename)
            #     print(f"New best model saved with reward: {best_total_reward} at episode: {i+1}")

            
            save_to_csv(file_path, game_id, game_stats, turn_count, players, agent.epsilon, avg_loss)
            game_id += 1


            print('Episode: ', i, ', Points: ', end_points, ', Turns: ', turn_count ,' Score: %.2f' % score, ', Epsilon:  %.2f' % agent.epsilon)

        torch.save(agent.policy_net.state_dict(), 'model_data_simplerinitbuild/dqn_model_final.pth')
        torch.save(agent.optimizer.state_dict(), 'model_data_simplerinitbuild/dqn_optimizer_final.pth')


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

        duration = timedelta(seconds=time.perf_counter()-starttime)
        print('Job took: ', duration)
