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


def train_dqn_agent(gym_env, episodes=1200):
    env = gym.make(gym_env)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(env, Color.BLUE, state_size, action_size)
    batch_size = 32

    # Initialize variables to track the best performance and episode
    best_total_reward = -float('inf')
    # Prepare to collect training data
    training_logs = []

    for e in range(episodes):
        observation, info = env.reset()
        state = np.reshape(observation, [1, state_size])
        episode_rewards = 0  # Sum of rewards within the episode
        episode_steps = 0  # Number of steps taken in the episode
        for time in range(1000):
            action = agent.act(state)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(observation, [1, state_size])
            agent.remember(state, action, reward, next_state, done) 
            state = next_state
            episode_rewards += reward
            episode_steps += 1
            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        training_logs.append({
            'episode': e,
            'total_reward': episode_rewards,
            'steps': episode_steps,
            'epsilon': agent.epsilon,
            # Include more metrics as needed
        })
        if (e + 1) % 1000 == 0:  # Checkpoint every 1000 episodes
            checkpoint_filename = f'dqn_model_checkpoint_{e+1}.pth'
            torch.save(agent.model.state_dict(), checkpoint_filename)
        # Check if this episode's reward is the best so far and save the model if so
        if episode_rewards > best_total_reward:
            best_total_reward = episode_rewards
            best_model_filename = f'dqn_best_model_{e+1}.pth'
            torch.save(agent.model.state_dict(), best_model_filename)
            print(f"New best model saved with reward: {best_total_reward} at episode: {e+1}")


    torch.save(agent.model.state_dict(), 'dqn_model.pth')
    # Convert logs to a DataFrame and save to CSV for analysis

    training_df = pd.DataFrame(training_logs)
    training_df.to_csv('training_logs.csv', index=False)

    env.close()

print("Training DQN agent")
print("------------------")
starttime = time.perf_counter()
print("train1 - balanced maps, random ops, random order")
train_dqn_agent("catanatron_gym:catanatron-v1")
duration = timedelta(seconds=time.perf_counter()-starttime)
print('Job took: ', duration)
# print("train2 - balanced maps 2nd")
# train_dqn_agent("catanatron_gym:catanatronp2-v3")
# duration = timedelta(seconds=time.perf_counter()-starttime)
# print('Job took: ', duration)
# print("train3 - balanced maps 1st")
# train_dqn_agent("catanatron_gym:catanatronp1-v1")
# duration = timedelta(seconds=time.perf_counter()-starttime)
# print('Job took: ', duration)
# print("train4 - balanced maps 4th")
# train_dqn_agent("catanatron_gym:catanatronp4-v4")
# duration = timedelta(seconds=time.perf_counter()-starttime)
# print('Job took: ', duration)


def train_dqn_agent_multi_env(envs, episodes=6000, checkpoint_interval=1000):
    # Assuming envs is a list of environment names
    state_sizes = []
    action_sizes = []
    for env_name in envs:
        env = gym.make(env_name)
        state_sizes.append(env.observation_space.shape[0])
        action_sizes.append(env.action_space.n)
        env.close()
    
    # Take the max of state_sizes and action_sizes to ensure compatibility across environments
    state_size = max(state_sizes)
    action_size = max(action_sizes)
    
    agent = DQNAgent(env=None, my_color=Color.BLUE, state_size=state_size, action_size=action_size)  # Modified to pass None as the initial environment
    batch_size = 32

    # Initialize variables to track the best performance and episode
    best_total_reward = -float('inf')
    training_logs = []

    for e in range(episodes):
        for env_name in envs:
            env = gym.make(env_name)
            agent.env = env  # Update the agent's environment
            
            observation, info = env.reset()
            state = np.reshape(observation, [1, state_size])
            episode_rewards = 0  # Sum of rewards within the episode
            for time in range(1000):
                action = agent.act(state)
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state = np.reshape(observation, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                episode_rewards += reward
                if done:
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

            # Log training data
            training_logs.append({
                'episode': e,
                'env': env_name,
                'total_reward': episode_rewards,
                'epsilon': agent.epsilon,
            })
            env.close()

            # Checkpoint and best model saving logic here
            if (e + 1) % checkpoint_interval == 0 or e == 0:  # Also save on the first episode
                checkpoint_filename = f'dqn_model_checkpoint_{env_name}_{e+1}.pth'
                torch.save(agent.model.state_dict(), checkpoint_filename)
            if episode_rewards > best_total_reward:
                best_total_reward = episode_rewards
                best_model_filename = f'dqn_best_model_{env_name}_{e+1}.pth'
                torch.save(agent.model.state_dict(), best_model_filename)
                print(f"New best model saved with reward: {best_total_reward} at episode: {e+1}, env: {env_name}")

    # Final model save
    torch.save(agent.model.state_dict(), 'dqn_model_final.pth')
    # Convert logs to DataFrame and save
    training_df = pd.DataFrame(training_logs)
    training_df.to_csv('training_logs_multi_env.csv', index=False)

# Example usage
# envs = [
#     # "catanatron_gym:catanatronp3-v1",
#     # "catanatron_gym:catanatronp2-v1",
#     "catanatron_gym:catanatron-v1",
#     # "catanatron_gym:catanatronp4-v1",
# ]
# envs = "catanatron_gym:catanatron-v1"
# print("Training DQN agent across multiple environments")
# starttime = time.perf_counter()
# train_dqn_agent_multi_env(envs)
# duration = timedelta(seconds=time.perf_counter()-starttime)
# print('Total training took: ', duration)


