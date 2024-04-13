from catanatron_server.models import GameState
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
import sqlite3
from sqlalchemy import Column, Integer, LargeBinary, MetaData, String, create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager


from catanatron import json
from catanatron.models.player import Color
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Setup for database connection
# metadata = MetaData()
# Base = declarative_base(metadata=metadata)
# ////
# Base = declarative_base()

# class GameState(Base):
#     __tablename__ = "game_states"

#     id = Column(Integer, primary_key=True)
#     uuid = Column(String(64), nullable=False)
#     state_index = Column(Integer, nullable=False)
#     state = Column(String, nullable=False)
#     pickle_data = Column(LargeBinary, nullable=False)

# SessionLocal = sessionmaker()

# @contextmanager
# def database_session():
#     database_url = "sqlite:///catanatron_local.db"
#     engine = create_engine(database_url, echo=True, connect_args={"check_same_thread": False})
#     SessionLocal.configure(bind=engine)
#     session = SessionLocal()
#     try:
#         yield session
#     finally:
#         session.close()

# def initialize_database():
#     engine = create_engine("sqlite:///catanatron_local.db")
#     Base.metadata.create_all(engine)


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

    def remember(self, state, info, action, reward, next_state, info_next, done):
        # state_json = json.dumps(state)
        # next_state_json = json.dumps(next_state)
        # with database_session() as session:
        #     new_state_record = GameState(
        #         game_id="game.id,  # You may want to dynamically set this",
        #         state_data=f"{state_json}|{action}|{reward}|{next_state_json}|{done}"
        #     )
        #     session.add(new_state_record)
        #     session.commit()
        # print("REMEMBER")
        # print("-----------------------------------------------------------\n")
        # print(state)
        # print("\n\n")
        # print(info)
        # print("\n\n")
        # print(action)
        # print("\n\n")
        # print(reward)
        # print("\n\n")
        # print(next_state)
        # print("\n\n")
        # print(info_next)
        # print("\n\n")
        # print(done)
        # print("\n\n")
        self.memory.append((state, info, action, reward, next_state, info_next, done))

    def act(self, state, info):
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
        # print("MEMRY")
        # print("-----------------------------------------------------------\n")
        # print(self.memory[0])
        # print("\n\nMinibatch")
        # print("-----------------------------------------------------------\n")
        # print(minibatch)
        # print("\n\nZIP Minibatch")
        # print("-----------------------------------------------------------\n")
        # print(zip(*minibatch))
        states, valid_actions, actions, rewards, next_states, next_valid_actions, dones = zip(*minibatch)

        print(valid_actions)

        states = np.array(states)
        states = torch.FloatTensor(states).squeeze().to(device)
        # states = torch.FloatTensor(np.concatenate(states)).squeeze().to(device)
        
        next_states = np.array(next_states)
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
        # Mask invalid actions for next states as well using the generate_playable_actions method
        valid_actions_masks = np.array([ns.generate_playable_actions() for ns in next_states])
        for index, valid_actions in enumerate(valid_actions_masks):
            next_q_values[index][~valid_actions] = -np.inf  # Assuming valid_actions is a boolean mask
        # Mask invalid actions for next states as well
        # valid_actions_masks = np.array([self.env.unwrapped.get_valid_actions(state_index) for state_index in range(batch_size)])
        # for index, valid_actions in enumerate(valid_actions_masks):
        #     next_q_values[index][~np.isin(np.arange(self.action_size), valid_actions)] = -np.inf

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
    observation, info = env.reset()

    # state_size = env.observation_space.shape[0]
    print("obs_space: ",env.observation_space.shape[0])
    state_size = 1046 # 4 player, # 3 player - 841, 2 player - 636
    action_size = env.action_space.n
    # action_size = 290
    agent = DQNAgent(env, Color.BLUE, state_size, action_size)
    print("sus....")
    batch_size = 32

    # Initialize variables to track the best performance and episode
    best_total_reward = -float('inf')
    # Prepare to collect training data
    training_logs = []

    for e in range(episodes):
        # observation, info = env.reset()
        # print("OBSERVATION")
        # print("-----------------------------------------------------------\n")
        # print(observation)
        state = np.reshape(observation, [1, state_size])
        episode_rewards = 0  # Sum of rewards within the episode
        episode_steps = 0  # Number of steps taken in the episode
        for time in range(1000):
            print("perhaps hither")
            action = agent.act(state, info)
            print("PRAY THE WHY")
            observation, reward, terminated, truncated, info_next = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(observation, [1, state_size])
            # print("BEFORE REMEMBER")
            # print("-----------------------------------------------------------\n")
            # print(state)
            # print("\n\n")
            # print(info)
            # print("\n\n")
            # print(action)
            # print("\n\n")
            # print(reward)
            # print("\n\n")
            # print(next_state)
            # print("\n\n")
            # print(info_next)
            # print("\n\n")
            # print(done)
            # print("\n\n")
            agent.remember(state, info, action, reward, next_state, info_next, done) 
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
# initialize_database()
starttime = time.perf_counter()
print("train1 - balanced maps, random ops, determined order")
train_dqn_agent("catanatron_gym:catanatronp3-v1")
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


# def train_dqn_agent_multi_env(envs, episodes=6000, checkpoint_interval=1000):
#     # Assuming envs is a list of environment names
#     state_sizes = []
#     action_sizes = []
#     for env_name in envs:
#         env = gym.make(env_name)
#         state_sizes.append(env.observation_space.shape[0])
#         action_sizes.append(env.action_space.n)
#         env.close()
    
#     # Take the max of state_sizes and action_sizes to ensure compatibility across environments
#     state_size = max(state_sizes)
#     action_size = max(action_sizes)
    
#     agent = DQNAgent(env=None, my_color=Color.BLUE, state_size=state_size, action_size=action_size)  # Modified to pass None as the initial environment
#     batch_size = 32

#     # Initialize variables to track the best performance and episode
#     best_total_reward = -float('inf')
#     training_logs = []

#     for e in range(episodes):
#         for env_name in envs:
#             env = gym.make(env_name)
#             agent.env = env  # Update the agent's environment
            
#             observation, info = env.reset()
#             state = np.reshape(observation, [1, state_size])
#             episode_rewards = 0  # Sum of rewards within the episode
#             for time in range(1000):
#                 action = agent.act(state)
#                 observation, reward, terminated, truncated, info = env.step(action)
#                 done = terminated or truncated
#                 next_state = np.reshape(observation, [1, state_size])
#                 agent.remember(state, action, reward, next_state, done)
#                 state = next_state
#                 episode_rewards += reward
#                 if done:
#                     break
#                 if len(agent.memory) > batch_size:
#                     agent.replay(batch_size)

#             # Log training data
#             training_logs.append({
#                 'episode': e,
#                 'env': env_name,
#                 'total_reward': episode_rewards,
#                 'epsilon': agent.epsilon,
#             })
#             env.close()

#             # Checkpoint and best model saving logic here
#             if (e + 1) % checkpoint_interval == 0 or e == 0:  # Also save on the first episode
#                 checkpoint_filename = f'dqn_model_checkpoint_{env_name}_{e+1}.pth'
#                 torch.save(agent.model.state_dict(), checkpoint_filename)
#             if episode_rewards > best_total_reward:
#                 best_total_reward = episode_rewards
#                 best_model_filename = f'dqn_best_model_{env_name}_{e+1}.pth'
#                 torch.save(agent.model.state_dict(), best_model_filename)
#                 print(f"New best model saved with reward: {best_total_reward} at episode: {e+1}, env: {env_name}")

#     # Final model save
#     torch.save(agent.model.state_dict(), 'dqn_model_final.pth')
#     # Convert logs to DataFrame and save
#     training_df = pd.DataFrame(training_logs)
#     training_df.to_csv('training_logs_multi_env.csv', index=False)

# envs = [
#     "catanatron_gym:catanatron-v3",
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


