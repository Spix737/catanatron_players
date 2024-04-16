import os
import pdb
from catanatron_server.models import GameState
import pandas as pd
from sklearn.model_selection import learning_curve
import torch
import torch.nn as nn
import torch.optim as optim
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
from catanatron.state import State
from catanatron.state_functions import player_key


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
    def __init__(self, state_size, n_actions, lr):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    

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
        self.model = DQN(state_size, n_actions, self.learning_rate) # .to(self.model.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)



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
        self.memory.append((state, info, action, reward, next_state, info_next, done))



    def act(self, state, info):
        valid_actions = self.env.unwrapped.get_valid_actions()
        if np.random.rand() <= self.epsilon:
            return np.random.choice(valid_actions)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.model.device)
        act_values = self.model(state)
        act_values = act_values.detach().numpy().squeeze()
        # Mask invalid actions by setting their Q-values to a large negative value
        act_values[~np.isin(np.arange(self.action_size), valid_actions)] = -np.inf
        return np.argmax(act_values)



    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, valid_actions, actions, rewards, next_states, next_valid_actions, dones = zip(*minibatch)

        states = np.array(states)
        states = torch.FloatTensor(states).squeeze().to(self.model.device)
        # states = torch.FloatTensor(np.concatenate(states)).squeeze().to(self.model.device)
        
        next_states = np.array(next_states)
        next_states = torch.FloatTensor(next_states).squeeze().to(self.model.device)
        actions = torch.LongTensor(actions).to(self.model.device)
        rewards = torch.FloatTensor(rewards).to(self.model.device)
        dones = torch.FloatTensor(dones).to(self.model.device)

        # Predict the Q-values of the current states
        q_values = self.model(states)
        # Select the Q-value for the action taken
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Predict the Q-values of the next states using the target network
        valid_actions_masks = np.zeros((batch_size, 290), dtype=bool)
        for i, action_dict in enumerate(next_valid_actions):
            valid_actions_masks[i, action_dict['valid_actions']] = True

        next_q_values = self.model(next_states).detach()
        next_q_values = np.where(valid_actions_masks, next_q_values, -np.inf)

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

        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min




def train_dqn_agent(gym_env, episodes=1200):
    env = gym.make(gym_env)
    observation, info = env.reset()
    # state_size = env.observation_space.shape[0]

    state_size = 1046 # 4 player, # 3 player - 841, 2 player - 636
    action_size = env.action_space.n
    # action_size = 290
    agent = DQNAgent(env, Color.BLUE, state_size, action_size)
    batch_size = 32
    # Initialize variables to track the best performance and episode
    best_total_reward = -float('inf')
    # Prepare to collect training data
    training_logs = []

    for e in range(episodes):
        state = np.reshape(observation, [1, state_size])
        episode_rewards = 0  # Sum of rewards within the episode
        episode_steps = 0  # Number of steps taken in the episode
        for time in range(1000):
            action = agent.act(state, info)
            observation, reward, terminated, truncated, info_next = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(observation, [1, state_size])
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

# print("Training DQN agent")
# print("------------------")
# # initialize_database()
# starttime = time.perf_counter()
# print("train1 - balanced maps, random ops, determined order")
# train_dqn_agent("catanatron_gym:catanatronp3-v1")
# duration = timedelta(seconds=time.perf_counter()-starttime)
# print('Job took: ', duration)

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


def train_dqn_agent_multi_env(envs, episodes=3, checkpoint_interval=1000):
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
        try:
            agent.model.load_state_dict(torch.load('dqn_model_final.pth'))
        except:
            print("No model found")
        for env_name in envs:
            env = gym.make(env_name)
            agent.env = env  # Update the agent's environment
            observation, info = env.reset()
            done = False
            state = np.reshape(observation, [1, state_size])
            episode_rewards = 0  # Sum of rewards within the episode
            while not done:
                action = agent.act(state, info)
                observation, reward, terminated, truncated, info_next = env.step(action)
                done = terminated or truncated
                next_state = np.reshape(observation, [1, state_size])
                agent.remember(state, info, action, reward, next_state, info_next, done) 
                state = next_state
                episode_rewards += reward
                if done:
                    print(f"GAME END: episode: {e}/{episodes}, env: {env_name}, score: {episode_rewards}, e: {agent.epsilon:.2}")
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

envs = [
    "catanatron_gym:catanatronp3-v1",
    "catanatron_gym:catanatronp2-v1",
    "catanatron_gym:catanatron-v1",
    # "catanatron_gym:catanatronp4-v1",
]
# envs = "catanatron_gym:catanatron-v1"
# print("Training DQN agent across multiple environments")
# starttime = time.perf_counter()
# train_dqn_agent_multi_env(envs)
# duration = timedelta(seconds=time.perf_counter()-starttime)
# print('Total training took: ', duration)

if __name__ == '__main__':
    starttime = time.perf_counter()

    env = gym.make('catanatron_gym:catanatron-v1')
    # agent = dqnAgent(gamma=0.99, epsilon=1.0, lr=0.001, input_dims=env.observation_space.shape)
    agent = DQNAgent(env=env, my_color=Color.BLUE, state_size=1046, gamma=0.99, epsilon=1.0, batch_size=64,
                      n_actions=290, eps_end=0.01, lr=0.003)
    best_total_reward = 0 # flawed as max = 1
    best_end_points = 0 # flawed as max = 10 
    scores, eps_history = [], []
    n_games = 50000

    for i in range(n_games):
        score = 0
        done = False
        observation, info = env.reset()
        while not done:
            action = agent.act(observation, info)
            observation_, reward, done, truncated, info_ = env.step(action)
            score += reward
            agent.remember(observation, info, action, reward, observation_, info_, done)
            agent.replay(64)
            observation = observation_
            info = info_

        # os.system('cls') # if os.name == 'nt' else 'clear')
        os.system('clear')
        scores.append(score)
        eps_history.append(agent.epsilon)

        key = player_key(env.unwrapped.game.state, Color.BLUE)
        end_points = env.unwrapped.game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
        turn_count = env.unwrapped.game.state.num_turns


        if (n_games + 1) % 1000 == 0:  # Checkpoint every 1000 episodes
            checkpoint_filename = f'dqn_model_checkpoint_{best_total_reward}.pth'
            torch.save(agent.model.state_dict(), checkpoint_filename)
        # Check if this episode's reward is the best so far and save the model if so
        if score >= best_total_reward and end_points >= best_end_points:
            best_total_reward = score
            best_model_filename = f'dqn_best_model_{best_total_reward}_{best_end_points}.pth'
            torch.save(agent.model.state_dict(), best_model_filename)
            print(f"New best model saved with reward: {best_total_reward} at episode: {best_total_reward}")


        avg_score = np.mean(scores[-100:])
        print('episode: ', i, ', points: ', end_points, ', turns: ', turn_count ,' score: %.2f' % score, ', average score: %.2f' % avg_score, ', epsilon:  %.2f' % agent.epsilon)

    x = [i+1 for i in range(n_games)]
    filename = 'learningcurve.png'
    try:
        learning_curve(x, scores, eps_history, filename)
    except Exception as e:
        print(e)

    duration = timedelta(seconds=time.perf_counter()-starttime)
    print('Job took: ', duration)
