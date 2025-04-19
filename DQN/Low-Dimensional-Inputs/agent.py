from typing import Sequence
import torch
import torch.nn.functional as F
from torch.optim import Adam
from network import QNetwork
import numpy as np
import gymnasium as gym
from gymnasium_wrappers import AutoResetEnv
from buffer import ReplayBuffer
from tqdm import tqdm
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self,
                 env_id: str,
                 env_num_observations: int,
                 env_num_actions: int,
                 env_kwargs: dict = {},
                 q_network_hidden_layer_sizes: int | Sequence[int] = [64, 64],
                 use_cuda: bool = True,
                 save_folder: str | None = None
    ):
        self.env_id = env_id
        self.env_num_observations = env_num_observations
        self.env_num_actions = env_num_actions
        self.env_kwargs = env_kwargs
        self.q_network_hidden_layer_sizes = q_network_hidden_layer_sizes

        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.q_network = QNetwork(env_num_observations, env_num_actions, q_network_hidden_layer_sizes).to(self.device)

        self.save_folder = f"DQN_New/DQN_Discrete_TargetNetwork/{env_id}" if save_folder is None else save_folder

    def train(self,
              total_steps: int,
              train_start_step: int,
              exploration_steps: int,
              epsilon_max: float = 1.0,
              epsilon_min: float = 0.1,
              alpha: float = 1e-4,
              gamma: float = 0.99,
              replay_buffer_size: int = 1_000_000,
              training_batch_size: int = 64,
              target_network_update_interval: int = 10_000,
              load_model: bool = False,
              save_start_reward: float = 0.0,
              plot_graph_interval: int = 1_000_000
    ):
        env = gym.make(self.env_id, **self.env_kwargs)
        env = AutoResetEnv(env)

        if load_model:
            self.load_model()

        optimizer = Adam(self.q_network.parameters(), lr=alpha)

        target_q_network = QNetwork(self.env_num_observations, self.env_num_actions, self.q_network_hidden_layer_sizes).to(self.device)
        target_q_network.load_state_dict(self.q_network.state_dict())

        replay_buffer = ReplayBuffer(replay_buffer_size, env.observation_space, env.action_space, optimize_memory_usage=True)

        episode_rewards = {}

        observation, _ = env.reset()
        current_episode_reward = 0

        for step in tqdm(range(total_steps)):
            epsilon = max(epsilon_min, epsilon_max - (epsilon_max - epsilon_min) / exploration_steps * step)

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = self.q_network(torch.tensor(observation).to(self.device).unsqueeze(dim=0))
                    action = torch.argmax(q_values, dim=1).squeeze().cpu().item()

            next_observation, reward, terminated, truncated, _ = env.step(action)

            current_episode_reward += reward
            done = terminated or truncated

            if done:
                episode_rewards[step + 1] = current_episode_reward

                if (step > train_start_step) and (current_episode_reward >= save_start_reward):
                    self.save_model()
                    save_start_reward = current_episode_reward

                    print(f"New Best Reward - Steps completed: {step + 1} - Reward: {current_episode_reward}")

                current_episode_reward = 0

            replay_buffer.add(observation, action, reward, next_observation, done)

            observation = next_observation

            if step > train_start_step:
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = replay_buffer.sample(training_batch_size)

                obs_batch = torch.tensor(obs_batch).to(self.device)
                action_batch = torch.tensor(action_batch).to(self.device)
                reward_batch = torch.tensor(reward_batch).to(self.device)
                next_obs_batch = torch.tensor(next_obs_batch).to(self.device)
                done_batch = torch.tensor(done_batch, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    max_next_q_values = target_q_network(next_obs_batch).max(dim=1).values

                    y = reward_batch + gamma * max_next_q_values * (1 - done_batch)

                current_q_values = self.q_network(obs_batch).gather(dim=1, index=action_batch.unsqueeze(dim=1)).squeeze()

                loss = F.huber_loss(current_q_values, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step + 1) % target_network_update_interval == 0:
                    target_q_network.load_state_dict(self.q_network.state_dict())

            if (step + 1) % plot_graph_interval == 0:
                self.plot_reward_graph(episode_rewards)

    def evaluate(self,
                 num_episodes: int,
                 render: bool = True,
                 load_model: bool = False
    ):
        env = gym.make(self.env_id, render_mode="human" if render else None, **self.env_kwargs)

        if load_model:
            self.load_model()

        for episode in range(num_episodes):
            observation, _ = env.reset()
            current_episode_steps = 0
            current_episode_reward = 0
            done = False

            while not done:
                with torch.no_grad():
                    q_values = self.q_network(torch.tensor(observation).to(self.device).unsqueeze(dim=0))
                    action = torch.argmax(q_values, dim=1).squeeze().cpu().item()

                observation, reward, terminated, truncated, _ = env.step(action)

                current_episode_steps += 1
                current_episode_reward += reward
                done = terminated or truncated

            print(f"Episode: {episode + 1} - Steps: {current_episode_steps} - Reward: {current_episode_reward}")

    def save_model(self,
                   save_file: str | None = None
    ):
        f = f"{self.save_folder}/Q_Network.pth" if save_file is None else save_file

        torch.save(self.q_network.state_dict(), f)
    
    def load_model(self,
                   save_file: str | None = None
    ):
        f = f"{self.save_folder}/Q_Network.pth" if save_file is None else save_file

        self.q_network.load_state_dict(torch.load(f))
    
    def plot_reward_graph(self,
                          episode_rewards: dict
    ):
        steps = list(episode_rewards.keys())
        rewards = list(episode_rewards.values())

        smoothed_rewards = [np.mean(rewards[max(0, i - 99):i + 1]) for i in range(len(rewards))]

        plt.plot(steps, rewards, "-b", label="Episode Rewards")
        plt.plot(steps, smoothed_rewards, "-r", label="Average Rewards")

        plt.title("Episode Rewards")
        plt.xlabel("Training Steps")
        plt.ylabel("Reward")
        plt.legend()

        plt.savefig(f"{self.save_folder}/Episode_Rewards_Graph.png")

        plt.close()