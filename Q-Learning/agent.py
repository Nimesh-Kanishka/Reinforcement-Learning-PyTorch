import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt

class Agent:
    def __init__(self,
                 env_id: str,
                 num_observations:int,
                 num_actions: int,
                 env_kwargs: dict = {},
                 save_folder: str | None = None
    ):
        self.env_id = env_id
        self.env_kwargs = env_kwargs

        self.save_folder = f"Q-Learning/{env_id}" if save_folder is None else save_folder

        self.q_table = np.zeros((num_observations, num_actions))
        
    def train(self,
              total_episodes: int,
              alpha: float = 0.1,
              gamma: float = 0.9,
              epsilon_max: float = 1.0,
              epsilon_min: float = 0.1,
              exploration_episodes: int | None = None,
              load_q_table: bool = False,
              save_start_reward: float = 0.0
    ):
        exploration_episodes = total_episodes if exploration_episodes is None else exploration_episodes

        env = gym.make(self.env_id, **self.env_kwargs)
        
        if load_q_table:
            self.q_table = np.load(f"{self.save_folder}/Q_Table.npy")

        episode_rewards = []

        for episode in tqdm(range(total_episodes)):
            epsilon = max(epsilon_min, epsilon_max - (epsilon_max - epsilon_min) / exploration_episodes * episode)

            ep_reward = 0
            done = False
            observation, _ = env.reset()

            while not done:
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[observation])

                next_observation, reward, terminated, truncated, _ = env.step(action)

                ep_reward += reward
                done = terminated or truncated

                self.q_table[observation, action] += alpha * (reward + gamma * np.max(self.q_table[next_observation]) - self.q_table[observation, action])

                observation = next_observation

            episode_rewards.append(ep_reward)

            if ep_reward >= save_start_reward:
                save_start_reward = ep_reward

                np.save(f"{self.save_folder}/Q_Table.npy", self.q_table)

                print(f"New Best Reward - Episode: {episode + 1} - Reward: {ep_reward}")

        self.plot_reward_graph(episode_rewards)

    def evaluate(self,
                 total_episodes: int,
                 render: bool = True,
                 load_q_table: bool = False
    ):
        env = gym.make(self.env_id, render_mode="human" if render else None, **self.env_kwargs)

        if load_q_table:
            self.q_table = np.load(f"{self.save_folder}/Q_Table.npy")

        for episode in range(total_episodes):
            ep_steps = 0
            ep_reward = 0
            done = False
            observation, _ = env.reset()

            while not done:
                action = np.argmax(self.q_table[observation])

                observation, reward, terminated, truncated, _ = env.step(action)

                ep_steps += 1
                ep_reward += reward
                done = terminated or truncated

            print(f"Episode: {episode + 1} - Steps: {ep_steps} - Reward: {ep_reward}")

    def plot_reward_graph(self,
                          episode_rewards: list
    ):
        x = [i for i in range(1, len(episode_rewards) + 1)]

        smoothed_episode_rewards = [np.mean(episode_rewards[max(0, i - 99):i + 1]) for i in range(1, len(episode_rewards) + 1)]

        plt.plot(x, episode_rewards, "-b", label="Episode Rewards")
        plt.plot(x, smoothed_episode_rewards, "-r", label="Average Rewards")

        plt.title("Episode Rewards")
        plt.xlabel("Training Steps")
        plt.ylabel("Reward")
        plt.legend()

        plt.savefig(f"{self.save_folder}/Episode_Rewards.png")

        plt.close()