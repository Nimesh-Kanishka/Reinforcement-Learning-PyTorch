import logging
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from networks import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer
from utils import OUActionNoise, DynamicRenderEnv

class Agent:
    def __init__(self,
                 env_id: str,
                 env_kwargs: dict = {},
                 alpha: float = 1e-4,
                 beta: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.001,
                 replay_buffer_size: int = 1000000,
                 batch_size: int = 64,
                 network_fc1_size: int = 400,
                 network_fc2_size: int = 300,
                 use_gpu: bool = False,
                 load_checkpoint: bool = False,
                 save_folder: str | None = None
    ):
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size

        self.save_folder = f"DDPG/DDPG_Low_Dimensional_Inputs/{env_id}" if save_folder is None else save_folder

        self.env = DynamicRenderEnv(env_id, **env_kwargs)

        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        self.actor = ActorNetwork(self.env.observation_space.shape[0], self.env.action_space.shape[0],
                                  network_fc1_size, network_fc2_size).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=alpha)

        self.critic = CriticNetwork(self.env.observation_space.shape[0], self.env.action_space.shape[0],
                                    network_fc1_size, network_fc2_size).to(self.device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=beta)

        self.target_actor = ActorNetwork(self.env.observation_space.shape[0], self.env.action_space.shape[0],
                                         network_fc1_size, network_fc2_size).to(self.device)
        
        self.target_critic = CriticNetwork(self.env.observation_space.shape[0], self.env.action_space.shape[0],
                                           network_fc1_size, network_fc2_size).to(self.device)
        
        if load_checkpoint:
            self.load_checkpoint()
        else:
            # Copy actor's weights to target actor and critic's weights to target critic
            self.update_target_network_parameters(tau=1)

        logging.basicConfig(filename=f"{self.save_folder}/Training_Data.log",
                            format="%(asctime)s - %(levelname)s: %(message)s",
                            level=logging.INFO,
                            datefmt="%I:%M:%S")
        
    def train(self,
              total_episodes: int,              
              train_start_step: int = 10000,
              plot_graph_interval: int = 100,
              save_start_reward: float = 0.0
    ):
        logging.info(f"----- Starting Training - Episodes: {total_episodes} -----")

        self.env.change_render_mode(render_mode=None)

        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()

        replay_buffer = ReplayBuffer(self.replay_buffer_size, self.env.observation_space,
                                     self.env.action_space, optimize_memory_usage=True)
        
        action_noise = OUActionNoise(mu=np.zeros(self.env.action_space.shape[0]))

        global_step = 0
        episode_reward_history = {}

        pbar = tqdm(range(total_episodes))

        for episode in pbar:
            episode_steps = 0
            episode_reward = 0
            episode_done = False
            observation, _ = self.env.reset()

            while not episode_done:
                with torch.no_grad():
                    mu = self.actor(torch.tensor(observation).to(self.device).unsqueeze(dim=0)).squeeze()
                    noise = torch.tensor(action_noise()).to(self.device)

                    action = (mu + noise).cpu().numpy()
                    
                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                global_step += 1
                episode_steps += 1
                episode_reward += reward
                episode_done = terminated or truncated

                replay_buffer.add(observation, action, reward, next_observation, episode_done)

                observation = next_observation

                if global_step > train_start_step:
                    observation_batch, action_batch, reward_batch, next_observation_batch, done_batch = \
                        replay_buffer.sample(self.batch_size)

                    observation_batch = torch.tensor(observation_batch).to(self.device)
                    action_batch = torch.tensor(action_batch).to(self.device)
                    reward_batch = torch.tensor(reward_batch).to(self.device)
                    next_observation_batch = torch.tensor(next_observation_batch).to(self.device)
                    done_batch = torch.tensor(done_batch, dtype=torch.float32).to(self.device)

                    with torch.no_grad():
                        target_actions = self.target_actor(next_observation_batch)
                        next_critic_values = self.target_critic(next_observation_batch, target_actions)
                                                
                        targets = reward_batch.unsqueeze(dim=1) + \
                            self.gamma * next_critic_values * (1.0 - done_batch.unsqueeze(dim=1))

                    critic_values = self.critic(observation_batch, action_batch)

                    self.critic.train()

                    critic_loss = F.mse_loss(critic_values, targets)

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    self.critic.eval()

                    new_actions = self.actor(observation_batch)

                    self.actor.train()

                    actor_loss = -self.critic(observation_batch, new_actions)
                    actor_loss = torch.mean(actor_loss)

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    self.actor.eval()

                    self.update_target_network_parameters()

            pbar.set_postfix(Reward=episode_reward)
            episode_reward_history[global_step] = episode_reward

            avg_reward = np.mean(list(episode_reward_history.values())[-100:])

            logging.info(f"Episode: {episode} - Episode steps: {episode_steps} - Total steps: {global_step} - Reward: {episode_reward} - Avg Reward: {avg_reward}")
            
            if avg_reward >= save_start_reward:
                save_start_reward = avg_reward
                self.save_checkpoint()

                logging.info("----- Saved Model Checkpoint -----")

            if (episode + 1) % plot_graph_interval == 0:
                self.plot_graphs(episode_reward_history)

        logging.info("----- Finished Training -----")

    def evaluate(self,
                 total_episodes: int
    ):
        print(f"----- Starting Evaluation - Episodes: {total_episodes} -----")

        self.env.change_render_mode(render_mode="human")

        self.actor.eval()

        episode_reward_history = []

        for episode in range(total_episodes):
            episode_steps = 0
            episode_reward = 0
            episode_done = False
            observation, _ = self.env.reset()

            while not episode_done:
                with torch.no_grad():
                    action = self.actor(torch.tensor(observation).to(self.device).unsqueeze(dim=0)).squeeze().cpu().numpy()
                    
                observation, reward, terminated, truncated, _ = self.env.step(action)

                episode_steps += 1
                episode_reward += reward
                episode_done = terminated or truncated

            episode_reward_history.append(episode_reward)

            print(f"Episode: {episode + 1}/{total_episodes} - Episode steps: {episode_steps} - Reward: {episode_reward}")

        print(f"Min reward: {min(episode_reward_history)} - Max reward: {max(episode_reward_history)} - Avg reward: {np.mean(episode_reward_history)}")
        print("----- Finished Evaluation -----")

    def update_target_network_parameters(self,
                                         tau: float | None = None
    ):
        if tau is None:
            tau = self.tau

        for target_critic_param, critic_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_critic_param.data.copy_(tau * critic_param.data + (1.0 - tau) * target_critic_param.data)

        for target_actor_param, actor_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_actor_param.data.copy_(tau * actor_param.data + (1.0 - tau) * target_actor_param.data)

    def save_checkpoint(self,
                        save_file: str | None = None
    ):
        file = f"{self.save_folder}/Model_Checkpoint.pth" if save_file is None else save_file

        torch.save({"actor_state_dict": self.actor.state_dict(),
                    "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                    "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                    "target_actor_state_dict": self.target_actor.state_dict(),
                    "target_critic_state_dict": self.target_critic.state_dict()},
                   file)
    
    def load_checkpoint(self,
                        save_file: str | None = None
    ):
        file = f"{self.save_folder}/Model_Checkpoint.pth" if save_file is None else save_file

        checkpoint = torch.load(file, weights_only=True)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
    
    def plot_graphs(self,
                    episode_reward_history: dict
    ):
        if not episode_reward_history:
            return

        episode_rewards = list(episode_reward_history.values())
        # Average reward from last 100 episodes
        smoothed_rewards = [np.mean(episode_rewards[max(0, i - 99):i + 1]) for i in range(len(episode_rewards))]

        # Episode vs Reward and Episode vs Average reward graphs
        episode_ids = np.arange(1, len(episode_rewards) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(episode_ids, episode_rewards, "-b", label="Episode Rewards")
        plt.plot(episode_ids, smoothed_rewards, "-r", label="Average Rewards")
        plt.title("Episode vs Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_folder}/Episode_vs_Reward_Graph.png")
        plt.close()

        # Total steps vs Reward and Total steps vs Average reward graphs
        episode_end_steps = list(episode_reward_history.keys())

        plt.figure(figsize=(10, 6))
        plt.plot(episode_end_steps, episode_rewards, "-b", label="Episode Rewards")
        plt.plot(episode_end_steps, smoothed_rewards, "-r", label="Average Rewards")
        plt.title("Total Steps vs Reward")
        plt.xlabel("Total Steps")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_folder}/Total_Steps_vs_Reward_Graph.png")
        plt.close()