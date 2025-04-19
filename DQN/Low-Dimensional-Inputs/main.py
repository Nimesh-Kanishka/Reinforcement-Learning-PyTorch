from agent import DQNAgent

if __name__ == "__main__":
    agent = DQNAgent(env_id="LunarLander-v3",
                     env_num_observations=8,
                     env_num_actions=4,
                     q_network_hidden_layer_sizes=[64, 64])
    """
    agent.train(total_steps=1_000_000,
                train_start_step=20_000,
                exploration_steps=100_000,
                epsilon_min=0.01,
                alpha=5e-4,
                replay_buffer_size=100_000,
                plot_graph_interval=100_000)
    """
    agent.evaluate(num_episodes=5,
                   load_model=True)