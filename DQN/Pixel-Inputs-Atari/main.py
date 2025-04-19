from agent import DQNAgent

if __name__ == "__main__":
    agent = DQNAgent(env_id="ALE/Breakout-v5",
                     env_num_actions=4,
                     env_kwargs={
                         "frameskip": 1,
                         "repeat_action_probability": 0.0
                     },
                     save_folder="DQN_New/DQN_Atari_TargetNetwork/Breakout-v5")
    """
    agent.train(total_steps=5_000_000,
                train_start_step=50_000,
                exploration_steps=500_000,
                epsilon_min=0.01,
                replay_buffer_size=100_000,
                plot_graph_frequency=100_000)
    """
    agent.evaluate(num_episodes=5,
                   load_model=True)