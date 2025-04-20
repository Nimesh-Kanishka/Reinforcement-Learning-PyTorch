from agent import Agent

if __name__ == "__main__":
    agent = Agent(env_id="Taxi-v3",
                  num_observations=500,
                  num_actions=6)
    """
    agent.train(total_episodes=2000,
                epsilon_min=0.05,
                exploration_episodes=1000)
    """
    agent.evaluate(total_episodes=10,
                   load_q_table=True)