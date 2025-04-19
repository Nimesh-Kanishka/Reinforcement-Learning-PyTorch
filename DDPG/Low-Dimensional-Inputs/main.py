from agent import Agent

if __name__ == "__main__":
    agent = Agent(env_id="LunarLander-v3",
                  env_kwargs={"continuous": True},
                  alpha=1.5e-5,
                  beta=1.5e-4,
                  use_gpu=True,
                  load_checkpoint=True,
                  save_folder="DDPG/DDPG_Low_Dimensional_Inputs/LunarLander-v3_Continuous")
    """
    agent.train(total_episodes=2000,
                save_start_reward=210)
    """
    agent.evaluate(total_episodes=10)