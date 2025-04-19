from agent import Agent

##### CONFIG #####
# Environment Params
ENV_ID = "CarRacing-v3"
# Other environment specific params
ENV_KWARGS = {
    "continuous": True,
    "domain_randomize": False
}
FRAME_STACK_SIZE = 4

# Hyperparameters
ACTOR_LR = 2e-5 # Actor learning rate
CRITIC_LR = 2e-4 # Critic learning rate
GAMMA = 0.99 # Discount factor
TAU = 0.003 # Soft update rate
REPLAY_BUFFER_SIZE = 250_000 # Maximum size of replay buffer
MINIBATCH_SIZE = 64 # Training batch size
# OU Noise Params
THETA = 0.15
SIGMA = 0.2

# Network Architecture
CONV_LAYER_PARAMS = [
    {"out_channels": 16, "kernel_size": 7, "stride": 2, "padding": 3},
    {"out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 2},
    {"out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 1}
]
FC_LAYER_SIZES = [400, 300]

# Device
USE_GPU = True

# Saving
# Folder to which training logs, models and graphs should be saved
SAVE_FOLDER = "DDPG/DDPG_Pixel_Inputs/CarRacing-v3_Continuous"
LOAD_CHECKPOINT = True

# Training
TRAIN = False
TOTAL_EPISODES = 10
TRAIN_START_STEP = 10_000
SAVE_START_REWARD = 450
PLOT_GRAPH_INTERVAL = 100

if __name__ == "__main__":
    agent = Agent(ENV_ID, ENV_KWARGS, FRAME_STACK_SIZE, ACTOR_LR,
                  CRITIC_LR, GAMMA, TAU, REPLAY_BUFFER_SIZE,
                  MINIBATCH_SIZE, THETA, SIGMA, CONV_LAYER_PARAMS,
                  FC_LAYER_SIZES, USE_GPU, SAVE_FOLDER, LOAD_CHECKPOINT)
    
    if TRAIN:
        agent.train(TOTAL_EPISODES, TRAIN_START_STEP,
                    SAVE_START_REWARD, PLOT_GRAPH_INTERVAL)
    else:
        agent.evaluate(TOTAL_EPISODES)