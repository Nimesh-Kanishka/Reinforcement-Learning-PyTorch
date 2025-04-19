import numpy as np
from gymnasium.spaces import Space

class ReplayBuffer:
    def __init__(self,
                 buffer_size: int,
                 env_observation_space: Space,
                 env_action_space: Space,
                 optimize_memory_usage: bool = False
    ):
        self.buffer_size = buffer_size
        self.optimize_memory_usage = optimize_memory_usage
        
        self.observation_buffer = np.zeros((buffer_size, *env_observation_space.shape),
                                           dtype=env_observation_space.dtype)
        self.action_buffer = np.zeros((buffer_size, *env_action_space.shape),
                                      dtype=env_action_space.dtype)
        self.reward_buffer = np.zeros(buffer_size,
                                      dtype=np.float32)
        self.done_buffer = np.zeros(buffer_size,
                                    dtype=np.bool_)
        if not optimize_memory_usage:
            self.next_observation_buffer = np.zeros((buffer_size, *env_observation_space.shape),
                                                    dtype=env_observation_space.dtype)
        
        self.mem_cntr = 0
        self.mem_full = False
        
    def add(self, observation, action, reward, next_observation, done):
        """
        Add a transition to the replay buffer
        """

        self.observation_buffer[self.mem_cntr] = observation
        self.action_buffer[self.mem_cntr] = action
        self.reward_buffer[self.mem_cntr] = reward
        self.done_buffer[self.mem_cntr] = done
        if self.optimize_memory_usage:
            self.observation_buffer[(self.mem_cntr + 1) % self.buffer_size] = next_observation
        else:
            self.next_observation_buffer[self.mem_cntr] = next_observation

        self.mem_cntr += 1
        if self.mem_cntr >= self.buffer_size:
            self.mem_cntr = 0
            self.mem_full = True

    def sample(self, batch_size: int):
        """
        Sample elements from the replay buffer
        """

        # If the next_observation_buffer exists:
        if not self.optimize_memory_usage:
            mem_indices = np.random.choice(self.buffer_size if self.mem_full else self.mem_cntr, size=batch_size, replace=False)

            return self.observation_buffer[mem_indices], \
                self.action_buffer[mem_indices], \
                self.reward_buffer[mem_indices], \
                self.next_observation_buffer[mem_indices], \
                self.done_buffer[mem_indices]
        
        # If the next_observation_buffer does not exist:
        if self.mem_full:
            # Cannot take the transition at the index self.mem_cntr as the observation and the other data will be from different timesteps
            mem_indices = (np.random.choice(np.arange(start=1, stop=self.buffer_size), size=batch_size, replace=False) + self.mem_cntr) % self.buffer_size
        else:
            mem_indices = np.random.choice(self.mem_cntr, size=batch_size, replace=False)
            
        return self.observation_buffer[mem_indices], \
            self.action_buffer[mem_indices], \
            self.reward_buffer[mem_indices], \
            self.observation_buffer[(mem_indices + 1) % self.buffer_size], \
            self.done_buffer[mem_indices]