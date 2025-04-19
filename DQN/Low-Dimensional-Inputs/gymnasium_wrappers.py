import gymnasium as gym

class AutoResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        # Call the built-in step function
        observation, reward, terminated, truncated, info = self.env.step(action)

        # If terminal condition is reached, reset the environment and obtain the new observation
        if terminated or truncated:
            observation, info = self.env.reset()

        return observation, reward, terminated, truncated, info