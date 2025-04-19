import numpy as np
import gymnasium as gym

class AutoResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Compute the episode_over flag based on the pre-reset state
        episode_over = truncated or (self.env.unwrapped.ale.lives() == 0)

        if terminated or truncated:
            observation, info = self.env.reset()

        info["episode_over"] = episode_over

        return observation, reward, terminated, truncated, info
    
class FireResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

        observation, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)

        # Some Atari games require a sequence of actions (like FIRE and then a second button press)
        # to get out of the waiting state
        observation, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)

        return observation, info
    
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

        self._lives = self.env.unwrapped.ale.lives()
        self._was_real_done = True

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self._was_real_done = terminated or truncated

        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self._lives:
            terminated = True
        self._lives = lives

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self._was_real_done:
            observation, info = self.env.reset(**kwargs)

        else:
            # No-op step to advance from terminal/lost life state
            observation, _, terminated, truncated, info = self.env.step(0)

            # If the no-op step has lead to a game over, we should reset the environment
            if terminated or truncated:
                observation, info = self.env.reset(**kwargs)

        self._lives = self.env.unwrapped.ale.lives()

        return observation, info
    
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        
        self._observation_buffer = np.zeros((2, *env.observation_space.shape),
                                            dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int):
        total_reward = 0.0
        terminated = truncated = False

        for i in range(self._skip):
            observation, reward, terminated, truncated, info = self.env.step(action)

            if i == self._skip - 2:
                self._observation_buffer[0] = observation
            if i == self._skip - 1:
                self._observation_buffer[1] = observation

            total_reward += float(reward)

            if terminated or truncated:
                break
        
        max_frame = self._observation_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info
    
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)

        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

        # As np_random.integers() function returns an integer in the range [low, high), to get an
        # integer in the range [1, noop_max], noop_max + 1 should be passed as the upper limit
        self._noop_max = noop_max + 1
        assert self._noop_max > 1        

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

        num_noops = self.unwrapped.np_random.integers(1, self._noop_max)

        for _ in range(num_noops):
            observation, _, terminated, truncated, info = self.env.step(0)

            if terminated or truncated:
                observation, info = self.env.reset(**kwargs)

        return observation, info