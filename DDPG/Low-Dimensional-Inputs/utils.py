import numpy as np
import gymnasium as gym

class OUActionNoise:
    def __init__(self,
                 mu: np.ndarray,
                 theta: float = 0.15,
                 sigma: float = 0.2,
                 dt: float = 1e-2,
                 x0: np.ndarray | None = None
    ):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0

        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        
        self.x_prev = x
        
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class DynamicRenderEnv:
    def __init__(self,
                 env_id: str,
                 **kwargs
    ):
        self.env_id = env_id
        self.env_kwargs = kwargs.copy()
        
        # If a render_mode is not specified, default to None
        if not("render_mode" in self.env_kwargs):
            self.env_kwargs["render_mode"] = None

        self.env = gym.make(env_id, **self.env_kwargs)

    def change_render_mode(self,
                           render_mode: str | None
    ):
        # If the new render_mode is the same as the current
        # render_mode of the environment, do nothing
        if self.env_kwargs["render_mode"] == render_mode:
            return
        
        # Close the current environment
        self.env.close()

        # Update render_mode in the dictionary and re-create
        # the environment with the new render_mode
        self.env_kwargs["render_mode"] = render_mode
        self.env = gym.make(self.env_id, **self.env_kwargs)

    # Delegate other calls to the env
    def __getattr__(self, name):
        return getattr(self.env, name)