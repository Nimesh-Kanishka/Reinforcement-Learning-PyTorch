import torch.nn as nn
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
                 frame_stack_size: int = 4,
                 **kwargs
    ):
        self.env_id = env_id
        self.env_kwargs = kwargs
        self.frame_stack_size = frame_stack_size
        
        # If a render_mode is not specified, default to None
        if "render_mode" not in self.env_kwargs:
            self.env_kwargs["render_mode"] = None

        self.env = self.make_env()

    def make_env(self):
        env = gym.make(self.env_id, **self.env_kwargs)

        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, self.frame_stack_size)

        return env

    def change_render_mode(self,
                           render_mode: str | None
    ):
        # If the new render mode is the same as the current one, do nothing
        if self.env_kwargs["render_mode"] == render_mode:
            return
        
        # Close the current environment
        self.env.close()

        # Update render mode and re-create the environment
        self.env_kwargs["render_mode"] = render_mode
        self.env = self.make_env()

    # Delegate other calls to the env
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    
def init_layer_weights_and_biases_uniform(layer: nn.Module,
                                          init_lower_bound: float | None = None,
                                          init_upper_bound: float | None = None
):
    if (init_lower_bound is None) and (init_upper_bound is None):
        if isinstance(layer, nn.Linear):
            fan_in = layer.in_features
        elif isinstance(layer, nn.Conv2d):
            fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
        else:
            raise ValueError(
                f"Automatic initialization is not supported for layer type '{type(layer).__name__}'. "
                "Please specify 'init_lower_bound' and 'init_upper_bound' manually."
            )
        
        init_upper_bound = 1 / np.sqrt(fan_in)
        init_lower_bound = -init_upper_bound

    elif (init_lower_bound is None) or (init_upper_bound is None):
        raise ValueError(
            "Provide both 'init_lower_bound' and 'init_upper_bound', or leave both as None for automatic initialization."
        )

    nn.init.uniform_(layer.weight, init_lower_bound, init_upper_bound)

    if layer.bias is not None:
        nn.init.uniform_(layer.bias, init_lower_bound, init_upper_bound)