from typing import Sequence
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self,
                 num_observations: int,
                 num_actions: int,
                 hidden_layer_sizes: int | Sequence[int] = [64, 64]):
        super().__init__()

        if isinstance(hidden_layer_sizes, int):
            hidden_layer_sizes = [hidden_layer_sizes]
        elif isinstance(hidden_layer_sizes, (list, tuple)):
            if not all(isinstance(i, int) for i in hidden_layer_sizes):
                raise TypeError("All elements in hidden_layer_sizes must be integers.")
            if len(hidden_layer_sizes) == 0:
                raise ValueError("hidden_layer_sizes cannot be an empty list or tuple.")
        else:
            raise TypeError("hidden_layer_sizes must be an int or a sequence of ints.")

        layers = []
        next_layer_in_features = num_observations
        # Hidden layers with ReLU activations
        for layer_size in hidden_layer_sizes:
            layers.append(nn.Linear(next_layer_in_features, layer_size))
            layers.append(nn.ReLU())
            next_layer_in_features = layer_size

        # Output layer with no activation
        layers.append(nn.Linear(next_layer_in_features, num_actions))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)