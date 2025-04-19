from typing import Sequence
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self,
                 num_actions: int,
                 fc_layer_sizes: int | Sequence[int] = [256]):
        super().__init__()

        if isinstance(fc_layer_sizes, int):
            fc_layer_sizes = [fc_layer_sizes]
        elif isinstance(fc_layer_sizes, (list, tuple)):
            if not all(isinstance(i, int) for i in fc_layer_sizes):
                raise TypeError("All elements in fc_layer_sizes must be integers.")
            if len(fc_layer_sizes) == 0:
                raise ValueError("fc_layer_sizes cannot be an empty list or tuple.")
        else:
            raise TypeError("fc_layer_sizes must be an int or a sequence of ints.")
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),  # Conv1: (4, 84, 84) -> (16, 20, 20)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # Conv2: (16, 20, 20) -> (32, 9, 9)
            nn.ReLU(),
            nn.Flatten(),  # Flatten: (32, 9, 9) -> 32*9*9 = 2592
        )

        fc_layers = []
        next_layer_in_features = 2592
        # Hidden layers with ReLU activations
        for layer_size in fc_layer_sizes:
            fc_layers.append(nn.Linear(next_layer_in_features, layer_size))
            fc_layers.append(nn.ReLU())
            next_layer_in_features = layer_size

        # Output layer with no activation
        fc_layers.append(nn.Linear(next_layer_in_features, num_actions))
        
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x / 255.0))