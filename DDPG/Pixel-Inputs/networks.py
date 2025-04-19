from typing import Sequence
import torch
import torch.nn as nn
from utils import init_layer_weights_and_biases_uniform

class ActorNetwork(nn.Module):
    def __init__(self,
                 observation_shape: tuple[int],
                 num_actions: int,
                 conv_layer_params: Sequence[dict] | None = None,
                 fc_layer_sizes: Sequence[int] | None = None
    ):
        super().__init__()

        # Set defaults if conv_layer_params or fc_layer_sizes are not provided
        if conv_layer_params is None:
            conv_layer_params = [
                {"out_channels": 16, "kernel_size": 7, "stride": 2, "padding": 3},
                {"out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 2},
                {"out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 1}
            ]

        if fc_layer_sizes is None:
            fc_layer_sizes = [200, 200]

        # Convolutional layers
        conv_layers = []

        in_channels, h, w = observation_shape

        for layer_params in conv_layer_params:
            out_channels, k, s, p = \
                layer_params["out_channels"], layer_params["kernel_size"], layer_params["stride"], layer_params["padding"]
            
            h = (h - k + 2 * p) // s + 1
            w = (w - k + 2 * p) // s + 1

            layer = nn.Conv2d(in_channels, out_channels, k, s, p)
            init_layer_weights_and_biases_uniform(layer)

            conv_layers.extend([
                layer,
                nn.LayerNorm([out_channels, h, w]),
                nn.ReLU(),
            ])

            in_channels = out_channels

        self.conv_layers = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
        )

        # Fully connected layers
        fc_layers = []

        in_features = in_channels * h * w

        for layer_size in fc_layer_sizes:
            layer = nn.Linear(in_features, layer_size)
            init_layer_weights_and_biases_uniform(layer)

            fc_layers.extend([
                layer,
                nn.LayerNorm(layer_size),
                nn.ReLU()
            ])

            in_features = layer_size

        mu = nn.Linear(in_features, num_actions)
        init_layer_weights_and_biases_uniform(mu, -0.0003, 0.0003)
        
        self.fc_layers = nn.Sequential(
            *fc_layers,
            mu,
            nn.Tanh(),
        )

    def forward(self, observations):
        return self.fc_layers(self.conv_layers(observations.float() / 255.0))
    
    
class CriticNetwork(nn.Module):
    def __init__(self,
                 observation_shape: tuple[int],
                 num_actions: int,
                 conv_layer_params: Sequence[dict] | None = None,
                 fc_layer_sizes: Sequence[int] | None = None
    ):
        super().__init__()

        # Set defaults if conv_layer_params or fc_layer_sizes are not provided
        if conv_layer_params is None:
            conv_layer_params = [
                {"out_channels": 16, "kernel_size": 7, "stride": 2, "padding": 3},
                {"out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 2},
                {"out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 1}
            ]

        if fc_layer_sizes is None:
            fc_layer_sizes = [200, 200]

        # Convolutional layers
        conv_layers = []

        in_channels, h, w = observation_shape

        for layer_params in conv_layer_params:
            out_channels, k, s, p = \
                layer_params["out_channels"], layer_params["kernel_size"], layer_params["stride"], layer_params["padding"]
            
            h = (h - k + 2 * p) // s + 1
            w = (w - k + 2 * p) // s + 1

            layer = nn.Conv2d(in_channels, out_channels, k, s, p)
            init_layer_weights_and_biases_uniform(layer)

            conv_layers.extend([
                layer,
                nn.LayerNorm([out_channels, h, w]),
                nn.ReLU(),
            ])

            in_channels = out_channels

        self.conv_layers = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
        )

        # Fully connected layers
        fc_layers = []

        in_features = in_channels * h * w + num_actions

        for layer_size in fc_layer_sizes:
            layer = nn.Linear(in_features, layer_size)
            init_layer_weights_and_biases_uniform(layer)

            fc_layers.extend([
                layer,
                nn.LayerNorm(layer_size),
                nn.ReLU()
            ])

            in_features = layer_size

        q = nn.Linear(in_features, 1)
        init_layer_weights_and_biases_uniform(q, -0.0003, 0.0003)
        
        self.fc_layers = nn.Sequential(
            *fc_layers,
            q,
        )

    def forward(self, observations, actions):
        # Normalize pixel observations and process through convolutional layers
        x = self.conv_layers(observations.float() / 255.0)

        # Concatenate the convolutional features with the actions along the feature dimension.
        # x shape: (batch_size, conv_features), actions shape: (batch_size, num_actions)
        x = torch.cat([x, actions], dim=1)

        return self.fc_layers(x)