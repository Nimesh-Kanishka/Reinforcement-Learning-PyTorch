import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self,
                 num_observations: int,
                 num_actions: int,
                 fc1_size: int = 400,
                 fc2_size: int = 300
    ):
        super().__init__()

        fc1 = nn.Linear(num_observations, fc1_size)
        f1 = 1 / np.sqrt(fc1.weight.data.size()[0])
        nn.init.uniform_(fc1.weight.data, -f1, f1)
        nn.init.uniform_(fc1.bias.data, -f1, f1)

        fc2 = nn.Linear(fc1_size, fc2_size)
        f2 = 1 / np.sqrt(fc2.weight.data.size()[0])
        nn.init.uniform_(fc2.weight.data, -f2, f2)
        nn.init.uniform_(fc2.bias.data, -f2, f2)

        mu = nn.Linear(fc2_size, num_actions)
        nn.init.uniform_(mu.weight.data, -0.003, 0.003)
        nn.init.uniform_(mu.bias.data, -0.003, 0.003)

        self.layers = nn.Sequential(
            fc1,
            nn.LayerNorm(fc1_size),
            nn.ReLU(),
            fc2,
            nn.LayerNorm(fc2_size),
            nn.ReLU(),
            mu,
            nn.Tanh(),
        )

    def forward(self, observations):
        return self.layers(observations)
    
class CriticNetwork(nn.Module):
    def __init__(self,
                 num_observations: int,
                 num_actions: int,
                 fc1_size: int = 400,
                 fc2_size: int = 300
    ):
        super().__init__()

        fc1 = nn.Linear(num_observations, fc1_size)
        f1 = 1 / np.sqrt(fc1.weight.data.size()[0])
        nn.init.uniform_(fc1.weight.data, -f1, f1)
        nn.init.uniform_(fc1.bias.data, -f1, f1)

        fc2 = nn.Linear(fc1_size, fc2_size)
        f2 = 1 / np.sqrt(fc2.weight.data.size()[0])
        nn.init.uniform_(fc2.weight.data, -f2, f2)
        nn.init.uniform_(fc2.bias.data, -f2, f2)
        
        self.state_value = nn.Sequential(
            fc1,
            nn.LayerNorm(fc1_size),
            nn.ReLU(),
            fc2,
            nn.LayerNorm(fc2_size),
        )

        self.action_value = nn.Sequential(
            nn.Linear(num_actions, fc2_size),
            nn.ReLU(),
        )

        self.q = nn.Linear(fc2_size, 1)
        nn.init.uniform_(self.q.weight.data, -0.003, 0.003)
        nn.init.uniform_(self.q.bias.data, -0.003, 0.003)

    def forward(self, observations, actions):
        state_value = self.state_value(observations)

        action_value = self.action_value(actions)

        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)
        return state_action_value