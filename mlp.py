import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, hidden_layers=1):
        super(MLP, self).__init__()
        
        # Initialize module list for dynamically adding layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))  # First layer

        # Add hidden layers based on hidden_layers parameter
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))  # Intermediate layers

        self.layers.append(nn.Linear(hidden_dim, out_dim))  # Output layer

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))  # Apply ReLU after each intermediate layer
        x = self.layers[-1](x)  # No activation after last layer
        return x
