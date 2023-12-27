import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

def aggregate_max(neighbors):
    return torch.max(neighbors, 0)[0]

def aggregate_min(neighbors):
    return torch.min(neighbors, 0)[0]

def aggregate_mean(neighbors):
    return torch.mean(neighbors, 0)

def aggregate_sum(neighbors):
    return torch.sum(neighbors, 0)

# Mapping from string names to functions
AGGREGATION_FUNCS = {
    'max': aggregate_max,
    'min': aggregate_min,
    'mean': aggregate_mean,
    'sum': aggregate_sum,
}

class Layer(nn.Module):
    def __init__(self, in_dim, out_dim, aggregation):
        super(Layer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)  # Adjust input dimension for concatenation
        self.aggregation = AGGREGATION_FUNCS[aggregation]

    def get_adjacency(self, num_nodes, edge_index):
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1  # Assuming undirected graph
        return adj

    def forward(self, h, edge_index, batch=None):
        # If batched, use the batch to find actual number of nodes
        num_nodes = h.size(0)
        
        # Construct adjacency matrix or use prebuilt methods for handling graphs
        adj = self.get_adjacency(num_nodes, edge_index)

        aggregated_feats = torch.zeros_like(h)
        for node in range(num_nodes):
            neighbors = adj[node] == 1
            # Ensure neighbors mask is correctly sized and aligned with h
            if neighbors.any():
                neighbor_feats = h[neighbors]
                aggregated_feats[node] = self.aggregation(neighbor_feats)

        combined_feats = h + aggregated_feats
        return self.fc(combined_feats)


class CustomLayerModel(nn.Module):
    def __init__(self, num_features, out_features, hid_features, num_layers, aggregation, output_type='node'):
        super(CustomLayerModel, self).__init__()
        self.layers = nn.ModuleList()
        self.output_type = output_type  # 'node' or 'graph'

        # Initialize layers with the specified aggregation function
        self.layers.append(Layer(num_features, hid_features, aggregation))
        for _ in range(num_layers - 2):
            self.layers.append(Layer(hid_features, hid_features, aggregation))
        self.fc_last = nn.Linear(hid_features, out_features)
        self.act = nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Pass input through each layer
        for layer in self.layers:
            x = self.act(layer(x, edge_index))

        # Conditionally apply global pooling for graph-level tasks
        if self.output_type == 'graph':
            x = global_add_pool(x, batch)

        # Final transformation
        x = self.fc_last(x)
        return x
