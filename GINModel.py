import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, BatchNorm
from torch_geometric.nn import Set2Set
from mlp import MLP

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
    def __init__(self, in_dim, out_dim, aggregation, mlp_hidden_layers):
        super(Layer, self).__init__()
        self.aggregation = AGGREGATION_FUNCS[aggregation]
        self.mlp = MLP(in_dim, out_dim, out_dim, hidden_layers=mlp_hidden_layers)

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
        combined_feats = self.mlp(combined_feats)
        return combined_feats


class CustomLayerModel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers, aggregation, output_type='node', mlp_hidden_layers=1):
        super(CustomLayerModel, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.output_type = output_type  # 'node' or 'graph'
        self.mlp_hidden_layers = mlp_hidden_layers
        self.gru = nn.GRUCell(hidden_channels, hidden_channels)  # Using GRUCell

        # Initialize layers with the specified aggregation function
        self.layers.append(Layer(in_channels, hidden_channels, aggregation, self.mlp_hidden_layers))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.layers.append(Layer(hidden_channels, hidden_channels, aggregation, self.mlp_hidden_layers))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        if self.output_type == 'graph':
            # Initialize Set2Set, here assuming LSTM processing steps as 3, and the LSTM hidden dim as hid_features
            self.set2set = Set2Set(hidden_channels, processing_steps=3, num_layers=1)
            self.mlp_last = MLP(2 * hidden_channels, hidden_channels, out_channels, hidden_layers=self.mlp_hidden_layers)
        else: 
            self.mlp_last = MLP(hidden_channels, hidden_channels, out_channels, hidden_layers=self.mlp_hidden_layers)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = x.new_zeros(x.size(0), self.gru.hidden_size)

        # Pass input through each layer
        for (layer, bn) in zip(self.layers, self.batch_norms):
            x = layer(x, edge_index)
            x = bn(x)
            x = F.dropout(x, training=self.training)
            
            h = self.gru(x, h)
            x = h  

        # Conditionally apply global pooling for graph-level tasks
        if self.output_type == 'graph':
            #x = global_add_pool(x, batch)
            x = self.set2set(x, batch)

        # Final transformation
        x = self.mlp_last(x)
        return x
