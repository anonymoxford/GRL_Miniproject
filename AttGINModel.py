import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Set2Set
from torch_geometric.nn import global_add_pool, BatchNorm
from mlp import MLP

def aggregate_max(neighbors):
    return torch.max(neighbors, 0)[0]

def aggregate_min(neighbors):
    return torch.min(neighbors, 0)[0]

def aggregate_mean(neighbors):
    return torch.mean(neighbors, 0)

def aggregate_sum(neighbors):
    return torch.sum(neighbors, 0)

# Attention Module
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        # Attention layer for two entities
        self.attention_fc = nn.Linear(in_channels, 2)

    def forward(self, features_1, features_2):
        # Concatenate node features and aggregated neighbors for attention computation
        combined_features = torch.cat([features_1, features_2], dim=1)
        # Calculate attention scores
        scores = self.attention_fc(combined_features)
        # Apply softmax to get two attention weights
        weights = F.softmax(scores, dim=1)
        # Apply weights to the respective features
        features_1_weighted = weights[:, 0].unsqueeze(1) * features_1
        features_2_weighted = weights[:, 1].unsqueeze(1) * features_2
        # Sum weighted features
        combined_output = features_1_weighted + features_2_weighted
        return combined_output


class CombinedLayer(nn.Module):
    def __init__(self, in_dim, out_dim, mlp_hidden_layers):
        super(CombinedLayer, self).__init__()
        self.attention = AttentionModule(2*in_dim)  # Attention over concatenated features
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

        aggregated_feats_sum = torch.zeros_like(h)
        aggregated_feats_mean = torch.zeros_like(h)
        aggregated_feats_max = torch.zeros_like(h)
        for node in range(num_nodes):
            neighbors = adj[node] == 1
            # Ensure neighbors mask is correctly sized and aligned with h
            if neighbors.any():
                neighbor_feats = h[neighbors]
                aggregated_feats_sum[node] = aggregate_sum(neighbor_feats)
                aggregated_feats_mean[node] = aggregate_mean(neighbor_feats)
                aggregated_feats_max[node] = aggregate_max(neighbor_feats)

        combined_output = h + self.attention(aggregated_feats_sum, aggregated_feats_max)
        combined_output = self.mlp(combined_output)
        return combined_output


# New Custom Model using the combined layer
class NewCustomModel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers, output_type='node', mlp_hidden_layers=1):
        super(NewCustomModel, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.output_type = output_type  # 'node' or 'graph'
        self.mlp_hidden_layers = mlp_hidden_layers
        self.gru = nn.GRUCell(hidden_channels, hidden_channels)  # Using GRUCell

        # Initialize layers with the specified aggregation function
        self.layers.append(CombinedLayer(in_channels, hidden_channels, self.mlp_hidden_layers))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.layers.append(CombinedLayer(hidden_channels, hidden_channels, self.mlp_hidden_layers))
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