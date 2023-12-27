import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_add_pool, BatchNorm
from pytorch_geometric.pna import PNAConvSimple
from torch_geometric.nn import Set2Set
from mlp import MLP

class PNAConvModel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers, deg, output_type='node', mlp_hidden_layers=1):
        super(PNAConvModel, self).__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.gru = nn.GRUCell(hidden_channels, hidden_channels)  # Using GRUCell
        self.mlps = nn.ModuleList()
        self.mlp_hidden_layers = mlp_hidden_layers
        self.output_type = output_type  # 'node' or 'graph'

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        # First layer
        self.convs.append(PNAConvSimple(in_channels, hidden_channels, aggregators, scalers, deg))
        self.batch_norms.append(BatchNorm(hidden_channels))
        self.mlps.append(MLP(hidden_channels, hidden_channels, hidden_channels, self.mlp_hidden_layers))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(PNAConvSimple(hidden_channels, hidden_channels, aggregators, scalers, deg))
            self.batch_norms.append(BatchNorm(hidden_channels))
            self.mlps.append(MLP(hidden_channels, hidden_channels, hidden_channels, self.mlp_hidden_layers))

        if self.output_type == 'graph':
            # Initialize Set2Set, here assuming LSTM processing steps as 3, and the LSTM hidden dim as hid_features
            self.set2set = Set2Set(hidden_channels, processing_steps=3, num_layers=1)
            self.mlp_last = MLP(2 * hidden_channels, hidden_channels, out_channels, hidden_layers=self.mlp_hidden_layers)
        else: 
            self.mlp_last = MLP(hidden_channels, hidden_channels, out_channels, hidden_layers=self.mlp_hidden_layers)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Initial hidden state for GRU
        h = x.new_zeros(x.size(0), self.gru.hidden_size)

        for (conv, bn, mlp) in zip(self.convs, self.batch_norms, self.mlps):
            x = conv(x, edge_index)
            x = mlp(x)
            x = bn(x)
            x = F.dropout(x, training=self.training)

            h = self.gru(x, h)
            x = h  
        
        # Conditionally apply global pooling for graph-level tasks
        if self.output_type == 'graph':
            #x = global_add_pool(x, batch)
            x = self.set2set(x, batch)

        x = self.mlp_last(x)
        return x