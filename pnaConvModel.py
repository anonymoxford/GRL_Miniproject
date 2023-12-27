import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_add_pool, BatchNorm
from pytorch_geometric.pna import PNAConvSimple

class PNAConvModel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers, deg, output_type='node'):
        super(PNAConvModel, self).__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.gru = nn.GRUCell(hidden_channels, hidden_channels)  # Using GRUCell
        self.fcs = nn.ModuleList()
        self.output_type = output_type  # 'node' or 'graph'

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        # First layer
        self.convs.append(PNAConvSimple(in_channels, hidden_channels, aggregators, scalers, deg))
        self.batch_norms.append(BatchNorm(hidden_channels))
        self.fcs.append(nn.Linear(hidden_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(PNAConvSimple(hidden_channels, hidden_channels, aggregators, scalers, deg))
            self.batch_norms.append(BatchNorm(hidden_channels))
            self.fcs.append(nn.Linear(hidden_channels, hidden_channels))

        self.fc_last = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Initial hidden state for GRU
        h = x.new_zeros(x.size(0), self.gru.hidden_size)

        for (conv, bn, fc) in zip(self.convs, self.batch_norms, self.fcs):
            x = conv(x, edge_index)
            x = bn(x)
            x = fc(x)
            x = F.relu(x) 
            x = F.dropout(x, training=self.training)

            h = self.gru(x, h)
            x = h  
        
        # Conditionally apply global pooling for graph-level tasks
        if self.output_type == 'graph':
            x = global_add_pool(x, batch)

        x = self.fc_last(x)
        return x