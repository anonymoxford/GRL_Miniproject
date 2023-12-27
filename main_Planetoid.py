import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
import matplotlib.pyplot as plt
from GINModel import CustomLayerModel
from PNAModel import PNAConvModel
import os
from traintest import train, test
from torch.optim.lr_scheduler import ReduceLROnPlateau
from saveloadplot import save_metrics, load_metrics, plot_metrics
import numpy as np
from AttGINModel import NewCustomModel

# Set the device and move data to the device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
dataset_names = ["Cora", "CiteSeer", "PubMed"]
num_epochs_list = [100, 100, 100]

random_feature_count = 0

for attempt in [11]:
    for dataset_name, num_epochs  in zip(dataset_names, num_epochs_list):
        # Load the dataset
        dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
        data = dataset[0]

        random_features = torch.rand((data.num_nodes, random_feature_count))
        data.x = torch.cat((data.x, random_features), dim=1)
        data = data.to(device)

        # Compute the degree for PNAConvSimple
        d = degree(data.edge_index[0], dtype=torch.long)
        max_degree = int(d.max())
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

        glb_hidden_features = 16
        glb_output_type = 'node'
        glb_num_layers = 2
        glb_mlp_hidden_layers = 1

        original_feature_dim = dataset.num_node_features  # or however you obtain it
        new_feature_dim = original_feature_dim + random_feature_count  # New dimension size

        models = {
            'PNA': PNAConvModel(
            in_channels=new_feature_dim, 
            out_channels=dataset.num_classes, 
            hidden_channels=glb_hidden_features, 
            num_layers=glb_num_layers,
            deg=deg,
            output_type=glb_output_type,
            mlp_hidden_layers=glb_mlp_hidden_layers).to(device),

            'GIN': CustomLayerModel(
            in_channels=new_feature_dim,
            out_channels=dataset.num_classes,
            hidden_channels=glb_hidden_features,
            num_layers=glb_num_layers,
            aggregation='sum',
            output_type=glb_output_type,
            mlp_hidden_layers=glb_mlp_hidden_layers).to(device),

            'Att-GIN': NewCustomModel(
            in_channels=new_feature_dim,
            out_channels=dataset.num_classes,
            hidden_channels=glb_hidden_features,
            num_layers=glb_num_layers,
            output_type=glb_output_type,
            mlp_hidden_layers=glb_mlp_hidden_layers).to(device)
        }

        # Define optimizers for each model
        glb_lr = 0.01
        glb_weight_decay = 1e-4
        optimizers = {model_name: torch.optim.Adam(model.parameters(), lr=glb_lr, weight_decay=glb_weight_decay) for model_name, model in models.items()}
        schedulers = {model_name: ReduceLROnPlateau(optimizers[model_name], mode='min', factor=0.5, patience=5, verbose=True) for model_name in models.keys()}
        # Initialize metrics for each model
        metrics = {
            model_name: {
                'train_losses': np.empty(num_epochs),
                'train_accuracies': np.empty(num_epochs),
                'test_accuracies': np.empty(num_epochs),
                'val_accuracies': np.empty(num_epochs),
                'lr': np.empty(num_epochs)
            } for model_name in models.keys()
        }
        for epoch in range(num_epochs):
            for model_name, model in models.items():
                optimizer = optimizers[model_name]
                current_lr = optimizer.param_groups[0]['lr']  # Assuming one parameter group

                # Training
                loss = train(model, data, optimizer, glb_output_type)

                # Evaluation
                train_acc, val_acc, test_acc = test(model, data, glb_output_type)
                schedulers[model_name].step(loss)

                # Print stats
                print(f'Epoch: {epoch:03d}, {model_name} Loss: {loss:.4f}', flush=True)
                print(f'{model_name} Train/Val/Test Acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}')
                
                metrics[model_name]['train_losses'][epoch] = loss
                metrics[model_name]['train_accuracies'][epoch] = train_acc
                metrics[model_name]['val_accuracies'][epoch] = val_acc
                metrics[model_name]['test_accuracies'][epoch] = test_acc
                metrics[model_name]['lr'][epoch] = current_lr

                save_metrics(metrics, dataset_name, model_name, attempt)

            print()
            if epoch % 2 == 0 or epoch == num_epochs - 1:
                plot_metrics(dataset_name, models.keys(), attempt)