import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
import matplotlib.pyplot as plt
from GINModel import CustomLayerModel
from PNAModel import PNAConvModel
from traintest import train, test  # Make sure these functions are compatible with batched data
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from saveloadplot import save_metrics, load_metrics, plot_metrics
from AttGINModel import NewCustomModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# Dataset specifics
dataset_names = []
dataset_names.append("MUTAG")
dataset_names.append("ENZYMES")
dataset_names.append("PROTEINS")
glb_epochs = 500
attempt = 11

# 10 - batch size 128
# 11 - batch size 256

random_feature_count = 0

for dataset_name in dataset_names:
    if dataset_name == "PROTEINS":
        num_epochs = 100
    else: 
        num_epochs = glb_epochs
    # Load the dataset
    dataset = TUDataset(root=f'/tmp/{dataset_name}', name=dataset_name)

    for graph in dataset:
        random_features = torch.rand((graph.num_nodes, random_feature_count))
        graph.x = torch.cat((graph.x, random_features), dim=1)

    num_graphs = len(dataset)
    train_size = int(num_graphs * 0.8)
    val_size = int(num_graphs * 0.1)
    test_size = num_graphs - (train_size + val_size)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    glb_batch_size = 256 

    # Create DataLoader for train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=glb_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=glb_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=glb_batch_size, shuffle=False)

    # Example data to calculate max_degree for PNAConvModel
    data = dataset[0]  
    d = degree(data.edge_index[0], dtype=torch.long)
    max_degree = int(d.max())
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())

    glb_hidden_features = 32
    glb_output_type = 'graph'
    glb_num_layers = 3
    glb_mlp_hidden_layers = 2

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
    glb_weight_decay = 0.001
    optimizers = {model_name: torch.optim.Adam(model.parameters(), lr=glb_lr, weight_decay=glb_weight_decay) for model_name, model in models.items()}
    # Define schedulers for each model
    schedulers = {model_name: ReduceLROnPlateau(optimizers[model_name], mode='min', factor=0.5, patience=25, verbose=True, min_lr=0.1) for model_name in models.keys()}
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
            total_loss = 0
            model.train()
            for data in train_loader:
                data = data.to(device)
                loss = train(model, data, optimizer, task_type=glb_output_type)
                total_loss += loss
            avg_loss = total_loss / len(train_loader)

            # Evaluation with validation and test loaders
            val_acc = test(model, val_loader, task_type=glb_output_type)
            test_acc = test(model, test_loader, task_type=glb_output_type)
            train_acc = test(model, train_loader, task_type=glb_output_type)

            schedulers[model_name].step(avg_loss)

            # Print stats
            print(f'Epoch: {epoch:03d}, {model_name} Loss: {avg_loss:.4f}', flush=True)
            print(f'{model_name} Val/Test Acc: {val_acc:.4f}/{test_acc:.4f}')

            metrics[model_name]['train_losses'][epoch] = loss
            metrics[model_name]['train_accuracies'][epoch] = train_acc
            metrics[model_name]['val_accuracies'][epoch] = val_acc
            metrics[model_name]['test_accuracies'][epoch] = test_acc
            metrics[model_name]['lr'][epoch] = current_lr

            save_metrics(metrics, dataset_name, model_name, attempt)

        print()
        if epoch % 2 == 0 or epoch == num_epochs - 1:
            plot_metrics(dataset_name, models.keys(), attempt)

