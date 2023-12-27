import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def train(model, data, optimizer, task_type='node'):
    model.train()
    optimizer.zero_grad()

    out = model(data)

    if task_type == 'node':
        # Node-level task: use masks for training
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    elif task_type == 'graph':
        # Graph-level task: assume entire graph is used for training
        loss = F.cross_entropy(out, data.y)

    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, loader, task_type='node'):
    model.eval()
    correct = 0
    total = 0

    if task_type == 'node':
        # Assuming 'loader' actually represents single data in this case
        data = loader.to(device)  # 'loader' is actually a single data object here
        logits = model(data)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    elif task_type == 'graph':
        for data in loader:  # Now it iterates over batches
            data = data.to(device)
            logits = model(data)
            pred = logits.max(1)[1]
            correct += pred.eq(data.y).sum().item()
            total += data.y.size(0)
        accs = correct / total  # Calculate combined accuracy over all batches

    return accs

