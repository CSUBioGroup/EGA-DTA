"""
Main：
1.  ESM-2 3B  (2560)
2.  CosineAnnealingWarmRestarts 
3.  MSELoss 
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import sys
import json
from datetime import datetime
import time

from utils import TestbedDataset, rmse, mse, pearson, spearman, ci
from BDEGNN import EnhancedGATNet


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_epoch(model, device, train_loader, optimizer, loss_fn, config):
    model.train()
    total_loss = 0
    valid_batches = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)
        
        loss = loss_fn(output, data.y.view(-1, 1).float())

        loss.backward()
        
        # clip_grad
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
        
        optimizer.step()

        total_loss += loss.item() * len(data.y)
        valid_batches += len(data.y)

    return total_loss / valid_batches

def evaluate(model, device, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            all_preds.extend(output.cpu().numpy().flatten())
            all_labels.extend(data.y.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return {
        'mse': mse(all_labels, all_preds),
        'rmse': rmse(all_labels, all_preds),
        'pearson': pearson(all_labels, all_preds),
        'ci': ci(all_labels, all_preds)
    }

def main():
    SEED = 42
    setup_seed(SEED)
    
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'davis'


    #  2560 (3B) 
    esm_dim = 1280 
    if os.path.exists(f'data/esm2_features_{dataset_name}_2560.pkl'):
        print(f"2560 (3B) features！")
        esm_dim = 2560
    elif os.path.exists(f'data/esm2_features_{dataset_name}_5120.pkl'):
        print(f" 5120 (15B) ！")
        esm_dim = 5120
    else:
        print(f"1280(650M) ")

    config = {
        'dataset': dataset_name,
        'lr': 1e-4, 
        'batch_size': 512, 
        'num_epochs': 1000,
        'patience': 100,
        'clip_grad': 1.0,
        'esm_dim': esm_dim, 
        'output_dim': 512,
        'dropout': 0.1
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'DTAoutputs/{dataset_name}_MSE_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | ESM Dim: {esm_dim} | Loss: MSE")


    train_dataset = TestbedDataset(root='data', dataset=dataset_name + '_train', use_precomputed_esm2=True)
    test_dataset = TestbedDataset(root='data', dataset=dataset_name + '_test', use_precomputed_esm2=True)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # ==================== Model ====================
    model = EnhancedGATNet(
        output_dim=config['output_dim'],
        dropout=config['dropout'],
        esm_dim=config['esm_dim'] # 传入正确维度
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )

    loss_fn = nn.MSELoss()


    best_mse = float('inf')
    best_ci = 0.0
    patience_count = 0

    print("Train...")
    for epoch in range(1, config['num_epochs'] + 1):
        start = time.time()
        
        train_loss = train_epoch(model, device, train_loader, optimizer, loss_fn, config)
        metrics = evaluate(model, device, test_loader)
        

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:03d} ({time.time()-start:.1f}s) | LR: {current_lr:.2e} | "
              f"Loss: {train_loss:.4f} | Test MSE: {metrics['mse']:.4f} | CI: {metrics['ci']:.4f}")

        if metrics['mse'] < best_mse:
            best_mse = metrics['mse']
            best_ci = metrics['ci']
            patience_count = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            print(f"  >>> ★ New Best: MSE {best_mse:.4f}")
        else:
            patience_count += 1
            if patience_count >= config['patience']:
                print("Early Stopping.")
                break

    print(f"Best MSE: {best_mse:.4f}, CI: {best_ci:.4f}")

if __name__ == "__main__":
    main()
