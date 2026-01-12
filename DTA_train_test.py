"""
DTA_EIAGNN.py (适配 ESM-2 3B & 优化训练策略)
功能：
1. 自动加载 ESM-2 3B 特征 (2560维)
2. 引入 CosineAnnealingWarmRestarts 调度器
3. 使用 MSELoss (回归任务标准损失)
"""

import os
# 解决某些环境下的库冲突
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

# ==================== 全局设置 ====================
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
        
        # [修改] 回归任务标准 MSE Loss
        loss = loss_fn(output, data.y.view(-1, 1).float())

        loss.backward()
        
        # 梯度裁剪 (防止梯度爆炸)
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

    # ==================== 智能配置检测 ====================
    # 检测是否存在 2560 (3B) 维特征文件
    esm_dim = 1280 # 默认
    if os.path.exists(f'data/esm2_features_{dataset_name}_2560.pkl'):
        print(f"★ 检测到 2560维 (3B) 特征，启用高维配置！")
        esm_dim = 2560
    elif os.path.exists(f'data/esm2_features_{dataset_name}_5120.pkl'):
        print(f"★ 检测到 5120维 (15B) 特征，启用旗舰配置！")
        esm_dim = 5120
    else:
        print(f"注意: 使用默认 1280维 (650M) 配置。")

    config = {
        'dataset': dataset_name,
        'lr': 1e-4, # 配合 Scheduler 使用
        'batch_size': 512, # 针对 3B 特征和深层网络调整
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
    print(f"运行设备: {device} | ESM Dim: {esm_dim} | Loss: MSE")

    # ==================== 加载数据 ====================
    train_dataset = TestbedDataset(root='data', dataset=dataset_name + '_train', use_precomputed_esm2=True)
    test_dataset = TestbedDataset(root='data', dataset=dataset_name + '_test', use_precomputed_esm2=True)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # ==================== 模型 ====================
    model = EnhancedGATNet(
        output_dim=config['output_dim'],
        dropout=config['dropout'],
        esm_dim=config['esm_dim'] # 传入正确维度
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    
    # 余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )
    
    # [修改] 切换回 MSE Loss
    loss_fn = nn.MSELoss()

    # ==================== 训练 ====================
    best_mse = float('inf')
    best_ci = 0.0
    patience_count = 0

    print("开始训练...")
    for epoch in range(1, config['num_epochs'] + 1):
        start = time.time()
        
        train_loss = train_epoch(model, device, train_loader, optimizer, loss_fn, config)
        metrics = evaluate(model, device, test_loader)
        
        # 更新学习率
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

    print(f"训练结束。最佳 MSE: {best_mse:.4f}, CI: {best_ci:.4f}")

if __name__ == "__main__":
    main()