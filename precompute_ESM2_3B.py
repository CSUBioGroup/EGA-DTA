"""
model: esm2_t36_3B_UR50D
ESM Dimensions: 2560

"""
import torch
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys

def precompute_esm2_features(dataset_name='davis', batch_size=4): 


    print("=" * 60)
    print(f"预计算 ESM2 3B (2560-dim) 特征 - {dataset_name}")
    print("模型: esm2_t36_3B_UR50D")
    print("=" * 60)


    try:
        import esm
        print("Loading ESM2 3B (3B params, 2560-dim) ... downloading")
        
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"✓ ESM2 3B 加载成功，使用设备: {device}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("  请确保已安装 esm: pip install fair-esm")
        return

    train_path = f'data/{dataset_name}_train.csv'
    test_path = f'data/{dataset_name}_test.csv'
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"错误: 找不到数据文件 {train_path} 或 {test_path}")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)


    all_sequences = set(train_df['target_sequence'].tolist() +
                        test_df['target_sequence'].tolist())
    all_sequences_list = list(all_sequences)
    print(f"\n总共 {len(all_sequences_list)} 个唯一蛋白质序列")

    esm2_features = {}
    esm_dim = 2560 
    esm_layer = 36 

    print(f"\n开始计算特征 (Batch Size: {batch_size})...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(all_sequences_list), batch_size)):
            batch_seqs = all_sequences_list[i:i+batch_size]

            batch_data = []
            for j, seq in enumerate(batch_seqs):
                batch_data.append((f"protein_{i+j}", seq[:1022]))

            try:
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
                batch_tokens = batch_tokens.to(device)

                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        results = model(batch_tokens, repr_layers=[esm_layer], return_contacts=False)
                else:
                    results = model(batch_tokens, repr_layers=[esm_layer], return_contacts=False)

                token_representations = results["representations"][esm_layer]


                for k, (_, seq) in enumerate(batch_data):
                    seq_len = min(len(seq), 1022)
                    embedding = token_representations[k, 1 : seq_len + 1].mean(0).cpu().numpy()
                    esm2_features[batch_seqs[k]] = embedding

                del batch_tokens, results, token_representations
                torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"✗ 批次 {i // batch_size} 失败 (通常是 OOM): {e}")
                print("  尝试减小 batch_size 重试。")
                continue 

    output_file = f'data/esm2_features_{dataset_name}_2560.pkl'
    os.makedirs('data', exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(esm2_features, f)

    print(f"\n✓ 特征已保存到: {output_file}")
    print(f"  特征维度: {list(esm2_features.values())[0].shape}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python precompute_ESM2_features.py [dataset] [batch_size]")
        print("示例: python precompute_ESM2_features.py davis 4")
    else:
        dataset = sys.argv[1]
        b_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        precompute_esm2_features(dataset, batch_size=b_size)
