"""
数据转换脚本 - 支持预计算 ESM2 特征 (适配 3B/15B 模型)
运行方法: python convert_pytorch_data.py --use-precomputed-esm2
修正说明: 
1. 标签列名已固定为 'affinity'
2. 数据集列表更新为 ['Metz', 'kiba', 'davis', 'bindingdb']
"""
import pandas as pd
import numpy as np
import os
import pickle
import argparse

from utils import TestbedDataset, seq_cat


def check_precomputed_features(dataset_name):
    """检查预计算特征是否存在 (优先级: 5120 > 2560 > 1280)"""
    files_to_check = [
        f'data/esm2_features_{dataset_name}_5120.pkl', # 15B
        f'data/esm2_features_{dataset_name}_2560.pkl', # 3B
        f'data/esm2_features_{dataset_name}_1280.pkl'  # 650M
    ]
    
    for f in files_to_check:
        if os.path.exists(f):
            print(f"  ✓ 检测到特征文件: {f}")
            return True
            
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-precomputed-esm2', action='store_true',
                        help='使用预计算的ESM2特征')
    args = parser.parse_args()

    print("=" * 60)
    print("数据转换脚本 (Multi-Dataset Support)")
    print("=" * 60)

    # 加载smile_graph.pkl
    print("\n正在加载 smile_graph.pkl...")
    try:
        with open("smile_graph.pkl", "rb") as f:
            smile_graph = pickle.load(f)
        print(f"✓ smile_graph.pkl 加载成功 ({len(smile_graph)} 个分子)")
    except FileNotFoundError:
        print("✗ 错误: 未找到 'smile_graph.pkl'!")
        print("  请先运行您的 `drug_graph_construct.py` 脚本来生成此文件。")
        exit(1)

    # 计算全局特征统计
    print("\n正在计算全局 BDE/BL 统计数据...")
    all_bdes = []
    all_bls = []
    for smile in smile_graph:
        num_edges_half = len(smile_graph[smile][2]) // 2
        all_bdes.extend(smile_graph[smile][2][:num_edges_half])
        all_bls.extend(smile_graph[smile][3][:num_edges_half])

    all_bdes = np.array(all_bdes)
    bde_mean = np.mean(all_bdes)
    bde_std = np.std(all_bdes)

    epsilon = 1e-8
    all_inv_bls = 1.0 / (np.array(all_bls) + epsilon)
    inv_bl_mean = np.mean(all_inv_bls)
    inv_bl_std = np.std(all_inv_bls)

    feature_stats = {
        'bde': {'mean': bde_mean, 'std': bde_std},
        'inv_bl': {'mean': inv_bl_mean, 'std': inv_bl_std}
    }
    print(f"  BDE (mean, std): ({bde_mean:.2f}, {bde_std:.2f})")
    print(f"  Inv-BL (mean, std): ({inv_bl_mean:.2f}, {inv_bl_std:.2f})")

    # [修改] 更新数据集列表
    datasets = ['Metz', 'kiba', 'davis', 'bindingdb']
    
    for dataset in datasets:
        print(f"\n{'=' * 60}")
        print(f"处理数据集: {dataset}")
        print(f"{'=' * 60}")

        processed_data_file_train = f'data/processed/{dataset}_train.pt'
        processed_data_file_test = f'data/processed/{dataset}_test.pt'

        os.makedirs('data/processed', exist_ok=True)

        # [检查] 预计算特征
        if args.use_precomputed_esm2:
            if not check_precomputed_features(dataset):
                print(f"⚠ 警告: 未找到任何 ESM2 特征文件 (5120/2560/1280).pkl")
                print(f"  请先为 {dataset} 运行预计算脚本 (precompute_ESM2_features.py)。")
                print(f"  跳过 {dataset} ...")
                continue

        # 检查原始CSV文件
        train_csv = f'data/{dataset}_train.csv'
        test_csv = f'data/{dataset}_test.csv'
        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            print(f"✗ 错误: 未找到 {train_csv} 或 {test_csv}")
            print(f"  跳过 {dataset} ...")
            continue

        if ((not os.path.isfile(processed_data_file_train)) or
                (not os.path.isfile(processed_data_file_test))):

            print(f"\n读取CSV文件...")
            # 读取训练集
            df_train = pd.read_csv(train_csv)
            train_drugs = list(df_train['compound_iso_smiles'])
            train_prots_raw = list(df_train['target_sequence'])
            
            # [修正] 使用 'affinity'
            try:
                train_Y = list(df_train['affinity'])
            except KeyError:
                print(f"✗ 错误: 在 {train_csv} 中找不到 'affinity' 列！")
                print(f"  现有列名: {list(df_train.columns)}")
                continue

            XT_train = [seq_cat(t) for t in train_prots_raw]
            train_drugs = np.asarray(train_drugs)
            train_prots = np.asarray(XT_train)
            train_Y = np.asarray(train_Y)

            # 读取测试集
            df_test = pd.read_csv(test_csv)
            test_drugs = list(df_test['compound_iso_smiles'])
            test_prots_raw = list(df_test['target_sequence'])
            
            # [修正] 使用 'affinity'
            try:
                test_Y = list(df_test['affinity']) 
            except KeyError:
                print(f"✗ 错误: 在 {test_csv} 中找不到 'affinity' 列！")
                continue

            XT_test = [seq_cat(t) for t in test_prots_raw]
            test_drugs = np.asarray(test_drugs)
            test_prots = np.asarray(XT_test)
            test_Y = np.asarray(test_Y)

            print(f'  训练集: {len(train_drugs)} 样本')
            print(f'  测试集: {len(test_drugs)} 样本')

            # 调用 TestbedDataset
            print(f'\n正在准备 {dataset}_train.pt ...')
            train_data = TestbedDataset(
                root='data',
                dataset=dataset + '_train',
                xd=train_drugs,
                xt=train_prots,
                y=train_Y,
                smile_graph=smile_graph,
                feature_stats=feature_stats,
                use_precomputed_esm2=args.use_precomputed_esm2
            )

            print(f'正在准备 {dataset}_test.pt ...')
            test_data = TestbedDataset(
                root='data',
                dataset=dataset + '_test',
                xd=test_drugs,
                xt=test_prots,
                y=test_Y,
                smile_graph=smile_graph,
                feature_stats=feature_stats,
                use_precomputed_esm2=args.use_precomputed_esm2
            )

            print(f'✓ {processed_data_file_train} 和 {processed_data_file_test} 已创建')
        else:
            print(f'  文件 {processed_data_file_train} 已存在，跳过生成。')
            print(f'  (如需重新生成，请先删除 data/processed/ 下的文件)')

    print("\n" + "=" * 60)
    print("所有任务完成!")
    print("=" * 60)

    if args.use_precomputed_esm2:
        print("\n✓ 数据已配置为使用预计算ESM2特征")
    else:
        print("\n✓ 数据已配置为标准模式")

if __name__ == "__main__":
    main()